from typing import Optional, Union
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from simulation_interface import FLORISFarmSim
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import networkx as nx
import torch

class RepositioningEnv(gym.Env):
    '''
    class for a Gym compatible FOWF repositioning environment that captures
    steady state behavior using wrapper classes for wind farm simulators
    '''
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(
            self,
            simulation_interface:Union[FLORISFarmSim]=None, # can add more sims
            yaw_bounds:list|tuple=(-20, 20),
            # max_num_steps:int=int(1e6)
    ):
        
        # set simulation interface and farm information
        self.SI = simulation_interface
        n_turbines = self.SI.n_turbines

        # set environment space parameters
        self.yaw_bounds = yaw_bounds
        # self.max_num_steps = max_num_steps

        # initialize learning loop values
        self.steps = 0
        self.episodic_reward = 0
        self.sequenced_yaws = np.full(n_turbines, np.nan)
        self.committed_turbines = 0
        self.committed_mask = np.zeros(n_turbines, dtype=bool)
        self.committed_values = np.zeros(n_turbines)

        # define environment spaces
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3+6*n_turbines,), 
            dtype=np.float64
        )

        # set state vector contains:
        #           ws, wd, ws_t, layout_x, layout_y, op., assigned, value
        # shape is: 2+6*N, observation is 3+6*N because sin/cos angle embedding
        self.state = np.zeros((2 + 6*n_turbines))

    def scale_action(
            self,
            normalized_action
    ):
        '''
        action space is [-1, 1], but want to scale to match yaw bounds
        
        :param normalized_action: yaw misalignment in normalized form [-1, 1]
        :return action: yaw misalignment in [lb, ub]
        '''
        low, high = self.yaw_bounds
        action = low + (normalized_action + 1) * 0.5 * (high - low)
        return action
    
    def _get_sorted_turbines(self):
        '''
        use freestream wind direction and turbine neutral positions to create
        sequence order

        :return sequence_order: wind direction aligned ordering over turbines

        NOTE: use of neutral positions neglects repositioning effects; may cause
        issues in some cases
        NOTE: turbine (x,y) position in farm used to break ties via lexsort
        '''
        wind_dir = np.deg2rad(270 - self.SI.ep_wind_dir)
        wind_vec = np.array([np.cos(wind_dir), np.sin(wind_dir)])

        neutral_x, neutral_y = self.SI.neutral_positions
        neutral_layout = self.SI.neutral_positions.T
        projected_positions = neutral_layout @ wind_vec
        return np.lexsort((neutral_y, neutral_x, projected_positions))
    
    def _parse_state_vec(
            self,
            state=None
    ):
        '''
        parse state vec into "meaningful" segments which are easier to work with
    
        :param state: full state vector
        :return parsed_state: tuple of sub-state vectors
        '''

        if state is None:
            state = self.state

        n_turbines = self.SI.n_turbines
        freestream_speed = state[0]
        freestream_direction = state[1]
        turbine_wind_speeds = state[2:2+n_turbines]
        turbine_reloc_x = state[2+n_turbines:2+2*n_turbines]
        turbine_reloc_y = state[2+2*n_turbines:2+3*n_turbines]
        turbine_available = state[2+3*n_turbines:2+4*n_turbines]
        turbine_committed = state[2+4*n_turbines:2+5*n_turbines]
        committed_values = state[2+5*n_turbines:]

        return(
            freestream_speed,
            freestream_direction,
            turbine_wind_speeds,
            turbine_reloc_x,
            turbine_reloc_y,
            turbine_available,
            turbine_committed,
            committed_values
        )
    
    def _normalize_state_vec(
            self,
            state=None,
            for_graph=False,
            vec=False
    ):
        '''
        handles normalizing state using wind farm specific values to aid in NN
        training 
        
        :param state: regular state vector
        :type state: np.ndarray
        :param for_graph: whether the normalized state will be used to construct
        a graph representation of state
        :type for_graph: bool
        :param vec: whether the normalized state needs to be a vector (when
        returned as an observation)
        :type vec: bool
        '''
        
        # parse state
        if state is None:
            state = self.state
        parsed_state = self._parse_state_vec(state)
        freestream_speed = parsed_state[0]
        freestream_direction = parsed_state[1]
        turbine_wind_speeds = parsed_state[2]
        turbine_reloc_x = parsed_state[3]
        turbine_reloc_y = parsed_state[4]
        turbine_available = parsed_state[5]
        turbine_committed = parsed_state[6]
        committed_values = parsed_state[7]

        # spacial normalization by farm extents
        neutral_x, neutral_y = self.SI.neutral_positions
        x_extent = np.max((neutral_x.max(), np.abs(neutral_x.min()), 1))
        y_extent = np.max((neutral_y.max(), np.abs(neutral_y.min()), 1))

        # other normalizing values
        v_rated = 11.4 #NOTE: assuming homogeneous turbines -> one v_rated
        yaw_bound = 20 #NOTE: ideal to extract from env, how to handle non symm.

        # (wind direction, speed) <- sin(wd), cos(wd), speed / v_rated
        freestream_direction = np.deg2rad(270 - freestream_direction) # to cart.
        wd_sin, wd_cos = np.sin(freestream_direction), np.cos(freestream_direction)
        freestream_speed = freestream_speed / v_rated

        # positions <- positions / farm extents
        neutral_x = neutral_x / x_extent
        neutral_y = neutral_y / y_extent
        turbine_reloc_x = turbine_reloc_x / x_extent
        turbine_reloc_y = turbine_reloc_y / y_extent

        # turbine wind speeds <- wind speed / v_rated
        turbine_wind_speeds = turbine_wind_speeds / v_rated

        # committed yaw values <- yaw value / yaw bound
        committed_values = committed_values / yaw_bound

        normalized_state = (
            freestream_speed,
            wd_sin,
            wd_cos,
            turbine_wind_speeds,
            turbine_reloc_x,
            turbine_reloc_y,
            turbine_available,
            turbine_committed,
            committed_values
        )

        if for_graph: # need extra info for graph construction
            normalizing_info = {
                'x_extent': x_extent,
                'y_extent': y_extent,
                'v_rated': v_rated,
                'yaw_bound': yaw_bound
            }
            return normalized_state, normalizing_info
        
        if vec: # return numpy array instead of parsed tuple
            normalized_state = np.concatenate(
                ([freestream_speed], [wd_sin], [wd_cos], turbine_wind_speeds,
                 turbine_reloc_x, turbine_reloc_y, turbine_available,
                 turbine_committed, committed_values)
            )
            return normalized_state
        
        else: # return the parsed tuple version of state
            return normalized_state
        
    def _update_state_vec(
            self,
            observation
    ):
        '''
        Updates state vector using observed values and sequence book keeping 
        states.
        
        :param observation: vector of observed values from simulation interface
        '''
        return np.concatenate(
            (observation, self.SI.available_turbines.astype(int), 
             self.committed_mask, self.committed_values)
        )

    def reset(
            self,
            seed:Optional[int]=None,
            options:dict={}
    ):
        '''
        reset the environment with random or specified freestream and f_op

        :param seed: seed used for rng
        :type seed: Optional[int]
        :param options: freestream and f_op options
        :type options: dict
        '''
        super().reset(seed=seed, options=options) # gym env reset

        n_turbines = self.SI.n_turbines

        # reset steps and yaw sequence
        self.steps = 0
        self.sequenced_yaws = np.full(n_turbines, np.nan)
        self.committed_turbines = 0
        self.committed_mask = np.zeros(n_turbines, dtype=bool)
        self.committed_values = np.zeros(n_turbines)
        
        # reset wind farm interface NOTE: can definitely clean this up...
        if 'n_active_turbines' in options:
            n_active_turbines = options['n_active_turbines']
            n_disabled_turbines = n_turbines - n_active_turbines
        else:
            n_disabled_turbines = np.random.randint(low=0, high=10+1)
            # n_disabled_turbines = 0

        if 'wind_direction' in options:
            wind_direction = options['wind_direction']
        else:
            wind_direction = np.random.uniform(265, 275)
            # wind_direction = 270

        if 'wind_speed' in options:
            wind_speed = options['wind_speed']
        else:
            wind_speed = np.random.uniform(8, 11)
            # wind_speed = 10
        
        if 'available_turbines' in options: # testing same f_op diff. controller
            available_idx = options['available_turbines']
            available_turbines = np.zeros(n_turbines, dtype=bool)
            available_turbines[available_idx] = True
        else: # randomly assign which turbines are active to create f_op
            available_turbines = np.concatenate(
                (np.zeros(n_disabled_turbines), 
                 np.ones(n_turbines - n_disabled_turbines))
            ).astype(bool)
            np.random.shuffle(available_turbines)

        # reset simulator with new f_op and freestream condition
        self.SI.reset_simulator(
            wind_direction=wind_direction,
            wind_speed=wind_speed,
            available_turbines=available_turbines
        )

        # compute the greedy baseline reward
        self.greedy_power = self.SI.get_powers().sum()

        # observe the environment
        observation = self.SI.make_observation()
        
        # committed values nan -> 0 for active turbines
        self.committed_values = np.where(
            self.committed_mask, self.sequenced_yaws, 0
        )

        # assign yaw angle sequencing order for active turbines
        full_sequence_order = self._get_sorted_turbines()
        available_turbines = self.SI.available_turbines
        sorted_available_turbines = available_turbines[full_sequence_order]
        self.sequence_order = full_sequence_order[sorted_available_turbines]

        # update state
        # self.state = np.concatenate(
        #     (observation, self.SI.available_turbines.astype(int),
        #      self.committed_mask, self.committed_values)
        # )
        self.state = self._update_state_vec(observation)

        self.episodic_reward = 0
        self.graph = self.build_graph()
        normalized_state = self._normalize_state_vec(vec=True)
        info = {}
        return normalized_state, info

    def step(
            self,
            action
    ):
        '''
        take one step in the sequence over active turbines with the provided
        action

        :param action: the nacelle yaw misalignment to be assigned to the
        current turbine in the sequence
        '''

        # get current position in the sequence over f_op
        turbine = self.sequence_order[self.committed_turbines]
        # update book keeping for sequence
        self.sequenced_yaws[turbine] = self.scale_action(action).squeeze()
        self.committed_mask[turbine] = True
        self.committed_turbines += 1
        self.steps += 1

        # check if the full yaw sequence is complete in this step
        if self.committed_turbines == self.SI.available_turbines.sum():
            # take ``large`` step including repositioning + wake solve
            info = {}
            action = self.sequenced_yaws # gamma_op
            self.SI.take_step(action)

            powers = self.SI.get_powers()
            self.repos_power = powers.sum() # repositioning control farm power

            # observe environment
            observation = self.SI.make_observation()

            # update state
            self.committed_values = np.where(
                self.committed_mask, self.sequenced_yaws, 0
            ) # fill nans with 0 
            self.state[:observation.shape[0]] = observation
            self.state[-3*self.SI.n_turbines:] = np.concatenate(
                (self.SI.available_turbines.astype(int), self.committed_mask,
                 self.committed_values)
            )

            truncated = False
            terminated = True

            # save repositioning controlled powers without wake
            self.SI.sim.run_no_wake()
            self.SI.repos_no_wake_power = self.SI.get_powers().sum()

            # compute reward
            penalty = np.linalg.norm(self.sequenced_yaws[turbine] / 20)
            r_RL = self.repos_power / self.SI.greedy_no_wake_power
            r_greedy = self.greedy_power / self.SI.greedy_no_wake_power

            reward = 2*(r_RL-r_greedy) + (2/3)*(r_greedy-1) - 0.01*penalty
            self.episodic_reward += reward

            info['faero_x'] = self.SI.faero_x
            info['faero_y'] = self.SI.faero_y
            info['episodic_reward'] = self.episodic_reward
            info['penalty'] = penalty
            info['r_RL'] = r_RL
            info['r_greedy'] = r_greedy

        else: # not done sequencing over f_op, state update is just book keeping
            self.committed_values = np.where(
                self.committed_mask, self.sequenced_yaws, 0
            ) # fill nans with 0

            self.state[-3*self.SI.n_turbines:] = np.concatenate(
                (self.SI.available_turbines.astype(int), self.committed_mask,
                 self.committed_values)
            )

            truncated = False
            terminated = False
            
            # compute reward (just penalty)
            penalty = np.linalg.norm(self.sequenced_yaws[turbine] / 20)
            reward = -0.01*penalty
            self.episodic_reward += reward

            info = {}
        # NOTE: could simplify state update to break redundant parts out of ifs

        # update graph
        self.graph = self.build_graph()

        normalized_state = self._normalize_state_vec(vec=True)
        return normalized_state, reward, terminated, truncated, info

    def build_graph(self):
        '''
        build the graph representation of the system state
        '''
        # read in current state and normalization information
        parsed_state, norm_info = self._normalize_state_vec(for_graph=True)
        freestream_speed = parsed_state[0]
        wd_sin = parsed_state[1]
        wd_cos = parsed_state[2]
        turbine_wind_speeds = parsed_state[3]
        repos_x = parsed_state[4]
        repos_y = parsed_state[5]
        turbine_available = parsed_state[6]
        committed_mask = parsed_state[7]
        committed_values = parsed_state[8]

        x_extent = norm_info['x_extent']
        y_extent = norm_info['y_extent']

        # get neutral positions
        neutral_x, neutral_y = self.SI.neutral_positions

        # get information about currently available turbines
        available_idx = np.nonzero(turbine_available)[0]
        n_available_turbines = self.get_n_available_turbines()

        # build adjacency
        row, col = torch.meshgrid(
            torch.arange(n_available_turbines), 
            torch.arange(n_available_turbines), 
            indexing='ij'
        )
        edge_index = torch.stack([row.reshape(-1), col.reshape(-1)], dim=0)
        self_loop_mask = edge_index[0] == edge_index[1]
        edge_index = edge_index[:, ~self_loop_mask]

        # build vertex features
        freestream_condition = torch.tensor([freestream_speed, wd_sin, wd_cos])
        # extend to include at every vertex NOTE: global graph features better...
        freestream_condition = freestream_condition.repeat(n_available_turbines, 1)

        # normalize neutral positions
        neutral_x = neutral_x / x_extent
        neutral_y = neutral_y / y_extent

        # wind direction projections
        wind_vector = np.array([wd_cos, wd_sin])
        wind_vector_n = np.array([-wd_sin, wd_cos])

        projected_neutral_positions = (
            np.vstack([neutral_x, neutral_y]).T @ wind_vector
        )
        projected_repositions = (
            np.vstack([repos_x, repos_y]).T @ wind_vector
        )

        projected_neutral_positions_normal = (
            np.vstack([neutral_x, neutral_y]).T @ wind_vector_n
        )
        projected_repositions_normal = (
            np.vstack([repos_x, repos_y]).T @ wind_vector_n
        )
        # NOTE: can clean the above by redefining neutral layout after
        #       normalizing and define a reposition layout matrix as well. Then
        #       don't need to construct inside inner product each time...

        # remove deactivated turbines from state
        turbine_wind_speeds = turbine_wind_speeds[available_idx]
        neutral_x = neutral_x[available_idx]
        neutral_y = neutral_y[available_idx]
        repos_x = repos_x[available_idx]
        repos_y = repos_y[available_idx]
        projected_neutral_positions = projected_neutral_positions[available_idx]
        projected_repositions = projected_repositions[available_idx]
        projected_neutral_positions_normal = projected_neutral_positions_normal[available_idx]
        projected_repositions_normal = projected_repositions_normal[available_idx]
        committed_mask = committed_mask[available_idx]
        committed_values = committed_values[available_idx]

        turbine_features = np.vstack(
            (turbine_wind_speeds, repos_x, repos_y, neutral_x, neutral_y,
             projected_neutral_positions, projected_repositions, committed_mask,
             committed_values)
        ).T
        turbine_features = torch.from_numpy(turbine_features)
        vertex_features = torch.cat(
            (freestream_condition, turbine_features), dim=1
        ).to(torch.float)
        # NOTE: ideal to change so that graph uses freestream globally instead
        #       of redundant info at every vertex
        
        # build edge features
        wd_neutral_distances = (
            projected_neutral_positions[:, None] - projected_neutral_positions
        )
        wd_repos_distances = (
            projected_repositions[:, None] - projected_repositions
        )
        wd_n_neutral_distances = (
            projected_neutral_positions_normal[:, None] - projected_neutral_positions_normal
        )
        wd_n_repos_distances = (
            projected_repositions_normal[:, None] - projected_repositions_normal
        )
        wd_neutral_distances = torch.tensor(
            wd_neutral_distances, dtype=torch.float
        ).reshape(-1, 1)
        wd_repos_distances = torch.tensor(
            wd_repos_distances, dtype=torch.float
        ).reshape(-1, 1)
        wd_n_neutral_distances = torch.tensor(
            wd_n_neutral_distances, dtype=torch.float
        ).reshape(-1, 1)
        wd_n_repos_distances = torch.tensor(
            wd_n_repos_distances, dtype=torch.float
        ).reshape(-1, 1)
        # NOTE: ``distances`` are relative position vectors between active turbines

        edge_attr = torch.cat(
            (wd_neutral_distances, wd_repos_distances, wd_n_neutral_distances, 
             wd_n_repos_distances), dim=-1
        )
        edge_attr = edge_attr[~self_loop_mask]

        # remove upstream connections to committed turbines
        upstream_sequence = self.sequence_order[:self.committed_turbines]
        upstream_sequence = np.where(
            np.isin(available_idx, upstream_sequence)
        )[0]
        upstream_limit_mask = ~torch.isin(
            edge_index[1], torch.from_numpy(upstream_sequence)
        )
        edge_index = edge_index.T[upstream_limit_mask].T
        edge_attr = edge_attr[upstream_limit_mask]
        # NOTE: way overcomplicated, can just check state for committed flag

        # build graph data object
        data = Data(
            x=vertex_features, edge_index=edge_index, edge_attr=edge_attr
        )
        return data

    # def inspect_graph()
        
    def get_n_available_turbines(self):
        return self.SI.available_turbines.sum()
        
    def get_step_mask(self):
        '''
        Return a mask over the graph with vertices = f_op that isolates the
        vertex corresponding to the current turbine in the sequence
        '''
        available_turbines = self.SI.available_turbines
        turbine = self.sequence_order[self.committed_turbines]
        available_idx = np.nonzero(available_turbines)[0]

        return available_idx == turbine
             
    # def graph_viz()

class RepositioningEnvFullFarm(RepositioningEnv):

    def __init__(
            self,
            simulation_interface:Union[FLORISFarmSim]=None,
            yaw_bounds:list|tuple=(-20,20)
    ):
        
        # Init general repositioning environment
        super().__init__(simulation_interface, yaw_bounds)

        n_turbines = self.SI.n_turbines

        # When only considering the full farm (during training), the z_op state
        # component no longer makes sense, as it is always an N-dim 1 vector. 
        # We opt to remove it from state/observations
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3+5*n_turbines,), # from 3+6N to 3+5N
            dtype=np.float64
        )

        # set state vector contains:
        #           ws, wd, ws_t, layout_x, layout_y, assigned, value
        # shape is: 2+5*N, observation is 3+5*N because sin/cos angle embedding
        self.state = np.zeros((2+5*n_turbines))

    def _parse_state_vec(
            self, 
            state=None
    ):
        # We need to adjust to not include z_op in state parsing
        '''
        parse state vec into "meaningful" segments which are easier to work with
    
        :param state: full state vector
        :return parsed_state: tuple of sub-state vectors
        '''

        if state is None:
            state = self.state
        
        n_turbines = self.SI.n_turbines
        freestream_speed = state[0]
        freestream_direction = state[1]
        turbine_wind_speeds = state[2:2+n_turbines]
        turbine_reloc_x = state[2+n_turbines:2+2*n_turbines]
        turbine_reloc_y = state[2+2*n_turbines:2+3*n_turbines]
        turbine_committed = state[2+3*n_turbines:2+4*n_turbines]
        committed_values = state[2+4*n_turbines:]

        return(
            freestream_speed,
            freestream_direction,
            turbine_wind_speeds,
            turbine_reloc_x,
            turbine_reloc_y,
            turbine_committed,
            committed_values
        )
    
    def _normalize_state_vec(
            self,
            state=None,
            for_graph=False,
            vec=False
    ):
        # We need to adjust to not include z_op in state normalization
        '''
        handles normalizing state using wind farm specific values to aid in NN
        training 
        
        :param state: regular state vector
        :type state: np.ndarray
        :param for_graph: whether the normalized state will be used to construct
        a graph representation of state
        :type for_graph: bool
        :param vec: whether the normalized state needs to be a vector (when
        returned as an observation)
        :type vec: bool
        '''

        # parse state
        if state is None:
            state = self.state
        parsed_state = self._parse_state_vec(state)
        freestream_speed = parsed_state[0]
        freestream_direction = parsed_state[1]
        turbine_wind_speeds = parsed_state[2]
        turbine_reloc_x = parsed_state[3]
        turbine_reloc_y = parsed_state[4]
        turbine_committed = parsed_state[5]
        committed_values = parsed_state[6]

        # spacial normalization by farm extents
        neutral_x, neutral_y = self.SI.neutral_positions
        x_extent = np.max((neutral_x.max(), np.abs(neutral_x.min()), 1))
        y_extent = np.max((neutral_y.max(), np.abs(neutral_y.min()), 1))

        # other normalizing values
        v_rated = 11.4 #NOTE: assuming homogenous turbines -> one v_rated
        yaw_bound = 20 #NOTE: ideal to extract from env, how to handle non symm.

        # (wind direction, speed) <- sin(wd), cos(wd), speed / v_rated
        freestream_direction = np.deg2rad(270 - freestream_direction) # to cart.
        wd_sin, wd_cos = np.sin(freestream_direction), np.cos(freestream_direction)
        freestream_speed  = freestream_speed / v_rated

        # positions <- positions / farm extents
        neutral_x = neutral_x / x_extent
        neutral_y = neutral_y / y_extent
        turbine_reloc_x = turbine_reloc_x / x_extent
        turbine_reloc_y = turbine_reloc_y / y_extent

        # turbine wind speeds <- wind speed / v_rated
        turbine_wind_speeds = turbine_wind_speeds / v_rated

        # committed yaw values <- yaw value / yaw_bound
        committed_values = committed_values / yaw_bound

        normalized_state = (
            freestream_speed,
            wd_sin,
            wd_cos,
            turbine_wind_speeds,
            turbine_reloc_x,
            turbine_reloc_y,
            turbine_committed,
            committed_values
        )
        
        if for_graph: # need extra info for graph construction
            normalizing_info = {
                'x_extent': x_extent,
                'y_extent': y_extent,
                'v_rated': v_rated,
                'yaw_bound': yaw_bound
            }
            return normalized_state, normalizing_info
        
        if vec: # return numpy array instead of parsed tuple
            normalized_state = np.concatenate(
                ([freestream_speed], [wd_sin], [wd_cos], turbine_wind_speeds,
                 turbine_reloc_x, turbine_reloc_y, turbine_committed,
                 committed_values)
            )
            return normalized_state
        
        else: # return the parsed tuple version of state
            return normalized_state
        
    def _update_state_vec(
            self, 
            observation
    ):
        '''
        Updates state vector using observed values and sequence book keeping 
        states. This method varies from that of the F_op-Aware version, as we
        no longer include the z_op element of the state.
        
        :param observation: vector of observed values from simulation interface
        '''
        return np.concatenate(
            (observation, self.committed_mask, self.committed_values)
        )
    
    def reset(
            self,
            seed:Optional[int]=None,
            options:dict={}
    ):
        # need to change sequence order to be over the full farm, not just F_op
        '''
        reset the environment with random or specified freestream and F_op

        :param seed: seed used for rng
        :type seed: Optional[int]
        :param options: freestream and F_op options
        :type options: dict
        '''
        obs, info = super().reset(seed=seed, options=options)

        full_sequence_order = self._get_sorted_turbines()
        self.sequence_order = full_sequence_order
        return obs, info
        
        
    def step(
            self,
            action
    ):
        # Due to messy code, need to re-implement step to adjust state updates
        '''
        take one step in the sequence over active turbines with the provided
        action

        :param action: the nacelle yaw misalignment to be assigned to the
        current turbine in the sequence
        '''

        # get current position in the sequence over f_op
        turbine = self.sequence_order[self.committed_turbines]
        # update book keeping for sequence
        self.sequenced_yaws[turbine] = self.scale_action(action).squeeze()
        self.committed_mask[turbine] = True
        self.committed_turbines += 1
        self.steps += 1

        # check if the full yaw sequence is complete in this step
        if self.committed_turbines == self.SI.n_turbines: # all assumed op.
            # take ``large`` step including repositioning + wake solve
            info = {}
            action = self.sequenced_yaws # gamma_op
            self.SI.take_step(action)

            powers = self.SI.get_powers()
            self.repos_power = powers.sum() # repositioning control farm power

            # observe environment
            observation = self.SI.make_observation()

            # update state
            self.committed_values = np.where(
                self.committed_mask, self.sequenced_yaws, 0
            )  # full nans with 0
            self.state[:observation.shape[0]] = observation
            self.state[-2*self.SI.n_turbines:] = np.concatenate(
                (self.committed_mask, self.committed_values)
            )

            truncated = False
            terminated = True

            # save repositioning controlled powers without wake
            self.SI.sim.run_no_wake()
            self.SI.repos_no_wake_power = self.SI.get_powers().sum()

            # compute reward
            penalty = np.linalg.norm(self.sequenced_yaws[turbine] / 20)
            r_RL = self.repos_power / self.SI.greedy_no_wake_power
            r_greedy = self.greedy_power / self.SI.greedy_no_wake_power

            reward = 2*(r_RL-r_greedy) + (2/3)*(r_greedy-1) - 0.01*penalty
            self.episodic_reward += reward

            info['faero_x'] = self.SI.faero_x
            info['faero_y'] = self.SI.faero_y
            info['episodic_reward'] = self.episodic_reward
            info['penalty'] = penalty
            info['r_RL'] = r_RL
            info['r_greedy'] = r_greedy

        else: # not done sequencing over f_op, state update is just book keeping
            self.committed_values = np.where(
                self.committed_mask, self.sequenced_yaws, 0
            ) # fill nans with 0

            self.state[-2*self.SI.n_turbines:] = np.concatenate(
                (self.committed_mask, self.committed_values)
            )

            truncated = False
            terminated = False

            # compute reward (just penalty)
            penalty = np.linalg.norm(self.sequenced_yaws[turbine] / 20)
            reward = -0.01*penalty
            self.episodic_reward += reward

            info = {}
        # NOTE: could simplify state update to break redundant parts out of ifs

        # update graph
        self.graph = self.build_graph()

        normalized_state = self._normalize_state_vec(vec=True)
        return normalized_state, reward, terminated, truncated, info
    
    def build_graph(self):
        # similarly need to re-implement graph construction to handle new 
        # form of state...
        '''
        build the graph representation of the system state
        '''
        # read in current state and normalization information
        parsed_state, norm_info = self._normalize_state_vec(for_graph=True)
        freestream_speed = parsed_state[0]
        wd_sin = parsed_state[1]
        wd_cos = parsed_state[2]
        turbine_wind_speeds = parsed_state[3]
        repos_x = parsed_state[4]
        repos_y = parsed_state[5]
        committed_mask = parsed_state[6]
        committed_values = parsed_state[7]

        x_extent = norm_info['x_extent']
        y_extent = norm_info['y_extent']

        # get neutral positions
        neutral_x, neutral_y = self.SI.neutral_positions

        # get information about currently available turbines (should be all 
        # during training, but vary in testing/deployment)
        n_available_turbines = self.get_n_available_turbines()
        n_turbines = self.SI.n_turbines

        # build adjacency
        row, col = torch.meshgrid(
            torch.arange(n_turbines),
            torch.arange(n_turbines),
            indexing='ij'
        )
        edge_index = torch.stack([row.reshape(-1), col.reshape(-1)], dim=0)
        self_loop_mask = edge_index[0] == edge_index[1]
        edge_index = edge_index[:, ~self_loop_mask]

        # build vertex features
        freestream_condition = torch.tensor([freestream_speed, wd_sin, wd_cos])
        # extend to include at every vertex NOTE: global graph features better...
        freestream_condition = freestream_condition.repeat(n_turbines, 1)

        # normalize neutral positions
        neutral_x = neutral_x / x_extent
        neutral_y = neutral_y / y_extent

        # wind direction projections
        wind_vector = np.array([wd_cos, wd_sin])
        wind_vector_n = np.array([-wd_sin, wd_cos])

        projected_neutral_positions = (
            np.vstack([neutral_x, neutral_y]).T @ wind_vector
        )
        projected_repositions = (
            np.vstack([repos_x, repos_y]).T @ wind_vector
        )
        projected_neutral_positions_normal = (
            np.vstack([neutral_x, neutral_y]).T @ wind_vector_n
        )
        projected_repositions_normal = (
            np.vstack([repos_x, repos_y]).T @ wind_vector_n
        )
        # NOTE: can clean the above by redefining neutral layout after
        #       normalizing and define a reposition layout matrix as well. Then
        #       don't need to construct inside inner product each time...

        # don't need to ``remove deactivated turbines from state`` as full farm
        # is active...
        
        turbine_features = np.vstack(
            (turbine_wind_speeds, repos_x, repos_y, neutral_x, neutral_y,
             projected_neutral_positions, projected_repositions, committed_mask,
             committed_values)
        ).T
        turbine_features = torch.from_numpy(turbine_features)
        vertex_features = torch.cat(
            (freestream_condition, turbine_features), dim=1
        ).to(torch.float)
        # NOTE: ideal to change so that graph uses freestream globally instead
        #       of redundant info at every vertex

        # build edge features
        wd_neutral_distances = (
            projected_neutral_positions[:, None] - projected_neutral_positions
        )
        wd_repos_distances = (
            projected_repositions[:, None] - projected_repositions
        )
        wd_n_neutral_distances = (
            projected_neutral_positions_normal[:, None] - projected_neutral_positions_normal
        )
        wd_n_repos_distances = (
            projected_repositions_normal[:, None] - projected_repositions_normal
        )
        wd_neutral_distances = torch.tensor(
            wd_neutral_distances, dtype=torch.float
        ).reshape(-1, 1)
        wd_repos_distances = torch.tensor(
            wd_repos_distances, dtype=torch.float
        ).reshape(-1, 1)
        wd_n_neutral_distances = torch.tensor(
            wd_n_neutral_distances, dtype=torch.float
        ).reshape(-1, 1)
        wd_n_repos_distances = torch.tensor(
            wd_n_repos_distances, dtype=torch.float
        ).reshape(-1, 1)
        # NOTE: ``distances`` are relative position vectors between active turbines

        edge_attr = torch.cat(
            (wd_neutral_distances, wd_repos_distances, wd_n_neutral_distances,
             wd_n_repos_distances), dim=-1
        )
        edge_attr = edge_attr[~self_loop_mask]

        # remove upstream connections to committed turbines
        upstream_sequence = self.sequence_order[:self.committed_turbines]
        upstream_sequence = np.where(
            np.isin(np.arange(n_available_turbines), upstream_sequence)
        )[0]
        upstream_limit_mask = ~torch.isin(
            edge_index[1], torch.from_numpy(upstream_sequence)
        )
        edge_index = edge_index.T[upstream_limit_mask].T
        edge_attr = edge_attr[upstream_limit_mask]
        # NOTE: way overcomplicated, can just check state for committed flag

        # build graph data object
        data = Data(
            x=vertex_features, edge_index=edge_index, edge_attr=edge_attr
        )
        return data