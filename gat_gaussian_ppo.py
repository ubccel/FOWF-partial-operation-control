# Modified Clean RL PPO for continuous action spaces implementation with 
# bare bones features and removed vectorized environments and added graph
# attention networks

from dataclasses import dataclass
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.data import Batch

import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
import random
import time
import torch
import torch_geometric
import tyro

import init_envs

# Experiment and PPo inputs
@dataclass
class Args:
    # Experiment arguments
    exp_name:str=os.path.basename(__file__)[:-len('.py')]
    '''the name of this experiment (run name is env-name + exp-name + time)'''
    seed:int=1
    '''seed of the experiment'''
    torch_deterministic:bool=True
    '''if toggled, `torch.backends.cudnn.deterministic=False'''
    cuda:bool=True
    '''if toggled, cuda will be enabled by default'''
    track:bool=False
    '''if toggled, this experiment will be tracked with Weights and Biases'''
    wandb_project_name:str='repositioning-ppo'
    '''the wandb's project name'''
    wandb_entity:str=None
    '''the entity (team) of wandb's project'''
    capture_video:bool=False
    '''whether or not to capture videos of the agent performances'''
    save_model:bool=False
    '''whether to save model into the `runs/{run_name}` folder'''
    upload_model:bool=False
    '''whether to upload the saved model to huggingface'''
    hf_entity:str=''
    '''the user or org name of the model repository from the Hugging Face Hub'''
    checkpoints:bool=True
    '''whether to save agents while training'''
    full_farm:bool=False
    '''whether or not the agent only trains on the full farm'''
    
    # Algorithm specific arguments
    env_id:str='FLORIS-repositioning-v0'
    '''the id of the environment'''
    total_timesteps:int=int(1e6)
    '''total timesteps of the experiment'''
    learning_rate:float=3e-4
    '''the learning rate of the optimizer'''
    num_steps:int=2048
    '''the number of steps to run in the environment per policy rollout'''
    anneal_lr:bool=False
    '''toggle learning rate annealing for policy and value networks'''
    gamma:float=0.99
    '''the discount factor gamma'''
    gae_lambda:float=0.95
    '''the lambda for the general advantage estimation'''
    num_minibatches:int=32
    '''the number of mini-batches'''
    update_epochs:int=10
    '''the K epochs used to update the policy'''
    norm_adv:bool=True
    '''toggles advantage normalization'''
    clip_coef:float=0.2
    '''the surrogate clipping coefficient'''
    clip_vloss:bool=True
    '''toggles whether or not to use a clipped loss for the value function'''
    ent_coef:float=0.0
    '''coefficient of the entropy'''
    vf_coef:float=0.5
    '''coefficient of the value function'''
    max_grad_norm:float=0.5
    '''the maximum norm for the gradient clipping'''
    target_kl:float=None
    '''the target KL divergence threshold'''

    # To be computed at runtime
    batch_size:int=0
    '''the batch size'''
    minibatch_size:int=0
    '''the mini-batch size'''
    num_iterations:int=0
    '''the number of iterations'''

# Function to make env and wrap with helper/utilities
def make_env(env_id, capture_video, run_name, gamma=0.99):
    if capture_video:
        raise NotImplementedError('Methods for rendering not added yet')
    
    else:
        env = gym.make(env_id)

    # Wrap the base environment with helpful Gym pre-built features
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)

    return env

# Function to outline layer initialization procedure
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

# Class for GAT-based PPO agent
class GATAgent(nn.Module):
    def __init__(
            self,
            state_dim:int=None,
            graph_feature_dim:int=None,
            gat_hidden_dim:int=64,
            gat_heads:int=1,
            actor_hidden_dim:int=64,
            value_hidden_dim:int=64,
    ):
        
        super().__init__()
        # Check presence of required dimensions
        assert state_dim is not None
        assert graph_feature_dim is not None

        # Graph attention network(s) to process graph representation of state
        # separate GATs used for actor and critic
        self.actor_GAT = nn.ModuleList([
            GATv2Conv(
                in_channels=graph_feature_dim,
                out_channels=gat_hidden_dim // gat_heads,
                edge_dim=4,
                heads=gat_heads,
                add_self_loops=False,
                concat=True,
                residual=True
            ),
            GATv2Conv(
                in_channels=gat_hidden_dim,
                out_channels=gat_hidden_dim // gat_heads,
                edge_dim=4,
                heads=gat_heads,
                add_self_loops=False,
                concat=True,
                residual=True
            )
        ])

        self.critic_GAT = nn.ModuleList([
            GATv2Conv(
                in_channels=graph_feature_dim,
                out_channels=gat_hidden_dim // gat_heads,
                edge_dim=4,
                heads=gat_heads,
                add_self_loops=False,
                concat=True,
                residual=True
            ),
            GATv2Conv(
                in_channels=gat_hidden_dim,
                out_channels=gat_hidden_dim // gat_heads,
                edge_dim=4,
                heads=gat_heads,
                add_self_loops=False,
                concat=True,
                residual=True
            )
        ])

        # Actor network
        actor_in_dim = 2*gat_hidden_dim + state_dim
        self.actor_mean = nn.Sequential(
            layer_init(
                nn.Linear(actor_in_dim, actor_hidden_dim),
            ),
            nn.Tanh(),
            layer_init(
                nn.Linear(actor_hidden_dim, actor_hidden_dim),
            ),
            nn.Tanh(),
            layer_init(
                nn.Linear(actor_hidden_dim, 1), std=0.01
            )
        )

        self.actor_logstd = nn.Parameter(
            torch.zeros(1, 1)
        )

        # Value network
        value_in_dim = 2*gat_hidden_dim + state_dim
        self.value_net = nn.Sequential(
            layer_init(
                nn.Linear(value_in_dim, value_hidden_dim),
            ),
            nn.Tanh(),
            layer_init(
                nn.Linear(value_hidden_dim, value_hidden_dim),
            ),
            nn.Tanh(),
            layer_init(
                nn.Linear(value_hidden_dim, 1), std=1.0
            )
        )


    def get_action_and_value(
            self,
            x,
            edge_index,
            edge_attr,
            batch,
            state,
            step_mask=None,
            action=None,
    ):
        
        assert step_mask is not None

        h_actor = x
        # not currently learning transformation on edge features
        for GAT_layer in self.actor_GAT:
            h_actor = GAT_layer(
                x=h_actor, edge_index=edge_index, edge_attr=edge_attr
            )
            h_actor = nn.functional.elu(h_actor)

        actor_summary = global_mean_pool(h_actor, batch)

        if state.ndim == 1:
            state = state.unsqueeze(0)
        state = state.float()

        actor_input = torch.cat(
            (actor_summary, h_actor[step_mask], state), dim=-1
        )

        action_mean = self.actor_mean(actor_input)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        self.dist = probs

        if action is None:
            action = probs.sample()

        value = self.get_value(x, edge_index, edge_attr, batch, state)

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value
    
    def get_deterministic_action(
            self,
            x,
            edge_index,
            edge_attr,
            batch,
            state,
            step_mask=None,
            action=None,
    ):
        '''
        returns the action mean instead of constructing the distribution and 
        sampling. Used for testing.
        '''
        assert step_mask is not None

        h_actor = x 
        for GAT_layer in self.actor_GAT:
            h_actor = GAT_layer(
                x=h_actor, edge_index=edge_index, edge_attr=edge_attr
            )
            h_actor = nn.functional.elu(h_actor)

        actor_summary = global_mean_pool(h_actor, batch)

        if state.ndim == 1:
            state = state.unsqueeze(0)
        state = state.float()

        actor_input = torch.cat(
            (actor_summary, h_actor[step_mask], state), dim=-1
        )
        action_mean = self.actor_mean(actor_input)

        return action_mean
    
    def get_value(
            self,
            x,
            edge_index,
            edge_attr,
            batch,
            state
    ):
        h_critic = x
        for GAT_layer in self.critic_GAT:
            h_critic = GAT_layer(
                x=h_critic, edge_index=edge_index, edge_attr=edge_attr
            )
            h_critic = nn.functional.elu(h_critic)

        critic_summary_1 = global_mean_pool(h_critic, batch)
        critic_summary_2 = global_max_pool(h_critic, batch)

        value_input = torch.cat(
            (critic_summary_1, critic_summary_2, state), dim=1
        )

        value = self.value_net(value_input)

        return value
    
# PPO training logic
if __name__ == '__main__':
    # read in args
    args = tyro.cli(Args)
    # compute runtime args
    args.batch_size = int(args.num_steps)
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f'{args.env_id}_{args.exp_name}_{args.seed}_{int(time.time())}'
    print(f'batch size: {args.batch_size}')
    print(f'minibatch size: {args.minibatch_size}')

    # initialize WandB if --track
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=False
        )

    # local logging init
    writer = SummaryWriter(f'runs/{run_name}')
    writer.add_text(
        'hyperparameters',
        '|param|value|\n|-|-|\n%s' % ('\n'.join(
            [f'|{key}|{value}' for key, value in vars(args).items()]
        ))
    )

    # check point initialization
    if args.checkpoints:
        args.best_avg_reward = -torch.inf
    else:
        args.best_avg_reward = torch.inf

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch_geometric.seed_everything(args.seed)

    # set device
    device = torch.device(
        'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'
    )

    # env setup
    env = make_env(args.env_id, args.capture_video, run_name, args.gamma)
    assert isinstance(env.action_space, gym.spaces.Box), \
         'repositioning environment requires continuous action space'

        # environment reset options initialization
    if args.full_farm:
        assert 'full-farm' in args.env_id, \
            'set to full-farm training without full-farm environment'
        options = {'n_active_turbines': env.unwrapped.SI.n_turbines}
    else:
        options = {}

    agent = GATAgent(
        state_dim=env.observation_space.shape[0],
        graph_feature_dim=12,
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    trainable_params = sum(
        p.numel() for p in agent.parameters() if p.requires_grad
    )

    # ALGO Logic: storage allocation setup
    obs = torch.zeros((args.num_steps, 1)+env.observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, 1)).to(device)
    logprobs = torch.zeros((args.num_steps, 1)).to(device)
    rewards = torch.zeros((args.num_steps, 1)).to(device)
    dones = torch.zeros((args.num_steps, 1)).to(device)
    values = torch.zeros((args.num_steps, 1)).to(device)
    # need to store the graph and current turbine index masks for each step
    graphs = [None] * args.num_steps
    masks = [None] * args.num_steps

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = env.reset(seed=args.seed, options=options)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(1).to(device)
    next_graph = env.unwrapped.graph.to(device)
    next_mask = env.unwrapped.get_step_mask()

    for iteration in range(1, args.num_iterations+1):
        # annealing the rate if set to do so
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]['lr'] = lrnow

        print('rolling out policy')
        for step in range(0, args.num_steps):
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done
            graphs[step] = next_graph
            masks[step] = next_mask

            # ALGO Logic: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    x=next_graph.x,
                    edge_index=next_graph.edge_index,
                    edge_attr=next_graph.edge_attr,
                    batch=torch.zeros(
                        (next_graph.num_nodes,), dtype=torch.long, device=device
                    ),
                    state=next_obs,
                    step_mask=next_mask
                )
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data
            next_obs, reward, termination, truncation, info = env.step(
                action=action.cpu().numpy()
            )

            next_done = torch.tensor([float(termination or truncation)], device=device)
            rewards[step] = torch.tensor(reward).to(device)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(next_done).to(device)

            if next_done:
                print('='*30)
                print(f'episode freestream: {env.unwrapped.state[0]:0.0f}, {env.unwrapped.state[1]:0.2f}')
                agent_power = env.unwrapped.repos_power
                percent_improve = (agent_power - env.unwrapped.greedy_power) / env.unwrapped.greedy_power
                print(f'episodic improvement: {percent_improve * 100:0.2f}% vs greedy')
                r_RL = info['r_RL']
                r_Greedy = info['r_greedy']
                print(f'r_RL: {r_RL:0.3f}')
                print(f'r_Greedy: {r_Greedy:0.3f}')
                print('='*30)

                next_obs, _ = env.reset(seed=args.seed, options=options)
                next_obs = torch.Tensor(next_obs).to(device)

            next_graph = env.unwrapped.graph.to(device)
            next_mask = env.unwrapped.get_step_mask()

            if 'episode' in info:
                writer.add_scalar(
                    'Charts/episodic_return', info['episode']['r'], global_step
                )
                writer.add_scalar(
                    'Charts/episodic_length', info['episode']['l'], global_step
                )
                writer.add_scalar(
                    'Charts/episodic_improvement', percent_improve, global_step
                )
                writer.add_scalar(
                    'Charts/episodic_r_RL', r_RL, global_step
                )
                writer.add_scalar(
                    'Charts/episodic_r_Greedy', r_Greedy, global_step
                )

        if rewards.mean() > args.best_avg_reward:
            args.best_avg_reward = rewards.mean().item()
            model_path = f'runs/{run_name}/{args.exp_name}_checkpoint.cleanrl_model'
            torch.save(agent.state_dict(), model_path)
            print(f'model checkpoint saved to {model_path}')

        print('updating policy...')
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(
                x=next_graph.x,
                edge_index=next_graph.edge_index,
                edge_attr=next_graph.edge_attr,
                batch=torch.zeros(
                    (next_graph.num_nodes,), dtype=torch.long, device=device
                ),
                state=next_obs.unsqueeze(0)
            ).reshape(1, -1)

            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + env.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + env.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # optimizing the policy and value networks
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                mb_states = b_obs[mb_inds].to(device)
                mb_actions = b_actions[mb_inds].to(device)
                mb_graphs_list = [graphs[i] for i in mb_inds]
                mb_graphs = Batch.from_data_list(mb_graphs_list).to(device)
                mb_masks_list = [torch.tensor(masks[i], dtype=torch.bool) for i in mb_inds]
                step_mask = torch.cat([m.to(device) for m in mb_masks_list])

                assert step_mask.shape[0] == mb_graphs.num_nodes, \
                    f'mask length {step_mask.shape[0]} != total nodes {mb_graphs.num_nodes}'
                assert mb_graphs.batch.dtype == torch.long
                assert mb_graphs.edge_index.dtype == torch.long

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    x=mb_graphs.x,
                    edge_index=mb_graphs.edge_index,
                    edge_attr=mb_graphs.edge_attr,
                    batch=mb_graphs.batch,
                    state=mb_states,
                    step_mask=step_mask,
                    action=mb_actions
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joshcu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (
                        (mb_advantages - mb_advantages.mean()) /
                        (mb_advantages.std() + 1e-8)
                    )
                
                # policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds])**2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds])**2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()

                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds])**2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            'Charts/learning_rate', optimizer.param_groups[0]['lr'], global_step
        )
        writer.add_scalar(
            'losses/value_loss', v_loss.item(), global_step
        )
        writer.add_scalar(
            'losses/entropy', entropy_loss.item(), global_step
        )
        writer.add_scalar(
            'losses/old_approx_kl', old_approx_kl.item(), global_step
        )
        writer.add_scalar(
            'losses/approx_kl', approx_kl.item(), global_step
        )
        writer.add_scalar(
            'losses/clipfrac', np.mean(clipfracs), global_step
        )
        writer.add_scalar(
            'losses/explained_variance', explained_var, global_step
        )
        SPS = int(global_step / (time.time() - start_time))
        print('SPS: ', SPS, global_step)
        writer.add_scalar(
            'Charts/SPS', SPS, global_step
        )

    if args.save_model:
        model_path = f'runs/{run_name}/{args.exp_name}.cleanrl.model'
        torch.save(agent.state_dict(), model_path)
        print(f'model saved to {model_path}')

    env.close()
    writer.close()