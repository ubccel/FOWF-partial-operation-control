# Power maximization for floating offshore wind farms under partial operating conditions with RL

## Introduction
This repository contains the Python implementation of an RL-based floating offshore wind farm controller. The controller is based around a Markov decision process meant to capture the sequential process of choosing nacelle yaw misalignments for each turbine, ordered in-line with the direction of the freestream wind. Graph attention networks aid the actor in critic in the PPO agent, based on [CleanRL's PPO with continuous actions](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py).

The simulation tools used to model the floating offshore wind farm include [FLORIS](https://github.com/NREL/floris) and [MoorPy](https://github.com/NREL/MoorPy).

## Agent training
 + An agent which considers variable farm subsets may be trained using:\
    ``python gat_gaussian_ppo.py --save-model``
+ An agent that only ever sees the entire farm may be trained using:\
    ``python gat_gaussian_ppo.py --save-model --full-farm``
+ A user-friendly evaluation script is under development and will be added to the repository when completed.

