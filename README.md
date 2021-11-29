# Promoting Coordination via Reward Sharing Networks in Multi-Agent Deep Reinforcement Learning

This is a fork of McGill University & MILA Qubec's research "Promoting Coordination Through Policy Regularization in Multi-Agent Deep Reinforcement Learning"
Original Github: https://github.com/julienroyd/coordination-marl 

In this repository I've resolved a few bugs in the original repository and added the experiments we're currently working on.

## Online vsualisations

Visualisations of rollouts and description is available on https://www.alokmalik.com/research
 
## Requirements

Open a terminal inside `coordination-marl` folder, then:
* Install a conda environment with Python 3.7: `conda create --name test_env python=3.7`
* Install the regular dependencies: `pip install -r requirements.txt`
* Trobuleshooring Tip: If you get error while installing requirements.txt you may have to install each library individually.
* Install the external dependencies: `pip install -e external_dependencies/multiagent_particle_environment_fork`
* Get clone of openai baselines : 'git clone https://github.com/openai/baselines'
* Install baselines 'pip install -e baselines'

## Main experiments

#### To visualize trained_models

1. Go in the code folder of the desired algorithm:
    * example1: `cd code/continuous_control/coach`
    * example2: `cd code/discrete_control/coach`

2. Run `evaluate.py` with the desired arguments:
    * example1: `python evaluate.py --root ../../../trained_models/continuous_control/chase --storage_name PB6_2bc3c27_5f7a15b_CoachMADDPG_chase_retrainBestPB3_review`
    * example2: `python evaluate.py --root ../../../trained_models/discrete_control/3v2football --storage_name Ju25_2667341_5e972b5_CoachMADDPG_3v2football_retrainBestJu24_benchmarkv3`


#### To train new models

1. Go in the code folder of the desired algorithm:
    * example1: `cd code/continuous_control/coach`
    * example2: `cd code/discrete_control/baselines`

2. Run `main.py` with the desired arguments:
    * example1: `python main.py --env_name s63_deliver --network communitarian --agent_alg CoachMADDPG`
    * example2: `python main.py --env_name s63_deliver --network survivalist --n_agents 2

Run `python main.py --help` for all options.
