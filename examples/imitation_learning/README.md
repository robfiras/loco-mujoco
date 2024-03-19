# Imitation Learning Experiments
This directory contains the code needed to launch the imitation learning experiments. 
We have a separate configurations defining the imitation 
learning algorithm to chose as well as the respective hyperparameters. For almost all 
environments, we provide the configuration of the best performing algorithm in `confs.yaml`.


To easily schedule the experiments on local PCs and Slurm compute clusters, we use 
the [experiment launcher](https://git.ias.informatik.tu-darmstadt.de/common/experiment_launcher) package,
which is installed with LocoMuJoCo. Two files are needed for the experiment launcher; `launcher.py` and 
`experiment.py`. Both can be found in this directory.
As the name suggest, the launcher file is used to launch the experiments. So once the `imitation_lib` is
installed you, can launch the experiments by running the following command in the terminal:

```bash
python launcher.py
```

It is suggested to modify `launcher.py` to choose the desired environment.

### Visualizing the Results

The results are saved in the `./logs` directory. To visualize the results, you can use the `tensorboard`.
To do so, run the following command in the terminal:

```bash
tensorboard --logdir ./logs
```

The focus should be put on the following three metrics: "Eval_R-stochastic", "Eval_J-stochastic", and "Eval_L-stochastic", 
which are the **mean undiscounted return**, **mean discounted return**, and the **mean length of an episode** the agent, respectively.
The return is calculated based on the reward specified for each environment. Note that the latter is not used for training
but only for evaluation.


### Tuning the Hyperparameters
If you want to change the hyperparameters or the algorithm, we suggest to copy the `confs.yaml` file and 
pass the new configuration file to the `get_agent` method in `experiment.py`.

Alternatively, you can also directly use the specific agent getter (e.g., `create_gail_agent`or `create_vail_agent`, 
which can be found in `utils.py`). This way you can also directly pass the hyperparameters to the agent. In doing so, 
you can easily loop over hyperparameters to perform a search. Therefore, specify the parameter 
you would like to perform hyperparameter in the launcher file. Here is an example. Let's say you 
want to perform a hyperparameter search over the critic's learning rate of a VAIL agent.

To do so, change the loop in `launcher.py` from:

```python
for env_id in env_ids:
    launcher.add_experiment(env_id__=env_id, **default_params)
```

to:

```python
from itertools import product

critic_lrs = [1e-3, 1e-4, 1e-5]
for env_id, critic_lr in product(env_ids, critic_lrs):
    launcher.add_experiment(env_id__=env_id, critic_lr__=critic_lr, **default_params)
```
The trailing underscores are important to have a separate logging directory for each experiment when looping 
over a parameter.

**Important**: You have to specify the new parameter with the type declaration in the `experiment.py` file.
Hence, the experiment file changes from:

```python
def experiment(env_id: str = None,
               n_epochs: int = 500,
               n_steps_per_epoch: int = 10000,
               n_steps_per_fit: int = 1024,
               n_eval_episodes: int = 50,
               n_epochs_save: int = 500,
               gamma: float = 0.99,
               results_dir: str = './logs',
               use_cuda: bool = False,
               seed: int = 0):
    # ...
```

to:

```python
def experiment(env_id: str = None,
               n_epochs: int = 500,
               n_steps_per_epoch: int = 10000,
               n_steps_per_fit: int = 1024,
               n_eval_episodes: int = 50,
               n_epochs_save: int = 500,
               lr_critic: float = 1e-3,     # WE ADDED THIS LINE
               gamma: float = 0.99,
               results_dir: str = './logs',
               use_cuda: bool = False,
               seed: int = 0):
    # ...

    # PASS THE LEARNING RATE TO THE AGENT
```


### Load and Evaluate a Trained Agent

The best agents are saved every `n_epochs_save` epochs at your specified directory or at the default directory
`./logs`. To load and evaluate a trained agent, you can use the following code:

```python
from mushroom_rl.core import Core, Agent
from loco_mujoco import LocoEnv


env = LocoEnv.make("Atlas.walk")

agent = Agent.load("./path/to/agent.msh")

core = Core(agent, env)

core.evaluate(n_episodes=10, render=True)
```

In the example above, first an Atlas environment is created. Then, the agent is loaded from the specified path. Finally,
the agent is evaluated for 10 episodes with rendering enabled.

### Continue Training from a Checkpoint


Similarly to above, if you want to continue training from a checkpoint, you can replace the line
`agent = get_agent(env_id, mdp, use_cuda, sw)` in the `experiment.py` file with the following line
`agent = Agent.load("./path/to/agent.msh")`. In that case, you will continue training from the specified
checkpoint.



