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

### Tuning the Hyperparameters
If you want to to change the hyperparameters or the algorithm, we suggest to copy the `confs.yaml` file and 
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



