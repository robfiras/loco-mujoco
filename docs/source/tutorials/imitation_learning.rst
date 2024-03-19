Imitation Learning
=================================

Basic Usage
-----------

LocoMuJoCo comes with many baseline algorithms. All baseline algorithms are implemented in MushroomRL.
Here we will show how to setup an experiment to train a policy using a baseline algorithm.
To easily schedule the experiments on local PCs and Slurm compute clusters, we use
the `experiment_launcher <https://git.ias.informatik.tu-darmstadt.de/common/experiment_launcher>`_ package,
which is installed with LocoMuJoCo. Two files are needed for the experiment launcher; :code:`launcher.py` and
:code:`experiment.py`. Note that the :code:`imitation_lib` has be be installed before running the experiment (checkout
:ref:`install-baseline-label`).

.. note:: All files shown here can be found under :code:`examples/imitation_learning` in the LocoMuJoCo repository.

The launcher file is used to define the parameters of the experiment and the experiment file is used to define the
experiment itself. Let's say you would like to train on almost all environments in LocoMuJoCo. Then the Launcher file
will look like this:

.. literalinclude:: ../../../examples/imitation_learning/launcher.py
    :language: python

In the launcher file, we defined information about the execution of the experiment (e.g., number of cores,
memory per core number of seeds to run, etc.). We also defined the parameter of the experiments. These parameters
are only the Task-IDs of the environments in LocoMuJoCo.

The experiment file will look like this:

.. literalinclude:: ../../../examples/imitation_learning/experiment.py
    :language: python

The main important part of the experiment file is the definition of the environment, the definition of the agent,
and the definition of the MushroomRL core. The definition of the environment is done as usual by using the :code:`make`
method from from LocoMuJoCo together with the desired task-ID. The definition of the agent is done by using the
a helper function, which can be found in the :code:`utils.py` file in the :code:`examples/imitation_learning` in the LocoMuJoCo
repository. This helper functions returns an agent with fine-tuned parameter for the respective environment. These
parameters can be found in the :code:`conf.yaml` file. The definition of the MushroomRL core is done by passing the agent
and the environment to the :code:`Core` class from MushroomRL. Finally, at each epoch, the agent is trained using the
:code:`core.learn` and is evaluated using the :code:`core.evaluate` method.

That's it! Now you can run the experiment by executing the following command in the terminal:

.. code-block:: bash

    python launcher.py


Visualizing the Results
-----------------------

The results are saved in the `./logs` directory. To visualize the results, you can use the `tensorboard`.
To do so, run the following command in the terminal:

.. code-block:: bash

    tensorboard --logdir ./logs


The focus should be put on the following three metrics: "Eval_R-stochastic", "Eval_J-stochastic", and "Eval_L-stochastic",
which are the **mean undiscounted return**, **mean discounted return**, and the **mean length of an episode** the agent, respectively.
The return is calculated based on the reward specified for each environment. Note that the latter is not used for training
but only for evaluation.

Tuning the Hyperparameters
--------------------------

If you want to to change the hyperparameters or the algorithm, we suggest to copy the :code:`confs.yaml` file and
pass the new configuration file to the :code:`get_agent` method in :code:`experiment.py`.

Alternatively, you can also directly use the specific agent getter (e.g., :code:`create_gail_agent`or :code:`create_vail_agent`,
which can be found in :code:`utils.py`). This way you can also directly pass the hyperparameters to the agent. In doing so,
you can easily loop over hyperparameters to perform a search. Therefore, specify the parameter
you would like to perform hyperparameter in the launcher file. Here is an example. Let's say you
want to perform a hyperparameter search over the critic's learning rate of a VAIL agent.

To do so, change the loop in :code:`launcher.py` from:

.. code-block:: python

    for env_id in env_ids:
        launcher.add_experiment(env_id__=env_id, **default_params)

to:


.. code-block:: python

    critic_lrs = [1e-3, 1e-4, 1e-5]
    for env_id, critic_lr in product(env_ids, critic_lrs):
        launcher.add_experiment(env_id__=env_id, critic_lr__=critic_lr, **default_params)


The trailing underscores are important to have a separate logging directory for each experiment when looping
over a parameter.

.. note:: You have to specify the new parameter **with the type declaration** in the `experiment.py` file.

Hence, the experiment file changes from:


.. code-block:: python

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


to:

.. code-block:: python

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

        # pass the new learning rate to the agent


Load and Evaluate a Trained Agent
---------------------------------

The best agents are saved every :code:`n_epochs_save` epochs at your specified directory or at the default directory
:code:`./logs`. To load and evaluate a trained agent, you can use the following code:

.. code-block:: python

    from mushroom_rl.core import Core, Agent
    from loco_mujoco import LocoEnv


    env = LocoEnv.make("Atlas.walk")

    agent = Agent.load("./path/to/agent.msh")

    core = Core(agent, env)

    core.evaluate(n_episodes=10, render=True)

In the example above, first an Atlas environment is created. Then, the agent is loaded from the specified path. Finally,
the agent is evaluated for 10 episodes with rendering enabled.

Continue Training from a Checkpoint
-----------------------------------

Similarly to above, if you want to continue training from a checkpoint, you can replace the line
:code:`agent = get_agent(env_id, mdp, use_cuda, sw)` in the :code:`experiment.py` file with the following line
:code:`agent = Agent.load("./path/to/agent.msh")`. In that case, you will continue training from the specified
checkpoint.