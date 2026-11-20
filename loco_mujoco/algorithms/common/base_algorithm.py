"""
Base classes shared by all JAX learning algorithms.

Every algorithm in :mod:`loco_mujoco.algorithms` is a subclass of :class:`JaxRLAlgorithmBase` that only consists
of class and static methods. The data of an agent is split into two containers:

- an *agent configuration* (:class:`AgentConfBase`): static, hashable objects such as the Hydra/OmegaConf
  configuration, the Flax network definitions and the Optax optimizers. It is closed over by the jitted
  training function.
- an *agent state* (:class:`AgentStateBase`): the dynamic JAX pytree (network parameters, optimizer state,
  running statistics, ...) that is passed into and returned by the training function.

The typical call sequence is ``init_agent_conf`` -> ``init_agent_state`` -> ``build_train_fn`` ->
``save_agent`` / ``play_policy``; see ``examples/training_examples``.

"""

import pickle
from pathlib import Path
from dataclasses import dataclass

from flax import struct

from loco_mujoco.utils import MetricsHandler


@dataclass(frozen=True)
class AgentConfBase:
    """
    Abstract base class for the *static* configuration of an agent.

    Subclasses are frozen dataclasses that hold everything that does not change during training (configuration,
    network definitions, optimizers). Any subclass must implement :meth:`serialize` and :meth:`from_dict`.

    """

    def serialize(self):
        """
        Serialize the agent configuration.

        Returns:
            dict: A picklable dictionary from which :meth:`from_dict` can rebuild the configuration.

        """
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d):
        """
        Create an instance of the configuration class from a dictionary.

        Args:
            d (dict): Dictionary created by :meth:`serialize`.

        Returns:
            AgentConfBase: The restored agent configuration.

        """
        raise NotImplementedError


@struct.dataclass
class AgentStateBase:
    """
    Abstract base class for the *dynamic* state of an agent.

    Subclasses are Flax struct dataclasses (JAX pytrees) holding network parameters, optimizer states and other
    quantities that change during training. Any subclass must implement :meth:`serialize` and :meth:`from_dict`.

    """

    def serialize(self):
        """
        Serialize the agent state.

        Returns:
            dict: A picklable dictionary from which :meth:`from_dict` can rebuild the state.

        """
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d, agent_conf):
        """
        Create an instance of the agent state from a dictionary.

        Args:
            d (dict): Dictionary created by :meth:`serialize`.
            agent_conf (AgentConfBase): Agent configuration the state belongs to; it provides the networks and
                optimizers needed to rebuild the train states.

        Returns:
            AgentStateBase: The restored agent state.

        """
        raise NotImplementedError


class JaxRLAlgorithmBase:

    """
    Base class defining the interface of all JAX learning algorithms.

    Algorithms are stateless: all methods are class or static methods, and the data of an agent lives in an
    :class:`AgentConfBase` (static) and an :class:`AgentStateBase` (dynamic pytree). Subclasses set the class
    attributes ``_agent_conf`` and ``_agent_state`` to their concrete container classes and implement
    :meth:`init_agent_conf`, :meth:`init_agent_state`, ``_train_fn``, ``_wrap_env`` and :meth:`play_policy`
    (and optionally ``_eval_fn``).

    Typical usage::

        agent_conf = Algo.init_agent_conf(env, config)
        agent_state = Algo.init_agent_state(env, agent_conf, rng)
        train_fn = jax.jit(Algo.build_train_fn(env, agent_conf, mh=mh))
        out = train_fn(rng, agent_state, traj)
        Algo.save_agent(result_dir, agent_conf, out["agent_state"])

    """

    # these class attributes must be set by subclasses
    _agent_conf: AgentConfBase
    _agent_state: AgentStateBase

    # prefix for all saved agent files
    _saved_agent_suffix = ".pkl"

    @classmethod
    def init_agent_conf(cls, env, config):
        """
        Initialize the static agent configuration.

        Args:
            env: The (unwrapped) environment the agent is trained on.
            config (DictConfig): Hydra/OmegaConf configuration; the hyperparameters live under ``config.experiment``.

        Returns:
            AgentConfBase: The agent configuration (configuration, networks and optimizers).

        """
        raise NotImplementedError

    @classmethod
    def init_agent_state(cls, env, agent_conf: AgentConfBase, rng) -> AgentStateBase:
        """
        Initialize the dynamic agent state (network parameters, optimizer state, etc.).

        Args:
            env: The (unwrapped) environment the agent is trained on.
            agent_conf (AgentConfBase): Agent configuration created by :meth:`init_agent_conf`.
            rng (jax.Array): PRNG key used to initialize the network parameters.

        Returns:
            AgentStateBase: The freshly initialized agent state.

        """
        raise NotImplementedError

    @classmethod
    def build_train_fn(cls, env, agent_conf: AgentConfBase, mh: MetricsHandler = None):
        """
        Build the training function.

        The returned function is pure and can be wrapped in ``jax.jit`` and, to train several seeds in parallel,
        in ``jax.vmap``.

        Args:
            env: The (unwrapped) environment the agent is trained on.
            agent_conf (AgentConfBase): Agent configuration created by :meth:`init_agent_conf`.
            mh (MetricsHandler): Optional metrics handler used to compute validation metrics during training.

        Returns:
            Callable: A function ``train_fn(rng, agent_state, traj)`` that runs one training chunk and returns the
            output of ``_train_fn`` (for the built-in algorithms a dictionary containing at least the new
            ``agent_state`` and the logged metrics).

        """
        return lambda rng_key, agent_state, traj: cls._train_fn(rng_key, env, agent_conf, agent_state, traj=traj, mh=mh)

    @classmethod
    def _train_fn(cls, rng, env,
                  agent_conf: AgentConfBase,
                  agent_state: AgentStateBase = None,
                  traj=None,
                  mh: MetricsHandler = None):
        """
        Run the training loop of the algorithm; to be implemented by subclasses.

        Args:
            rng (jax.Array): PRNG key.
            env: The (unwrapped) environment.
            agent_conf (AgentConfBase): Static agent configuration.
            agent_state (AgentStateBase): Agent state to start (or resume) from.
            traj: Trajectory data passed on to ``env.reset`` and ``env.step``; ``None`` if the task has no trajectory.
            mh (MetricsHandler): Optional metrics handler for validation.

        Returns:
            The training output, including the updated agent state.

        """
        raise NotImplementedError

    @classmethod
    def build_eval_fn(cls, env, agent_conf: AgentConfBase, mh: MetricsHandler = None):
        """
        Build the evaluation function.

        It has the same signature as the function returned by :meth:`build_train_fn`. The agent's training state is
        not modified; updates of the running observation statistics made during the evaluation rollout are discarded.

        Args:
            env: The (unwrapped) environment the agent is evaluated on.
            agent_conf (AgentConfBase): Agent configuration created by :meth:`init_agent_conf`.
            mh (MetricsHandler): Optional metrics handler used to compute validation metrics.

        Returns:
            Callable: A function ``eval_fn(rng, agent_state, traj)`` that runs an evaluation rollout and returns a
            dictionary with metric summaries.

        """
        return lambda rng_key, agent_state, traj: cls._eval_fn(rng_key, env, agent_conf, agent_state, traj=traj, mh=mh)

    @classmethod
    def _eval_fn(cls, rng, env,
                 agent_conf: AgentConfBase,
                 agent_state: AgentStateBase,
                 traj=None,
                 mh: MetricsHandler = None):
        """
        Run an evaluation rollout; to be implemented by subclasses that support :meth:`build_eval_fn`.

        Subclasses run the agent (greedy/deterministic if applicable) for a fixed number of parallel environment
        steps and return a dictionary of metrics.

        Args:
            rng (jax.Array): PRNG key.
            env: The (unwrapped) environment.
            agent_conf (AgentConfBase): Static agent configuration.
            agent_state (AgentStateBase): Agent state to evaluate.
            traj: Trajectory data passed on to ``env.reset`` and ``env.step``; ``None`` if the task has no trajectory.
            mh (MetricsHandler): Optional metrics handler for validation.

        Returns:
            dict: Metric summaries of the rollout.

        """
        raise NotImplementedError

    @classmethod
    def play_policy(cls, train_state, env, config, n_envs, n_steps=None, record=False, key=None):
        """
        Roll out a trained policy for visualization or recording; to be implemented by subclasses.

        Note:
            The concrete algorithms override this method with their own signature, typically
            ``play_policy(env, agent_conf, agent_state, n_envs, n_steps=None, render=True, record=False, ...)``.
            Refer to the docstring of the algorithm you use.

        """
        raise NotImplementedError

    @classmethod
    def save_agent(cls, path, agent_conf: AgentConfBase, agent_state: AgentStateBase):
        """
        Save the configuration and the state of an agent to a pickle file.

        The file is named ``<ClassName>_saved.pkl`` and contains the dictionary returned by :meth:`serialize`.

        Args:
            path (Union[str, Path]): Existing directory in which the file is created.
            agent_conf (AgentConfBase): Agent configuration to save.
            agent_state (AgentStateBase): Agent state to save.

        Returns:
            Path: Path of the written file.

        """
        path = Path(path)
        path = path / (cls.__name__ + "_saved")
        path = path.with_suffix(cls._saved_agent_suffix)
        # serialize agent state
        serialized_state = cls.serialize(agent_conf, agent_state)
        # save agent state
        with open(path, 'wb') as file:
            pickle.dump(serialized_state, file)
        print(f"\nSaved agent to: {path}\n")
        return path

    @classmethod
    def load_agent(cls, path):
        """
        Load the configuration and the state of an agent from a file written by :meth:`save_agent`.

        Args:
            path (Union[str, Path]): Path to the ``.pkl`` file.

        Returns:
            Tuple[AgentConfBase, AgentStateBase]: The restored agent configuration and agent state.

        Raises:
            ValueError: If ``path`` is not a file or does not have the ``.pkl`` suffix.

        """
        if isinstance(path, str):
            path = Path(path)
        if not path.is_file():
            raise ValueError(f'Not a file: {path}')
        if path.suffix != cls._saved_agent_suffix:
            raise ValueError(f'Not a {cls._saved_agent_suffix} file: {path}')
        with open(path, 'rb') as file:
            data = pickle.load(file)
        return cls.from_dict(data)

    @classmethod
    def serialize(cls, agent_conf: AgentConfBase, agent_state: AgentStateBase):
        """
        Serialize the configuration and the state of an agent.

        Args:
            agent_conf (AgentConfBase): Agent configuration.
            agent_state (AgentStateBase): Agent state.

        Returns:
            dict: Dictionary with the keys ``"agent_conf"`` and ``"agent_state"`` holding the serialized containers.

        """
        serialized_agent_conf = agent_conf.serialize()
        serialized_agent_state = agent_state.serialize()
        return {"agent_conf": serialized_agent_conf,
                "agent_state": serialized_agent_state}

    @classmethod
    def from_dict(cls, d):
        """
        Restore the configuration and the state of an agent from a dictionary.

        Args:
            d (dict): Dictionary created by :meth:`serialize`.

        Returns:
            Tuple[AgentConfBase, AgentStateBase]: The restored agent configuration and agent state.

        """
        agent_conf = cls._agent_conf.from_dict(d["agent_conf"])
        agent_state = cls._agent_state.from_dict(d["agent_state"], agent_conf)
        return agent_conf, agent_state

    @staticmethod
    def _wrap_env(env, config):
        """
        Wrap the environment with the wrappers the algorithm needs; to be implemented by subclasses.

        Args:
            env: The (unwrapped) environment.
            config (DictConfig): The ``experiment`` node of the configuration.

        Returns:
            The wrapped environment.

        """
        raise NotImplementedError

    @staticmethod
    def _validation_schedule(num_updates, validation_num):
        """
        Compute how often a validation rollout is run during training.

        Args:
            num_updates (int): Total number of updates of the training run.
            validation_num (int): Requested number of validation rollouts. Zero (or a negative number) disables
                validation.

        Returns:
            Tuple[int, int]: The number of updates between two validation rollouts and the number of validation
            rollouts that are actually run. If validation is disabled, the interval is ``num_updates + 1``, which
            is never reached, and the number is 0.

        """
        if validation_num is None or validation_num <= 0:
            return int(num_updates) + 1, 0
        validation_interval = max(int(num_updates) // int(validation_num), 1)
        return validation_interval, int(num_updates) // validation_interval

    @classmethod
    def _linear_lr_schedule(cls, count, num_minibatches, update_epochs, lr, num_updates):
        """
        Linearly anneal the learning rate from ``lr`` to zero over ``num_updates`` updates.

        Args:
            count (jax.Array): Number of optimizer steps taken so far.
            num_minibatches (int): Number of minibatches per epoch.
            update_epochs (int): Number of epochs per update.
            lr (float): Initial learning rate.
            num_updates (int): Total number of updates.

        Returns:
            jax.Array: The learning rate for the current optimizer step.

        """
        frac = (
                1.0
                - (count // (num_minibatches * update_epochs))
                / num_updates
        )
        return lr * frac
