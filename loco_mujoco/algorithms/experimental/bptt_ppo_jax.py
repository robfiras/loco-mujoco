"""
PPO with backpropagation through time (BPTT-PPO) for recurrent policies in JAX.

A variant of :class:`~loco_mujoco.algorithms.ppo_jax.PPOJax` whose actor carries a deterministic recurrent
hidden state (GRU, LSTM or an MLP-based recurrence). During the policy update the stored rollout is replayed
from the hidden state at the start of the rollout with ``jax.lax.scan``, so that gradients flow through the
whole rollout of ``num_steps`` steps. Minibatches are therefore formed over environments instead of over
individual time steps.

This algorithm is experimental. Example: ``examples/training_examples/jax_rl_mimic_bptt``.

"""

import ast
import warnings
from omegaconf import open_dict
from dataclasses import dataclass
from typing import Any, NamedTuple, Tuple, Sequence
from omegaconf import DictConfig, OmegaConf, ListConfig

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
import flax
import optax
import distrax

from loco_mujoco.algorithms import (JaxRLAlgorithmBase, AgentConfBase, AgentStateBase,
                                    TrainState, TrainStateBuffer, MetricHandlerTransition)
from loco_mujoco.algorithms.common.networks import FullyConnectedNet, RunningMeanStd, get_activation_fn
from loco_mujoco.algorithms.ppo_jax import PPOJax, PPOSummaryMetrics
from loco_mujoco.core.wrappers import LogWrapper, NStepWrapper, LogEnvState, VecEnv
from loco_mujoco.environments.base import TrajState
from loco_mujoco.core.wrappers.mjx import Metrics
from loco_mujoco.utils import MetricsHandler, ValidationSummary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stored_hidden_dim(hidden_state_dim: int, rnn_type: str) -> int:
    """Total stored hidden-state size: 2x for LSTM (h and c), 1x otherwise."""
    return 2 * hidden_state_dim if rnn_type == "lstm" else hidden_state_dim


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class ActorCriticBPTT(nn.Module):
    """
    Recurrent actor-critic network for BPTT-PPO.

    **Actor.** With ``rnn_type`` set to ``"gru"`` or ``"lstm"``:

    1. The normalized observation and the previous action are embedded separately by a dense layer of width
       ``n_features``.
    2. The concatenated embeddings are fed into a GRU / LSTM cell whose carry is the current hidden state.
    3. The new carry is the (deterministic) new hidden state.
    4. A second embedding of the observation is concatenated with the RNN output and mapped linearly to the
       action mean.

    With ``rnn_type="vanilla"`` the two embeddings are concatenated with the hidden state and passed through an
    MLP that outputs the action mean and the new hidden state.

    The actor yields a diagonal Gaussian over actions with a state-independent standard deviation
    (parameter ``log_std_a``); the new hidden state is returned deterministically.

    **Critic.** An MLP over the concatenation of the normalized observation, the hidden state and the previous
    action, i.e. ``V(s, h, a_prev)``.

    The network has a ``run_stats`` variable collection (observation normalization) and has to be applied with
    ``mutable=["run_stats"]``.

    Attributes:
        action_dim (int): Dimension of the action space.
        hidden_state_dim (int): Size of the RNN hidden state. For an LSTM the stored hidden state is the
            concatenation of ``h`` and ``c`` and thus twice as large.
        rnn_type (str): ``"gru"``, ``"lstm"`` or ``"vanilla"``.
        n_features (int): Width of the embeddings before and after the RNN.
        activation (str): Name of the activation function in ``flax.linen``.
        init_std_a (float): Initial standard deviation of the action distribution.
        learnable_std (bool): If False, the gradient with respect to ``log_std_a`` is stopped.
        hidden_layer_dims (Sequence[int]): Hidden layers of the critic MLP and of the ``"vanilla"`` actor MLP.
        ema_coef (float): Exponential smoothing of the hidden-state transition,
            ``h' = (1 - ema_coef) * h + ema_coef * f(obs, h, a_prev)``. The default ``1.0`` disables smoothing.

    """

    action_dim: int
    hidden_state_dim: int           # actual RNN hidden size (per vector; LSTM stores 2x)
    rnn_type: str = "gru"           # "gru" | "lstm" | "vanilla"
    n_features: int = 256           # pre / post-RNN embedding width
    activation: str = "tanh"
    init_std_a: float = 1.0
    learnable_std: bool = True
    hidden_layer_dims: Sequence[int] = (512, 256)   # critic MLP + "vanilla" actor MLP
    ema_coef: float = 1.0           # exponential smoothing for hidden state transition

    def setup(self):
        """Resolve the activation function from its name."""
        self.activation_fn = get_activation_fn(self.activation)

    @nn.compact
    def __call__(self, obs, hidden_state, prev_action):
        """
        Compute the action distribution, the value estimate and the next hidden state.

        Args:
            obs (jax.Array): Observations, shape ``(batch, obs_dim)``.
            hidden_state (jax.Array): Current hidden states, shape ``(batch, stored_hidden_dim)``, where
                ``stored_hidden_dim`` is ``2 * hidden_state_dim`` for an LSTM (``h`` and ``c`` concatenated) and
                ``hidden_state_dim`` otherwise.
            prev_action (jax.Array): Previous actions, shape ``(batch, action_dim)``.

        Returns:
            Tuple[distrax.MultivariateNormalDiag, jax.Array, jax.Array]: The action distribution, the value
            estimates of shape ``(batch,)`` and the new hidden states of shape ``(batch, stored_hidden_dim)``.

        Raises:
            ValueError: If ``rnn_type`` is unknown.

        """
        obs_norm = RunningMeanStd()(obs)

        stored_z_dim = _stored_hidden_dim(self.hidden_state_dim, self.rnn_type)

        if self.rnn_type == "vanilla":
            # ---- MLP actor ----
            # Embed obs and prev_action separately, then combine with hidden state
            obs_embed = self.activation_fn(nn.Dense(self.n_features)(obs_norm))
            prev_a_embed = self.activation_fn(nn.Dense(self.n_features)(prev_action))
            actor_in = jnp.concatenate([obs_embed, prev_a_embed, hidden_state], axis=-1)
            actor_out = FullyConnectedNet(
                self.hidden_layer_dims,
                self.action_dim + stored_z_dim,
                self.activation, None, False, False
            )(actor_in)
            action_mean = actor_out[..., :self.action_dim]
            new_hidden = actor_out[..., self.action_dim:]

        else:
            # ---- RNN actor ----
            # Pre-RNN embeddings
            obs_embed = self.activation_fn(nn.Dense(self.n_features)(obs_norm))
            prev_a_embed = self.activation_fn(nn.Dense(self.n_features)(prev_action))
            rnn_input = jnp.concatenate([obs_embed, prev_a_embed], axis=-1)

            if self.rnn_type == "gru":
                new_carry, rnn_out = nn.GRUCell(self.hidden_state_dim)(hidden_state, rnn_input)
                new_hidden = new_carry   # shape (..., hidden_state_dim)

            elif self.rnn_type == "lstm":
                h = hidden_state[..., :self.hidden_state_dim]
                c = hidden_state[..., self.hidden_state_dim:]
                (new_h, new_c), rnn_out = nn.LSTMCell(self.hidden_state_dim)((h, c), rnn_input)
                new_hidden = jnp.concatenate([new_h, new_c], axis=-1)
                rnn_out = new_h   # use h for downstream heads

            else:
                raise ValueError(f"Unknown rnn_type: {self.rnn_type!r}")

            # Post-RNN obs re-embedding → action mean
            obs_post = self.activation_fn(nn.Dense(self.n_features)(obs_norm))
            action_mean = nn.Dense(
                self.action_dim,
                kernel_init=nn.initializers.orthogonal(0.01),
                bias_init=nn.initializers.constant(0.0),
            )(jnp.concatenate([obs_post, rnn_out], axis=-1))

        # Exponential smoothing on the hidden state transition:
        #   new_h = (1 - ema_coef) * h_t + ema_coef * f_theta(...)
        # ema_coef=1.0 recovers the unsmoothed transition (default).
        if self.ema_coef < 1.0:
            new_hidden = (1.0 - self.ema_coef) * hidden_state + self.ema_coef * new_hidden

        # Action distribution (independent Gaussian)
        log_std_a = self.param("log_std_a",
                               nn.initializers.constant(jnp.log(self.init_std_a)),
                               (self.action_dim,))
        if not self.learnable_std:
            log_std_a = jax.lax.stop_gradient(log_std_a)

        pi = distrax.MultivariateNormalDiag(action_mean, jnp.exp(log_std_a))

        # Critic — MLP over (obs, h, prev_action)
        critic_in = jnp.concatenate([obs_norm, hidden_state, prev_action], axis=-1)
        critic = FullyConnectedNet(
            self.hidden_layer_dims, 1, self.activation, None, False, False
        )(critic_in)

        return pi, jnp.squeeze(critic, axis=-1), new_hidden


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class BPTTPolicy:
    """
    Policy wrapper for BPTT-PPO.

    All methods take the observation, the hidden state and the previous action as separate arguments. The hidden
    state is propagated deterministically (there is no stochastic transition).

    """

    def __init__(self, network: nn.Module, action_dim: int):
        """
        Initialize the policy wrapper.

        Args:
            network (nn.Module): The :class:`ActorCriticBPTT` network.
            action_dim (int): Dimension of the action space.

        """
        self.network = network
        self.action_dim = action_dim

    def get_action(
        self, obs, hidden_state, prev_action, train_state, rng
    ):
        """
        Sample an action and compute the next hidden state.

        Args:
            obs (jax.Array): Observations, shape ``(n_envs, obs_dim)``.
            hidden_state (jax.Array): Current hidden states, shape ``(n_envs, stored_hidden_dim)``.
            prev_action (jax.Array): Previous actions, shape ``(n_envs, action_dim)``.
            train_state (TrainState): Current train state with parameters and running statistics.
            rng (jax.Array): PRNG key used for action sampling.

        Returns:
            Tuple[jax.Array, jax.Array, jax.Array, jax.Array, TrainState]: The sampled actions, the new hidden
            states, the log probabilities of the actions, the value estimates and the train state with updated
            running statistics.

        """
        pi, value, new_hidden, train_state = self._forward_pass(obs, hidden_state, prev_action, train_state)
        action = pi.sample(seed=rng)
        log_prob = pi.log_prob(action)
        return action, new_hidden, log_prob, value, train_state

    def get_dist_and_value(self, obs, hidden_state, prev_action, train_state):
        """
        Compute the action distribution, the value estimate and the next hidden state.

        Args:
            obs (jax.Array): Observations, shape ``(n_envs, obs_dim)``.
            hidden_state (jax.Array): Current hidden states, shape ``(n_envs, stored_hidden_dim)``.
            prev_action (jax.Array): Previous actions, shape ``(n_envs, action_dim)``.
            train_state (TrainState): Current train state with parameters and running statistics.

        Returns:
            Tuple[distrax.Distribution, jax.Array, jax.Array, TrainState]: The action distribution, the value
            estimates, the new hidden states and the train state with updated running statistics.

        """
        return self._forward_pass(obs, hidden_state, prev_action, train_state)

    def _forward_pass(self, obs, hidden_state, prev_action, train_state):
        """Run a forward pass of the network and write the updated running statistics to the train state."""
        (pi, value, new_hidden), updates = self.network.apply(
            {"params": train_state.params, "run_stats": train_state.run_stats},
            obs, hidden_state, prev_action,
            mutable=["run_stats"],
        )
        return pi, value, new_hidden, train_state.replace(run_stats=updates["run_stats"])


# ---------------------------------------------------------------------------
# Transition
# ---------------------------------------------------------------------------

class BPTTTransition(NamedTuple):
    """
    One step of experience collected by BPTT-PPO.

    The hidden state is not stored per step; it is recomputed during the update by replaying the sequence from
    the hidden state at the start of the rollout.

    Attributes:
        done (jax.Array): Whether the episode ended at this step.
        absorbing (jax.Array): Whether the state reached at this step is absorbing.
        action (jax.Array): Action ``a_t`` applied to the environment.
        prev_action (jax.Array): Previous action ``a_{t-1}`` (zeros at the start of an episode).
        value (jax.Array): Critic estimate at this step.
        reward (jax.Array): Reward received.
        log_prob (jax.Array): Log probability of ``action`` under the data-collecting policy.
        obs (jax.Array): Raw environment observation the action was computed from.
        info (dict): Info dictionary returned by the environment step.
        traj_state (TrajState): Trajectory state of the environment after the step.
        metrics (Metrics): Episode statistics of the :class:`~loco_mujoco.core.wrappers.mjx.LogWrapper`.

    """
    done: jnp.ndarray
    absorbing: jnp.ndarray
    action: jnp.ndarray         # env action a_t
    prev_action: jnp.ndarray    # a_{t-1} (zeros at episode start)
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray            # raw env obs (not augmented)
    info: jnp.ndarray
    traj_state: TrajState
    metrics: Metrics


# ---------------------------------------------------------------------------
# Agent conf / state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BPTTAgentConf(AgentConfBase):
    """
    Static configuration of a BPTT-PPO agent.

    Attributes:
        config (DictConfig): The full configuration; the hyperparameters live under ``config.experiment``.
        network (ActorCriticBPTT): The recurrent actor-critic network definition.
        tx (optax.GradientTransformation): The optimizer.

    """
    config: DictConfig
    network: ActorCriticBPTT
    tx: Any

    def serialize(self):
        """
        Serialize the agent configuration and the network definition.

        Returns:
            dict: Dictionary with the keys ``"config"`` (the resolved configuration as a plain container) and
            ``"network"`` (the state dictionary of the network definition).

        """
        conf_dict = OmegaConf.to_container(self.config, resolve=True, throw_on_missing=True)
        serialized_network = flax.serialization.to_state_dict(self.network)
        return {"config": conf_dict, "network": serialized_network}

    @classmethod
    def from_dict(cls, d):
        """
        Restore the agent configuration from a dictionary.

        Args:
            d (dict): Dictionary created by :meth:`serialize`.

        Returns:
            BPTTAgentConf: The restored configuration with a newly built optimizer.

        """
        config = OmegaConf.create(d["config"])
        tx = BPTTPPOJax._get_optimizer(config)
        return cls(config=config,
                   network=flax.serialization.from_state_dict(ActorCriticBPTT, d["network"]),
                   tx=tx)


@struct.dataclass
class BPTTAgentState(AgentStateBase):
    """
    Dynamic state of a BPTT-PPO agent.

    Attributes:
        train_state (TrainState): Parameters, optimizer state and running observation statistics of the network.
        env_state: Environment state at the end of the last training chunk; not saved to disk.
        last_obs (jax.Array): Last observations of the previous training chunk; not saved to disk.
        hidden_state (jax.Array): Hidden states at the end of the last training chunk, shape
            ``(num_envs, stored_hidden_dim)``; not saved to disk.
        prev_action (jax.Array): Last actions of the previous training chunk, shape ``(num_envs, action_dim)``;
            not saved to disk.

    """
    train_state: TrainState
    env_state: Any = None
    last_obs: Any = None
    hidden_state: Any = None    # h carried across chunks
    prev_action: Any = None     # a_{t-1} carried across chunks

    def serialize(self):
        """
        Serialize the agent state.

        Returns:
            dict: Dictionary with the key ``"train_state"``; the carried environment state, observation, hidden
            state and previous action are dropped.

        """
        return {"train_state": flax.serialization.to_state_dict(self.train_state)}

    @classmethod
    def from_dict(cls, d, agent_conf):
        """
        Restore the agent state from a dictionary.

        Args:
            d (dict): Dictionary created by :meth:`serialize`.
            agent_conf (BPTTAgentConf): Agent configuration providing the network and the optimizer.

        Returns:
            BPTTAgentState: The restored state, including the optimizer state.

        """
        train_state = TrainState.create(
            apply_fn=agent_conf.network.apply,
            tx=agent_conf.tx,
            params=d["train_state"]["params"],
            run_stats=d["train_state"]["run_stats"],
        )
        opt_state = flax.serialization.from_state_dict(
            train_state.opt_state, d["train_state"]["opt_state"]
        )
        return cls(train_state=train_state.replace(opt_state=opt_state))


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------

class BPTTPPOJax(PPOJax):
    """
    BPTT-PPO: Proximal Policy Optimization with backpropagation through time.

    Key properties:

    * The policy uses a GRU, an LSTM or an MLP-based recurrence to maintain a deterministic hidden state; the
      importance-sampling ratio is over actions only, as in standard PPO.
    * During the update the rollout is replayed with ``jax.lax.scan`` so that gradients flow through time over
      the whole rollout (``num_steps`` steps).
    * Minibatches are formed over environments (not over individual time steps), so sequences stay intact during
      the gradient computation. ``num_envs`` must be divisible by ``num_minibatches``.
    * The hidden state at the start of each rollout is saved and used to seed the replay; hidden state and
      previous action are reset to zero when an episode ends.

    Supported ``rnn_type`` values: ``"gru"`` (default), ``"lstm"`` and ``"vanilla"`` (MLP fallback).

    The interface is that of :class:`~loco_mujoco.algorithms.ppo_jax.PPOJax`; see
    ``examples/training_examples/jax_rl_mimic_bptt`` for a training script. Reward normalization
    (``normalize_env``) is not applied by this algorithm.

    """

    _agent_conf = BPTTAgentConf
    _agent_state = BPTTAgentState

    @classmethod
    def init_agent_conf(cls, env, config):
        """
        Create the static agent configuration: derived hyperparameters, the network and the optimizer.

        In addition to the PPO hyperparameters (see :meth:`PPOJax.init_agent_conf
        <loco_mujoco.algorithms.ppo_jax.PPOJax.init_agent_conf>`), the following keys are read from
        ``config.experiment``:

        - ``hidden_state_dim``: size of the RNN hidden state.
        - ``rnn_type`` (optional, default ``"gru"``): ``"gru"``, ``"lstm"`` or ``"vanilla"``.
        - ``n_features`` (optional, default 256): width of the embeddings before and after the RNN.
        - ``ema_coef`` (optional, default 1.0): exponential smoothing of the hidden-state transition.

        The keys ``num_updates``, ``minibatch_size``, ``validation_interval``, ``action_dim`` and a recomputed
        ``validation.num`` are written to ``config.experiment`` in place. Here ``minibatch_size`` is the number of
        environments (i.e. sequences) per minibatch, ``num_envs // num_minibatches``.

        Args:
            env: The (unwrapped) environment the agent is trained on.
            config (DictConfig): Configuration with the hyperparameters under ``config.experiment``.

        Returns:
            BPTTAgentConf: The agent configuration.

        """
        with open_dict(config.experiment):
            config.experiment.num_updates = (
                config.experiment.total_timesteps // config.experiment.num_steps // config.experiment.num_envs)
            # For BPTT, minibatch_size is over environments not flattened steps
            config.experiment.minibatch_size = (
                config.experiment.num_envs // config.experiment.num_minibatches)
            config.experiment.validation_interval = config.experiment.num_updates // config.experiment.validation.num
            config.experiment.validation.num = int(
                config.experiment.num_updates // config.experiment.validation_interval)
            config.experiment.action_dim = env.info.action_space.shape[0]

        hidden_layers = config.experiment.hidden_layers \
            if isinstance(config.experiment.hidden_layers, (list, ListConfig)) \
            else ast.literal_eval(config.experiment.hidden_layers)

        hidden_state_dim = config.experiment.hidden_state_dim
        rnn_type = getattr(config.experiment, 'rnn_type', 'gru')
        n_features = getattr(config.experiment, 'n_features', 256)
        ema_coef = getattr(config.experiment, 'ema_coef', 1.0)

        network = ActorCriticBPTT(
            action_dim=env.info.action_space.shape[0],
            hidden_state_dim=hidden_state_dim,
            rnn_type=rnn_type,
            n_features=n_features,
            activation=config.experiment.activation,
            init_std_a=config.experiment.init_std,
            learnable_std=config.experiment.learnable_std,
            hidden_layer_dims=hidden_layers,
            ema_coef=ema_coef,
        )
        tx = cls._get_optimizer(config)
        return cls._agent_conf(config, network, tx)

    @classmethod
    def init_agent_state(cls, env, agent_conf: BPTTAgentConf, rng) -> BPTTAgentState:
        """
        Initialize the agent state (network parameters, running statistics and optimizer state).

        Args:
            env: The (unwrapped) environment the agent is trained on.
            agent_conf (BPTTAgentConf): Agent configuration created by :meth:`init_agent_conf`.
            rng (jax.Array): PRNG key used to initialize the network parameters.

        Returns:
            BPTTAgentState: The initial agent state; the carried environment state, observation, hidden state and
            previous action are ``None`` and get initialized by the training function.

        """
        config, network, tx = agent_conf.config.experiment, agent_conf.network, agent_conf.tx
        wrapped_env = cls._wrap_env(env, config)
        obs_dim = wrapped_env.info.observation_space.shape[0]
        action_dim = config.action_dim
        stored_z_dim = _stored_hidden_dim(config.hidden_state_dim, network.rnn_type)
        rng, _rng = jax.random.split(rng)
        network_params = network.init(
            _rng,
            jnp.zeros((1, obs_dim)),
            jnp.zeros((1, stored_z_dim)),
            jnp.zeros((1, action_dim)),
        )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params["params"],
            run_stats=network_params["run_stats"],
            tx=tx,
        )
        return cls._agent_state(train_state=train_state)

    @classmethod
    def _train_fn(cls, rng, env,
                  agent_conf: BPTTAgentConf,
                  agent_state: BPTTAgentState = None,
                  traj=None,
                  mh: MetricsHandler = None):
        """
        Run BPTT-PPO for ``num_updates`` updates.

        Every update collects a rollout of ``num_steps`` steps, computes advantages with GAE and runs
        ``update_epochs`` epochs over ``num_minibatches`` minibatches of whole environment sequences. In the loss,
        each sequence is replayed from the hidden state it had at the start of the rollout, with the running
        observation statistics held fixed, and the PPO losses are evaluated on the replayed log probabilities and
        values. If ``agent_state`` carries an environment state, hidden state and previous action from a previous
        call, training continues from them.

        Args:
            rng (jax.Array): PRNG key.
            env: The (unwrapped) environment.
            agent_conf (BPTTAgentConf): Static agent configuration.
            agent_state (BPTTAgentState): Agent state to start or resume from.
            traj: Trajectory data passed on to ``env.reset`` and ``env.step``; ``None`` if the task has no trajectory.
            mh (MetricsHandler): Optional metrics handler computing the validation metrics.

        Returns:
            dict: Dictionary with the keys ``"agent_state"`` (:class:`BPTTAgentState` after training),
            ``"training_metrics"`` (:class:`~loco_mujoco.algorithms.ppo_jax.PPOSummaryMetrics` with a leading axis
            of length ``num_updates``) and ``"validation_metrics"``
            (:class:`~loco_mujoco.utils.metrics.ValidationSummary` with a leading axis of length ``num_updates``;
            entries of updates without validation are zero).

        """

        config, network, tx = agent_conf.config.experiment, agent_conf.network, agent_conf.tx
        action_dim = config.action_dim
        hidden_state_dim = config.hidden_state_dim
        rnn_type = network.rnn_type
        stored_z_dim = _stored_hidden_dim(hidden_state_dim, rnn_type)

        assert config.num_envs % config.num_minibatches == 0, (
            f"num_envs ({config.num_envs}) must be divisible by num_minibatches ({config.num_minibatches})"
        )
        mb_size = config.num_envs // config.num_minibatches

        env = cls._wrap_env(env, config)
        policy = BPTTPolicy(network, action_dim)

        # ---- init train state ----
        if agent_state is not None:
            train_state = agent_state.train_state.replace(apply_fn=network.apply)
        else:
            rng, _rng1 = jax.random.split(rng)
            obs_dim = env.info.observation_space.shape[0]
            network_params = network.init(
                _rng1,
                jnp.zeros(obs_dim),
                jnp.zeros(stored_z_dim),
                jnp.zeros(action_dim),
            )
            train_state = TrainState.create(
                apply_fn=network.apply,
                params=network_params["params"],
                run_stats=network_params["run_stats"],
                tx=tx,
            )

        # ---- init env state ----
        if agent_state is not None and agent_state.env_state is not None:
            env_state = agent_state.env_state
            obsv = agent_state.last_obs
        else:
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config.num_envs)
            obsv, env_state = env.reset(reset_rng, traj)

        # ---- init hidden state and prev_action ----
        if agent_state is not None and agent_state.hidden_state is not None:
            hidden_state = agent_state.hidden_state
        else:
            hidden_state = jnp.zeros((config.num_envs, stored_z_dim))

        if agent_state is not None and agent_state.prev_action is not None:
            prev_action = agent_state.prev_action
        else:
            prev_action = jnp.zeros((config.num_envs, action_dim))

        train_state_buffer = TrainStateBuffer.create(train_state, config.validation.num)

        # ---- training loop ----
        def _update_step(runner_state, unused):

            # -- save h_init before rollout for BPTT replay --
            train_state, env_state, last_obs, hidden_state, prev_action, train_state_buffer, rng = runner_state
            h_init = hidden_state  # (num_envs, stored_z_dim)

            # -- trajectory collection --
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, hidden_state, prev_action, train_state_buffer, rng = runner_state

                rng, _rng = jax.random.split(rng)
                action, next_hidden, log_prob, value, train_state = \
                    policy.get_action(last_obs, hidden_state, prev_action, train_state, _rng)

                obsv, reward, absorbing, done, info, env_state = env.step(env_state, action, traj)

                # reset hidden state and prev_action on episode termination
                next_hidden = next_hidden * (1 - done)[..., None]
                next_prev_action = action * (1 - done)[..., None]

                log_env_state = env_state.find(LogEnvState)
                logged_metrics = log_env_state.metrics

                transition = BPTTTransition(
                    done, absorbing, action, prev_action,
                    value, reward, log_prob, last_obs, info,
                    env_state.additional_carry.traj_state, logged_metrics
                )
                runner_state = (
                    train_state, env_state, obsv, next_hidden, next_prev_action,
                    train_state_buffer, rng
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config.num_steps)

            # -- advantage estimation --
            train_state, env_state, last_obs, hidden_state, prev_action, train_state_buffer, rng = runner_state
            _, last_val, _, _ = policy.get_dist_and_value(last_obs, hidden_state, prev_action, train_state)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    delta = transition.reward + config.gamma * next_value * (1 - transition.absorbing) - transition.value
                    gae = delta + config.gamma * config.gae_lambda * (1 - transition.done) * gae
                    return (gae, transition.value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # -- policy / value update --
            def _update_epoch(update_state, unused):
                def _update_minibatch(train_state, batch_info):
                    traj_mb, adv_mb, tgt_mb, h_init_mb = batch_info
                    # traj_mb.*: (mb_size, num_steps, ...)
                    # adv_mb, tgt_mb: (mb_size, num_steps)
                    # h_init_mb: (mb_size, stored_z_dim)

                    # Capture run_stats outside the grad (fixed during grad computation)
                    run_stats = train_state.run_stats

                    def _loss_fn(params):
                        # Transpose sequences to (num_steps, mb_size, ...) for scan
                        obs_seq = jnp.transpose(traj_mb.obs, (1, 0) + tuple(range(2, traj_mb.obs.ndim)))
                        done_seq = jnp.transpose(traj_mb.done, (1, 0) + tuple(range(2, traj_mb.done.ndim)))
                        action_seq = jnp.transpose(traj_mb.action, (1, 0) + tuple(range(2, traj_mb.action.ndim)))
                        prev_action_seq = jnp.transpose(traj_mb.prev_action, (1, 0) + tuple(range(2, traj_mb.prev_action.ndim)))

                        def _rnn_step(h, data):
                            obs_t, done_t, action_t, prev_action_t = data
                            (pi, value_t, new_h), _ = network.apply(
                                {"params": params, "run_stats": run_stats},
                                obs_t, h, prev_action_t,
                                mutable=["run_stats"],
                            )
                            log_prob_t = pi.log_prob(action_t)
                            entropy_t = pi.entropy()
                            # Reset hidden state on episode termination (after step)
                            new_h = new_h * (1 - done_t)[..., None]
                            return new_h, (log_prob_t, value_t, entropy_t)

                        _, (log_probs, values, entropies) = jax.lax.scan(
                            _rnn_step, h_init_mb, (obs_seq, done_seq, action_seq, prev_action_seq)
                        )
                        # log_probs, values, entropies: (num_steps, mb_size)

                        # Transpose back to (mb_size, num_steps) then flatten for losses
                        log_probs = jnp.transpose(log_probs, (1, 0))       # (mb_size, num_steps)
                        values = jnp.transpose(values, (1, 0))             # (mb_size, num_steps)
                        entropies = jnp.transpose(entropies, (1, 0))       # (mb_size, num_steps)

                        # old stored quantities are already (mb_size, num_steps)
                        old_log_probs = traj_mb.log_prob                    # (mb_size, num_steps)
                        old_values = traj_mb.value                          # (mb_size, num_steps)

                        # Flatten to (mb_size * num_steps,) for loss computation
                        log_probs_flat = log_probs.reshape(-1)
                        values_flat = values.reshape(-1)
                        entropies_flat = entropies.reshape(-1)
                        old_log_probs_flat = old_log_probs.reshape(-1)
                        old_values_flat = old_values.reshape(-1)
                        adv_flat = adv_mb.reshape(-1)
                        tgt_flat = tgt_mb.reshape(-1)

                        # Value loss (clipped)
                        value_pred_clipped = old_values_flat + (
                            values_flat - old_values_flat
                        ).clip(-config.clip_eps, config.clip_eps)
                        value_loss = 0.5 * jnp.maximum(
                            jnp.square(values_flat - tgt_flat),
                            jnp.square(value_pred_clipped - tgt_flat),
                        ).mean()

                        # PPO actor loss (IS ratio over actions only)
                        ratio = jnp.exp(log_probs_flat - old_log_probs_flat)
                        gae_norm = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
                        loss_actor = -jnp.minimum(
                            ratio * gae_norm,
                            jnp.clip(ratio, 1 - config.clip_eps, 1 + config.clip_eps) * gae_norm,
                        ).mean()

                        entropy = entropies_flat.mean()
                        total_loss = loss_actor + config.vf_coef * value_loss - config.ent_coef * entropy
                        old_approx_kl = (old_log_probs_flat - log_probs_flat).mean()
                        clip_fraction = jnp.mean(jnp.abs(ratio - 1.0) > config.clip_eps)
                        return total_loss, (value_loss, loss_actor, entropy, old_approx_kl, clip_fraction)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, h_init, rng = update_state
                rng, _rng = jax.random.split(rng)

                # Permute over environments (not individual timesteps)
                permutation = jax.random.permutation(_rng, config.num_envs)

                # traj_batch.*: (num_steps, num_envs, ...)
                # Transpose to (num_envs, num_steps, ...), shuffle, reshape to (num_minibatches, mb_size, num_steps, ...)
                def _prepare_traj(x):
                    # x: (num_steps, num_envs, ...)
                    # → (num_envs, num_steps, ...) → shuffle → (num_minibatches, mb_size, num_steps, ...)
                    x_T = jnp.swapaxes(x, 0, 1)  # (num_envs, num_steps, ...)
                    x_shuffled = jnp.take(x_T, permutation, axis=0)
                    return x_shuffled.reshape((config.num_minibatches, mb_size) + x_shuffled.shape[1:])

                def _prepare_flat(x):
                    # x: (num_steps, num_envs) or (num_envs, num_steps) — advantages/targets are (num_steps, num_envs)
                    x_T = jnp.swapaxes(x, 0, 1)  # (num_envs, num_steps)
                    x_shuffled = jnp.take(x_T, permutation, axis=0)
                    return x_shuffled.reshape((config.num_minibatches, mb_size) + x_shuffled.shape[1:])

                def _prepare_h_init(x):
                    # x: (num_envs, stored_z_dim)
                    x_shuffled = jnp.take(x, permutation, axis=0)
                    return x_shuffled.reshape((config.num_minibatches, mb_size) + x_shuffled.shape[1:])

                minibatch_traj = jax.tree.map(_prepare_traj, traj_batch)
                minibatch_adv = _prepare_flat(advantages)   # (num_minibatches, mb_size, num_steps)
                minibatch_tgt = _prepare_flat(targets)      # (num_minibatches, mb_size, num_steps)
                minibatch_h_init = _prepare_h_init(h_init)  # (num_minibatches, mb_size, stored_z_dim)

                minibatches = (minibatch_traj, minibatch_adv, minibatch_tgt, minibatch_h_init)
                train_state, total_loss = jax.lax.scan(_update_minibatch, train_state, minibatches)

                # adaptive KL learning rate
                desired_kl = getattr(config, 'desired_kl', None)
                if desired_kl is not None:
                    mean_kl = jnp.mean(total_loss[1][3])
                    current_lr = train_state.opt_state.inner_state.hyperparams['learning_rate']
                    new_lr = jax.lax.cond(
                        mean_kl > desired_kl * 2.0,
                        lambda lr: jnp.maximum(1e-5, lr / 1.5),
                        lambda lr: jax.lax.cond(
                            (mean_kl < desired_kl / 2.0) & (mean_kl > 0.0),
                            lambda lr: jnp.minimum(1e-2, lr * 1.5),
                            lambda lr: lr, lr,
                        ),
                        current_lr,
                    )
                    new_opt_state = train_state.opt_state._replace(
                        inner_state=train_state.opt_state.inner_state._replace(
                            hyperparams={'learning_rate': new_lr}
                        )
                    )
                    train_state = train_state.replace(opt_state=new_opt_state)

                update_state = (train_state, traj_batch, advantages, targets, h_init, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, h_init, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.update_epochs
            )
            train_state = update_state[0]
            rng = update_state[-1]

            # -- metrics --
            counter = ((train_state.step + 1) // config.num_minibatches) // config.update_epochs
            logged_metrics = traj_batch.metrics

            mean_value_loss = jnp.mean(loss_info[1][0])
            mean_actor_loss = jnp.mean(loss_info[1][1])
            mean_entropy = jnp.mean(loss_info[1][2])
            mean_approx_kl = jnp.mean(loss_info[1][3])
            mean_clip_fraction = jnp.mean(loss_info[1][4])

            desired_kl = getattr(config, 'desired_kl', None)
            if config.anneal_lr:
                current_lr = cls._linear_lr_schedule(
                    train_state.step, config.num_minibatches,
                    config.update_epochs, config.lr, config.num_updates
                )
            elif desired_kl is not None:
                current_lr = train_state.opt_state.inner_state.hyperparams['learning_rate']
            else:
                current_lr = jnp.array(config.lr)

            metric = PPOSummaryMetrics(
                mean_episode_return=jnp.sum(
                    jnp.where(logged_metrics.done, logged_metrics.returned_episode_returns, 0.0)
                ) / jnp.sum(logged_metrics.done),
                mean_episode_length=jnp.sum(
                    jnp.where(logged_metrics.done, logged_metrics.returned_episode_lengths, 0.0)
                ) / jnp.sum(logged_metrics.done),
                max_timestep=jnp.max(logged_metrics.timestep * config.num_envs),
                mean_value_loss=mean_value_loss,
                mean_actor_loss=mean_actor_loss,
                mean_entropy=mean_entropy,
                mean_approx_kl=mean_approx_kl,
                mean_clip_fraction=mean_clip_fraction,
                learning_rate=current_lr,
            )

            # -- validation --
            def _evaluation_step():
                eval_hidden = jnp.zeros((config.validation.num_envs, stored_z_dim))
                eval_prev_action = jnp.zeros((config.validation.num_envs, action_dim))

                def _eval_env(eval_runner_state, unused):
                    train_state, env_state, last_obs, eval_hidden, eval_prev_action, rng = eval_runner_state

                    rng, _rng = jax.random.split(rng)
                    action, next_hidden, _, _, train_state = \
                        policy.get_action(last_obs, eval_hidden, eval_prev_action, train_state, _rng)

                    obsv, reward, absorbing, done, info, env_state = env.step(env_state, action, traj)
                    next_hidden = next_hidden * (1 - done)[..., None]
                    next_prev_action = action * (1 - done)[..., None]

                    log_env_state = env_state.find(LogEnvState)
                    transition = MetricHandlerTransition(env_state, log_env_state.metrics)
                    eval_runner_state = (train_state, env_state, obsv, next_hidden, next_prev_action, rng)
                    return eval_runner_state, transition

                rng = runner_state[-1]
                reset_rng = jax.random.split(rng, config.validation.num_envs)
                obsv, eval_env_state = env.reset(reset_rng, traj)
                eval_runner_state = (train_state, eval_env_state, obsv, eval_hidden, eval_prev_action, rng)

                _, traj_batch_eval = jax.lax.scan(
                    _eval_env, eval_runner_state, None, config.validation.num_steps
                )
                return mh(traj_batch_eval.env_state)

            if mh is None:
                validation_metrics = ValidationSummary()
            else:
                validation_metrics = jax.lax.cond(
                    counter % config.validation_interval == 0,
                    _evaluation_step, mh.get_zero_container
                )

            if config.debug:
                def callback(metrics):
                    return_values = metrics.returned_episode_returns[metrics.done]
                    timesteps = metrics.timestep[metrics.done] * config.num_envs
                    for t in range(len(timesteps)):
                        print(f"global step={timesteps[t]}, episodic return={return_values[t]}")
                jax.debug.callback(callback, env_state.metrics)

            train_state_buffer = jax.lax.cond(
                counter % config.validation_interval == 0,
                lambda x, y: TrainStateBuffer.add(x, y),
                lambda x, y: x,
                train_state_buffer, train_state,
            )

            runner_state = (
                train_state, env_state, last_obs, hidden_state, prev_action,
                train_state_buffer, rng
            )
            return runner_state, (metric, validation_metrics)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, hidden_state, prev_action, train_state_buffer, _rng)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config.num_updates)

        agent_state = cls._agent_state(
            train_state=runner_state[0],
            env_state=runner_state[1],
            last_obs=runner_state[2],
            hidden_state=runner_state[3],
            prev_action=runner_state[4],
        )
        return {
            "agent_state": agent_state,
            "training_metrics": metrics[0],
            "validation_metrics": metrics[1],
        }

    @classmethod
    def play_policy(cls, env,
                    agent_conf: BPTTAgentConf,
                    agent_state: BPTTAgentState,
                    n_envs: int, n_steps=None, render=True,
                    record=False, rng=None, deterministic=False,
                    use_mujoco=False, wrap_env=True,
                    train_state_seed=None, traj=None):
        """
        Roll out a trained recurrent policy in a Python loop and render it.

        Hidden state and previous action start at zero and are reset to zero whenever an episode ends. By default
        the policy runs in ``n_envs`` parallel MJX environments; with ``use_mujoco`` it runs in the CPU MuJoCo
        environment instead. The environment is stopped at the end.

        Args:
            env: The (unwrapped) environment.
            agent_conf (BPTTAgentConf): Agent configuration.
            agent_state (BPTTAgentState): Agent state holding the trained policy.
            n_envs (int): Number of parallel environments; must be 1 with ``use_mujoco``.
            n_steps (int): Number of steps to run. If ``None``, the loop runs until it is interrupted.
            render (bool): Only used to warn about a call that neither renders, records nor terminates; the
                environment is rendered at every step regardless.
            record (bool): Whether to record a video of the MJX rollout. The MuJoCo rollout is always recorded.
            rng (jax.Array): PRNG key for environment resets and action sampling. Defaults to ``jax.random.key(0)``.
            deterministic (bool): If True, the standard deviation of the action distribution is set to zero so that
                the mean action is applied. This overwrites ``log_std_a`` in the parameters of ``agent_state`` in
                place.
            use_mujoco (bool): If True, use the CPU MuJoCo backend instead of MJX.
            wrap_env (bool): If True (and not ``use_mujoco``), the environment is wrapped as during training.
                Pass False if ``env`` is already wrapped.
            train_state_seed (int): Index of the seed to play if the agent was trained with ``n_seeds > 1``.
            traj: Trajectory data passed on to ``env.reset`` and ``env.step`` of the MJX environment.

        """

        if use_mujoco:
            assert n_envs == 1, "Only one mujoco env can be run at a time."

        config = agent_conf.config.experiment
        action_dim = config.action_dim
        stored_z_dim = _stored_hidden_dim(config.hidden_state_dim, agent_conf.network.rnn_type)
        _policy = BPTTPolicy(agent_conf.network, action_dim)

        train_state = agent_state.train_state
        if deterministic:
            train_state.params["log_std_a"] = np.ones_like(train_state.params["log_std_a"]) * -np.inf

        if config.n_seeds > 1:
            assert train_state_seed is not None, (
                "Loaded train state has multiple seeds. Please specify train_state_seed for replay."
            )
            train_state = jax.tree.map(lambda x: x[train_state_seed], train_state)

        if not render and n_steps is None and not record:
            warnings.warn("No rendering, no record, no n_steps specified. This will run forever with no effect.")

        if wrap_env and not use_mujoco:
            env = cls._wrap_env(env, config)

        if rng is None:
            rng = jax.random.key(0)

        keys = jax.random.split(rng, n_envs + 1)
        rng, env_keys = keys[0], keys[1:]

        hidden_state = jnp.zeros((n_envs, stored_z_dim))
        prev_action = jnp.zeros((n_envs, action_dim))

        def sample_actions(ts, obs, hidden, prev_a, _rng):
            action, next_h, _, _, ts = _policy.get_action(obs, hidden, prev_a, ts, _rng)
            return action, next_h, ts

        plcy_call = jax.jit(sample_actions)

        if use_mujoco:
            obs = env.reset()
            env_state = None
        else:
            obs, env_state = env.reset(env_keys, traj)

        if n_steps is None:
            n_steps = np.iinfo(np.int32).max

        for i in range(n_steps):
            rng, _rng = jax.random.split(rng)
            action, next_hidden, train_state = plcy_call(train_state, obs, hidden_state, prev_action, _rng)
            action = jnp.atleast_2d(action)

            if use_mujoco:
                obs, reward, absorbing, done, info = env.step(action)
                done_f = jnp.array(done, dtype=jnp.float32)
                hidden_state = next_hidden * (1 - done_f)[..., None]
                prev_action = action * (1 - done_f)[..., None]
            else:
                obs, reward, absorbing, done, info, env_state = env.step(env_state, action, traj)
                hidden_state = next_hidden * (1 - done)[..., None]
                prev_action = action * (1 - done)[..., None]

            if use_mujoco:
                env.render(record=True)
            else:
                env.mjx_render(env_state, record=record)

            if use_mujoco and done:
                obs = env.reset()
                hidden_state = jnp.zeros((n_envs, stored_z_dim))
                prev_action = jnp.zeros((n_envs, action_dim))

        env.stop()
