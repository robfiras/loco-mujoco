"""
PPO with the Stochastic Stateful Policy Gradient (S2PG-PPO) in JAX.

Implements the stochastic stateful policy of Al-Hafez et al., "Time-Efficient Reinforcement Learning with
Stochastic Stateful Policies" (arXiv:2311.04082), on top of :class:`~loco_mujoco.algorithms.ppo_jax.PPOJax`.
The policy has an internal state ``z`` whose transition is stochastic: at every step the policy samples the
action and the next internal state jointly. The joint sample is treated as the action of a stateless policy,
so that the policy can be trained with the ordinary PPO update and without backpropagation through time.

This algorithm is experimental. Example: ``examples/training_examples/jax_rl_mimic_s2pg``.

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


class IndependentJointDist:
    """
    Joint distribution of two independent distributions (action and next hidden state).

    Samples are the concatenation ``[a, z']``. Because the two parts are independent,
    ``log p(a, z') = log p_a(a) + log p_z(z')`` and ``entropy(a, z') = entropy(a) + entropy(z')``.

    """

    def __init__(self, dist_a: distrax.Distribution, dist_z: distrax.Distribution, action_dim: int):
        """
        Initialize the joint distribution.

        Args:
            dist_a (distrax.Distribution): Distribution over actions.
            dist_z (distrax.Distribution): Distribution over the next hidden state.
            action_dim (int): Dimension of the action, used to split joint samples.

        """
        self.dist_a = dist_a
        self.dist_z = dist_z
        self.action_dim = action_dim

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the log probability of a joint sample.

        Args:
            x (jax.Array): Joint samples ``[a, z']``, shape ``(..., action_dim + stored_hidden_dim)``.

        Returns:
            jax.Array: Sum of the log probabilities of the two parts, shape ``(...)``.

        """
        return self.dist_a.log_prob(x[..., :self.action_dim]) \
             + self.dist_z.log_prob(x[..., self.action_dim:])

    def sample(self, seed: jnp.ndarray) -> jnp.ndarray:
        """
        Sample an action and a next hidden state.

        Args:
            seed (jax.Array): PRNG key; it is split into one key per part.

        Returns:
            jax.Array: Joint samples ``[a, z']``, shape ``(..., action_dim + stored_hidden_dim)``.

        """
        seed_a, seed_z = jax.random.split(seed)
        return jnp.concatenate([self.dist_a.sample(seed=seed_a),
                                 self.dist_z.sample(seed=seed_z)], axis=-1)

    def entropy(self) -> jnp.ndarray:
        """
        Compute the entropy of the joint distribution.

        Returns:
            jax.Array: Sum of the entropies of the two parts.

        """
        return self.dist_a.entropy() + self.dist_z.entropy()


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class ActorCriticS2PG(nn.Module):
    """
    Actor-critic network for S2PG-PPO.

    **Actor.** With ``rnn_type`` set to ``"gru"`` or ``"lstm"``:

    1. The normalized observation and the previous action are embedded separately by a dense layer of width
       ``n_features``.
    2. The concatenated embeddings are fed into a GRU / LSTM cell whose carry is the current hidden state ``z``.
    3. The new carry is the mean of the stochastic next hidden state ``z'``.
    4. A second embedding of the observation is concatenated with the RNN output and mapped linearly to the
       action mean.

    With ``rnn_type="vanilla"`` the two embeddings are concatenated with the hidden state and passed through an
    MLP that outputs the action mean and the mean of the next hidden state.

    The actor yields an :class:`IndependentJointDist` over ``(a, z')`` made of two diagonal Gaussians with
    state-independent standard deviations (parameters ``log_std_a`` and ``log_std_z``).

    **Critic.** An MLP over the concatenation of the normalized observation, the hidden state and the previous
    action, i.e. ``V(s, z, a_prev)``.

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
        init_std_z (float): Initial standard deviation of the hidden-state distribution.
        learnable_std (bool): If False, the gradients with respect to ``log_std_a`` and ``log_std_z`` are
            stopped.
        hidden_layer_dims (Sequence[int]): Hidden layers of the critic MLP and of the ``"vanilla"`` actor MLP.
        ema_coef (float): Exponential smoothing of the mean of the hidden-state transition,
            ``z'_mean = (1 - ema_coef) * z + ema_coef * f(obs, z, a_prev)``. The default ``1.0`` disables
            smoothing; values between 0 and 1 add inertia.

    """

    action_dim: int
    hidden_state_dim: int           # actual RNN hidden size (per vector; LSTM stores 2x)
    rnn_type: str = "gru"           # "gru" | "lstm" | "vanilla"
    n_features: int = 256           # pre / post-RNN embedding width (GRU/LSTM only)
    activation: str = "tanh"
    init_std_a: float = 1.0
    init_std_z: float = 0.1
    learnable_std: bool = True
    hidden_layer_dims: Sequence[int] = (512, 256)   # critic MLP + "vanilla" actor MLP
    ema_coef: float = 1.0           # exponential smoothing for hidden state transition:
                                    # next_z_mean = (1-ema_coef)*z_t + ema_coef*f_theta(...)
                                    # 1.0 = no smoothing (default), 0 < ema_coef < 1 adds inertia

    def setup(self):
        """Resolve the activation function from its name."""
        self.activation_fn = get_activation_fn(self.activation)

    @nn.compact
    def __call__(self, obs, hidden_state, prev_action):
        """
        Compute the joint distribution over action and next hidden state, and the value estimate.

        Args:
            obs (jax.Array): Observations, shape ``(batch, obs_dim)``.
            hidden_state (jax.Array): Current hidden states, shape ``(batch, stored_hidden_dim)``, where
                ``stored_hidden_dim`` is ``2 * hidden_state_dim`` for an LSTM (``h`` and ``c`` concatenated) and
                ``hidden_state_dim`` otherwise.
            prev_action (jax.Array): Previous actions, shape ``(batch, action_dim)``.

        Returns:
            Tuple[IndependentJointDist, jax.Array]: The joint distribution over ``(a, z')`` and the value estimates
            of shape ``(batch,)``.

        Raises:
            ValueError: If ``rnn_type`` is unknown.

        """
        # RunningMeanStd applies atleast_2d then squeeze internally, which
        # strips the batch dim on 1D inputs.  Re-promote all three tensors to
        # 2D so every downstream layer sees a consistent (batch, feat) shape.
        obs_norm = RunningMeanStd()(obs)

        stored_z_dim = _stored_hidden_dim(self.hidden_state_dim, self.rnn_type)

        if self.rnn_type == "vanilla":
            # ---- MLP actor (mirrors the RNN structure) ----
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
            next_hidden_mean = actor_out[..., self.action_dim:]

        else:
            # ---- RNN actor ----
            # Pre-RNN embeddings
            obs_embed = self.activation_fn(nn.Dense(self.n_features)(obs_norm))
            prev_a_embed = self.activation_fn(nn.Dense(self.n_features)(prev_action))
            rnn_input = jnp.concatenate([obs_embed, prev_a_embed], axis=-1)

            if self.rnn_type == "gru":
                new_carry, rnn_out = nn.GRUCell(self.hidden_state_dim)(hidden_state, rnn_input)
                next_hidden_mean = new_carry   # shape (..., hidden_state_dim)

            elif self.rnn_type == "lstm":
                h = hidden_state[..., :self.hidden_state_dim]
                c = hidden_state[..., self.hidden_state_dim:]
                (new_h, new_c), rnn_out = nn.LSTMCell(self.hidden_state_dim)((h, c), rnn_input)
                next_hidden_mean = jnp.concatenate([new_h, new_c], axis=-1)
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

        # Exponential smoothing on the hidden state transition mean:
        #   next_z_mean = (1 - ema_coef) * z_t + ema_coef * f_theta(...)
        # ema_coef=1.0 recovers the unsmoothed transition (default).
        if self.ema_coef < 1.0:
            next_hidden_mean = (1.0 - self.ema_coef) * hidden_state + self.ema_coef * next_hidden_mean

        # Distributions (independent Gaussians for action and next hidden state)
        log_std_a = self.param("log_std_a",
                               nn.initializers.constant(jnp.log(self.init_std_a)),
                               (self.action_dim,))
        log_std_z = self.param("log_std_z",
                               nn.initializers.constant(jnp.log(self.init_std_z)),
                               (stored_z_dim,))
        if not self.learnable_std:
            log_std_a = jax.lax.stop_gradient(log_std_a)
            log_std_z = jax.lax.stop_gradient(log_std_z)

        pi = IndependentJointDist(
            distrax.MultivariateNormalDiag(action_mean, jnp.exp(log_std_a)),
            distrax.MultivariateNormalDiag(next_hidden_mean, jnp.exp(log_std_z)),
            self.action_dim,
        )

        # Critic — MLP over (obs, z, prev_action)
        critic_in = jnp.concatenate([obs_norm, hidden_state, prev_action], axis=-1)
        critic = FullyConnectedNet(
            self.hidden_layer_dims, 1, self.activation, None, False, False
        )(critic_in)

        return pi, jnp.squeeze(critic, axis=-1)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class S2PGPolicy:
    """
    Policy wrapper for S2PG-PPO.

    All methods take the observation, the hidden state and the previous action as separate arguments.

    """

    def __init__(self, network: nn.Module, action_dim: int):
        """
        Initialize the policy wrapper.

        Args:
            network (nn.Module): The :class:`ActorCriticS2PG` network.
            action_dim (int): Dimension of the action space, used to split joint samples.

        """
        self.network = network
        self.action_dim = action_dim

    def get_env_action_and_next_hidden(
        self, obs, hidden_state, prev_action, train_state, rng
    ):
        """
        Sample the joint ``(a, z')`` and split it into the environment action and the next hidden state.

        Args:
            obs (jax.Array): Observations, shape ``(n_envs, obs_dim)``.
            hidden_state (jax.Array): Current hidden states, shape ``(n_envs, stored_hidden_dim)``.
            prev_action (jax.Array): Previous actions, shape ``(n_envs, action_dim)``.
            train_state (TrainState): Current train state with parameters and running statistics.
            rng (jax.Array): PRNG key used for sampling.

        Returns:
            Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, TrainState]: The environment actions, the
            sampled next hidden states, the log probabilities of the joint samples, the value estimates, the joint
            samples ``[a, z']`` and the train state with updated running statistics.

        """
        pi, value, train_state = self._forward_pass(obs, hidden_state, prev_action, train_state)
        joint_action = pi.sample(seed=rng)
        log_prob = pi.log_prob(joint_action)
        env_action = joint_action[..., :self.action_dim]
        next_hidden = joint_action[..., self.action_dim:]
        return env_action, next_hidden, log_prob, value, joint_action, train_state

    def get_dist_and_value(self, obs, hidden_state, prev_action, train_state):
        """
        Compute the joint distribution and the value estimate.

        Args:
            obs (jax.Array): Observations, shape ``(n_envs, obs_dim)``.
            hidden_state (jax.Array): Current hidden states, shape ``(n_envs, stored_hidden_dim)``.
            prev_action (jax.Array): Previous actions, shape ``(n_envs, action_dim)``.
            train_state (TrainState): Current train state with parameters and running statistics.

        Returns:
            Tuple[IndependentJointDist, jax.Array, TrainState]: The joint distribution over ``(a, z')``, the value
            estimates and the train state with updated running statistics.

        """
        return self._forward_pass(obs, hidden_state, prev_action, train_state)

    def _forward_pass(self, obs, hidden_state, prev_action, train_state):
        """Run a forward pass of the network and write the updated running statistics to the train state."""
        y, updates = self.network.apply(
            {"params": train_state.params, "run_stats": train_state.run_stats},
            obs, hidden_state, prev_action,
            mutable=["run_stats"],
        )
        pi, value = y
        return pi, value, train_state.replace(run_stats=updates["run_stats"])


# ---------------------------------------------------------------------------
# Transition
# ---------------------------------------------------------------------------

class S2PGTransition(NamedTuple):
    """
    One step of experience collected by S2PG-PPO.

    Observation, hidden state and previous action are stored separately.

    Attributes:
        done (jax.Array): Whether the episode ended at this step.
        absorbing (jax.Array): Whether the state reached at this step is absorbing.
        action (jax.Array): Joint sample ``[a_t, z_{t+1}]``, shape ``(action_dim + stored_hidden_dim,)``.
        hidden_state (jax.Array): Hidden state ``z_t`` the policy was conditioned on at this step.
        prev_action (jax.Array): Previous environment action ``a_{t-1}`` (zeros at the start of an episode).
        value (jax.Array): Critic estimate at this step.
        reward (jax.Array): Reward received.
        log_prob (jax.Array): Log probability of the joint sample under the data-collecting policy.
        obs (jax.Array): Raw environment observation.
        info (dict): Info dictionary returned by the environment step.
        traj_state (TrajState): Trajectory state of the environment after the step.
        metrics (Metrics): Episode statistics of the :class:`~loco_mujoco.core.wrappers.mjx.LogWrapper`.

    """
    done: jnp.ndarray
    absorbing: jnp.ndarray
    action: jnp.ndarray         # joint action concat([a, z'])
    hidden_state: jnp.ndarray   # z_t (current hidden state at this step)
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
class S2PGAgentConf(AgentConfBase):
    """
    Static configuration of an S2PG-PPO agent.

    Attributes:
        config (DictConfig): The full configuration; the hyperparameters live under ``config.experiment``.
        network (ActorCriticS2PG): The actor-critic network definition.
        tx (optax.GradientTransformation): The optimizer.

    """
    config: DictConfig
    network: ActorCriticS2PG
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
            S2PGAgentConf: The restored configuration with a newly built optimizer.

        """
        config = OmegaConf.create(d["config"])
        tx = S2PGPPOJax._get_optimizer(config)
        return cls(config=config,
                   network=flax.serialization.from_state_dict(ActorCriticS2PG, d["network"]),
                   tx=tx)


@struct.dataclass
class S2PGAgentState(AgentStateBase):
    """
    Dynamic state of an S2PG-PPO agent.

    Attributes:
        train_state (TrainState): Parameters, optimizer state and running observation statistics of the network.
        env_state: Environment state at the end of the last training chunk; not saved to disk.
        last_obs (jax.Array): Last observations of the previous training chunk; not saved to disk.
        hidden_state (jax.Array): Hidden states ``z`` at the end of the last training chunk, shape
            ``(num_envs, stored_hidden_dim)``; not saved to disk.
        prev_action (jax.Array): Last actions of the previous training chunk, shape ``(num_envs, action_dim)``;
            not saved to disk.

    """
    train_state: TrainState
    env_state: Any = None
    last_obs: Any = None
    hidden_state: Any = None    # z carried across chunks
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
            agent_conf (S2PGAgentConf): Agent configuration providing the network and the optimizer.

        Returns:
            S2PGAgentState: The restored state, including the optimizer state.

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

class S2PGPPOJax(PPOJax):
    """
    S2PG-PPO: Proximal Policy Optimization with the Stochastic Stateful Policy Gradient.

    Key properties:

    * The policy jointly models ``(action, next_hidden_state)`` as two independent Gaussians; their summed log
      probability enters the importance-sampling ratio, so no backpropagation through time is required.
    * The hidden state ``z`` is propagated as the GRU/LSTM carry; its stochastic transition replaces BPTT with
      a score-function estimator.
    * The previous action ``a_{t-1}`` is fed to the RNN input and to the critic.
    * The value function is ``V(s, z, a_prev)``.
    * Hidden state and previous action are reset to zero when an episode ends.

    Supported ``rnn_type`` values: ``"gru"`` (default), ``"lstm"`` and ``"vanilla"`` (MLP fallback).

    Reference: Firas Al-Hafez et al., "Time-Efficient Reinforcement Learning with Stochastic Stateful Policies",
    arXiv:2311.04082.

    The interface is that of :class:`~loco_mujoco.algorithms.ppo_jax.PPOJax`; see
    ``examples/training_examples/jax_rl_mimic_s2pg`` for a training script. Reward normalization
    (``normalize_env``) is not applied by this algorithm.

    """

    _agent_conf = S2PGAgentConf
    _agent_state = S2PGAgentState

    @classmethod
    def init_agent_conf(cls, env, config):
        """
        Create the static agent configuration: derived hyperparameters, the network and the optimizer.

        In addition to the PPO hyperparameters (see :meth:`PPOJax.init_agent_conf
        <loco_mujoco.algorithms.ppo_jax.PPOJax.init_agent_conf>`), the following keys are read from
        ``config.experiment``:

        - ``hidden_state_dim``: size of the internal state ``z`` (the RNN hidden state).
        - ``rnn_type`` (optional, default ``"gru"``): ``"gru"``, ``"lstm"`` or ``"vanilla"``.
        - ``n_features`` (optional, default 256): width of the embeddings before and after the RNN.
        - ``init_std_z`` (optional, default 0.1): initial standard deviation of the hidden-state distribution.
        - ``ema_coef`` (optional, default 1.0): exponential smoothing of the hidden-state transition mean.

        The keys ``num_updates``, ``minibatch_size``, ``validation_interval``, ``action_dim`` and a recomputed
        ``validation.num`` are written to ``config.experiment`` in place.

        Args:
            env: The (unwrapped) environment the agent is trained on.
            config (DictConfig): Configuration with the hyperparameters under ``config.experiment``.

        Returns:
            S2PGAgentConf: The agent configuration.

        """
        with open_dict(config.experiment):
            config.experiment.num_updates = (
                config.experiment.total_timesteps // config.experiment.num_steps // config.experiment.num_envs)
            config.experiment.minibatch_size = (
                config.experiment.num_envs * config.experiment.num_steps // config.experiment.num_minibatches)
            config.experiment.validation_interval, config.experiment.validation.num = cls._validation_schedule(
                config.experiment.num_updates, config.experiment.validation.num)
            config.experiment.action_dim = env.info.action_space.shape[0]

        hidden_layers = config.experiment.hidden_layers \
            if isinstance(config.experiment.hidden_layers, (list, ListConfig)) \
            else ast.literal_eval(config.experiment.hidden_layers)

        hidden_state_dim = config.experiment.hidden_state_dim
        rnn_type = getattr(config.experiment, 'rnn_type', 'gru')
        n_features = getattr(config.experiment, 'n_features', 256)
        init_std_z = getattr(config.experiment, 'init_std_z', 0.1)
        ema_coef = getattr(config.experiment, 'ema_coef', 1.0)

        network = ActorCriticS2PG(
            action_dim=env.info.action_space.shape[0],
            hidden_state_dim=hidden_state_dim,
            rnn_type=rnn_type,
            n_features=n_features,
            activation=config.experiment.activation,
            init_std_a=config.experiment.init_std,
            init_std_z=init_std_z,
            learnable_std=config.experiment.learnable_std,
            hidden_layer_dims=hidden_layers,
            ema_coef=ema_coef,
        )
        tx = cls._get_optimizer(config)
        return cls._agent_conf(config, network, tx)

    @classmethod
    def init_agent_state(cls, env, agent_conf: S2PGAgentConf, rng) -> S2PGAgentState:
        """
        Initialize the agent state (network parameters, running statistics and optimizer state).

        Args:
            env: The (unwrapped) environment the agent is trained on.
            agent_conf (S2PGAgentConf): Agent configuration created by :meth:`init_agent_conf`.
            rng (jax.Array): PRNG key used to initialize the network parameters.

        Returns:
            S2PGAgentState: The initial agent state; the carried environment state, observation, hidden state and
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
                  agent_conf: S2PGAgentConf,
                  agent_state: S2PGAgentState = None,
                  traj=None,
                  mh: MetricsHandler = None):
        """
        Run S2PG-PPO for ``num_updates`` updates.

        Every update collects a rollout of ``num_steps`` steps in which the policy samples the environment action
        and the next hidden state jointly, computes advantages with GAE and runs ``update_epochs`` epochs of
        minibatch updates over individually shuffled time steps. The PPO losses use the log probability of the
        stored joint sample given the stored observation, hidden state and previous action. If ``agent_state``
        carries an environment state, hidden state and previous action from a previous call, training continues
        from them.

        Args:
            rng (jax.Array): PRNG key.
            env: The (unwrapped) environment.
            agent_conf (S2PGAgentConf): Static agent configuration.
            agent_state (S2PGAgentState): Agent state to start or resume from.
            traj: Trajectory data passed on to ``env.reset`` and ``env.step``; ``None`` if the task has no trajectory.
            mh (MetricsHandler): Optional metrics handler computing the validation metrics.

        Returns:
            dict: Dictionary with the keys ``"agent_state"`` (:class:`S2PGAgentState` after training),
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

        env = cls._wrap_env(env, config)
        policy = S2PGPolicy(network, action_dim)

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

        train_state_buffer = TrainStateBuffer.create(train_state, max(config.validation.num, 1))

        # ---- training loop ----
        def _update_step(runner_state, unused):

            # -- trajectory collection --
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, hidden_state, prev_action, train_state_buffer, rng = runner_state

                rng, _rng = jax.random.split(rng)
                env_action, next_hidden, log_prob, value, joint_action, train_state = \
                    policy.get_env_action_and_next_hidden(
                        last_obs, hidden_state, prev_action, train_state, _rng
                    )

                obsv, reward, absorbing, done, info, env_state = env.step(env_state, env_action, traj)

                # reset hidden state and prev_action on episode termination
                next_hidden = next_hidden * (1 - done)[..., None]
                next_prev_action = env_action * (1 - done)[..., None]

                log_env_state = env_state.find(LogEnvState)
                logged_metrics = log_env_state.metrics

                transition = S2PGTransition(
                    done, absorbing, joint_action, hidden_state, prev_action,
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
            _, last_val, _ = policy.get_dist_and_value(last_obs, hidden_state, prev_action, train_state)

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
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        pi, value, _ = policy.get_dist_and_value(
                            traj_batch.obs,
                            traj_batch.hidden_state,
                            traj_batch.prev_action,
                            train_state.replace(params=params),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # value loss
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config.clip_eps, config.clip_eps)
                        value_loss = 0.5 * jnp.maximum(
                            jnp.square(value - targets),
                            jnp.square(value_pred_clipped - targets),
                        ).mean()

                        # PPO actor loss with joint IS ratio
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor = -jnp.minimum(
                            ratio * gae,
                            jnp.clip(ratio, 1 - config.clip_eps, 1 + config.clip_eps) * gae,
                        ).mean()

                        entropy = pi.entropy().mean()
                        total_loss = loss_actor + config.vf_coef * value_loss - config.ent_coef * entropy
                        old_approx_kl = (traj_batch.log_prob - log_prob).mean()
                        clip_fraction = jnp.mean(jnp.abs(ratio - 1.0) > config.clip_eps)
                        return total_loss, (value_loss, loss_actor, entropy, old_approx_kl, clip_fraction)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config.minibatch_size * config.num_minibatches
                assert (
                    batch_size == config.num_steps * config.num_envs
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(x, [config.num_minibatches, -1] + list(x.shape[1:])),
                    shuffled_batch,
                )
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

                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
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
                    env_action, next_hidden, _, _, _, train_state = \
                        policy.get_env_action_and_next_hidden(
                            last_obs, eval_hidden, eval_prev_action, train_state, _rng
                        )

                    obsv, reward, absorbing, done, info, env_state = env.step(env_state, env_action, traj)
                    next_hidden = next_hidden * (1 - done)[..., None]
                    next_prev_action = env_action * (1 - done)[..., None]

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
                    agent_conf: S2PGAgentConf,
                    agent_state: S2PGAgentState,
                    n_envs: int, n_steps=None, render=True,
                    record=False, rng=None, deterministic=False,
                    use_mujoco=False, wrap_env=True,
                    train_state_seed=None, traj=None):
        """
        Roll out a trained stateful policy in a Python loop and render it.

        Hidden state and previous action start at zero and are reset to zero whenever an episode ends. By default
        the policy runs in ``n_envs`` parallel MJX environments; with ``use_mujoco`` it runs in the CPU MuJoCo
        environment instead. The environment is stopped at the end.

        Args:
            env: The (unwrapped) environment.
            agent_conf (S2PGAgentConf): Agent configuration.
            agent_state (S2PGAgentState): Agent state holding the trained policy.
            n_envs (int): Number of parallel environments; must be 1 with ``use_mujoco``.
            n_steps (int): Number of steps to run. If ``None``, the loop runs until it is interrupted.
            render (bool): Only used to warn about a call that neither renders, records nor terminates; the
                environment is rendered at every step regardless.
            record (bool): Whether to record a video of the MJX rollout. The MuJoCo rollout is always recorded.
            rng (jax.Array): PRNG key for environment resets and action sampling. Defaults to ``jax.random.key(0)``.
            deterministic (bool): If True, the standard deviation of the action distribution is set to zero so that
                the mean action is applied. This overwrites ``log_std_a`` in the parameters of ``agent_state`` in
                place. The transition of the hidden state remains stochastic.
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
        _policy = S2PGPolicy(agent_conf.network, action_dim)

        train_state = agent_state.train_state
        if deterministic:
            # replace log_std in a copy, so that the parameters of the given agent_state are not modified
            train_state = train_state.replace(params={
                **train_state.params, "log_std_a": np.ones_like(train_state.params["log_std_a"]) * -np.inf})

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
            env_a, next_h, _, _, _, ts = _policy.get_env_action_and_next_hidden(
                obs, hidden, prev_a, ts, _rng
            )
            return env_a, next_h, ts

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
