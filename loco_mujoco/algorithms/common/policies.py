"""Policy wrappers that turn a Flax network and a train state into action and value queries."""

import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Tuple
import distrax

from loco_mujoco.algorithms.common.dataclasses import TrainState



class PPOPolicy:
    """
    Thin policy wrapper around a Flax actor-critic network.

    The wrapped network is expected to return ``(pi, value)`` and to expose a mutable ``run_stats`` collection
    when applied with ``mutable=["run_stats"]``, as :class:`~loco_mujoco.algorithms.common.networks.ActorCritic`
    does. All methods return the train state with the updated running statistics.

    """

    def __init__(self, network: nn.Module):
        """
        Initialize the policy wrapper.

        Args:
            network (nn.Module): Flax module used for policy/value inference.

        """
        self.network = network

    def get_action_and_value(
        self, obs: jnp.ndarray, train_state: TrainState, rng: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, TrainState]:
        """
        Sample an action and return its log probability and the value estimate.

        Args:
            obs (jax.Array): Observation batch of shape ``(n_envs, obs_dim)``.
            train_state (TrainState): Current train state with parameters and running statistics.
            rng (jax.Array): PRNG key used for action sampling.

        Returns:
            Tuple[jax.Array, jax.Array, jax.Array, TrainState]: The sampled actions ``(n_envs, action_dim)``, their
            log probabilities ``(n_envs,)``, the value estimates ``(n_envs,)`` and the train state with updated
            running statistics.

        """
        pi, value, train_state = self._forward_pass(obs, train_state)
        action = pi.sample(seed=rng)
        log_prob = pi.log_prob(action)
        return action, log_prob, value, train_state

    def get_dist_and_value(
        self, obs: jnp.ndarray, train_state: TrainState
    ) -> Tuple[distrax.Distribution, jnp.ndarray, TrainState]:
        """
        Compute the action distribution and the value estimate for a batch of observations.

        Args:
            obs (jax.Array): Observation batch of shape ``(n_envs, obs_dim)``.
            train_state (TrainState): Current train state with parameters and running statistics.

        Returns:
            Tuple[distrax.Distribution, jax.Array, TrainState]: The action distribution, the value estimates
            ``(n_envs,)`` and the train state with updated running statistics.

        """
        pi, value, train_state = self._forward_pass(obs, train_state)
        return pi, value, train_state

    def _forward_pass(
        self, obs: jnp.ndarray, train_state: TrainState
    ) -> Tuple[distrax.Distribution, jnp.ndarray, TrainState]:
        """
        Run a forward pass of the network and update the running statistics.

        Args:
            obs (jax.Array): Observation batch of shape ``(n_envs, obs_dim)``.
            train_state (TrainState): Current train state.

        Returns:
            Tuple[distrax.Distribution, jax.Array, TrainState]: The action distribution, the value estimates and
            the train state with updated running statistics.

        """
        y, updates = self.network.apply(
            {"params": train_state.params, "run_stats": train_state.run_stats},
            obs,
            mutable=["run_stats"],
        )
        pi, value = y
        return pi, value, train_state.replace(run_stats=updates["run_stats"])
