import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Tuple
import distrax

from loco_mujoco.algorithms.common.dataclasses import TrainState


class PPOPolicy:
    """
    Thin policy wrapper around a Flax actor-critic network.

    The wrapped network is expected to return `(pi, value)` and expose
    mutable `run_stats` when called with `mutable=["run_stats"]`.
    """

    def __init__(self, network: nn.Module):
        """
        Initialize the policy wrapper.

        Args:
            network: Flax module used for policy/value inference.
        """
        self.network = network

    def get_action_and_value(
        self, obs: jnp.ndarray, train_state: TrainState, rng: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, TrainState]:
        """
        Sample an action and return the log probability and value estimate.

        Args:
            obs: Observation batch.
            train_state: Current training state with params and run stats.
            rng: PRNG key used for action sampling.

        Returns:
            A tuple `(action, log_prob, value, updated_train_state)`.
        """
        pi, value, train_state = self._forward_pass(obs, train_state)
        action = pi.sample(seed=rng)
        log_prob = pi.log_prob(action)
        return action, log_prob, value, train_state

    def get_dist_and_value(
        self, obs: jnp.ndarray, train_state: TrainState
    ) -> Tuple[distrax.Distribution, jnp.ndarray, TrainState]:
        """
        Compute action distribution and value estimate for observations.

        Args:
            obs: Observation batch.
            train_state: Current training state with params and run stats.

        Returns:
            A tuple `(distribution, value, updated_train_state)`.
        """
        pi, value, train_state = self._forward_pass(obs, train_state)
        return pi, value, train_state

    def _forward_pass(
        self, obs: jnp.ndarray, train_state: TrainState
    ) -> Tuple[distrax.Distribution, jnp.ndarray, TrainState]:
        """
        Run a forward pass and update running statistics.

        Args:
            obs: Observation batch.
            train_state: Current training state.

        Returns:
            A tuple `(distribution, value, updated_train_state)`.
        """
        y, updates = self.network.apply(
            {"params": train_state.params, "run_stats": train_state.run_stats},
            obs,
            mutable=["run_stats"],
        )
        pi, value = y
        return pi, value, train_state.replace(run_stats=updates["run_stats"])

