"""
Adversarial Motion Priors (AMP) in JAX.

Implements the adversarial imitation scheme of Peng et al., "AMP: Adversarial Motion Priors for Stylized
Physics-Based Character Control" (2021): a least-squares discriminator provides a style reward that is
combined with the task reward of the environment. The implementation reuses the training loop of
:class:`~loco_mujoco.algorithms.gail_jax.GAILJax` and only replaces the discriminator loss and the reward
computed from the discriminator output.

Example: ``examples/training_examples/jax_amp``.

"""

import jax.numpy as jnp

from loco_mujoco.algorithms import GAILJax


class AMPJax(GAILJax):
    """
    Adversarial Motion Priors (Peng et al., 2021) with PPO as policy optimizer.

    Subclass of :class:`~loco_mujoco.algorithms.gail_jax.GAILJax` with the same interface, configuration keys and
    training loop. It differs in two points: the discriminator is trained with a least-squares loss instead of
    the binary cross entropy, and the style reward is ``max(0, 1 - 0.25 * (D(s) - 1)^2)``, where ``D(s)`` is the
    raw discriminator output. The style reward is mixed with the environment (task) reward through
    ``proportion_env_reward``, e.g. a velocity-tracking reward in ``examples/training_examples/jax_amp``.

    """

    @classmethod
    def _predict_rewards(cls, obs, discriminator, disc_train_state):
        """
        Compute the style reward ``max(0, 1 - 0.25 * (D(s) - 1)^2)`` for a batch of observations.

        The running input statistics of the discriminator are not updated.

        Args:
            obs (jax.Array): Observations, shape ``(batch, obs_dim)``.
            discriminator (FullyConnectedNet): The discriminator network definition.
            disc_train_state (TrainState): Train state of the discriminator.

        Returns:
            jax.Array: Rewards in ``[0, 1]``, shape ``(batch,)``.

        """
        logits, _ = discriminator.apply({'params': disc_train_state.params,
                                         'run_stats': disc_train_state.run_stats},
                                        obs, mutable=["run_stats"])

        reward = jnp.maximum(0.0, 1 - 0.25*jnp.square(logits - 1))

        return reward

    @classmethod
    def _discriminator_loss(cls, config, logits, targets):
        """
        Compute the least-squares discriminator loss.

        Args:
            config (DictConfig): The ``experiment`` node of the configuration (unused).
            logits (jax.Array): Raw discriminator outputs, shape ``(batch,)``.
            targets (jax.Array): Regression targets, shape ``(batch,)``.

        Returns:
            Tuple[jax.Array, jax.Array]: The scalar mean squared error and the unchanged discriminator outputs.

        """

        # least squares loss
        total_loss = jnp.mean(jnp.square(logits - targets))

        return total_loss, logits

    @classmethod
    def _get_discriminator_targets(cls, plcy_batch_size, expert_batch_size):
        """
        Create the least-squares targets: -1 for policy samples, +1 for expert samples.

        Note:
            The inherited training function currently does not call this method; it uses the labels 0 (policy)
            and 1 (expert).

        Args:
            plcy_batch_size (int): Number of policy samples.
            expert_batch_size (int): Number of expert samples.

        Returns:
            Tuple[jax.Array, jax.Array]: Targets of the policy samples ``(plcy_batch_size,)`` and of the expert
            samples ``(expert_batch_size,)``.

        """
        plcy_target = -1 * jnp.ones(shape=(plcy_batch_size,))
        expert_target = 1 * jnp.ones(shape=(expert_batch_size,))
        return plcy_target, expert_target
