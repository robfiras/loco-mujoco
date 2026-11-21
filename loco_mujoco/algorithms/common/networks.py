"""
Flax network building blocks shared by the learning algorithms.

Contains a fully connected network, the Gaussian actor-critic used by PPO and the algorithms built on it, and
a running mean/standard-deviation layer for input normalization.

"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence
import distrax


def get_activation_fn(name: str):
    """
    Get an activation function by its name from the ``flax.linen`` module.

    Args:
        name (str): Name of the activation function as defined in ``flax.linen``, e.g. ``"tanh"`` or ``"relu"``.

    Returns:
        Callable: The activation function.

    Raises:
        ValueError: If ``flax.linen`` has no attribute with that name.

    """
    try:
        # Use getattr to dynamically retrieve the activation function from jax.nn
        return getattr(nn, name)
    except AttributeError:
        raise ValueError(f"Activation function '{name}' not found. Name must be the same as in flax.linen!")


class FullyConnectedNet(nn.Module):
    """
    Fully connected network (MLP) with optional input normalization.

    Hidden layers are initialized orthogonally with gain ``sqrt(2)``, the output layer orthogonally with gain
    ``0.01``; all biases are initialized to zero.

    Attributes:
        hidden_layer_dims (Sequence[int]): Number of units of each hidden layer.
        output_dim (int): Number of output units.
        activation (str): Name of the hidden-layer activation function in ``flax.linen``.
        output_activation (str): Name of the output activation function in ``flax.linen``; ``None`` means a
            linear output.
        use_running_mean_stand (bool): If True, the input is normalized by a :class:`RunningMeanStd` layer. The
            network then has a ``run_stats`` variable collection and has to be applied with
            ``mutable=["run_stats"]``.
        squeeze_output (bool): If True, all singleton dimensions are removed from the output.

    """

    hidden_layer_dims: Sequence[int]
    output_dim: int
    activation: str = "tanh"
    output_activation: str = None    # none means linear activation
    use_running_mean_stand: bool = True
    squeeze_output: bool = True

    def setup(self):
        """Resolve the activation functions from their names."""
        self.activation_fn = get_activation_fn(self.activation)
        self.output_activation_fn = get_activation_fn(self.output_activation) \
            if self.output_activation is not None else lambda x: x

    @nn.compact
    def __call__(self, x):
        """
        Apply the network.

        Args:
            x (jax.Array): Input of shape ``(batch, input_dim)`` or ``(input_dim,)``. With input normalization, the
                output always has a leading batch dimension (unless it is squeezed away).

        Returns:
            jax.Array: Output of shape ``(batch, output_dim)``, with singleton dimensions removed if
            ``squeeze_output`` is True.

        """

        if self.use_running_mean_stand:
            x = RunningMeanStd()(x)

        # build network
        for i, dim_layer in enumerate(self.hidden_layer_dims):
            x = nn.Dense(dim_layer, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = self.activation_fn(x)

        # add last layer
        x = nn.Dense(self.output_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        x = self.output_activation_fn(x)

        return jnp.squeeze(x) if self.squeeze_output else x


class ActorCritic(nn.Module):
    """
    Actor-critic network with a diagonal Gaussian policy and a state-value function.

    The input observation is normalized by a :class:`RunningMeanStd` layer, then actor and critic, two separate
    :class:`FullyConnectedNet` of the same hidden layout, each read their own subset of the normalized
    observation. The standard deviation of the policy is a state-independent parameter ``log_std``. The network
    has a ``run_stats`` variable collection and has to be applied with ``mutable=["run_stats"]``.

    Attributes:
        action_dim (int): Dimension of the action space.
        activation (str): Name of the hidden-layer activation function in ``flax.linen``.
        init_std (float): Initial standard deviation of the policy (same for all action dimensions).
        learnable_std (bool): If False, the gradient with respect to ``log_std`` is stopped, so the standard
            deviation stays at ``init_std``.
        hidden_layer_dims (Sequence[int]): Number of units of each hidden layer of the actor and of the critic.
        actor_obs_ind (jax.Array): Indices of the observation entries the actor sees; ``None`` means all.
        critic_obs_ind (jax.Array): Indices of the observation entries the critic sees; ``None`` means all.

    """
    action_dim: Sequence[int]
    activation: str = "tanh"
    init_std: float = 1.0
    learnable_std: bool = True
    hidden_layer_dims: Sequence[int] = (1024, 512)
    actor_obs_ind: jnp.ndarray = None
    critic_obs_ind: jnp.ndarray = None

    def setup(self):
        """Resolve the activation function from its name."""
        self.activation_fn = get_activation_fn(self.activation)

    @nn.compact
    def __call__(self, x):
        """
        Compute the action distribution and the value estimate.

        Args:
            x (jax.Array): Observations of shape ``(batch, obs_dim)`` or ``(obs_dim,)``; a single observation is
                treated as a batch of size one.

        Returns:
            Tuple[distrax.MultivariateNormalDiag, jax.Array]: The action distribution with event shape
            ``(action_dim,)`` and batch shape ``(batch,)``, and the value estimates of shape ``(batch,)``.

        """

        x = RunningMeanStd()(x)

        # build actor
        actor_x = x if self.actor_obs_ind is None else x[..., self.actor_obs_ind]
        actor_mean = FullyConnectedNet(self.hidden_layer_dims, self.action_dim, self.activation,
                                       None, False, False)(actor_x)
        actor_logtstd = self.param("log_std", nn.initializers.constant(jnp.log(self.init_std)),
                                   (self.action_dim,))
        if not self.learnable_std:
            actor_logtstd = jax.lax.stop_gradient(actor_logtstd)

        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        # build critic
        critic_x = x if self.critic_obs_ind is None else x[..., self.critic_obs_ind]
        critic = FullyConnectedNet(self.hidden_layer_dims, 1, self.activation, None, False, False)(critic_x)

        return pi, jnp.squeeze(critic, axis=-1)


class RunningMeanStd(nn.Module):
    """
    Layer that normalizes its input with a running estimate of the mean and variance.

    The statistics are stored in the ``run_stats`` variable collection (``mean``, ``var`` and ``count``). Every
    call first updates them with the incoming batch and then normalizes the batch with the updated statistics, so
    the module has to be applied with ``mutable=["run_stats"]``. Whether the update persists is up to the caller:
    it does only if the returned ``run_stats`` are written back to the train state.

    """

    @nn.compact
    def __call__(self, x):
        """
        Update the running statistics with ``x`` and return the normalized input.

        Args:
            x (jax.Array): Input of shape ``(batch, dim)`` or ``(dim,)``; the statistics are taken over the batch
                axis.

        Returns:
            jax.Array: Normalized input of shape ``(batch, dim)`` (a single input gets a batch dimension of one).

        """

        x = jnp.atleast_2d(x)

        # Initialize running mean, variance, and count
        mean = self.variable('run_stats', 'mean', lambda: jnp.zeros(x.shape[-1]))
        var = self.variable('run_stats', 'var', lambda: jnp.ones(x.shape[-1]))
        count = self.variable('run_stats', 'count', lambda: jnp.array(1e-6))

        # Compute batch mean and variance
        batch_mean = jnp.mean(x, axis=0)
        batch_var = jnp.var(x, axis=0) + 1e-6  # Add epsilon for numerical stability
        batch_count = x.shape[0]

        # Update counts
        updated_count = count.value + batch_count

        # Numerically stable mean and variance update
        delta = batch_mean - mean.value
        new_mean = mean.value + delta * batch_count / updated_count

        # Compute the new variance using Welford's method
        m_a = var.value * count.value
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * count.value * batch_count / updated_count
        new_var = M2 / updated_count

        # Normalize input
        normalized_x = (x - new_mean) / jnp.sqrt(new_var + 1e-8)

        # Update state variables
        mean.value = new_mean
        var.value = new_var
        count.value = updated_count

        return normalized_x
