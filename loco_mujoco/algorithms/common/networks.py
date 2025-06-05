import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence
import distrax


def get_activation_fn(name: str):
    """ Get activation function by name from the flax.linen module."""
    try:
        # Use getattr to dynamically retrieve the activation function from jax.nn
        return getattr(nn, name)
    except AttributeError:
        raise ValueError(f"Activation function '{name}' not found. Name must be the same as in flax.linen!")


class FullyConnectedNet(nn.Module):

    hidden_layer_dims: Sequence[int]
    output_dim: int
    activation: str = "tanh"
    output_activation: str = None    # none means linear activation
    use_running_mean_stand: bool = True
    squeeze_output: bool = True

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation)
        self.output_activation_fn = get_activation_fn(self.output_activation) \
            if self.output_activation is not None else lambda x: x

    @nn.compact
    def __call__(self, x):

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


class FullyConnectedNetSkeletonMuscle(nn.Module):
    hidden_layer_dims: Sequence[int]
    output_dim: int
    number_upper_body_activation: int  # number of upper body motors
    custom_output_activation: Sequence[str] #= ("tanh", "sigmoid") # #str = None    # none means linear activation
    activation: str = "tanh"
    output_activation: str = None    # none means linear activation
    use_running_mean_stand: bool = True
    squeeze_output: bool = True

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation)
        
        self.output_activation_fn = get_activation_fn(self.output_activation) \
            if self.output_activation is not None else lambda x: x
        # jax.debug.print("custom_output_activation: {custom_output_activation}", custom_output_activation=self.custom_output_activation)
        self.custom_output_activation_fns = [get_activation_fn(act) for act in self.custom_output_activation]

    @nn.compact
    def __call__(self, x):

        if self.use_running_mean_stand:
            x = RunningMeanStd()(x)

        # build network
        for i, dim_layer in enumerate(self.hidden_layer_dims):
            x = nn.Dense(dim_layer, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = self.activation_fn(x)

        # add last layer
        x = nn.Dense(self.output_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        
        # Apply the first activation function (e.g., tanh) to the first 14 outputs
        first_activation_fn = self.custom_output_activation_fns[0]
        second_activation_fn = self.custom_output_activation_fns[1]

        x_first = jnp.stack([first_activation_fn(x[..., i]) for i in range(self.number_upper_body_activation)], axis=-1)

        jax.debug.print("x_first: {x_first}", x_first=x_first)

        jax.debug.print("Inputs to sigmoid: {inputs}", inputs=x[..., i])

        # Apply the second activation function (e.g., sigmoid) to the remaining outputs
        x_rest = jnp.stack([second_activation_fn(x[..., i]) for i in range(self.number_upper_body_activation, self.output_dim)], axis=-1)

        # Check id x_rest is between 0 and 1 
        # jax.debug.print("x_rest: {x_rest}", x_rest=x_rest)
        x_rest_out_of_bounds = jnp.logical_or(jnp.any(x_rest <= 0), jnp.any(x_rest >= 1))
        jax.debug.print("x_rest_out_of_bounds: {x_rest_out_of_bounds}", x_rest_out_of_bounds=x_rest_out_of_bounds)

        jax.debug.print("x_rest: {x_rest}", x_rest=x_rest)

        # Concatenate the results
        x = jnp.concatenate([x_first, x_rest], axis=-1)

        # jax.debug.print("x: {x}", x=x)

        return jnp.squeeze(x) if self.squeeze_output else x




class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    init_std: float = 1.0
    learnable_std: bool = True
    hidden_layer_dims: Sequence[int] = (1024, 512)
    actor_obs_ind: jnp.ndarray = None
    critic_obs_ind: jnp.ndarray = None

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation)

    @nn.compact
    def __call__(self, x):

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


class ActorCriticSkeletonMuscle(nn.Module):
    action_dim: Sequence[int]
    number_upper_body_activation: int # number of upper body motors 
    custom_output_activation: Sequence[str]  # e.g. ("tanh", "sigmoid")
    activation: str = "tanh"
    init_std: float = 1.0
    learnable_std: bool = True
    hidden_layer_dims: Sequence[int] = (1024, 512)
    actor_obs_ind: jnp.ndarray = None
    critic_obs_ind: jnp.ndarray = None

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation)

    @nn.compact
    def __call__(self, x):

        x = RunningMeanStd()(x)

        # build actor
        actor_x = x if self.actor_obs_ind is None else x[..., self.actor_obs_ind]
        actor_mean = FullyConnectedNetSkeletonMuscle(self.hidden_layer_dims, self.action_dim, self.number_upper_body_activation , self.custom_output_activation, self.activation,
                                       None, False, False)(actor_x)
        actor_logtstd = self.param("log_std", nn.initializers.constant(jnp.log(self.init_std)),
                                   (self.action_dim,))
        
        # check that actor_mean is between -1 and 1 for [0:14] and between 0 and 1 for [14:]
        actor_mean_first = actor_mean[..., :self.number_upper_body_activation]
        actor_mean_rest = actor_mean[..., self.number_upper_body_activation:]
        actor_mean_first_out_of_bounds = jnp.logical_or(jnp.any(actor_mean_first < -1), jnp.any(actor_mean_first > 1))
        actor_mean_rest_out_of_bounds = jnp.logical_or(jnp.any(actor_mean_rest < 0), jnp.any(actor_mean_rest > 1))
        jax.debug.print("actor_mean_first_out_of_bounds: {actor_mean_first_out_of_bounds}", actor_mean_first_out_of_bounds=actor_mean_first_out_of_bounds)
        jax.debug.print("actor_mean_rest_out_of_bounds: {actor_mean_rest_out_of_bounds}", actor_mean_rest_out_of_bounds=actor_mean_rest_out_of_bounds)
        jax.debug.print("actor_mean: {actor_mean}", actor_mean=actor_mean)

        # debug print actor output layer activation functions
        jax.debug.print("actor_logtstd: {actor_logtstd}", actor_logtstd=actor_logtstd)
        jax.debug.print("custom_output_activation: {custom_output_activation}", custom_output_activation=self.custom_output_activation)

        # ValueError if actor_mean is not in the correct range
        actor_mean_out_of_bounds = jax.numpy.logical_or(actor_mean_first_out_of_bounds, actor_mean_rest_out_of_bounds)
        jax.debug.print("actor_mean_out_of_bounds: {actor_mean_out_of_bounds}", actor_mean_out_of_bounds=actor_mean_out_of_bounds)

        if not self.learnable_std:
            actor_logtstd = jax.lax.stop_gradient(actor_logtstd)

        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        # build critic
        critic_x = x if self.critic_obs_ind is None else x[..., self.critic_obs_ind]
        critic = FullyConnectedNet(self.hidden_layer_dims, 1, self.activation, None, False, False)(critic_x)

        return pi, jnp.squeeze(critic, axis=-1)    

class RunningMeanStd(nn.Module):
    """Layer that maintains running mean and variance for input normalization."""

    @nn.compact
    def __call__(self, x):

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

        return jnp.squeeze(normalized_x)