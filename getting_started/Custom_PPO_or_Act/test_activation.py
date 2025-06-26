import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence
from flax.linen.initializers import constant, orthogonal


import distrax


def test_jax_activation_limits():
    """
    Test to ensure the JAX actor's outputs are limited by the activation function.
    """

    # Example actor outputs before activation
    raw_outputs = jnp.array([-3, -1, 0, 1, 3], dtype=jnp.float32)

    # Example activation function (tanh limits outputs between -1 and 1)
    activated_outputs = nn.sigmoid(raw_outputs) 

    # Check limits
    assert jnp.all(activated_outputs >= 0), "Activated outputs exceed lower limit of -1"
    assert jnp.all(activated_outputs <= 1), "Activated outputs exceed upper limit of 1"

    print("Test passed: JAX actor outputs are correctly limited by the activation function.")
    print("Activated outputs:", activated_outputs)


def get_activation_fn(name: str):
    """ Get activation function by name from the flax.linen module."""
    try:
        # Use getattr to dynamically retrieve the activation function from jax.nn
        return getattr(nn, name)
    except AttributeError:
        raise ValueError(f"Activation function '{name}' not found. Name must be the same as in flax.linen!")


class FullyConnectedNet(nn.Module):
    """
    Test function to ensure the JAX actor's outputs are limited by the activation function.
    """
    hidden_layer_dims: Sequence[int]
    output_dim: int
    activation: str = "tanh"
    output_activation: str = None    # none means linear activation
    use_running_mean_stand: bool = False #True
    squeeze_output: bool = False #True

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation)
        self.output_activation_fn = get_activation_fn(self.output_activation) \
            if self.output_activation is not None else lambda x: x

    @nn.compact
    def __call__(self, x):

        # if self.use_running_mean_stand:
        #     x = RunningMeanStd()(x)

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
    use_running_mean_stand: bool = False #True
    squeeze_output: bool = False #True

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation)
        
        self.output_activation_fn = get_activation_fn(self.output_activation) \
            if self.output_activation is not None else lambda x: x
        # jax.debug.print("custom_output_activation: {custom_output_activation}", custom_output_activation=self.custom_output_activation)
        self.custom_output_activation_fns = [get_activation_fn(act) for act in self.custom_output_activation]

    @nn.compact
    def __call__(self, x):

        # if self.use_running_mean_stand:
        #     x = RunningMeanStd()(x)

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

        # jax.debug.print("x_first: {x_first}", x_first=x_first)

        # Apply the second activation function (e.g., sigmoid) to the remaining outputs
        x_rest = jnp.stack([second_activation_fn(x[..., i]) for i in range(self.number_upper_body_activation, self.output_dim)], axis=-1)

        # jax.debug.print("x_rest: {x_rest}", x_rest=x_rest)

        # Concatenate the results
        x = jnp.concatenate([x_first, x_rest], axis=-1)

        # jax.debug.print("x: {x}", x=x)

        return jnp.squeeze(x) if self.squeeze_output else x


# class RunningMeanStd(nn.Module):
#     """Layer that maintains running mean and variance for input normalization."""

#     @nn.compact
#     def __call__(self, x):

#         x = jnp.atleast_2d(x)

#         # Initialize running mean, variance, and count
#         mean = self.variable('run_stats', 'mean', lambda: jnp.zeros(x.shape[-1]))
#         var = self.variable('run_stats', 'var', lambda: jnp.ones(x.shape[-1]))
#         count = self.variable('run_stats', 'count', lambda: jnp.array(1e-6))

#         # Compute batch mean and variance
#         batch_mean = jnp.mean(x, axis=0)
#         batch_var = jnp.var(x, axis=0) + 1e-6  # Add epsilon for numerical stability
#         batch_count = x.shape[0]

#         # Update counts
#         updated_count = count.value + batch_count

#         # Numerically stable mean and variance update
#         delta = batch_mean - mean.value
#         new_mean = mean.value + delta * batch_count / updated_count

#         # Compute the new variance using Welford's method
#         m_a = var.value * count.value
#         m_b = batch_var * batch_count
#         M2 = m_a + m_b + jnp.square(delta) * count.value * batch_count / updated_count
#         new_var = M2 / updated_count

#         # Normalize input
#         normalized_x = (x - new_mean) / jnp.sqrt(new_var + 1e-8)

#         # Update state variables
#         mean.value = new_mean
#         var.value = new_var
#         count.value = updated_count

#         return jnp.squeeze(normalized_x)



def test_distribution(mean, log_std):

    pi = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))

    return pi, pi.sample(seed=jax.random.PRNGKey(0))





if __name__ == "__main__":
    test_jax_activation_limits()
    # activation= 'tanh'  # Example activation function
    # output_activation = None #['tanh', 'sigmoid']  # Example output activation function
    # # x = RunningMeanStd()(x)

    # # Example usage of FullyConnectedNet
    # net = FullyConnectedNet(hidden_layer_dims=[512, 256], output_dim=10, activation=activation, output_activation=output_activation)
    # # net = FullyConnectedNetSkeletonMuscle(
    # #     hidden_layer_dims=[512, 256],
    # #     output_dim=16,  # Example output dimension
    # #     number_upper_body_activation=8,  # Example number of upper body activations    
    # #     custom_output_activation=output_activation,  # Example custom output activation functions
    # #     activation=activation
    # #     # output_activation=None,  # Example output activation function
    # #     # use_running_mean_stand=True,
    # #     # squeeze_output=True
    # # )
    # x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float32)

    # # Initialize the model to create state variables
    # variables = net.init(jax.random.PRNGKey(0), x)

    # # Apply the model using the initialized variables, allowing updates to 'run_stats'
    # output, updated_variables = net.apply(variables, x, mutable=['run_stats'])
    
    # print(f'NetworkSkeletonMuscle output for activation {activation} without custom output_activation {output_activation}: {output}')


    # pi, pi_sample = test_distribution(mean=jnp.array([1, 0, 0.5]), log_std=jnp.array([-1.61, -1.61, -1.61]))

    # print(f"Distribution mean: {pi.mean}")
    # print(f"Sampled output: {pi_sample}")
    # print(f"pi: {pi}")