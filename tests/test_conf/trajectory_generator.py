import jax
import numpy.random

from .dummy_humanoid_env import DummyHumamoidEnv

DEFAULTS = {"horizon": 1000, "gamma": 0.99, "n_envs": 1,
            "th_params": {"random_start": False, "fixed_start_conf": (0, 0)}}


def generate_test_trajectories(expert_traj, nominal_traj, backend, horizon=None, **kwargs):
    seed = 0
    key = jax.random.PRNGKey(seed)
    numpy.random.seed(seed)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(enable_mjx=backend == "jax",
                               **kwargs,
                               **DEFAULTS)

    if backend == "numpy":
        expert_traj = expert_traj.replace(data=expert_traj.data.to_numpy())
        nominal_traj = nominal_traj.replace(data=nominal_traj.data.to_numpy())

    mjx_env.process_trajectory(expert_traj)

    # Create dataset of transitions using the nominal trajectory
    if backend == "numpy":
        return mjx_env.generate_trajectory_from_nominal(nominal_traj, horizon, rng_key=key)
    else:
        return mjx_env.mjx_generate_trajectory_from_nominal(nominal_traj, horizon, rng_key=key)
