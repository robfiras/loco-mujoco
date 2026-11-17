import pytest
import numpy as np
import jax.numpy as jnp
import jax

from test_conf import DummyHumamoidEnv
from loco_mujoco.core.observations.goals import Goal
from loco_mujoco.environments.base import TrajState

from loco_mujoco.core.trajectory import Trajectory
from test_conf import *

DEFAULTS = {"horizon": 1000, "gamma": 0.99, "n_envs": 1}

# set Jax-backend to CPU
jax.config.update('jax_platform_name', 'cpu')
print(f"Jax backend device: {jax.default_backend()} \n")

_GOAL_ATOL = 1e-7
_GOAL_RTOL = 1e-7


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_NoGoal(backend):
    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        goal_type="NoGoal",
        reward_type="NoReward",
        **DEFAULTS,
    )

    backend = jnp if backend == "jax" else np

    current_goal: Goal = mjx_env._goal
    dim = current_goal.dim
    assert dim == 0, "The dimension has to be 0"

    goal, carry = current_goal.get_obs_and_update_state(
        mjx_env, mjx_env._model, mjx_env._data, None, backend
    )

    assert goal.shape == (0,), "NoGoal should return an empty observation"
    assert carry is None, "Carry should remain unchanged"


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_GoalRandomRootVelocity(backend):
    seed = 0
    key = jax.random.PRNGKey(seed)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        goal_type="GoalRandomRootVelocity",
        reward_type="NoReward",
        **DEFAULTS,
    )

    backend = jnp if backend == "jax" else np

    current_goal: Goal = mjx_env._goal
    dim = current_goal.dim
    assert dim == 3, "The dimension has to be 3"

    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)
        carry = mjx_env._additional_carry
    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        obs = state.observation
        carry = state.additional_carry

    obs = obs[-dim:]

    assert (
        -current_goal.max_x_vel <= obs[0] <= current_goal.max_x_vel
    ), "X velocity out of bounds"
    assert (
        -current_goal.max_y_vel <= obs[1] <= current_goal.max_y_vel
    ), "Y velocity out of bounds"
    assert (
        -current_goal.max_yaw_vel <= obs[2] <= current_goal.max_yaw_vel
    ), "Yaw velocity out of bounds"

    goal, carry = current_goal.get_obs_and_update_state(
        mjx_env, mjx_env._model, mjx_env._data, carry, backend
    )
    # check the observation
    np.testing.assert_allclose(
        obs,
        goal,
        err_msg="Mismatch between Mujoco observation and goal",
        atol=_GOAL_ATOL,
        rtol=_GOAL_RTOL,
    )

    data, carry = current_goal.reset_state(
        mjx_env, mjx_env._model, mjx_env._data, carry, backend
    )

    if backend == jnp:
        obs, carry = mjx_env._mjx_create_observation(mjx_env._model, data, carry)
    else:
        obs, carry = mjx_env._create_observation(mjx_env._model, data, carry)

    obs = obs[-dim:]
    goal, carry = current_goal.get_obs_and_update_state(
        mjx_env, mjx_env._model, mjx_env._data, carry, backend
    )
    # check the observation
    np.testing.assert_allclose(
        obs,
        goal,
        err_msg="Mismatch between Mujoco observation and goal",
        atol=_GOAL_ATOL,
        rtol=_GOAL_RTOL,
    )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_GoalTrajRootVelocity(backend, standing_trajectory):
    seed = 0
    key = jax.random.PRNGKey(seed)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        goal_type="GoalTrajRootVelocity",
        reward_type="NoReward",
        **DEFAULTS,
    )

    trajectory: Trajectory = standing_trajectory
    mjx_env.process_trajectory(trajectory)

    backend = jnp if backend == "jax" else np

    goal: Goal = mjx_env._goal
    dim = goal.dim
    assert dim == 6, "The dimension has to be 6"
    assert goal.requires_trajectory == True
    assert goal.has_visual == True

    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)
        carry = mjx_env._additional_carry
    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        obs = state.observation
        carry = state.additional_carry
    obs = obs[-dim:]

    # Simulate a scenario where the trajectory is ending
    traj_no = carry.traj_state.traj_no
    idx_of_next_traj = mjx_env._traj.data.split_points[traj_no + 1]
    current_step = idx_of_next_traj - 1

    carry = carry.replace(
        traj_state=TrajState(
            traj_no=traj_no,
            subtraj_step_no=current_step,
            subtraj_step_no_init=current_step,
        )
    )

    if backend == np:
        # Check is_done function
        done = goal.is_done(mjx_env, mjx_env._model, mjx_env._data, carry, backend)
    else:
        # Check JAX-compatible version
        done = goal.mjx_is_done(mjx_env, mjx_env._model, mjx_env._data, carry, backend)

    assert (
        done == True
    ), "Goal should be marked as done when steps till end < _n_steps_average"

    goal, _ = goal.get_obs_and_update_state(
        mjx_env, mjx_env._model, mjx_env._data, carry, backend
    )

    # check the observation
    np.testing.assert_allclose(
        obs,
        goal,
        err_msg="Mismatch between Mujoco observation and goal",
        atol=_GOAL_ATOL,
        rtol=_GOAL_RTOL,
    )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_GoalTrajMimic(backend, standing_trajectory):
    seed = 0
    key = jax.random.PRNGKey(seed)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        goal_type="GoalTrajMimic",
        reward_type="NoReward",
        **DEFAULTS,
    )

    trajectory: Trajectory = standing_trajectory
    mjx_env.process_trajectory(trajectory)

    backend = jnp if backend == "jax" else np

    goal: Goal = mjx_env._goal
    dim = goal.dim

    assert goal.requires_trajectory == True
    assert goal.has_visual == True
    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)
        carry = mjx_env._additional_carry
    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        obs = state.observation
        carry = state.additional_carry

    obs = obs[-dim:]

    goal, carry = goal.get_obs_and_update_state(
        mjx_env, mjx_env._model, mjx_env._data, carry, backend
    )

    # check the observation
    np.testing.assert_allclose(
        obs, goal, err_msg="Mismatch between Mujoco observation and goal",
        atol=_GOAL_ATOL, rtol=_GOAL_RTOL,
    )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_GoalTrajMimicv2(backend, standing_trajectory):
    seed = 0
    key = jax.random.PRNGKey(seed)

    # GoalTrajMimicv2 adds robot-geom visualization on top of GoalTrajMimic;
    # the observation contract is identical, so it must match the env obs too.
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        goal_type="GoalTrajMimicv2",
        reward_type="NoReward",
        **DEFAULTS,
    )

    trajectory: Trajectory = standing_trajectory
    mjx_env.process_trajectory(trajectory)

    backend = jnp if backend == "jax" else np

    goal: Goal = mjx_env._goal
    dim = goal.dim

    assert goal.requires_trajectory == True
    assert goal.has_visual == True

    if backend == np:
        obs = mjx_env.reset(key)
        carry = mjx_env._additional_carry
    else:
        state = mjx_env.mjx_reset(key)
        obs = state.observation
        carry = state.additional_carry

    obs = obs[-dim:]

    goal, carry = goal.get_obs_and_update_state(
        mjx_env, mjx_env._model, mjx_env._data, carry, backend
    )

    np.testing.assert_allclose(
        obs, goal, err_msg="Mismatch between Mujoco observation and goal",
        atol=_GOAL_ATOL, rtol=_GOAL_RTOL,
    )
