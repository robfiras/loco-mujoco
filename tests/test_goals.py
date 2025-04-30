import pytest
import numpy as np
import jax.numpy as jnp
import jax

from test_conf import DummyHumamoidEnv
from loco_mujoco.core.observations.goals import Goal
from loco_mujoco.environments.base import TrajState

from loco_mujoco.trajectory import Trajectory
from test_conf import *

DEFAULTS = {"horizon": 1000, "gamma": 0.99, "n_envs": 1}

# set Jax-backend to CPU
jax.config.update('jax_platform_name', 'cpu')
print(f"Jax backend device: {jax.default_backend()} \n")


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
    mjx_env.load_trajectory(trajectory)

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
    idx_of_next_traj = mjx_env.th.traj.data.split_points[traj_no + 1]
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
    mjx_env.load_trajectory(trajectory)

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
        obs, goal, err_msg="Mismatch between Mujoco observation and goal", atol=1e-7
    )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
@pytest.mark.parametrize("adaptation_rate", [0.1, 0.5, 0.9])
def test_GoalAdaptiveTargeting(backend, standing_trajectory, walking_trajectory, adaptation_rate):
    seed = 42
    key = jax.random.PRNGKey(seed)

    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        goal_type="GoalAdaptiveTargeting",
        reward_type="RewTargetTracking",
        adaptation_rate=adaptation_rate,
        **DEFAULTS,
    )

    trajectories = [standing_trajectory, walking_trajectory]
    mjx_env.load_trajectories(trajectories)
    
    backend = jnp if backend == "jax" else np
    
    goal: Goal = mjx_env._goal
    dim = goal.dim
    assert dim == 9, "The dimension for adaptive targeting should be 9"
    assert goal.requires_trajectory == True
    assert goal.has_visual == True
    
    if backend == np:
        obs = mjx_env.reset(key)
        carry = mjx_env._additional_carry
        original_target = carry.target_state.copy()
    else:
        state = mjx_env.mjx_reset(key)
        obs = state.observation
        carry = state.additional_carry
        original_target = backend.copy(carry.target_state)
    
    goal_obs = obs[-dim:]
    
    # Simulate agent action that deviates from target
    deviation = backend.array([0.2, -0.1, 0.15, 0.1, -0.05, 0.12, 0.08, -0.15, 0.1])
    
    # Mock current state deviating from target
    mock_state = original_target + deviation
    if backend == np:
        mjx_env._data.qpos[:dim] = mock_state
    else:
        data = mjx_env._data.replace(qpos=mjx_env._data.qpos.at[:dim].set(mock_state))
        mjx_env._data = data
    
    # Update goal state with adaptation
    goal_obs, updated_carry = goal.get_obs_and_update_state(
        mjx_env, mjx_env._model, mjx_env._data, carry, backend
    )
    
    # Verify target adaptation
    expected_target = original_target + adaptation_rate * deviation
    actual_target = updated_carry.target_state
    
    np.testing.assert_allclose(
        actual_target, 
        expected_target, 
        err_msg=f"Target not adapted correctly with rate {adaptation_rate}",
        atol=1e-6
    )
    
    # Check if adaptation continues properly across multiple steps
    for _ in range(3):
        # Simulate more deviation
        new_deviation = backend.array([0.1, 0.05, -0.1, -0.05, 0.1, -0.08, 0.05, 0.1, -0.05])
        current_state = actual_target + new_deviation
        
        if backend == np:
            mjx_env._data.qpos[:dim] = current_state
        else:
            data = mjx_env._data.replace(qpos=mjx_env._data.qpos.at[:dim].set(current_state))
            mjx_env._data = data
            
        goal_obs, updated_carry = goal.get_obs_and_update_state(
            mjx_env, mjx_env._model, mjx_env._data, updated_carry, backend
        )
        
        actual_target = updated_carry.target_state
        expected_target = expected_target + adaptation_rate * new_deviation
        
        np.testing.assert_allclose(
            actual_target, 
            expected_target, 
            err_msg=f"Target adaptation not maintained over multiple steps with rate {adaptation_rate}",
            atol=1e-6
        )
    
    # Test goal termination when reaching target
    if backend == np:
        mjx_env._data.qpos[:dim] = actual_target
        done = goal.is_done(mjx_env, mjx_env._model, mjx_env._data, updated_carry, backend)
    else:
        data = mjx_env._data.replace(qpos=mjx_env._data.qpos.at[:dim].set(actual_target))
        mjx_env._data = data
        done = goal.mjx_is_done(mjx_env, mjx_env._model, mjx_env._data, updated_carry, backend)
    
    assert done == True, "Goal should be marked as done when target is reached"
