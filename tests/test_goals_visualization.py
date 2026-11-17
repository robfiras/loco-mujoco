"""Coverage for the goal *visualization* paths (Goal.set_visuals variants).

The existing test_goals.py exercises every goal's observation contract but
never sets ``visualize_goal=True``, so the large ``set_visuals`` methods
(arrow visualizer for the velocity goals, box-site visualizer for
GoalTrajMimic, robot-geom visualizer for GoalTrajMimicv2) are never entered.

These paths are pure array manipulation on ``carry.user_scene.geoms`` (no live
renderer), so they run fine headless. Enabling the flag and resetting the env
triggers ``get_obs_and_update_state`` -> ``set_visuals`` for both the numpy and
the MJX backends.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from test_conf import DummyHumamoidEnv
from test_conf import *  # noqa: F401,F403  (brings in fixtures like standing_trajectory)

jax.config.update('jax_platform_name', 'cpu')

DEFAULTS = {"horizon": 1000, "gamma": 0.99, "n_envs": 1}

# (goal_type, requires_trajectory)
VISUAL_GOALS = [
    ("GoalRandomRootVelocity", False),  # arrow visualizer
    ("GoalTrajRootVelocity", True),     # arrow visualizer (traj-driven)
    ("GoalTrajMimic", True),            # box-site visualizer
    ("GoalTrajMimicv2", True),          # robot-geom visualizer (FK-based)
]


@pytest.mark.parametrize("backend", ["jax", "numpy"])
@pytest.mark.parametrize("goal_type,requires_traj", VISUAL_GOALS)
def test_goal_set_visuals(goal_type, requires_traj, backend, standing_trajectory):
    key = jax.random.PRNGKey(0)

    env = DummyHumamoidEnv(
        enable_mjx=True,
        goal_type=goal_type,
        goal_params={"visualize_goal": True},
        reward_type="NoReward",
        **DEFAULTS,
    )

    if requires_traj:
        env.process_trajectory(standing_trajectory)

    # reset drives _create_observation -> goal.get_obs_and_update_state -> set_visuals
    if backend == "numpy":
        obs = env.reset(key)
        carry = env._additional_carry
    else:
        state = env.mjx_reset(key)
        obs = state.observation
        carry = state.additional_carry

    # the goal must have registered visual geoms when visualize_goal is on
    assert env._goal.n_visual_geoms > 0
    assert env._goal.visual_geoms_idx is not None

    # the observation is still finite (visualization must not corrupt the obs)
    assert np.all(np.isfinite(np.asarray(obs)))

    # set_visuals wrote into the user scene geom buffers
    geom_pos = np.asarray(carry.user_scene.geoms.pos)
    assert geom_pos.shape[0] >= env._goal.n_visual_geoms
    assert np.all(np.isfinite(geom_pos))
