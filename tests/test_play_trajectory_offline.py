"""Headless coverage of ``LocoEnv.play_trajectory`` / ``play_trajectory_from_velocity``.

``play_trajectory`` only touches OpenGL when ``render=True`` -- every ``self.render``
/ recorder call is guarded by the ``render`` / ``record`` flags, and ``stop()`` is a
no-op without a live viewer. So the whole replay loop (set sim state from traj data,
pre/post step, forward, observation creation, and the velocity-integration branch)
runs fine under headless CI. We drive it on a CPU ``DummyHumamoidEnv`` with the
synthetic ``standing_trajectory`` fixture -- no dataset download.
"""
import numpy as np
import jax

from test_conf import DummyHumamoidEnv
from test_conf import *  # noqa: F401,F403  (standing_trajectory fixture, np/jnp/jax/pytest)

jax.config.update('jax_platform_name', 'cpu')

_DEFAULTS = {"horizon": 1000, "gamma": 0.99, "n_envs": 1}


def _env_with_traj(traj):
    env = DummyHumamoidEnv(enable_mjx=False, goal_type="NoGoal",
                           reward_type="NoReward", **_DEFAULTS)
    env.process_trajectory(traj)
    assert env.th is not None
    return env


def test_play_trajectory_headless(standing_trajectory):
    env = _env_with_traj(standing_trajectory)
    # render=False keeps everything off the GL path; a couple of short episodes
    # exercise the outer episode loop + reset-between-episodes.
    env.play_trajectory(n_episodes=2, n_steps_per_episode=5,
                        render=False, record=False, quiet=True,
                        key=jax.random.key(0))


def test_play_trajectory_from_velocity_headless(standing_trajectory):
    env = _env_with_traj(standing_trajectory)
    # Exercises the deprecated play_trajectory_from_velocity wrapper (from_velocity=True).
    # NOTE: the velocity-integration branch itself (base.py 512-532) is currently
    # unreachable -- the local `subtraj_step_no` is initialised to 0 and never
    # incremented, so the `subtraj_step_no != 0` guard is always False. This test
    # therefore only confirms the wrapper + replay loop run headlessly; it does not
    # (and cannot, until that guard is fixed) cover the integration math.
    env.play_trajectory_from_velocity(n_episodes=1, n_steps_per_episode=5,
                                      render=False, record=False, quiet=True,
                                      key=jax.random.key(0))
