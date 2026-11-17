"""Branch-coverage tests for LocomotionReward penalty/regularization terms.

The existing test_reward.py::test_LocomotionReward only exercises the default
coefficients (joint_position_limit + a few defaults). This file drives the
remaining branches of LocomotionReward.__call__:

* the air-time loop body + symmetry-violation branch (needs >=4 foot geoms,
  since the symmetry term indexes foots_on_ground[0..3]),
* the energy / joint-velocity / nominal-joint-position penalty terms,
* the "coefficient == 0" *else* branches for every penalty term.

The stock DummyHumamoidEnv reports ``foot_geom_names == []`` (base default), so
the air-time loop never iterates. QuadFootDummyEnv overrides it with four real
geom names from humanoid_test.xml so the loop + symmetry logic run.
"""
import numpy as np
from test_conf import *
from loco_mujoco.core.utils import info_property
from test_conf.dummy_humanoid_env import DummyHumamoidEnv

# set Jax-backend to CPU
jax.config.update('jax_platform_name', 'cpu')

DEFAULTS = {"horizon": 1000, "gamma": 0.99, "n_envs": 1,
            "th_params": {"random_start": False, "fixed_start_conf": (0, 0)}}


class QuadFootDummyEnv(DummyHumamoidEnv):
    """DummyHumamoidEnv reporting four (real) foot geoms so the air-time loop
    iterates and the four-foot symmetry branch is reachable."""

    @info_property
    def foot_geom_names(self):
        return ["right_foot", "left_foot", "right_shin1", "left_shin1"]


def _generate(env_cls, expert_traj, nominal_traj, backend, horizon, **kwargs):
    """Mirror of test_conf.generate_test_trajectories but for a custom env class."""
    key = jax.random.PRNGKey(0)
    np.random.seed(0)

    env = env_cls(enable_mjx=(backend == "jax"), **kwargs, **DEFAULTS)

    if backend == "numpy":
        expert_traj = expert_traj.replace(data=expert_traj.data.to_numpy())
        nominal_traj = nominal_traj.replace(data=nominal_traj.data.to_numpy())

    env.process_trajectory(expert_traj)

    if backend == "numpy":
        return env.generate_trajectory_from_nominal(nominal_traj, horizon, rng_key=key)
    else:
        return env.mjx_generate_trajectory_from_nominal(nominal_traj, horizon, rng_key=key)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_LocomotionReward_all_penalties_on(standing_trajectory, falling_trajectory, backend, mock_random):
    """Every penalty coefficient > 0: covers the air-time loop body, the
    four-foot symmetry-violation branch, and the energy / joint-vel / nominal
    joint-position terms."""
    reward_params = dict(
        z_vel_coeff=2.0,
        roll_pitch_vel_coeff=5e-2,
        roll_pitch_pos_coeff=2e-1,
        nominal_joint_pos_coeff=1.0,
        nominal_joint_pos_names=["right_hip_x", "left_hip_x"],  # exercises the named-joint init branch
        joint_position_limit_coeff=10.0,
        joint_vel_coeff=1e-3,
        joint_acc_coeff=2e-7,
        joint_torque_coeff=2e-5,
        action_rate_coeff=1e-2,
        air_time_max=0.1,
        air_time_coeff=1.0,
        symmetry_air_coeff=1.0,
        energy_coeff=1e-3,
    )
    transitions = _generate(QuadFootDummyEnv, standing_trajectory, falling_trajectory,
                            backend, horizon=20,
                            goal_type="GoalRandomRootVelocity",
                            reward_type="LocomotionReward",
                            reward_params=reward_params)

    assert len(transitions.rewards) == 19
    if backend == "numpy":
        assert np.all(np.isfinite(transitions.rewards))
        assert np.all(transitions.rewards >= 0.0)  # total reward is clipped at 0
    else:
        assert jnp.all(jnp.isfinite(transitions.rewards))
        assert jnp.all(transitions.rewards >= 0.0)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_LocomotionReward_all_penalties_off(standing_trajectory, falling_trajectory, backend, mock_random):
    """Every penalty coefficient == 0: covers the ``else`` branch of every
    penalty term (including the no-air-time/no-symmetry tslt-copy branch)."""
    reward_params = dict(
        z_vel_coeff=0.0,
        roll_pitch_vel_coeff=0.0,
        roll_pitch_pos_coeff=0.0,
        nominal_joint_pos_coeff=0.0,
        joint_position_limit_coeff=0.0,
        joint_vel_coeff=0.0,
        joint_acc_coeff=0.0,
        joint_torque_coeff=0.0,
        action_rate_coeff=0.0,
        air_time_coeff=0.0,
        symmetry_air_coeff=0.0,
        energy_coeff=0.0,
    )
    transitions = _generate(QuadFootDummyEnv, standing_trajectory, falling_trajectory,
                            backend, horizon=20,
                            goal_type="GoalRandomRootVelocity",
                            reward_type="LocomotionReward",
                            reward_params=reward_params)

    assert len(transitions.rewards) == 19
    if backend == "numpy":
        assert np.all(np.isfinite(transitions.rewards))
    else:
        assert jnp.all(jnp.isfinite(transitions.rewards))
