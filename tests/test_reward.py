from math import isclose

from test_conf import *

# set Jax-backend to CPU
jax.config.update('jax_platform_name', 'cpu')
print(f"Jax backend device: {jax.default_backend()} \n")

# Tolerance for reward comparisons; 1e-5 allows for cross-platform float variance (CI vs local)
_ATOL = 1e-5


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_NoReward(standing_trajectory, falling_trajectory, backend):
    expert_traj: Trajectory = standing_trajectory
    nominal_traj: Trajectory = falling_trajectory

    transitions = generate_test_trajectories(expert_traj, nominal_traj, backend, horizon=100,
                                             reward_type="NoReward")

    if backend == "numpy":
        assert len(transitions.rewards) == 99
        assert np.all(transitions.rewards == 0)
    else:
        assert len(transitions.rewards) == 99
        assert jnp.all(transitions.rewards == 0)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_TargetXVelocityReward(standing_trajectory, falling_trajectory, backend, mock_random):
    expert_traj: Trajectory = standing_trajectory
    nominal_traj: Trajectory = falling_trajectory

    reward_params = dict(target_velocity=1.0)
    transitions = generate_test_trajectories(expert_traj, nominal_traj, backend, horizon=100,
                                             reward_type="TargetXVelocityReward",
                                             reward_params=reward_params)

    if backend == "numpy":
        assert len(transitions.rewards) == 99
        reward_sum = transitions.rewards.sum()
        reward_42 = transitions.rewards[42]
        assert isclose(reward_sum, 18.869667053222656, abs_tol=_ATOL)
        assert isclose(reward_42, 0.1381128579378128, abs_tol=_ATOL)
    else:
        assert len(transitions.rewards) == 99
        reward_sum = float(jnp.sum(transitions.rewards))
        reward_42 = float(transitions.rewards[42])
        assert isclose(reward_sum, 18.86966896057129, abs_tol=_ATOL)
        assert isclose(reward_42, 0.1381128579378128, abs_tol=_ATOL)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_TargetVelocityGoalReward(standing_trajectory, falling_trajectory, backend, mock_random):
    expert_traj: Trajectory = standing_trajectory
    nominal_traj: Trajectory = falling_trajectory

    transitions = generate_test_trajectories(expert_traj, nominal_traj, backend, horizon=100,
                                             goal_type="GoalRandomRootVelocity",
                                             reward_type="TargetVelocityGoalReward")

    if backend == "numpy":
        assert len(transitions.rewards) == 99
        reward_sum = transitions.rewards.sum()
        reward_42 = transitions.rewards[42]
        assert isclose(reward_sum, 40.313209533691406, abs_tol=_ATOL)
        assert isclose(reward_42, 0.8057810068130493, abs_tol=_ATOL)
    else:
        assert len(transitions.rewards) == 99
        reward_sum = float(jnp.sum(transitions.rewards))
        reward_42 = float(transitions.rewards[42])
        assert isclose(reward_sum, 40.313209533691406, abs_tol=_ATOL)
        assert isclose(reward_42, 0.8057810664176941, abs_tol=_ATOL)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_LocomotionReward(standing_trajectory, falling_trajectory, backend, mock_random):
    expert_traj: Trajectory = standing_trajectory
    nominal_traj: Trajectory = falling_trajectory

    reward_params = dict(joint_position_limit_coeff=1.0)
    transitions = generate_test_trajectories(expert_traj, nominal_traj, backend, horizon=100,
                                             goal_type="GoalRandomRootVelocity",
                                             reward_type="LocomotionReward",
                                             reward_params=reward_params)

    if backend == "numpy":
        assert len(transitions.rewards) == 99
        reward_sum = transitions.rewards.sum()
        reward_42 = transitions.rewards[42]
        assert isclose(reward_sum, 17.761075973510742, abs_tol=_ATOL)
        assert isclose(reward_42, 0.0, abs_tol=_ATOL)
    else:
        assert len(transitions.rewards) == 99
        reward_sum = float(jnp.sum(transitions.rewards))
        reward_42 = float(transitions.rewards[42])
        assert isclose(reward_sum, 17.761056900024414, abs_tol=_ATOL)
        assert isclose(reward_42, 0.0, abs_tol=_ATOL)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_TargetVelocityTrajReward(standing_trajectory, falling_trajectory, backend, mock_random):
    expert_traj: Trajectory = standing_trajectory
    nominal_traj: Trajectory = falling_trajectory

    transitions = generate_test_trajectories(expert_traj, nominal_traj, backend, horizon=100,
                                             reward_type="TargetVelocityTrajReward")

    if backend == "numpy":
        assert len(transitions.rewards) == 99
        reward_sum = transitions.rewards.sum()
        reward_42 = transitions.rewards[42]
        assert isclose(reward_sum, 41.20881271362305, abs_tol=_ATOL)
        assert isclose(reward_42, 0.4217287302017212, abs_tol=_ATOL)
    else:
        assert len(transitions.rewards) == 99
        reward_sum = float(jnp.sum(transitions.rewards))
        reward_42 = float(transitions.rewards[42])
        assert isclose(reward_sum, 41.20880889892578, abs_tol=_ATOL)
        assert isclose(reward_42, 0.4217287302017212, abs_tol=_ATOL)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_MimicReward(standing_trajectory, falling_trajectory, backend, mock_random):
    expert_traj: Trajectory = standing_trajectory
    nominal_traj: Trajectory = falling_trajectory

    transitions = generate_test_trajectories(expert_traj, nominal_traj, backend, horizon=100,
                                             reward_type="MimicReward")

    if backend == "numpy":
        assert len(transitions.rewards) == 99
        reward_sum = transitions.rewards.sum()
        reward_42 = transitions.rewards[42]
        assert isclose(reward_sum, 35.697357177734375, abs_tol=_ATOL)
        assert isclose(reward_42, 0.5023975968360901, abs_tol=_ATOL)
    else:
        assert len(transitions.rewards) == 99
        reward_sum = float(jnp.sum(transitions.rewards))
        reward_42 = float(transitions.rewards[42])
        assert isclose(reward_sum, 35.697357177734375, abs_tol=_ATOL)
        assert isclose(reward_42, 0.5023977160453796, abs_tol=_ATOL)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_SiteTrackingReward(standing_trajectory, falling_trajectory, backend, mock_random):
    expert_traj: Trajectory = standing_trajectory
    nominal_traj: Trajectory = falling_trajectory

    transitions = generate_test_trajectories(expert_traj, nominal_traj, backend, horizon=100,
                                             reward_type="SiteTrackingReward")

    if backend == "numpy":
        assert len(transitions.rewards) == 99
        reward_sum = transitions.rewards.sum()
        reward_42 = transitions.rewards[42]
        assert isclose(reward_sum, 2.501685380935669, abs_tol=_ATOL)
        assert isclose(reward_42, 0.03555460646748543, abs_tol=_ATOL)
    else:
        assert len(transitions.rewards) == 99
        reward_sum = float(jnp.sum(transitions.rewards))
        reward_42 = float(transitions.rewards[42])
        assert isclose(reward_sum, 2.501685380935669, abs_tol=_ATOL)
        assert isclose(reward_42, 0.03555460274219513, abs_tol=_ATOL)
