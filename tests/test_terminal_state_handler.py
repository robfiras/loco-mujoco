from test_conf import *

# set Jax-backend to CPU
jax.config.update('jax_platform_name', 'cpu')
print(f"Jax backend device: {jax.default_backend()} \n")


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_HeightBasedTerminalStateHandler(standing_trajectory, falling_trajectory, backend, mock_random):
    expert_traj: Trajectory = standing_trajectory
    nominal_traj: Trajectory = falling_trajectory

    transitions = generate_test_trajectories(expert_traj, nominal_traj, backend,
                                             terminal_state_type="HeightBasedTerminalStateHandler")

    assert len(transitions.absorbings) == 999

    if backend == "numpy":
        assert np.sum(transitions.absorbings) == 927
        assert np.all(transitions.absorbings[0:72] == 0)
        assert np.all(transitions.absorbings[72:] == 1)
    else:
        assert jnp.sum(transitions.absorbings) == 927
        assert jnp.all(transitions.absorbings[0:72] == 0)
        assert jnp.all(transitions.absorbings[72:] == 1)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_NoTerminalStateHandler(standing_trajectory, falling_trajectory, backend, mock_random):
    expert_traj: Trajectory = standing_trajectory
    nominal_traj: Trajectory = falling_trajectory

    transitions = generate_test_trajectories(expert_traj, nominal_traj, backend,
                                             terminal_state_type="NoTerminalStateHandler")

    assert len(transitions.absorbings) == 999

    if backend == "numpy":
        assert np.sum(transitions.absorbings) == 0
        assert np.all(transitions.absorbings == 0)
    else:
        assert jnp.sum(transitions.absorbings) == 0
        assert jnp.all(transitions.absorbings == 0)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_RootPoseTrajTerminalStateHandler(standing_trajectory, falling_trajectory, backend, mock_random):
    expert_traj: Trajectory = standing_trajectory
    nominal_traj: Trajectory = falling_trajectory


    transitions = generate_test_trajectories(expert_traj, nominal_traj, backend,
                                             terminal_state_type="RootPoseTrajTerminalStateHandler")

    assert len(transitions.absorbings) == 999

    if backend == "numpy":
        assert np.sum(transitions.absorbings) == 941
        assert np.all(transitions.absorbings[0:58] == 0)
        assert np.all(transitions.absorbings[58:] == 1)
    else:
        assert jnp.sum(transitions.absorbings) == 941
        assert jnp.all(transitions.absorbings[0:58] == 0)
        assert jnp.all(transitions.absorbings[58:] == 1)

@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_RootTrackingTerminalStateHandler(standing_trajectory, falling_trajectory, backend, mock_random):
    # following the reference never terminates, a fall away from it does and stays terminal
    expert_traj: Trajectory = standing_trajectory
    nominal_traj: Trajectory = falling_trajectory

    transitions = generate_test_trajectories(expert_traj, expert_traj, backend,
                                             terminal_state_type="RootTrackingTerminalStateHandler")
    assert np.sum(np.asarray(transitions.absorbings)) == 0

    transitions = generate_test_trajectories(expert_traj, nominal_traj, backend,
                                             terminal_state_type="RootTrackingTerminalStateHandler")
    absorbings = np.asarray(transitions.absorbings)
    first = int(np.argmax(absorbings))
    assert 0 < first < len(absorbings) - 1
    assert np.all(absorbings[:first] == 0) and np.all(absorbings[first:] == 1)
