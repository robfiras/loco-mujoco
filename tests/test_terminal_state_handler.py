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


@pytest.mark.parametrize("terminal_state_type", ["NoTerminalStateHandler", "HeightBasedTerminalStateHandler",
                                                 "RootPoseTrajTerminalStateHandler",
                                                 "RootTrackingTerminalStateHandler"])
def test_absorbing_prob_matches_hard_check(standing_trajectory, falling_trajectory, terminal_state_type,
                                           mock_random):
    # the smooth probability agrees with the hard check away from the thresholds and has finite gradients,
    # also on the reference itself (equal rotations)
    env = DummyHumamoidEnv(enable_mjx=True, terminal_state_type=terminal_state_type, horizon=1000, gamma=0.99,
                           n_envs=1, th_params={"random_start": False, "fixed_start_conf": (0, 0)})
    env.process_trajectory(standing_trajectory)
    traj = env._traj
    state = env.mjx_reset(jax.random.PRNGKey(0), traj)
    handler = env._terminal_state_handler

    def check(qpos):
        def prob(q):
            return handler.mjx_absorbing_prob(env, state.observation, state.info, state.data.replace(qpos=q),
                                              state.additional_carry, traj)
        hard = handler.mjx_is_absorbing(env, state.observation, state.info, state.data.replace(qpos=qpos),
                                        state.additional_carry, traj)[0]
        return hard, prob(qpos), jax.grad(prob)(qpos)

    hard, soft, grad = jax.jit(jax.vmap(check))(jnp.asarray(falling_trajectory.data.qpos))
    hard, soft, grad = np.asarray(hard), np.asarray(soft), np.asarray(grad)
    assert np.all(np.isfinite(soft)), np.flatnonzero(~np.isfinite(soft))
    assert np.all(np.isfinite(grad)), np.argwhere(~np.isfinite(grad))[:10]
    assert np.all((soft >= 0.0) & (soft <= 1.0))
    assert np.mean((soft > 0.5) == hard) > 0.98
    if terminal_state_type != "NoTerminalStateHandler":
        assert hard.any() and not hard.all()

    _, p0, g0 = check(jnp.asarray(standing_trajectory.data.qpos[0]))
    assert float(p0) < 0.01 and np.all(np.isfinite(np.asarray(g0)))
