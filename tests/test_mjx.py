from loco_mujoco.core.observations import ObservationType
from loco_mujoco.core.wrappers import MjxRolloutWrapper, RolloutWrapper

from test_conf import *


N_EPISODES = 500
N_STEPS = 1000
N_EPISODES_REP = 5
N_STEPS_REP = 500

jax.config.update('jax_platform_name', 'cpu')
print(f"Jax backend device: {jax.default_backend()} \n")


def test_mjx_simto_mujoco():
    """
    Test whether the Mjx simulation is equal/similar to the Mujoco simulation.
    """

    # tolerance for difference between MuJoCo and MJX forward calculations
    _TOLERANCE = 1e-4

    # set Jax-backend to CPU
    jax.config.update('jax_platform_name', 'cpu')
    print(f"Jax backend device: {jax.default_backend()} \n")

    N_ENVS = 1
    MODEL_OPTION = dict(iterations=100, ls_iterations=50)

    def test_dummy_task(n_steps, **task_params):

        key = jax.random.PRNGKey(94)
        keys = jax.random.split(key, N_ENVS)

        # Mujoco
        task_env = DummyHumamoidEnv(enable_mjx=False, **task_params)
        rollout_env = RolloutWrapper(task_env)

        qpos_idx = np.concatenate([obs.obs_ind for obs in task_env.obs_container.values()
                                   if isinstance(obs, ObservationType.JointPos)])
        qvel_idx = np.concatenate([obs.obs_ind for obs in task_env.obs_container.values()
                                   if isinstance(obs, ObservationType.JointVel)])

        obs, action, reward, next_obs, absorbing, done, cum_return = (
            rollout_env.batch_rollout(rng_keys=keys, n_steps=n_steps, policy_params=None))

        # Mjx rollout
        mjx_task_env = DummyHumamoidEnv(enable_mjx=True, **task_params)

        mjx_rollout_env = MjxRolloutWrapper(mjx_task_env)

        mjx_obs, mjx_action, mjx_reward, mjx_next_obs, mjx_absorbing, mjx_done, mjx_cum_return = (
            mjx_rollout_env.batch_rollout(rng_keys=keys, n_steps=n_steps, policy_params=None))

        #print("obs max diff: ", jax.numpy.max(obs - mjx_obs, axis=(0, 1)))
        #print("next obs max diff: ", jax.numpy.max(next_obs - mjx_next_obs, axis=(0, 1)))

        # check qpos
        assert jax.numpy.allclose(obs[:, :, qpos_idx], mjx_obs[:, :, qpos_idx],
                                  atol=_TOLERANCE, rtol=_TOLERANCE)
        assert jax.numpy.allclose(next_obs[:, :, qpos_idx], mjx_next_obs[:, :, qpos_idx],
                                  atol=_TOLERANCE, rtol=_TOLERANCE)

        # check qvel
        _QVEL_TOLERANCE = _TOLERANCE * 10  # velocities are higher in magnitude
        assert jax.numpy.allclose(obs[:, :, qvel_idx], mjx_obs[:, :, qvel_idx],
                                  atol=_QVEL_TOLERANCE, rtol=_QVEL_TOLERANCE)
        assert jax.numpy.allclose(next_obs[:, :, qvel_idx], mjx_next_obs[:, :, qvel_idx],
                                  atol=_QVEL_TOLERANCE, rtol=_QVEL_TOLERANCE)

        # check actions
        assert jax.numpy.allclose(action, mjx_action, atol=_TOLERANCE, rtol=_TOLERANCE)

        # check rewards
        assert jax.numpy.allclose(reward, mjx_reward, atol=_TOLERANCE, rtol=_TOLERANCE)

    n_steps = 4
    n_substeps = 2

    test_dummy_task(n_steps=n_steps, n_substeps=n_substeps, model_option_conf=MODEL_OPTION)

    # todo: currently disabled due to time constraints
    # By setting short horizons we test the resetting mechanism.
    # Note: we do not test it with random resets from trajectory, as the rng_keys propagate
    # slightly different in Mujoco and Mjx (due to _mjx_reset_in_step), so the sampled init
    # states wouldn't be equivalent. However, the main reset function _reset(key) and _mjx_reset(key)
    # are equivalent, which is why the above works.
    #
    # short_horizon = 2
    #
    # test_dummy_task(n_steps=n_steps, n_substeps=n_substeps,
    #                 model_option_conf=MODEL_OPTION, horizon=short_horizon)


def test_mjx_warp_backend_smoke():
    """
    Smoke test that the MjWarp backend can be initialized and stepped.

    Skipped only when MuJoCo Warp is not installed; a failing initialisation is a failure.
    """
    pytest.importorskip("mujoco_warp")
    # enable MJX and request the MjWarp backend
    env = DummyHumamoidEnv(enable_mjx=True, use_mjwarp=True, horizon=10, gamma=0.99, n_envs=1)

    key = jax.random.PRNGKey(0)
    # reset once
    state = env.mjx_reset(key)

    # take a random action and perform a single MJX step
    key, subkey = jax.random.split(key)
    action = env.sample_action_space(subkey)
    _ = env.mjx_step(state, action)
