import numpy.random

from loco_mujoco.core.utils import mj_jntname2qposid

from test_conf import *

# set Jax-backend to CPU
jax.config.update('jax_platform_name', 'cpu')
print(f"Jax backend device: {jax.default_backend()} \n")

# Strict tolerance; expected values from current physics (mujoco/mjx)
_ATOL = 1e-7


def _create_env(backend, init_state_type, init_state_params=None, trajectory=None):
    DEFAULTS = {"horizon": 1000, "gamma": 0.99, "n_envs": 1}

    mjx_env = DummyHumamoidEnv(enable_mjx=backend == "jax",
                               init_state_type=init_state_type,
                               init_state_params=init_state_params,
                               **DEFAULTS)

    if trajectory is not None:
        if backend == "numpy":
            trajectory.data = trajectory.data.to_numpy()

        mjx_env.process_trajectory(trajectory)

    return mjx_env


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_DefaultInitialStateHandler(falling_trajectory, backend):
    seed = 0
    key = jax.random.PRNGKey(seed)

    mjx_env_1 = _create_env(backend, "DefaultInitialStateHandler")

    qpos_init = np.zeros(18)
    qpos_init[:7] = np.array([1.5, 1.2, 0.3, 1., 0,  0., 0.])
    j_id = mj_jntname2qposid("abdomen_z", mjx_env_1._model)
    qpos_init[j_id] = 0.3

    qvel_init = 0.1 * np.ones(17)

    init_state_params = dict(qpos_init=qpos_init, qvel_init=qvel_init)

    mjx_env_2 = _create_env(backend, "DefaultInitialStateHandler", init_state_params=init_state_params)

    if backend == "numpy":
        state_0 = mjx_env_1.reset(key)
        state_1 = mjx_env_2.reset(key)

        # Expected from current mujoco physics
        state_0_test = np.array([1.293, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 -0.027269288753425325, 0.09849757187845201, 0.8279999999999998,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        state_1_test = np.array([0.3, 1.0, 0.0, 0.0, 0.0, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                 -0.027269288753425325, 0.09849757187845201, 0.8279999999999998,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        assert np.allclose(state_0, state_0_test, atol=_ATOL)
        assert np.allclose(state_1, state_1_test, atol=_ATOL)
        assert np.allclose(mjx_env_2._data.qpos, qpos_init, atol=_ATOL)
        assert np.allclose(mjx_env_2._data.qvel, qvel_init, atol=_ATOL)
    else:
        state_0 = mjx_env_1.mjx_reset(key)
        state_1 = mjx_env_2.mjx_reset(key)

        # Expected from current mjx physics (BodyPos/BodyVel differ from mujoco)
        state_0_test = jnp.array([1.2929999828338623, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        state_1_test = jnp.array([0.30000001192092896, 1.0, 0.0, 0.0, 0.0, 0.30000001192092896,
                                 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                                 0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        assert jnp.allclose(state_0.observation, state_0_test, atol=_ATOL)
        assert jnp.allclose(state_1.observation, state_1_test, atol=_ATOL)
        assert jnp.allclose(state_1.data.qpos, qpos_init, atol=_ATOL)
        assert jnp.allclose(state_1.data.qvel, qvel_init, atol=_ATOL)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_TrajInitialStateHandler(falling_trajectory, backend, mock_random):
    mjx_env = _create_env(backend, "TrajInitialStateHandler", trajectory=falling_trajectory)

    seed = 0
    key = jax.random.PRNGKey(seed)
    numpy.random.seed(seed)

    if backend == "numpy":
        state_0 = mjx_env.reset(key)
        state_1 = mjx_env.reset(key)
        state_2 = mjx_env.reset(key)

        # Expected from current mujoco physics (lmj env)
        state_test = np.array([0.11782673746347427, 0.5276333689689636, -0.06301149725914001, -0.7512574791908264,
                               0.3914649188518524, 0.027165060862898827, 0.00221033813431859, 0.0005365639226511121,
                               0.0002241866895928979, -0.004811504390090704, 0.0026350426487624645, 0.0048562875017523766,
                               -0.015604501590132713, -0.17573639750480652, 0.26319870352745056, 0.07775566726922989,
                               -0.0013760229339823127, -0.003615611232817173, -0.009761915542185307, -0.003593593370169401,
                               0.00025337518309243023, -0.00022906356025487185])

        assert np.allclose(state_0, state_test, atol=_ATOL)
        assert np.allclose(state_1, state_test, atol=_ATOL)
        assert np.allclose(state_2, state_test, atol=_ATOL)
    else:
        state_0 = mjx_env.mjx_reset(key)
        state_1 = mjx_env.mjx_reset(jax.random.PRNGKey(seed + 1))
        state_2 = mjx_env.mjx_reset(jax.random.PRNGKey(seed + 2))

        # Expected from current mjx physics (lmj env)
        state_test = jnp.array([0.11782893538475037, 0.5276318192481995, -0.06304890662431717, -0.7512582540512085,
                                0.3914594352245331, 0.027006052434444427, 0.0021996304858475924, 0.000567301525734365,
                                0.00021841752459295094, -0.004885685630142689, 0.0027152535039931536, 0.005478315055370331,
                                -0.016209162771701813, -0.1757616102695465, 0.263202428817749, 0.07775142788887024,
                                -0.0013723289594054222, -0.003627696307376027, -0.009739872999489307, -0.0035867858678102493,
                                0.0002513109357096255, -0.00023198407143354416])

        assert jnp.allclose(state_0.observation, state_test, atol=_ATOL)
        assert jnp.allclose(state_1.observation, state_test, atol=_ATOL)
        assert jnp.allclose(state_2.observation, state_test, atol=_ATOL)
