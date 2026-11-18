"""Trajectory handlers: the registry, the start of an episode each handler picks (NumPy and jitted JAX), a start
chosen by an algorithm, and the ``th_params`` of earlier versions."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from loco_mujoco.core.trajectory import (Trajectory, TrajectoryHandler, RandomStartTrajectoryHandler,
                                         RandomTrajFixedStepTrajectoryHandler, FixedStartTrajectoryHandler)

from conftest import _synthetic_traj_from_env
from test_conf import DummyHumamoidEnv

jax.config.update("jax_platform_name", "cpu")

DEFAULTS = {"horizon": 1000, "gamma": 0.99, "n_envs": 1}
LENGTHS = (30, 50, 40)   # three trajectories of different lengths
BACKENDS = ["numpy", "jax"]


def make_env(th_params=None, backend="numpy"):
    env = DummyHumamoidEnv(enable_mjx=backend == "jax", init_state_type="TrajInitialStateHandler",
                           th_params=th_params, **DEFAULTS)
    traj = Trajectory.concatenate([_synthetic_traj_from_env(env, n) for n in LENGTHS])
    if backend == "numpy":
        traj = traj.replace(data=traj.data.to_numpy())
    env.process_trajectory(traj)
    return env


def starts(env, backend, n=64):
    """(traj_no, subtraj_step_no, subtraj_step_no_init) of n resets."""
    if backend == "numpy":
        np.random.seed(0)
        out = []
        for _ in range(n):
            env.reset()
            ts = env._additional_carry.traj_state
            out.append((int(ts.traj_no), int(ts.subtraj_step_no), int(ts.subtraj_step_no_init)))
        return np.array(out)
    state = jax.jit(jax.vmap(env.mjx_reset))(jax.random.split(jax.random.PRNGKey(0), n))
    ts = state.additional_carry.traj_state
    return np.stack([np.asarray(ts.traj_no), np.asarray(ts.subtraj_step_no), np.asarray(ts.subtraj_step_no_init)], -1)


def test_handlers_are_registered():
    names = {"RandomStartTrajectoryHandler", "RandomTrajFixedStepTrajectoryHandler", "FixedStartTrajectoryHandler"}
    assert names <= set(TrajectoryHandler.list_registered())
    assert type(make_env().th) is RandomStartTrajectoryHandler       # the default
    with pytest.raises(ValueError, match="Unknown TrajectoryHandler 'NoSuchHandler'"):
        make_env(dict(name="NoSuchHandler"))


def test_the_base_class_does_not_sample():
    env = make_env()
    with pytest.raises(NotImplementedError, match="sample_start"):
        TrajectoryHandler(env.th.traj_info).sample_start(None, np, env._traj.data)


@pytest.mark.parametrize("backend", BACKENDS)
def test_random_start(backend):
    s = starts(make_env(dict(name="RandomStartTrajectoryHandler"), backend), backend)
    assert set(s[:, 0]) == {0, 1, 2}
    assert (s[:, 1] < np.array(LENGTHS)[s[:, 0]]).all()
    assert (s[:, 1] == s[:, 2]).all()
    assert len(set(s[:, 1])) > 10


@pytest.mark.parametrize("backend", BACKENDS)
def test_random_trajectory_fixed_step(backend):
    s = starts(make_env(dict(name="RandomTrajFixedStepTrajectoryHandler", fixed_step=7), backend), backend)
    assert set(s[:, 0]) == {0, 1, 2}
    assert (s[:, 1:] == 7).all()


@pytest.mark.parametrize("backend", BACKENDS)
def test_fixed_start(backend):
    s = starts(make_env(dict(name="FixedStartTrajectoryHandler", start_conf=(1, 12)), backend), backend)
    assert (s == [1, 12, 12]).all()


@pytest.mark.parametrize("th_params", [dict(name="RandomStartTrajectoryHandler"),
                                       dict(name="RandomTrajFixedStepTrajectoryHandler", fixed_step=3),
                                       dict(name="FixedStartTrajectoryHandler", start_conf=(1, 4))])
def test_a_start_chosen_by_the_algorithm_wins(th_params):
    """An algorithm that sets next_traj_no / next_subtraj_step_no gets that start, once, with every handler."""
    env = make_env(th_params)
    env.reset()
    carry = env._additional_carry
    carry = carry.replace(traj_state=carry.traj_state.replace(next_traj_no=2, next_subtraj_step_no=5))
    _, carry = env.th.reset_state(env, env._model, env._data, carry, np, traj_data=env._traj.data)
    ts = carry.traj_state
    assert (int(ts.traj_no), int(ts.subtraj_step_no), int(ts.subtraj_step_no_init)) == (2, 5, 5)
    assert (int(ts.next_traj_no), int(ts.next_subtraj_step_no)) == (-1, -1)


def test_fixed_start_set_per_env_in_a_vmapped_reset():
    """As examples/.../eval_clips.py does: swap in a fixed-start handler and start every env on its own clip."""
    env = make_env(backend="jax")
    th = env.th
    env.th = FixedStartTrajectoryHandler(th.traj_info, control_dt=th.control_dt, max_n_samples=th.max_n_samples,
                                         max_n_trajs=th.max_n_trajs)

    def reset(key, clip):
        env.th.start_conf = (clip, 0)
        return env.mjx_reset(key)

    state = jax.vmap(reset)(jax.random.split(jax.random.PRNGKey(0), 3), jnp.arange(3))
    ts = state.additional_carry.traj_state
    np.testing.assert_array_equal(np.asarray(ts.traj_no), np.arange(3))
    assert not np.asarray(ts.subtraj_step_no).any()


@pytest.mark.parametrize("old, cls, attrs", [
    (dict(random_start=False, fixed_start_conf=(1, 4)), FixedStartTrajectoryHandler, dict(start_conf=(1, 4))),
    (dict(fixed_start_conf=(2, 0)), FixedStartTrajectoryHandler, dict(start_conf=(2, 0))),
    (dict(random_start=True, max_n_trajs=50), RandomStartTrajectoryHandler, dict(max_n_trajs=50)),
])
def test_old_th_params_still_work(old, cls, attrs):
    with pytest.warns(DeprecationWarning, match="random_start / fixed_start_conf are deprecated"):
        env = make_env(old)
    assert type(env.th) is cls
    for k, v in attrs.items():
        assert getattr(env.th, k) == v


def test_old_th_params_errors_and_the_fixed_start_conf_alias():
    with pytest.raises(ValueError, match="random_start=False needs a fixed_start_conf"):
        make_env(dict(random_start=False))
    env = make_env(dict(name="FixedStartTrajectoryHandler"))
    with pytest.warns(DeprecationWarning, match="use start_conf"):
        env.th.fixed_start_conf = (2, 3)
    env.reset()
    ts = env._additional_carry.traj_state
    assert (int(ts.traj_no), int(ts.subtraj_step_no)) == (2, 3)
