from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from mujoco import mjx
from omegaconf import OmegaConf, open_dict

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms.experimental import SuperMimicJax
from loco_mujoco.algorithms.experimental.supermimic_jax import (WorldModelSpec, WindowBuffer, state_is_valid,
                                                                 skip_gradient_spikes, _quat_mul, _quat_from_rotvec,
                                                                 _rotvec_from_quat)

from test_conf import *


def _make(config):
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    return factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)


def _params_allclose(a, b):
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    return len(la) == len(lb) and all(np.allclose(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb))


def _random_state(rng, nj, batch=5):
    k1, k2, k3 = jax.random.split(rng, 3)
    quat = jax.random.normal(k1, (batch, 4))
    quat = quat / jnp.linalg.norm(quat, axis=-1, keepdims=True)
    qpos = jnp.concatenate([jax.random.normal(k2, (batch, 3)), quat, jax.random.normal(k2, (batch, nj))], -1)
    qvel = jax.random.normal(k3, (batch, 6 + nj))
    return qpos, qvel


def test_world_model_spec():
    nj, dt = 4, 0.02
    spec = WorldModelSpec(nj, dt, (1.0,) * 6, (1.0,) * 6, (1.0,) * 6)
    qpos, qvel = _random_state(jax.random.PRNGKey(0), nj)

    # zero network output = constant-velocity extrapolation
    qpos1, qvel1 = spec.integrate(qpos, qvel, jnp.zeros((qpos.shape[0], spec.delta_dim)))
    np.testing.assert_allclose(qvel1, qvel, atol=1e-6)
    np.testing.assert_allclose(qpos1[:, :3], qpos[:, :3] + dt * qvel[:, :3], atol=1e-6)
    np.testing.assert_allclose(qpos1[:, 7:], qpos[:, 7:] + dt * qvel[:, 6:], atol=1e-6)
    expected_quat = _quat_mul(qpos[:, 3:7], _quat_from_rotvec(dt * qvel[:, 3:6]))
    np.testing.assert_allclose(qpos1[:, 3:7], expected_quat, atol=1e-6)

    # error of a state with itself is zero; rotvec round-trips
    np.testing.assert_allclose(spec.error(qpos, qvel, qpos, qvel), 0.0, atol=1e-5)
    v = jnp.array([[0.3, -0.2, 0.1]])
    np.testing.assert_allclose(_rotvec_from_quat(_quat_from_rotvec(v)), v, atol=1e-6)

    # features are invariant to a yaw rotation and a horizontal shift of the root
    yaw = _quat_from_rotvec(jnp.array([[0.0, 0.0, 0.7]]))
    qpos_r = qpos.at[:, 3:7].set(_quat_mul(jnp.broadcast_to(yaw, (qpos.shape[0], 4)), qpos[:, 3:7]))
    qpos_r = qpos_r.at[:, :2].add(1.5)
    c, s = np.cos(0.7), np.sin(0.7)
    rot = jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    qvel_r = qvel.at[:, :3].set(qvel[:, :3] @ rot.T)  # world-frame lin vel rotates with the root
    np.testing.assert_allclose(spec.features(qpos_r, qvel_r), spec.features(qpos, qvel), atol=1e-5)

    # gradients stay finite at zero rotation / zero velocity, and at a perfect prediction
    qpos_id = qpos.at[:, 3:7].set(jnp.array([1.0, 0.0, 0.0, 0.0]))
    zero_delta = jnp.zeros((qpos.shape[0], spec.delta_dim))
    g = jax.grad(lambda d: jnp.sum(spec.integrate(qpos_id, jnp.zeros_like(qvel), d)[0]))(zero_delta)
    assert np.all(np.isfinite(g))
    g = jax.grad(lambda q: jnp.sum(spec.error(q, qvel, qpos, qvel) ** 2))(qpos)
    assert np.all(np.isfinite(g))


def test_skip_gradient_spikes():
    tx = skip_gradient_spikes(optax.sgd(1.0), factor=10.0, warmup=5)
    params = jnp.zeros(3)
    state = tx.init(params)
    step = lambda g, s: tx.update(g, s, params)  # noqa: E731
    ordinary = jnp.ones(3)
    for _ in range(20):
        updates, state = step(ordinary, state)
        assert np.all(np.asarray(updates) != 0.0)
    assert int(state.skipped) == 0

    updates, state = step(100.0 * ordinary, state)  # a spike is dropped
    assert np.all(np.asarray(updates) == 0.0) and int(state.skipped) == 1
    updates, state = step(ordinary, state)
    assert np.all(np.asarray(updates) != 0.0)

    # a lasting rise is a new scale, not a spike: updates resume
    for _ in range(300):
        updates, state = step(100.0 * ordinary, state)
    assert np.all(np.asarray(updates) != 0.0)


def _kinematics(sys, data):
    return mjx.com_vel(sys, mjx.com_pos(sys, mjx.kinematics(sys, data)))


def test_buffers_skip_diverged_states():
    """A simulator blow-up in one env must not reach world-model batches or imagination starts."""
    T, E, nq, nv, A = 6, 3, 9, 8, 2
    buf = WindowBuffer.create(T, E, nq, nv, A, 0, 4)
    qpos, qvel = jnp.ones((E, nq)), jnp.ones((E, nv))
    bad_qvel = qvel.at[1].set(jnp.array([jnp.nan] + [1e15] * (nv - 1)))
    for t in range(T):
        nqv = bad_qvel if t == 3 else qvel
        valid = state_is_valid(qpos, qvel, 100.0) & state_is_valid(qpos, nqv, 100.0)
        buf = buf.add(valid, qpos=qpos, qvel=qvel, next_qpos=qpos, next_qvel=nqv, action=jnp.zeros((E, A)),
                      xi=jnp.zeros((E, 0)), next_term_obs=jnp.zeros((E, 4)), absorbing=jnp.zeros(E),
                      done=jnp.zeros(E, dtype=bool))
    assert np.all(np.isfinite(np.asarray(buf.next_qvel)))
    batch, mask = buf.sample_windows(jax.random.PRNGKey(0), 512, 3)
    for v in batch.values():
        assert np.all(np.isfinite(np.asarray(v)))
    assert 0 < int(mask.sum()) < 512  # windows through env 1 at t=3 are masked, the rest kept

    from loco_mujoco.algorithms.experimental.supermimic_jax import StartBuffer, StartState
    template = SimpleNamespace(data=SimpleNamespace(qpos=qpos, qvel=qvel), observation=jnp.zeros((E, 4)),
                               info={}, additional_carry={"x": jnp.zeros(E)})
    sbuf = StartBuffer.create(4, template)
    for t in range(4):
        st = SimpleNamespace(data=SimpleNamespace(qpos=qpos, qvel=bad_qvel), observation=jnp.zeros((E, 4)),
                             info={}, additional_carry={"x": jnp.arange(E, dtype=jnp.float32)})
        sbuf = sbuf.add(st, state_is_valid(qpos, bad_qvel, 100.0))
    starts = sbuf.sample(jax.random.PRNGKey(1), 256)
    assert np.all(np.isfinite(np.asarray(starts.qvel)))
    assert not np.any(np.asarray(starts.carry["x"]) == 1.0)  # env 1 is never sampled


def test_physics_fn_reuses_env_pipeline(supermimic_config):
    """Everything around the simulation is the env's own code, and the observation and reward only
    need (qpos, qvel) plus the kinematics recomputed from them: that is what lets the world model
    reuse the env's observation and reward functions."""
    env, traj = _make(supermimic_config)
    state = env.mjx_reset(jax.random.PRNGKey(0), traj)
    action = 0.3 * jnp.ones(env.info.action_space.shape)
    ref = env.mjx_step(state, action, traj, auto_reset=False)

    # handing back the simulator's own data reproduces the normal step exactly
    same = env.mjx_step(state, action, traj, auto_reset=False, physics_fn=lambda sys, d, c, a: (ref.data, c))
    np.testing.assert_array_equal(same.observation, ref.observation)
    np.testing.assert_array_equal(same.reward, ref.reward)

    # writing only qpos/qvel into the *start* data and recomputing the kinematics gives the same result
    # as refreshing the kinematics of the simulator's data. (Not the same as `ref` itself: mjx.step runs
    # the kinematics before integrating, so after a step they lag qpos/qvel by one physics substep.)
    def from_qpos_qvel(sys, data, carry, a):
        return _kinematics(sys, data.replace(qpos=ref.data.qpos, qvel=ref.data.qvel)), carry

    def refreshed(sys, data, carry, a):
        return _kinematics(sys, ref.data), carry

    out = env.mjx_step(state, action, traj, auto_reset=False, physics_fn=from_qpos_qvel)
    expected = env.mjx_step(state, action, traj, auto_reset=False, physics_fn=refreshed)
    np.testing.assert_allclose(out.observation, expected.observation, atol=1e-5)
    np.testing.assert_allclose(out.reward, expected.reward, atol=1e-5)
    assert bool(out.absorbing) == bool(expected.absorbing)


def test_mjx_step_auto_reset_flag(supermimic_config):
    env, traj = _make(supermimic_config)
    state = env.mjx_reset(jax.random.PRNGKey(0), traj)
    state = state.replace(additional_carry=state.additional_carry.replace(cur_step_in_episode=env.info.horizon))
    action = jnp.zeros(env.info.action_space.shape)
    kept = env.mjx_step(state, action, traj, auto_reset=False)
    reset = env.mjx_step(state, action, traj)
    assert bool(kept.done) and bool(reset.done)
    assert int(kept.additional_carry.cur_step_in_episode) == env.info.horizon + 1
    assert int(reset.additional_carry.cur_step_in_episode) == 1


def test_SuperMimic_resume_in_chunks(supermimic_config, tmp_path):
    """A run split into calls continues the step count and the update pacing instead of starting over."""
    config = OmegaConf.create(OmegaConf.to_container(supermimic_config, resolve=True))
    env, traj = _make(config)
    agent_conf = SuperMimicJax.init_agent_conf(env, config)
    exp = agent_conf.config.experiment
    with open_dict(exp):
        exp.iterations_per_call = int(exp.num_iterations) // 2
    rng = jax.random.PRNGKey(0)
    train_fn = jax.jit(SuperMimicJax.build_train_fn(env, agent_conf))

    first = train_fn(rng, SuperMimicJax.init_agent_state(env, agent_conf, rng), traj)
    state1, m1 = first["agent_state"], first["training_metrics"]
    assert int(state1.iteration) == exp.iterations_per_call
    assert int(m1.num_updates[-1]) > 0

    # in the same process the buffers carry over: no second warm-up
    second = train_fn(jax.random.PRNGKey(1), state1, traj)
    state2, m2 = second["agent_state"], second["training_metrics"]
    assert int(state2.iteration) == 2 * exp.iterations_per_call
    assert int(m2.max_timestep[0]) > int(m1.max_timestep[-1])
    assert int(m2.num_updates[0]) > int(m1.num_updates[-1])

    # from a checkpoint the step count survives; the empty buffers are refilled before updating again
    path = SuperMimicJax.save_agent(tmp_path, agent_conf, state1)
    _, loaded = SuperMimicJax.load_agent(path)
    assert int(loaded.iteration) == int(state1.iteration)
    third = train_fn(jax.random.PRNGKey(1), loaded, traj)
    m3 = third["training_metrics"]
    assert int(m3.max_timestep[-1]) == int(m2.max_timestep[-1])
    assert int(m3.num_updates[0]) == int(m1.num_updates[-1])  # still warming up
    assert int(m3.num_updates[-1]) > int(m1.num_updates[-1])
    for name in ("mean_wm_loss", "mean_imag_return", "mean_critic_loss"):
        assert np.all(np.isfinite(np.asarray(getattr(m3, name)))), name


def test_handler_absorbing_prob_replaces_learned_head(supermimic_config):
    config = OmegaConf.create(OmegaConf.to_container(supermimic_config, resolve=True))
    env, traj = _make(config)
    assert env._terminal_state_handler.mjx_absorbing_prob(env, None, None, None, None, traj) is None

    p = 0.25
    env._terminal_state_handler.mjx_absorbing_prob = lambda *args: jnp.array(p)
    agent_conf = SuperMimicJax.init_agent_conf(env, config)
    rng = jax.random.PRNGKey(0)
    agent_state = SuperMimicJax.init_agent_state(env, agent_conf, rng)
    m = jax.jit(SuperMimicJax.build_train_fn(env, agent_conf))(rng, agent_state, traj)["training_metrics"]

    assert int(m.num_updates[-1]) > 0
    expected = (1.0 - p) ** (int(config.experiment.imag_horizon) - 1)  # survival before the last imagined step
    np.testing.assert_allclose(float(m.mean_imag_survival[-1]), expected, rtol=1e-5)


@pytest.mark.parametrize("domain_randomization", (False, True))
def test_SuperMimic_train_save_load(supermimic_config, domain_randomization, tmp_path):
    config = OmegaConf.create(OmegaConf.to_container(supermimic_config, resolve=True))
    if domain_randomization:
        dr_conf = OmegaConf.load(Path(__file__).resolve().parent / "test_conf/default_dom_rand_conf.yaml")
        with open_dict(config.experiment.env_params):
            config.experiment.env_params.domain_randomization_type = "DefaultRandomizer"
            config.experiment.env_params.domain_randomization_params = dr_conf
    env, traj = _make(config)
    agent_conf = SuperMimicJax.init_agent_conf(env, config)
    assert (int(agent_conf.config.experiment.xi_dim) > 0) == domain_randomization

    rng = jax.random.PRNGKey(0)
    agent_state = SuperMimicJax.init_agent_state(env, agent_conf, rng)
    train_fn = jax.jit(SuperMimicJax.build_train_fn(env, agent_conf))
    out = train_fn(rng, agent_state, traj)
    trained = out["agent_state"]
    m = out["training_metrics"]

    assert int(m.num_updates[-1]) > 0
    for name in ("mean_wm_loss", "mean_imag_return", "mean_critic_loss", "mean_actor_grad_norm"):
        assert np.all(np.isfinite(np.asarray(getattr(m, name)))), name
    assert float(m.mean_actor_grad_norm[-1]) > 0.0
    assert not _params_allclose(trained.actor_state.params, agent_state.actor_state.params)
    assert not _params_allclose(trained.wm_state.params, agent_state.wm_state.params)

    path = SuperMimicJax.save_agent(tmp_path, agent_conf, trained)
    loaded_conf, loaded = SuperMimicJax.load_agent(path)
    assert _params_allclose(trained.actor_state.params, loaded.actor_state.params)
    assert _params_allclose(trained.critic_state.params, loaded.critic_state.params)
    assert _params_allclose(trained.wm_state.params, loaded.wm_state.params)
    np.testing.assert_allclose(np.asarray(trained.xi_std), np.asarray(loaded.xi_std))
