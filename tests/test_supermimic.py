from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import optax
import pytest
from mujoco import mjx
from omegaconf import OmegaConf, open_dict

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms.experimental import SuperMimicJax
from loco_mujoco.core.mujoco_base import Mujoco
from loco_mujoco.algorithms.experimental.supermimic_jax import (BinResets, WorldModelSpec, WindowBuffer, state_is_valid,
                                                                 skip_gradient_spikes, muon_labels, scale_by_muon,
                                                                 _quat_mul, _quat_from_rotvec, _rotvec_from_quat, _Packed)

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


def test_packed_scan_matches_plain_scan():
    """Packing the carry and outputs of a scan changes neither its values nor its gradients."""
    c0 = dict(x=jnp.arange(6.0).reshape(2, 3), n=jnp.array([1, 2], jnp.int32), ok=jnp.array([True, False]),
              key=jax.random.split(jax.random.key(0), 2), empty=jnp.zeros((2, 0)), none=None)

    def step(w, c):
        x = jnp.tanh(c["x"] * w) + c["n"][:, None]
        return dict(c, x=x, n=c["n"] + 1, ok=~c["ok"]), dict(x=x, s=jnp.sum(x, -1), n=c["n"], none=None)

    def run(w, pack):
        if not pack:
            return jax.lax.scan(lambda c, _: step(w, c), c0, None, 4)
        c_shape, o_shape = jax.eval_shape(lambda c: step(w, c), c0)
        pc, po = _Packed(c_shape), _Packed(o_shape)
        cH, ro = jax.lax.scan(lambda c, _: (lambda r: (pc.pack(r[0]), po.pack(r[1])))(step(w, pc.unpack(c))),
                              pc.pack(c0), None, 4)
        return pc.unpack(cH), po.unpack(ro, (4,))

    for a, b in zip(jax.tree.leaves(run(0.3, False)), jax.tree.leaves(run(0.3, True))):
        assert a.shape == b.shape and a.dtype == b.dtype
        if jax.dtypes.issubdtype(a.dtype, jax.dtypes.extended):
            a, b = jax.random.key_data(a), jax.random.key_data(b)
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6)
    loss = lambda w, pack: jnp.sum(run(w, pack)[1]["s"] ** 2)  # noqa: E731
    np.testing.assert_allclose(jax.grad(loss)(0.3, True), jax.grad(loss)(0.3, False), rtol=1e-5)
    assert len(_Packed(c0).pack(c0)[0]) == 3  # float, int and bool arrays; the typed keys stay separate


def _kinematics(sys, data):
    return mjx.com_vel(sys, mjx.com_pos(sys, mjx.kinematics(sys, data)))


def test_world_model_targets_invert_integrate():
    spec = WorldModelSpec(2, 0.02, (1.0,) * 6, (0.5,) * 6, (1.0,) * 6, drift=0.625)
    rng = jax.random.PRNGKey(3)
    qpos, qvel = _random_state(rng, 2)
    delta = 0.1 * jax.random.normal(rng, (qpos.shape[0], spec.delta_dim))
    qpos1, qvel1 = spec.integrate(qpos, qvel, delta)
    np.testing.assert_allclose(np.asarray(spec.targets(qpos, qvel, qpos1, qvel1)), np.asarray(delta), atol=1e-4)


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


def test_recenter_root_xy_off_starts_on_the_reference(supermimic_config):
    config = OmegaConf.create(OmegaConf.to_container(supermimic_config, resolve=True))
    with open_dict(config.experiment.env_params):
        config.experiment.env_params.recenter_root_xy = False
    env, traj = _make(config)
    traj = traj.replace(data=traj.data.replace(qpos=traj.data.qpos.at[:, 0].add(1.0)))  # the clip is not at the origin
    state = jax.jit(lambda k: env.mjx_reset(k, traj))(jax.random.PRNGKey(3))
    ts = state.additional_carry.traj_state
    ref = traj.data.get(ts.traj_no, ts.subtraj_step_no, jnp)
    assert float(ref.qpos[0]) > 0.5
    np.testing.assert_allclose(np.asarray(state.data.qpos[:2]), np.asarray(ref.qpos[:2]), atol=1e-5)


def test_modify_contact_spec():
    spec = mujoco.MjSpec.from_string("""
    <mujoco><worldbody><geom name="floor" type="plane" size="1 1 .1"/>
      <body><freejoint/><geom name="a" size=".1" condim="1"/><geom name="vis" size=".1" contype="0" conaffinity="0"/>
        <body pos="0 0 .05"><joint axis="1 0 0"/><geom name="b" size=".1"/></body></body>
    </worldbody></mujoco>""")
    model = Mujoco._modify_contact_spec(spec, dict(self_collisions=False, condim=3)).compile()
    get = lambda n: (int(model.geom(n).contype[0]), int(model.geom(n).conaffinity[0]), int(model.geom(n).condim[0]))
    assert get("a") == get("b") == (0, 1, 3)
    assert get("floor") == (1, 1, 3)
    assert get("vis")[:2] == (0, 0)
    data = mujoco.MjData(model)
    data.qpos[2] = 0.05
    mujoco.mj_forward(model, data)
    pairs = {tuple(sorted((int(c.geom1), int(c.geom2)))) for c in data.contact[:data.ncon]}
    floor = model.geom("floor").id
    assert pairs and all(floor in p for p in pairs)


def test_bin_resets():
    # two clips of 25 and 10 frames, padded, in bins of 10 frames
    bins = BinResets(jnp.array([0, 25, 35, -1]), 40, 10, 10.0, 0.3)
    assert int(jnp.sum(bins.valid)) == 4
    assert [int(bins.bin_of(c, f)) for c, f in ((0, 0), (0, 24), (1, 0), (1, 9))] == [0, 2, 3, 3]
    assert [int(b) for b in bins.prev_bin[:4]] == [0, 0, 1, 3]

    state = bins.init_state()
    p, _ = bins.probs(state)
    np.testing.assert_allclose(np.asarray(p[:4]), 0.25, rtol=1e-5)
    assert float(jnp.sum(p[4:])) == 0.0

    # failures in bin 2 send resets to the bin before it, and the share fades with them
    failed = state.replace(fail=state.fail.at[2].set(1.0))
    assert float(bins.probs(failed)[0][1] - p[1]) == pytest.approx(0.3 * 0.75, rel=1e-5)
    faded = state.replace(fail=state.fail.at[2].set(0.1))
    assert float(bins.probs(faded)[0][1] - p[1]) == pytest.approx(0.03 * 0.75, rel=1e-5)

    clip, frame, b = bins.sample(failed, jax.random.PRNGKey(0), 4000)
    assert np.all(np.asarray(frame) < np.asarray(bins.clip_len)[np.asarray(clip)] - 0)
    assert np.all(np.asarray(bins.bin_of(clip, frame)) == np.asarray(b))
    assert 0.40 < float(jnp.mean(b == 1)) < 0.55 and int(jnp.max(b)) == 3

    done = jnp.array([True, True, False])
    new = bins.update(state, done, jnp.array([True, False, True]), jnp.array([0, 0, 3]), jnp.array([1, 2, 3]),
                      jnp.array([5.0, 3.0, 100.0]))
    assert float(new.ret[0]) == pytest.approx(0.2 * 4.0) and float(new.ret[3]) == 0.0
    assert float(new.fail[1]) == pytest.approx(0.1) and float(new.fail[2]) == 0.0 and float(new.fail[3]) == 0.0


def test_chosen_reset_and_noise(supermimic_config):
    config = OmegaConf.create(OmegaConf.to_container(supermimic_config, resolve=True))
    with open_dict(config.experiment.env_params):
        config.experiment.env_params.init_state_params = dict(root_pos_noise=[0.05, 0.05, 0.01], joint_pos_noise=0.1)
    env, traj = _make(config)

    def reset_at(key):
        s = env.mjx_reset(key, traj)
        carry = s.additional_carry
        s = s.replace(additional_carry=carry.replace(traj_state=carry.traj_state.replace(
            next_traj_no=jnp.array(0), next_subtraj_step_no=jnp.array(7))))
        return env._mjx_reset_in_step(s, traj)

    state = jax.jit(jax.vmap(reset_at))(jax.random.split(jax.random.PRNGKey(1), 16))
    ts = state.additional_carry.traj_state
    assert np.all(np.asarray(ts.subtraj_step_no) == 7) and np.all(np.asarray(ts.next_traj_no) == -1)
    ref = traj.data.get(0, 7, jnp)
    diff = np.abs(np.asarray(state.data.qpos[:, 7:] - ref.qpos[7:]))
    assert diff.max() <= 0.1 + 1e-5 and diff.std() > 0.01
    z = np.abs(np.asarray(state.data.qpos[:, 2] - ref.qpos[2]))
    assert 0 < z.max() <= 0.01 + 1e-5


def test_refresh_kinematics(supermimic_config):
    config = OmegaConf.create(OmegaConf.to_container(supermimic_config, resolve=True))
    with open_dict(config.experiment.env_params):
        config.experiment.env_params.refresh_kinematics = True
    env, traj = _make(config)
    state = env.mjx_reset(jax.random.PRNGKey(0), traj)
    state = jax.jit(lambda s: env.mjx_step(s, jnp.zeros(env.info.action_space.shape), traj, auto_reset=False))(state)
    fresh = _kinematics(env.sys, state.data.replace(xpos=state.data.xpos * 0.0))
    np.testing.assert_allclose(np.asarray(state.data.xpos), np.asarray(fresh.xpos), atol=1e-5)


@pytest.mark.parametrize("fail_reset", (0.0, 0.3))
def test_SuperMimic_bin_resets_train(supermimic_config, fail_reset):
    config = OmegaConf.create(OmegaConf.to_container(supermimic_config, resolve=True))
    with open_dict(config.experiment):
        config.experiment.bin_resets = True
        config.experiment.bin_seconds = 0.2
        config.experiment.max_step_reward = 1.0
        config.experiment.fail_reset = fail_reset
    env, traj = _make(config)
    agent_conf = SuperMimicJax.init_agent_conf(env, config)
    rng = jax.random.PRNGKey(0)
    agent_state = SuperMimicJax.init_agent_state(env, agent_conf, rng)
    out = jax.jit(SuperMimicJax.build_train_fn(env, agent_conf))(rng, agent_state, traj)
    assert int(out["training_metrics"].num_updates[-1]) > 0
    assert np.all(np.isfinite(np.asarray(out["agent_state"].bin_resets.ret)))


def test_muon():
    params = {f"Dense_{i}": {"kernel": jnp.ones((4, 4)), "bias": jnp.ones(4)} for i in range(4)}
    labels = muon_labels(params)
    assert [labels[f"Dense_{i}"]["kernel"] for i in range(4)] == ["adam", "muon", "muon", "adam"]
    assert {labels[f"Dense_{i}"]["bias"] for i in range(4)} == {"adam"}
    assert muon_labels({"a": params, "b": params}, "a")["b"]["Dense_1"]["kernel"] == "adam"

    g = jax.random.normal(jax.random.PRNGKey(0), (64, 128))
    tx = scale_by_muon()
    u, _ = tx.update(g, tx.init(g))
    # orthogonalised, then scaled to an update RMS of 0.2
    assert 0.15 < float(jnp.sqrt(jnp.mean(u ** 2))) < 0.25


def test_SuperMimic_blink_options_train(supermimic_config, tmp_path):
    config = OmegaConf.create(OmegaConf.to_container(supermimic_config, resolve=True))
    with open_dict(config.experiment):
        config.experiment.update(dict(policy_ema=0.9, actor_optimizer="muon", wm_optimizer="muon", policy_lr_mult=4.0,
                                      lr_decay_by=2, lr_decay_len=3, wm_layer_norm=True, wm_drift=0.625,
                                      start_buffer_stride=2, net_dtype="bfloat16", wm_running_norm=True,
                                      obs_norm=True, imag_soft_termination_only=True))
    env, traj = _make(config)
    agent_conf = SuperMimicJax.init_agent_conf(env, config)
    rng = jax.random.PRNGKey(0)
    agent_state = SuperMimicJax.init_agent_state(env, agent_conf, rng)
    out = jax.jit(SuperMimicJax.build_train_fn(env, agent_conf))(rng, agent_state, traj)
    trained, m = out["agent_state"], out["training_metrics"]
    assert int(m.num_updates[-1]) > 0 and float(m.mean_actor_grad_norm[-1]) > 0.0
    assert all(x.dtype == jnp.float32 for x in jax.tree.leaves(trained.actor_state.params))
    assert not _params_allclose(trained.actor_ema, trained.actor_state.params)
    assert int(trained.lr_decay_start) >= 0

    _, loaded = SuperMimicJax.load_agent(SuperMimicJax.save_agent(tmp_path, agent_conf, trained))
    assert _params_allclose(trained.deployed_params, loaded.deployed_params)
    assert int(loaded.lr_decay_start) == int(trained.lr_decay_start)


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


def test_traj_future_frames_matches_frame_by_frame():
    from types import SimpleNamespace
    from loco_mujoco.core.observations.goals import TrajFutureFrames
    from loco_mujoco.core.utils.math import heading_mat as _heading_mat, quat_to_mat as _quat_to_mat, rot6d_from_mat as _rot6d

    rng = np.random.default_rng(0)
    n_future, nj, T = 4, 3, 12
    quat = rng.normal(size=(T, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    qpos = np.concatenate([rng.normal(size=(T, 3)), quat, rng.normal(size=(T, nj))], axis=1)
    qvel = rng.normal(size=(T, 6 + nj))
    obs = TrajFutureFrames(n_future=n_future, joint_vel_scale=0.1)
    obs._dim = n_future * (15 + 2 * nj)
    td = SimpleNamespace(qpos=qpos, qvel=qvel, split_points=np.array([0, 5, T]))
    step = 4  # two frames before the end of the second clip, so the last frames repeat
    carry = SimpleNamespace(traj_state=SimpleNamespace(traj_no=1, subtraj_step_no=step))
    out, _ = obs.get_obs_and_update_state(None, None, None, carry, np, SimpleNamespace(data=td))

    ref = 5 + step
    heading_inv = _heading_mat(_quat_to_mat(qpos[ref, 3:7], np), np).T
    expected = []
    for k in range(1, n_future + 1):
        f = min(ref + k, T - 1)
        R = _quat_to_mat(qpos[f, 3:7], np)
        expected += [heading_inv @ (qpos[f, :3] - qpos[ref, :3]), _rot6d(heading_inv @ R, np), R.T @ qvel[f, :3],
                     qvel[f, 3:6], qpos[f, 7:], 0.1 * qvel[f, 6:]]
    np.testing.assert_allclose(out, np.concatenate(expected), rtol=1e-6, atol=1e-6)


def test_take_columns_gradient():
    from loco_mujoco.core.utils.math import take_columns

    x = jnp.arange(12.0).reshape(3, 4)
    ind = np.array([3, 0])
    w = jnp.array([2.0, 5.0])
    np.testing.assert_array_equal(take_columns(x, ind), x[:, ind])
    np.testing.assert_array_equal(jax.grad(lambda y: jnp.sum(take_columns(y, ind) * w))(x),
                                  jax.grad(lambda y: jnp.sum(y[:, ind] * w))(x))


def test_flat_kinematics_matches_mjx():
    from mujoco import mjx
    from loco_mujoco.core.utils.flat_kinematics import FlatKinematics

    xml = """
    <mujoco><worldbody><body name="root" pos="0 0 1"><freejoint/><geom size=".1" mass="3"/>
      <site name="s0" pos=".1 0 .2" quat="1 1 0 0"/>
      <body name="a" pos=".2 .1 0" quat=".9 .1 .2 .3"><joint type="hinge" axis="0 1 0" pos="0 0 .05"/><geom size=".1" mass="2" pos=".1 0 0"/>
        <body name="fixed" pos="0 .2 0" quat=".8 0 .6 0"><geom size=".05" mass="1"/>
          <body name="b" pos=".1 0 0"><joint type="hinge" axis="1 0 0" ref=".3"/><geom size=".1" mass="1"/>
            <site name="s1" pos="0 .1 0"/></body></body></body>
      <body name="c" pos="-.2 0 0"><joint type="hinge" axis="0 0 1"/><geom size=".1" mass="1.5"/></body>
    </body></worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    sys = mjx.put_model(model, impl="jax")
    rng = np.random.default_rng(0)
    quat = rng.normal(size=4)
    data = mjx.make_data(model, impl="jax").replace(
        qpos=jnp.array(np.concatenate([rng.normal(size=3), quat / np.linalg.norm(quat), rng.normal(size=3)])),
        qvel=jnp.array(rng.normal(size=model.nv)))
    ref = mjx.com_vel(sys, mjx.com_pos(sys, mjx.kinematics(sys, data)))
    out = FlatKinematics(model)(data)
    for name in ("xpos", "xquat", "xmat", "site_xpos", "site_xmat", "cvel"):
        np.testing.assert_allclose(getattr(out, name), getattr(ref, name), atol=2e-5, err_msg=name)
    np.testing.assert_allclose(out.subtree_com[1], ref.subtree_com[1], atol=2e-5)
