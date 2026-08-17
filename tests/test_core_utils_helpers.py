"""Pure-logic coverage for small helper modules that need no full env build:

* ``loco_mujoco.core.utils.mujoco`` -- joint/geom name<->id lookups and the
  numpy-backend collision helpers, driven on tiny inline MjModels.
* ``loco_mujoco.utils.running_stats`` -- the working parts of the running-stat
  utilities (Welford standardization + window reset/mean).

NOTE: ``RunningAveragedWindow.update_state`` is left uncovered on purpose -- it is
currently broken (see comment in the window test) and is dead code besides.
"""
import mujoco
import numpy as np
import jax.numpy as jnp
import pytest

from loco_mujoco.core.utils.mujoco import (
    mj_jnt_name2id,
    mj_jntname2qposid,
    mj_jntname2qvelid,
    mj_jntid2qposid,
    mj_jntid2qvelid,
    mj_spec_find_geom_id,
    mj_get_collision_dist_and_normal,
    mj_check_collisions,
)
from loco_mujoco.utils.running_stats import (
    RunningStandardization,
    RunningAveragedWindow,
    RunningAverageWindowState,
)

# one free joint (b0) + one hinge joint (b1); b0 and b1 boxes overlap -> contact,
# the "far" box collides with nothing.
_COLLISION_XML = """
<mujoco>
  <worldbody>
    <body name="b0"><freejoint name="free0"/><geom name="g0" type="box" size="0.1 0.1 0.1"/></body>
    <body name="b1" pos="0.08 0 0">
      <joint name="hinge1" type="hinge" axis="0 0 1"/>
      <geom name="g1" type="box" size="0.1 0.1 0.1"/>
    </body>
    <body name="b2" pos="10 0 0"><freejoint/><geom name="far" type="box" size="0.1 0.1 0.1"/></body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def coll_model():
    return mujoco.MjModel.from_xml_string(_COLLISION_XML)


# ------------------------------- joint lookups -------------------------------

def test_jnt_name2id_and_qpos_qvel(coll_model):
    # free joint on b0 is joint 0, hinge1 is joint 1
    hid = mj_jnt_name2id("hinge1", coll_model)
    assert coll_model.joint(hid).name == "hinge1"

    # free joint -> 7 qpos ids / 6 qvel ids; hinge -> 1 each
    free_qpos = mj_jntname2qposid("free0", coll_model)
    free_qvel = mj_jntname2qvelid("free0", coll_model)
    assert len(free_qpos) == 7 and len(free_qvel) == 6
    assert len(mj_jntid2qposid(hid, coll_model)) == 1
    assert len(mj_jntid2qvelid(hid, coll_model)) == 1


def test_joint_lookups_raise_on_unknown(coll_model):
    with pytest.raises(ValueError, match="not found"):
        mj_jnt_name2id("nope", coll_model)
    with pytest.raises(ValueError, match="not found"):
        mj_jntname2qposid("nope", coll_model)
    with pytest.raises(ValueError, match="not found"):
        mj_jntname2qvelid("nope", coll_model)


# ------------------------------- geom-in-spec lookup -------------------------------

def test_spec_find_geom_id():
    spec = mujoco.MjSpec.from_string(_COLLISION_XML)
    gid = mj_spec_find_geom_id(spec, "g1")
    assert spec.geoms[gid].name == "g1"
    with pytest.raises(ValueError, match="not found"):
        mj_spec_find_geom_id(spec, "no_such_geom")


# ------------------------------- numpy collision helpers -------------------------------

def _forward(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return data


def test_collision_dist_and_normal_numpy(coll_model):
    data = _forward(coll_model)
    g0 = mujoco.mj_name2id(coll_model, mujoco.mjtObj.mjOBJ_GEOM, "g0")
    g1 = mujoco.mj_name2id(coll_model, mujoco.mjtObj.mjOBJ_GEOM, "g1")
    far = mujoco.mj_name2id(coll_model, mujoco.mjtObj.mjOBJ_GEOM, "far")

    # overlapping boxes -> penetration (negative distance) and a unit-ish normal
    dist, normal = mj_get_collision_dist_and_normal(g0, g1, data, np)
    assert dist < 0
    assert normal.shape == (3,)
    assert np.linalg.norm(normal) > 0

    # non-colliding pair -> the "not found" fallthrough: dist 0, zero normal
    dist2, normal2 = mj_get_collision_dist_and_normal(g0, far, data, np)
    assert dist2 == 0.0
    assert np.allclose(normal2, 0.0)


def test_check_collisions_numpy(coll_model):
    data = _forward(coll_model)
    g0 = mujoco.mj_name2id(coll_model, mujoco.mjtObj.mjOBJ_GEOM, "g0")
    g1 = mujoco.mj_name2id(coll_model, mujoco.mjtObj.mjOBJ_GEOM, "g1")
    far = mujoco.mj_name2id(coll_model, mujoco.mjtObj.mjOBJ_GEOM, "far")

    assert bool(mj_check_collisions(g0, g1, data, np)) is True
    assert bool(mj_check_collisions(g0, far, data, np)) is False


# ------------------------------- running stats -------------------------------

def test_running_standardization_welford():
    with pytest.raises(AssertionError):
        RunningStandardization(shape=(), alpha=0.0)  # alpha must be in (0, 1)

    rs = RunningStandardization(shape=(), alpha=1e-3)
    state = rs.reset()
    assert float(state.count) == 1
    for v in (1.0, 2.0, 3.0):
        state = rs.update_state(jnp.array(v), state)
    # count advanced by one per scalar update; mean tracks the stream
    assert float(state.count) == 4
    assert jnp.isfinite(state.mean).all() and jnp.isfinite(state.std).all()


def test_running_average_window_reset_and_mean():
    raw = RunningAveragedWindow(shape=(), window_size=3)
    state = raw.reset()
    assert state.storage.shape == (3,)
    assert int(state.index) == 0 and int(state.curr_size) == 0

    # mean() is a pure staticmethod over the stored window; build a filled state
    # directly (update_state is currently broken -- it atleast_2d's the value to
    # (1,1) which cannot scatter into the scalar storage slot, and reset() only
    # supports scalar shape, so it is effectively dead).
    filled = RunningAverageWindowState(storage=jnp.array([2.0, 4.0, 6.0]),
                                       index=0, curr_size=3)
    assert float(RunningAveragedWindow.mean(filled)) == pytest.approx(4.0)
