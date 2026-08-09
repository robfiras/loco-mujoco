"""Tests for the pure helpers in loco_mujoco.datasets.data_generation.utils.

These cover the finite-difference velocity computation, the ``!expr`` YAML
loader, and the body/site name-and-id lookups -- none of which need a full
environment build. The heavier replay callbacks (ExtendTrajData /
CollisionExtender / optimize_for_collisions) are driven elsewhere via the
retargeting pipeline.
"""
from pathlib import Path

import mujoco
import numpy as np
import pytest
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as sRot

from loco_mujoco.datasets.data_generation.utils import (
    ExtendTrajData,
    add_mocap_bodies,
    calculate_qvel_with_finite_difference,
    expression_constructor,  # noqa: F401 (kept for import coverage / clarity)
    load_robot_conf,
    load_dataset_conf,
)
from loco_mujoco.core.utils.math import quat_scalarfirst2scalarlast

_MODEL_XML = (Path(__file__).resolve().parent / "test_conf" / "humanoid_test.xml").as_posix()


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(_MODEL_XML)


# --------------------------- calculate_qvel_with_finite_difference ---------------------------

def test_finite_difference_shapes_and_translation():
    n, njnt = 5, 3
    freq = 10.0
    qpos = np.zeros((n, 7 + njnt))
    # constant unit velocity along x for the free-joint position
    qpos[:, 0] = np.arange(n) * (1.0 / freq)  # x moves 1/freq per step -> vel 1.0
    # identity quaternion (scalar-first)
    qpos[:, 3] = 1.0
    # a hinge joint moving linearly
    qpos[:, 7] = np.arange(n) * 0.5

    qpos_out, qvel = calculate_qvel_with_finite_difference(qpos, freq)

    # trims the first and last frame
    assert qpos_out.shape == (n - 2, 7 + njnt)
    assert qvel.shape == (n - 2, 6 + njnt)
    # translational x-velocity is the finite difference of x -> 1.0
    np.testing.assert_allclose(qvel[:, 0], 1.0, atol=1e-6)
    # identity orientation -> zero angular velocity
    np.testing.assert_allclose(qvel[:, 3:6], 0.0, atol=1e-6)
    # hinge velocity = 0.5 * freq
    np.testing.assert_allclose(qvel[:, 6], 0.5 * freq, atol=1e-6)


def test_finite_difference_rotation():
    n = 3
    freq = 100.0
    qpos = np.zeros((n, 7))
    qpos[:, 0:3] = 0.0
    # rotate about z by an increasing angle
    angles = np.array([0.0, 0.01, 0.02])
    for i, a in enumerate(angles):
        quat_xyzw = sRot.from_euler("z", a).as_quat()  # scalar-last
        # store scalar-first
        qpos[i, 3:7] = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

    _, qvel = calculate_qvel_with_finite_difference(qpos, freq)
    # angular velocity about z should be positive, x/y ~ 0
    assert qvel.shape == (1, 6)
    assert qvel[0, 5] > 0
    np.testing.assert_allclose(qvel[0, 3:5], 0.0, atol=1e-6)


# --------------------------- YAML loaders ---------------------------

def test_load_robot_conf_evaluates_expr(tmp_path):
    p = tmp_path / "conf.yaml"
    p.write_text("angle: !expr pi/2\nscale: !expr np.sqrt(4)\nplain: 3\n")
    conf = load_robot_conf(p.as_posix())
    assert conf["angle"] == pytest.approx(np.pi / 2)
    assert conf["scale"] == pytest.approx(2.0)
    assert conf["plain"] == 3


def test_load_robot_conf_bad_expr_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    # builtins are disabled, so open(...) is not available -> ValueError
    p.write_text("x: !expr open('/etc/passwd')\n")
    with pytest.raises(ValueError, match="Error evaluating expression"):
        load_robot_conf(p.as_posix())


def test_load_dataset_conf_plain(tmp_path):
    p = tmp_path / "ds.yaml"
    p.write_text("a: 1\nb: [1, 2, 3]\n")
    conf = load_dataset_conf(p.as_posix())
    assert conf == {"a": 1, "b": [1, 2, 3]}


# --------------------------- body/site name+id lookups ---------------------------

def test_get_body_names_and_ids_all(model):
    names, ids = ExtendTrajData.get_body_names_and_ids(model, keys=None)
    assert len(names) == model.nbody
    assert ids == list(range(model.nbody))
    assert "world" in names


def test_get_body_names_and_ids_validates_keys(model):
    # existing keys are accepted; the full name list is still returned
    names, ids = ExtendTrajData.get_body_names_and_ids(model, keys=["torso"])
    assert "torso" in names
    with pytest.raises(AssertionError, match="Could not find"):
        ExtendTrajData.get_body_names_and_ids(model, keys=["not_a_body"])


def test_get_site_names_and_ids(model):
    names, ids = ExtendTrajData.get_site_names_and_ids(model, keys=None)
    assert len(names) == model.nsite
    assert ids == list(range(model.nsite))
    with pytest.raises(AssertionError, match="Could not find"):
        ExtendTrajData.get_site_names_and_ids(model, keys=["not_a_site"])


# --------------------------- add_mocap_bodies ---------------------------

_SITES = ["torso_site", "pelvis_site"]


def _fresh_spec():
    return mujoco.MjSpec.from_file(_MODEL_XML)


def test_add_mocap_bodies_no_conf():
    spec = _fresh_spec()
    n_eq_before = spec.compile().neq
    mocap = ["target_mocap_body_" + s for s in _SITES]

    spec = add_mocap_bodies(spec, _SITES, mocap, add_equality_constraint=True)
    model = spec.compile()

    # both mocap bodies were added and a WELD equality created per site
    assert model.nmocap == len(mocap)
    assert model.neq == n_eq_before + len(_SITES)


def test_add_mocap_bodies_without_equality():
    spec = _fresh_spec()
    n_eq_before = spec.compile().neq
    mocap = ["target_mocap_body_" + s for s in _SITES]

    spec = add_mocap_bodies(spec, _SITES, mocap, add_equality_constraint=False)
    model = spec.compile()

    assert model.nmocap == len(mocap)
    assert model.neq == n_eq_before  # no equalities added


def test_add_mocap_bodies_with_robot_conf():
    spec = _fresh_spec()
    mocap = ["target_mocap_body_" + s for s in _SITES]

    robot_conf = OmegaConf.create({
        "optimization_params": {
            "disable_joint_limits": True,
            "disable_collisions": True,
        },
        "site_joint_matches": {
            "torso_site": {
                "equality_constraint_type": "mjEQ_WELD",
                "torque_scale": 0.5,
                "solref": [0.02, 1.0],
                "solimp": [0.9, 0.95, 0.001, 0.5, 2.0],
            },
            "pelvis_site": {
                "equality_constraint_type": "mjEQ_WELD",
                "torque_scale": 1.0,
            },
        },
    })

    spec = add_mocap_bodies(
        spec, _SITES, mocap, robot_conf=robot_conf,
        add_equality_constraint=True,
        height_adjustment_geom_names=["right_foot", "left_foot"],
        max_height_adjustment=0.5,
    )
    model = spec.compile()

    assert model.nmocap == len(mocap)
    assert model.neq >= len(_SITES)
    # disable_joint_limits took effect -> no joint remains limited
    assert not np.any(model.jnt_limited)
    # disable_collisions zeroed contype/conaffinity on all geoms
    assert not np.any(model.geom_contype)
    assert not np.any(model.geom_conaffinity)
