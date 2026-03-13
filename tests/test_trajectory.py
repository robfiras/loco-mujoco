import pytest
import jax.numpy as jnp
import os
import numpy as np
import jax
import mujoco
from mujoco import MjModel, MjData, mj_id2name
from jax import lax
from loco_mujoco.core.trajectory.dataclasses import interpolate_trajectories


from loco_mujoco.core.trajectory import (
    Trajectory,
    TrajectoryInfo,
    TrajectoryData,
)

from test_conf import *

# set Jax-backend to CPU
jax.config.update('jax_platform_name', 'cpu')
print(f"Jax backend device: {jax.default_backend()} \n")

_TRAJ_ATOL = 1e-7
_TRAJ_RTOL = 1e-7


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_trajectory_info_save_and_load(
    backend,
    input_trajectory_info_data,
    input_trajectory_data,
    input_trajectory_transitions,
    tmp_path,
):
    path = tmp_path / "test.npz"
    info: TrajectoryInfo = input_trajectory_info_data(backend)
    data: TrajectoryData = input_trajectory_data(backend)
    transitions: TrajectoryTransitions = input_trajectory_transitions(backend)
    object_test = Trajectory(info, data, transitions)
    object_test.save(path)

    assert path.exists(), "File was not created"

    trajectory = Trajectory.load(path)

    # info test
    assert trajectory.info.joint_names == info.joint_names
    assert trajectory.info.frequency == info.frequency
    assert trajectory.info.metadata == info.metadata
    assert trajectory.info.site_names == info.site_names
    assert trajectory.info.body_names == info.body_names
    assert trajectory.info.model.njnt == info.model.njnt
    assert trajectory.info.model.nbody == info.model.nbody
    assert trajectory.info.model.nsite == info.model.nsite

    np.testing.assert_allclose(
        trajectory.info.model.jnt_type, info.model.jnt_type, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL
    )
    np.testing.assert_allclose(
        trajectory.info.model.body_rootid, info.model.body_rootid,
        atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL,
    )
    np.testing.assert_allclose(
        trajectory.info.model.body_weldid, info.model.body_weldid,
        atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL,
    )
    np.testing.assert_allclose(
        trajectory.info.model.body_mocapid, info.model.body_mocapid,
        atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL,
    )
    np.testing.assert_allclose(
        trajectory.info.model.site_bodyid, info.model.site_bodyid,
        atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL,
    )
    np.testing.assert_allclose(
        trajectory.info.model.body_pos, info.model.body_pos, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL
    )
    np.testing.assert_allclose(
        trajectory.info.model.body_quat, info.model.body_quat, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL
    )
    np.testing.assert_allclose(
        trajectory.info.model.body_ipos, info.model.body_ipos, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL
    )
    np.testing.assert_allclose(
        trajectory.info.model.body_iquat, info.model.body_iquat, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL
    )
    np.testing.assert_allclose(
        trajectory.info.model.site_pos, info.model.site_pos, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL
    )
    np.testing.assert_allclose(
        trajectory.info.model.site_quat, info.model.site_quat, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL
    )

    # data test
    np.testing.assert_allclose(trajectory.data.qpos, data.qpos, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL)
    np.testing.assert_allclose(trajectory.data.qvel, data.qvel, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL)
    np.testing.assert_allclose(trajectory.data.xpos, data.xpos, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL)
    np.testing.assert_allclose(trajectory.data.xquat, data.xquat, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL)
    np.testing.assert_allclose(trajectory.data.cvel, data.cvel, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL)
    np.testing.assert_allclose(trajectory.data.subtree_com, data.subtree_com, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL)
    np.testing.assert_allclose(trajectory.data.site_xpos, data.site_xpos, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL)
    np.testing.assert_allclose(trajectory.data.site_xmat, data.site_xmat, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL)

    # transition tests
    np.testing.assert_allclose(
        trajectory.transitions.observations, transitions.observations, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL
    )
    np.testing.assert_allclose(
        trajectory.transitions.next_observations,
        transitions.next_observations,
        atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL,
    )
    np.testing.assert_allclose(
        trajectory.transitions.absorbings, transitions.absorbings, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL
    )
    np.testing.assert_allclose(
        trajectory.transitions.dones, transitions.dones, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL
    )
    np.testing.assert_allclose(
        trajectory.transitions.actions, transitions.actions, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL
    )
    np.testing.assert_allclose(
        trajectory.transitions.rewards, transitions.rewards, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL
    )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_trajectory_info__eq__(input_trajectory_info_data, backend):
    trajectoryInfo: TrajectoryInfo = input_trajectory_info_data(backend)
    trajectoryInfo_2: TrajectoryInfo = input_trajectory_info_data(backend)
    trajectoryInfo_3: TrajectoryInfo = input_trajectory_info_data(backend)

    backend_type = jnp if backend == "jax" else np

    assert (
        trajectoryInfo.__eq__(trajectoryInfo_2) == True
    ), "These objects are not equal!"

    last_joint = trajectoryInfo.joint_names[-1]
    trajectoryInfo_3 = trajectoryInfo_3.remove_joints([last_joint], backend_type)

    assert trajectoryInfo.__eq__(trajectoryInfo_3) == False


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_trajectory_info__post_init__(
    backend,
    input_trajectory_info_data,
    input_expected_joint_name2ind_qpos,
    input_body_name2ind,
    input_site_name2ind,
):
    trajectoryInfo: TrajectoryInfo = input_trajectory_info_data(backend)

    expected_joint_name2ind_qpos = input_expected_joint_name2ind_qpos(backend)
    expected_body_name2ind = input_body_name2ind(backend)
    expected_site_name2ind = input_site_name2ind(backend)

    # Validate joint name to index mapping
    for key, expected_value in expected_joint_name2ind_qpos.items():
        np.testing.assert_allclose(
            expected_value,
            trajectoryInfo.joint_name2ind_qpos[key],
            err_msg=f"Mismatch for joint '{key}' in backend {backend}",
        )

    # Validate body name to index mapping
    for key, expected_value in expected_body_name2ind.items():
        np.testing.assert_allclose(
            expected_value,
            trajectoryInfo.body_name2ind[key],
            err_msg=f"Mismatch for body '{key}' in backend {backend}",
        )

    # Validate site name to index mapping
    for key, expected_value in expected_site_name2ind.items():
        np.testing.assert_allclose(
            expected_value,
            trajectoryInfo.site_name2ind[key],
            err_msg=f"Mismatch for site '{key}' in backend {backend}",
        )


def test_trajectory_info_get_attribute_names(input_trajectory_info_field_names):
    attribute_names = TrajectoryInfo.get_attribute_names()

    assert (
        attribute_names == input_trajectory_info_field_names
    ), f"Expected {input_trajectory_info_field_names}, but got {attribute_names}"


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_trajectory_info_add_joint(backend, input_trajectory_info_data):
    trajectoryInfo: TrajectoryInfo = input_trajectory_info_data(backend)

    backend_type = jnp if backend == "jax" else np

    jnt_names = trajectoryInfo.joint_names
    jnt_type = trajectoryInfo.model.jnt_type

    trajectoryInfo = trajectoryInfo.add_joint("joint1", 3, backend_type)
    jnt_names.append("joint1")
    jnt_type = backend_type.append(jnt_type, 3)

    assert trajectoryInfo.joint_names == jnt_names

    np.testing.assert_allclose(trajectoryInfo.model.jnt_type, jnt_type)

    trajectoryInfo = trajectoryInfo.add_joint("joint2", 0)
    jnt_names.append("joint2")
    jnt_type = backend_type.append(jnt_type, 0)

    assert trajectoryInfo.joint_names == jnt_names
    np.testing.assert_allclose(trajectoryInfo.model.jnt_type, jnt_type)


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_trajectory_info_add_body(backend, input_trajectory_info_data):
    trajectoryInfo: TrajectoryInfo = input_trajectory_info_data(backend)

    backend_type = jnp if backend == "jax" else np

    nbody = trajectoryInfo.model.nbody
    body_rootid = trajectoryInfo.model.body_rootid
    body_weldid = trajectoryInfo.model.body_weldid
    body_mocapid = trajectoryInfo.model.body_mocapid
    body_pos = trajectoryInfo.model.body_pos
    body_quat = trajectoryInfo.model.body_quat
    body_ipos = trajectoryInfo.model.body_ipos
    body_iquat = trajectoryInfo.model.body_iquat

    trajectoryInfo = trajectoryInfo.add_body(
        "head",
        15,
        16,
        17,
        backend_type.array([0.0, 0.0, 0.0]),
        backend_type.array([1.0, 0.0, 0.0, 0.0]),
        backend_type.array([0.0, 0.0, 0.0]),
        backend_type.array([1.0, 0.0, 0.0, 0.0]),
        backend_type,
    )

    body_rootid = backend_type.append(body_rootid, 15)
    body_weldid = backend_type.append(body_weldid, 16)
    body_mocapid = backend_type.append(body_mocapid, 17)
    new_row = backend_type.array([0.0, 0.0, 0.0])
    body_pos = backend_type.vstack([body_pos, new_row])
    new_row = backend_type.array([1.0, 0.0, 0.0, 0.0])
    body_quat = backend_type.vstack([body_quat, new_row])
    new_row = backend_type.array([0.0, 0.0, 0.0])
    body_ipos = backend_type.vstack([body_ipos, new_row])
    new_row = backend_type.array([1.0, 0.0, 0.0, 0.0])
    body_iquat = backend_type.vstack([body_iquat, new_row])

    assert trajectoryInfo.model.nbody == nbody + 1
    np.testing.assert_allclose(trajectoryInfo.model.body_rootid, body_rootid)
    np.testing.assert_allclose(trajectoryInfo.model.body_weldid, body_weldid)
    np.testing.assert_allclose(trajectoryInfo.model.body_mocapid, body_mocapid)
    np.testing.assert_allclose(trajectoryInfo.model.body_pos, body_pos)
    np.testing.assert_allclose(trajectoryInfo.model.body_quat, body_quat)
    np.testing.assert_allclose(trajectoryInfo.model.body_ipos, body_ipos)
    np.testing.assert_allclose(trajectoryInfo.model.body_iquat, body_iquat)


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_trajectory_info_add_site(backend, input_trajectory_info_data):
    trajectoryInfo: TrajectoryInfo = input_trajectory_info_data(backend)

    backend_type = jnp if backend == "jax" else np

    site_names = trajectoryInfo.site_names
    site_bodyid = trajectoryInfo.model.site_bodyid
    site_pos = trajectoryInfo.model.site_pos
    site_quat = trajectoryInfo.model.site_quat

    trajectoryInfo = trajectoryInfo.add_site(
        "new_site",
        backend_type.array([0.0, 0.0, 0.0]),
        backend_type.array([1.0, 0.0, 0.0, 0.0]),
        2,
        backend_type,
    )

    site_names.append("new_site")
    site_bodyid = backend_type.append(site_bodyid, 2)
    new_row = backend_type.array([0.0, 0.0, 0.0])
    site_pos = backend_type.vstack([site_pos, new_row])
    new_row = backend_type.array([1.0, 0.0, 0.0, 0.0])
    site_quat = backend_type.vstack([site_quat, new_row])

    assert trajectoryInfo.site_names == site_names
    np.testing.assert_allclose(trajectoryInfo.model.site_bodyid, site_bodyid)
    np.testing.assert_allclose(trajectoryInfo.model.site_pos, site_pos)
    np.testing.assert_allclose(trajectoryInfo.model.site_quat, site_quat)


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_trajectory_info_remove_joints(backend, input_trajectory_info_data):
    trajectoryInfo: TrajectoryInfo = input_trajectory_info_data(backend)

    backend_type = jnp if backend == "jax" else np

    last_joint = trajectoryInfo.joint_names[-1]
    joint_names = trajectoryInfo.joint_names[:-1]
    jnt_type = trajectoryInfo.model.jnt_type[:-1]
    njnt = trajectoryInfo.model.njnt

    trajectoryInfo = trajectoryInfo.remove_joints([last_joint], backend_type)

    assert trajectoryInfo.joint_names == joint_names
    assert trajectoryInfo.model.njnt == njnt - 1
    np.testing.assert_allclose(trajectoryInfo.model.jnt_type, jnt_type)


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_trajectory_info_remove_bodies(backend, input_trajectory_info_data):
    trajectoryInfo: TrajectoryInfo = input_trajectory_info_data(backend)

    backend_type = jnp if backend == "jax" else np
    last_body_name = trajectoryInfo.body_names[-1]
    body_names = trajectoryInfo.body_names[:-1]
    nbody = trajectoryInfo.model.nbody - 1
    body_rootid = trajectoryInfo.model.body_rootid[:-1]
    body_weldid = trajectoryInfo.model.body_weldid[:-1]
    body_mocapid = trajectoryInfo.model.body_mocapid[:-1]
    body_pos = trajectoryInfo.model.body_pos[:-1]
    body_quat = trajectoryInfo.model.body_quat[:-1]
    body_ipos = trajectoryInfo.model.body_ipos[:-1]
    body_iquat = trajectoryInfo.model.body_iquat[:-1]
    trajectoryInfo = trajectoryInfo.remove_bodies([last_body_name], backend_type)

    assert trajectoryInfo.body_names == body_names
    assert trajectoryInfo.model.nbody == nbody
    np.testing.assert_allclose(trajectoryInfo.model.body_rootid, body_rootid)
    np.testing.assert_allclose(trajectoryInfo.model.body_weldid, body_weldid)
    np.testing.assert_allclose(trajectoryInfo.model.body_mocapid, body_mocapid)
    np.testing.assert_allclose(trajectoryInfo.model.body_pos, body_pos)
    np.testing.assert_allclose(trajectoryInfo.model.body_quat, body_quat)
    np.testing.assert_allclose(trajectoryInfo.model.body_ipos, body_ipos)
    np.testing.assert_allclose(trajectoryInfo.model.body_iquat, body_iquat)


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_trajectory_info_remove_sites(backend, input_trajectory_info_data):
    trajectoryInfo: TrajectoryInfo = input_trajectory_info_data(backend)

    backend_type = jnp if backend == "jax" else np
    last_site_name = trajectoryInfo.site_names[-1]
    site_names = trajectoryInfo.site_names[:-1]
    nsite = trajectoryInfo.model.nsite - 1
    site_bodyid = trajectoryInfo.model.site_bodyid[:-1]
    site_pos = trajectoryInfo.model.site_pos[:-1]
    site_quat = trajectoryInfo.model.site_quat[:-1]
    trajectoryInfo = trajectoryInfo.remove_sites([last_site_name], backend_type)

    assert trajectoryInfo.site_names == site_names
    assert trajectoryInfo.model.nsite == nsite
    np.testing.assert_allclose(trajectoryInfo.model.site_bodyid, site_bodyid)
    np.testing.assert_allclose(trajectoryInfo.model.site_pos, site_pos)
    np.testing.assert_allclose(trajectoryInfo.model.site_quat, site_quat)


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_trajectory_info_reorder_joints(backend, input_trajectory_info_data):
    trajectoryInfo: TrajectoryInfo = input_trajectory_info_data(backend)

    backend_type = jnp if backend == "jax" else np

    trajectoryInfo = trajectoryInfo.reorder_joints(
        [1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], backend_type
    )

    assert trajectoryInfo.joint_names == [
        "abdomen_z",
        "root",
        "abdomen_y",
        "abdomen_x",
        "right_hip_x",
        "right_hip_z",
        "right_hip_y",
        "right_knee",
        "left_hip_x",
        "left_hip_z",
        "left_hip_y",
        "left_knee",
    ]

    np.testing.assert_allclose(
        trajectoryInfo.model.jnt_type,
        backend_type.array([3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]),
    )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_trajectory_info_reorder_bodies(backend, input_trajectory_info_data):
    trajectoryInfo: TrajectoryInfo = input_trajectory_info_data(backend)

    backend_type = jnp if backend == "jax" else np

    body_names = trajectoryInfo.body_names.copy()
    body_names[0], body_names[1] = body_names[1], body_names[0]

    trajectoryInfo = trajectoryInfo.reorder_bodies(
        [1, 0, 2, 3, 4, 5, 6, 7, 8, 9], backend_type
    )

    assert trajectoryInfo.body_names == body_names
    np.testing.assert_allclose(
        trajectoryInfo.model.body_rootid,
        backend_type.array([1, 0, 1, 1, 1, 1, 1, 1, 1, 1]),
        atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL,
    )
    np.testing.assert_allclose(
        trajectoryInfo.model.body_weldid,
        backend_type.array([1, 0, 2, 3, 4, 5, 5, 7, 8, 8]),
        atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL,
    )
    np.testing.assert_allclose(
        trajectoryInfo.model.body_mocapid,
        backend_type.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]),
        atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL,
    )
    np.testing.assert_allclose(
        trajectoryInfo.model.body_pos,
        backend_type.array(
            [
                [0.0, 0.0, 1.293],
                [0.0, 0.0, 0.0],
                [-0.01, 0.0, -0.26],
                [0.0, 0.0, -0.165],
                [0.0, -0.1, -0.04],
                [0.0, 0.01, -0.403],
                [0.0, 0.0, -0.45],
                [0.0, 0.1, -0.04],
                [0.0, -0.01, -0.403],
                [0.0, 0.0, -0.45],
            ]
        ),
        atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL,
    )
    np.testing.assert_allclose(
        trajectoryInfo.model.body_quat,
        backend_type.array(
            [
                [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [1.0, 0.0, 0.0, 0.0],
                [0.99623686, 0, 0, 0.0866726],
                [1, 0, 0, 0],
                [0.9998001, 0, 0, 0.019996],
                [0.99911916, 0, 0, 0.041963],
                [1, 0, 0, 0],
                [0.99975806, 0, 0, 0.02199468],
                [0.9994884, 0, 0, 0.03198363],
                [1, 0, 0, 0],
            ]
        ),
        atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL,
    )
    np.testing.assert_allclose(
        trajectoryInfo.model.body_ipos,
        backend_type.array(
            [
                [-0.00253938, 0.0, 0.03466259],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [-0.02, 0.0, 0.0],
                [0.0, 0.005, -0.17],
                [0.0, 0.0, -0.15],
                [0.0, 0.0, 0.1],
                [0.0, -0.005, -0.17],
                [0.0, 0.0, -0.15],
                [0.0, 0.0, 0.1],
            ]
        ),
        atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL,
    )
    np.testing.assert_allclose(
        trajectoryInfo.model.body_iquat,
        backend_type.array(
            [
                [0.99991226, 0.0, 0.01324499, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.70710677, 0.70710677, 0.0, -0.0],
                [0.70710677, 0.70710677, 0.0, -0.0],
                [0.99989194, 0.01470111, 0.0, -0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.99989194, -0.01470111, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        ),
        atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL,
    )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_trajectory_info_reorder_sites(backend, input_trajectory_info_data):
    trajectoryInfo: TrajectoryInfo = input_trajectory_info_data(backend)

    backend_type = jnp if backend == "jax" else np

    site_names = trajectoryInfo.site_names.copy()
    site_names[-1], site_names[-2] = site_names[-2], site_names[-1]

    trajectoryInfo = trajectoryInfo.reorder_sites([0, 1, 2, 3, 5, 4], backend_type)

    assert trajectoryInfo.site_names == site_names
    np.testing.assert_allclose(
        trajectoryInfo.model.site_bodyid,
        backend_type.array([1, 3, 4, 6, 9, 7]),
    )
    np.testing.assert_allclose(
        trajectoryInfo.model.site_pos,
        backend_type.array(
            [
                [0.0, 0.0, 0.1],
                [0.0, 0.0, -0.1],
                [0.0, 0.0, 0.1],
                [0.0, 0.0, 0.15],
                [0.0, 0.0, 0.15],
                [0.0, 0.0, 0.1],
            ]
        ),
        atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL,
    )
    np.testing.assert_allclose(
        trajectoryInfo.model.site_quat,
        backend_type.array(
            [
                [0.995037, 0.09950372, 0.0, 0],
                [0.980581, 0.19611613, 0.0, 0],
                [0.980581, 0.19611613, 0.0, 0],
                [1, 0, 0, 0],
                [0.928477, 0.37139067, 0.0, 0],
                [0.957826, 0.28734789, 0.0, 0],
            ]
        ),
        atol=1e-5,
    )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_trajectory_model__eq__(input_trajectory_info_data, backend):
    trajectoryInfo: TrajectoryInfo = input_trajectory_info_data(backend)
    trajectoryInfo_2: TrajectoryInfo = input_trajectory_info_data(backend)
    trajectoryInfo_3: TrajectoryInfo = input_trajectory_info_data(backend)

    backend_type = jnp if backend == "jax" else np

    assert (
        trajectoryInfo.model.__eq__(trajectoryInfo_2.model) == True
    ), "These objects are not equal!"

    trajectoryInfo_3 = trajectoryInfo_3.add_joint("test", 0, backend_type)
    assert trajectoryInfo.model.__eq__(trajectoryInfo_3.model) == False


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_trajectory_model_get_attribute_names(backend, input_trajectory_info_data):
    trajectoryInfo: TrajectoryInfo = input_trajectory_info_data(backend)

    attribute_names = trajectoryInfo.model.get_attribute_names()

    assert attribute_names == [
        "njnt",
        "jnt_type",
        "nbody",
        "body_rootid",
        "body_weldid",
        "body_mocapid",
        "body_pos",
        "body_quat",
        "body_ipos",
        "body_iquat",
        "nsite",
        "site_bodyid",
        "site_pos",
        "site_quat",
    ]


@pytest.mark.parametrize("backend", ["jax"])
def test_trajectory_model_to_numpy(backend, input_trajectory_info_data):
    trajectoryInfo: TrajectoryInfo = input_trajectory_info_data(backend)

    trajectoryModelNumpy = trajectoryInfo.model.to_numpy()

    assert isinstance(trajectoryModelNumpy.jnt_type, np.ndarray)
    assert isinstance(trajectoryModelNumpy.body_rootid, np.ndarray)
    assert isinstance(trajectoryModelNumpy.body_weldid, np.ndarray)
    assert isinstance(trajectoryModelNumpy.body_mocapid, np.ndarray)
    assert isinstance(trajectoryModelNumpy.body_pos, np.ndarray)
    assert isinstance(trajectoryModelNumpy.body_quat, np.ndarray)
    assert isinstance(trajectoryModelNumpy.body_ipos, np.ndarray)
    assert isinstance(trajectoryModelNumpy.body_iquat, np.ndarray)
    assert isinstance(trajectoryModelNumpy.site_bodyid, np.ndarray)
    assert isinstance(trajectoryModelNumpy.site_pos, np.ndarray)
    assert isinstance(trajectoryModelNumpy.site_quat, np.ndarray)


@pytest.mark.parametrize("backend", ["numpy"])
def test_trajectory_model_to_jax(backend, input_trajectory_info_data):
    trajectoryInfo: TrajectoryInfo = input_trajectory_info_data(backend)

    trajectoryModelJax = trajectoryInfo.model.to_jax()

    assert isinstance(trajectoryModelJax.jnt_type, jax.Array)
    assert isinstance(trajectoryModelJax.body_rootid, jax.Array)
    assert isinstance(trajectoryModelJax.body_weldid, jax.Array)
    assert isinstance(trajectoryModelJax.body_mocapid, jax.Array)
    assert isinstance(trajectoryModelJax.body_pos, jax.Array)
    assert isinstance(trajectoryModelJax.body_quat, jax.Array)
    assert isinstance(trajectoryModelJax.body_ipos, jax.Array)
    assert isinstance(trajectoryModelJax.body_iquat, jax.Array)
    assert isinstance(trajectoryModelJax.site_bodyid, jax.Array)
    assert isinstance(trajectoryModelJax.site_pos, jax.Array)
    assert isinstance(trajectoryModelJax.site_quat, jax.Array)


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_trajectory_data__eq__(input_trajectory_data, input_trajectory_data_2, backend):
    trajectory_data: TrajectoryData = input_trajectory_data(backend)
    trajectory_data_2: TrajectoryData = input_trajectory_data(backend)
    trajectory_data_3: TrajectoryData = input_trajectory_data_2(backend)

    assert (
        trajectory_data.__eq__(trajectory_data_2) == True
    ), "These objects are not equal!"
    assert trajectory_data.__eq__(trajectory_data_3) == False


@pytest.mark.parametrize("backend", ["jax", "numpy"])
@pytest.mark.parametrize(
    "traj_index, sub_traj_index", [(0, 0), (0, 1), (0, 999), (1, 0), (1, 999)]
)
def test_trajectory_data_get(
    backend, traj_index, sub_traj_index, input_trajectory_data_2
):
    trajectory_data: TrajectoryData = input_trajectory_data_2(backend)

    backend_type = jnp if backend == "jax" else np

    data = trajectory_data.get(traj_index, sub_traj_index, backend_type)
    start_idx = trajectory_data.split_points[traj_index]
    ind = start_idx + sub_traj_index

    expected_qpos = trajectory_data.qpos[ind]
    expected_qvel = trajectory_data.qvel[ind]
    expected_xpos = (
        trajectory_data.xpos[ind]
        if trajectory_data.xpos.size > 0
        else backend_type.empty(0)
    )
    expected_xquat = (
        trajectory_data.xquat[ind]
        if trajectory_data.xquat.size > 0
        else backend_type.empty(0)
    )
    expected_cvel = (
        trajectory_data.cvel[ind]
        if trajectory_data.cvel.size > 0
        else backend_type.empty(0)
    )
    expected_subtree_com = (
        trajectory_data.subtree_com[ind]
        if trajectory_data.subtree_com.size > 0
        else backend_type.empty(0)
    )
    expected_site_xpos = (
        trajectory_data.site_xpos[ind]
        if trajectory_data.site_xpos.size > 0
        else backend_type.empty(0)
    )
    expected_site_xmat = (
        trajectory_data.site_xmat[ind]
        if trajectory_data.site_xmat.size > 0
        else backend_type.empty(0)
    )

    np.testing.assert_allclose(data.qpos, expected_qpos, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL)
    np.testing.assert_allclose(data.qvel, expected_qvel, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL)
    np.testing.assert_allclose(data.xpos, expected_xpos, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL)
    np.testing.assert_allclose(data.xquat, expected_xquat, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL)
    np.testing.assert_allclose(data.cvel, expected_cvel, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL)
    np.testing.assert_allclose(data.subtree_com, expected_subtree_com, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL)
    np.testing.assert_allclose(data.site_xpos, expected_site_xpos, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL)
    np.testing.assert_allclose(data.site_xmat, expected_site_xmat, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL)


@pytest.mark.parametrize("backend", ["jax", "numpy"])
@pytest.mark.parametrize(
    "traj_index, sub_traj_start_index, slice_length", [(0, 0, 2), (1, 0, 1000)]
)
def test_dynamic_slice_in_dim(
    backend, traj_index, sub_traj_start_index, slice_length, input_trajectory_data_2
):

    trajectory_data: TrajectoryData = input_trajectory_data_2(backend)

    backend_type = jnp if backend == "jax" else np

    sliced_data = TrajectoryData.dynamic_slice_in_dim(
        trajectory_data,
        traj_index,
        sub_traj_start_index,
        slice_length,
        backend_type,
    )

    expected_qpos = backend_type.squeeze(
        trajectory_data.qpos[sub_traj_start_index : sub_traj_start_index + slice_length]
    )
    np.testing.assert_allclose(sliced_data.qpos, expected_qpos, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL)

    expected_qvel = backend_type.squeeze(
        trajectory_data.qvel[sub_traj_start_index : sub_traj_start_index + slice_length]
    )
    np.testing.assert_allclose(sliced_data.qvel, expected_qvel, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL)

    if trajectory_data.xpos.size > 0:
        expected_xpos = backend_type.squeeze(
            trajectory_data.xpos[
                sub_traj_start_index : sub_traj_start_index + slice_length
            ]
        )
        np.testing.assert_allclose(sliced_data.xpos, expected_xpos, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL)

    if trajectory_data.xquat.size > 0:
        expected_xquat = backend_type.squeeze(
            trajectory_data.xquat[
                sub_traj_start_index : sub_traj_start_index + slice_length
            ]
        )
        np.testing.assert_allclose(sliced_data.xquat, expected_xquat, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL)

    if trajectory_data.cvel.size > 0:
        expected_cvel = backend_type.squeeze(
            trajectory_data.cvel[
                sub_traj_start_index : sub_traj_start_index + slice_length
            ]
        )
        np.testing.assert_allclose(sliced_data.cvel, expected_cvel, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL)

    if trajectory_data.subtree_com.size > 0:
        expected_subtree_com = backend_type.squeeze(
            trajectory_data.subtree_com[
                sub_traj_start_index : sub_traj_start_index + slice_length
            ]
        )
        np.testing.assert_allclose(
            sliced_data.subtree_com, expected_subtree_com, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL
        )

    if trajectory_data.site_xpos.size > 0:
        expected_site_xpos = backend_type.squeeze(
            trajectory_data.site_xpos[
                sub_traj_start_index : sub_traj_start_index + slice_length
            ]
        )
        np.testing.assert_allclose(sliced_data.site_xpos, expected_site_xpos, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL)

    if trajectory_data.site_xmat.size > 0:
        expected_site_xmat = backend_type.squeeze(
            trajectory_data.site_xmat[
                sub_traj_start_index : sub_traj_start_index + slice_length
            ]
        )
        np.testing.assert_allclose(sliced_data.site_xmat, expected_site_xmat, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL)


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_dynamic_slice_in_dim_compat(backend, input_trajectory_data):
    trajectory_data: TrajectoryData = input_trajectory_data(backend)

    backend_type = jnp if backend == "jax" else np

    start = 2
    length = 2

    sliced_data = TrajectoryData._dynamic_slice_in_dim_compat(
        trajectory_data.qpos, start, length, backend_type
    )

    if backend == "jax":
        expected_slice = lax.dynamic_slice_in_dim(trajectory_data.qpos, start, length)
    else:
        expected_slice = trajectory_data.qpos[start : start + length].copy()

    np.testing.assert_allclose(sliced_data, expected_slice, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL)


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_get_single_attribute(backend, input_trajectory_data):
    trajectory_data: TrajectoryData = input_trajectory_data(backend)

    backend_type = jnp if backend == "jax" else np

    split_points = trajectory_data.split_points
    traj_index = 0
    sub_traj_index = 2

    attribute = trajectory_data.qpos
    result = TrajectoryData._get_single_attribute(
        attribute, split_points, traj_index, sub_traj_index, backend_type
    )

    start_idx = split_points[traj_index] + sub_traj_index
    expected_value = backend_type.squeeze(attribute[start_idx].copy())

    np.testing.assert_allclose(result, expected_value)


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_dynamic_slice_in_dim_single(backend, input_trajectory_data):
    trajectory_data: TrajectoryData = input_trajectory_data(backend)

    backend_type = jnp if backend == "jax" else np

    attribute = trajectory_data.site_xpos
    split_points = trajectory_data.split_points
    traj_index = 0
    sub_traj_index = 0
    slice_length = 5

    slice = trajectory_data._dynamic_slice_in_dim_single(
        attribute, split_points, traj_index, sub_traj_index, slice_length, backend_type
    )

    start_idx = split_points[traj_index]
    slice_start = start_idx + sub_traj_index
    if backend == "jax":
        expected_slice = lax.dynamic_slice_in_dim(
            trajectory_data.site_xpos, slice_start, slice_length
        )
    else:
        expected_slice = trajectory_data.site_xpos[
            slice_start : slice_start + slice_length
        ].copy()

    np.testing.assert_allclose(slice, expected_slice, atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL)


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_trajectory_data_add_joint(backend, input_trajectory_data):
    trajectory_data: TrajectoryData = input_trajectory_data(backend)

    backend_type = jnp if backend == "jax" else np

    original_qpos_shape = trajectory_data.qpos.shape
    original_qvel_shape = trajectory_data.qvel.shape

    new_qpos_value = 1.0
    new_qvel_value = 2.0
    updated_trajectory_data = trajectory_data.add_joint(
        new_qpos_value, new_qvel_value, backend_type
    )

    updated_qpos_shape = updated_trajectory_data.qpos.shape
    updated_qvel_shape = updated_trajectory_data.qvel.shape

    assert updated_qpos_shape == (original_qpos_shape[0], original_qpos_shape[1] + 1)
    assert updated_qvel_shape == (original_qvel_shape[0], original_qvel_shape[1] + 1)

    # Check that the last column contains the new values
    np.testing.assert_allclose(
        updated_trajectory_data.qpos[:, -1],
        backend_type.full((original_qpos_shape[0],), new_qpos_value),
        atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL,
    )
    np.testing.assert_allclose(
        updated_trajectory_data.qvel[:, -1],
        backend_type.full((original_qvel_shape[0],), new_qvel_value),
    )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_trajectory_data_add_body(backend, input_trajectory_data):
    trajectory_data: TrajectoryData = input_trajectory_data(backend)

    backend_type = jnp if backend == "jax" else np

    original_xpos_shape = trajectory_data.xpos.shape
    original_xquat_shape = trajectory_data.xquat.shape
    original_cvel_shape = trajectory_data.cvel.shape
    original_subtree_com_shape = trajectory_data.subtree_com.shape

    new_xpos_value = 1.0
    new_cvel_value = 2.0
    new_subtree_com_value = 3.0

    updated_trajectory_data: TrajectoryData = trajectory_data.add_body(
        new_xpos_value, new_cvel_value, new_subtree_com_value, backend_type
    )

    updated_xpos_shape = updated_trajectory_data.xpos.shape
    updated_xquat_shape = updated_trajectory_data.xquat.shape
    updated_cvel_shape = updated_trajectory_data.cvel.shape
    updated_subtree_com_shape = updated_trajectory_data.subtree_com.shape

    assert updated_xpos_shape == (
        original_xpos_shape[0],
        original_xpos_shape[1] + 1,
        original_xpos_shape[2],
    )
    assert updated_xquat_shape == (
        original_xquat_shape[0],
        original_xquat_shape[1] + 1,
        original_xquat_shape[2],
    )
    assert updated_cvel_shape == (
        original_cvel_shape[0],
        original_cvel_shape[1] + 1,
        original_cvel_shape[2],
    )
    assert updated_subtree_com_shape == (
        original_subtree_com_shape[0],
        original_subtree_com_shape[1] + 1,
        original_subtree_com_shape[2],
    )

    # Check values for the added body
    np.testing.assert_allclose(
        updated_trajectory_data.xpos[:, -1, :],
        backend_type.full((original_xpos_shape[0], 3), new_xpos_value),
        atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL,
    )
    np.testing.assert_allclose(
        updated_trajectory_data.cvel[:, -1, :],
        backend_type.full((original_cvel_shape[0], 6), new_cvel_value),
        atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL,
    )
    np.testing.assert_allclose(
        updated_trajectory_data.subtree_com[:, -1, :],
        backend_type.full((original_subtree_com_shape[0], 3), new_subtree_com_value),
        atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL,
    )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_trajectory_data_add_site(backend, input_trajectory_data):
    trajectory_data: TrajectoryData = input_trajectory_data(backend)

    backend_type = jnp if backend == "jax" else np

    original_site_xpos_shape = trajectory_data.site_xpos.shape
    original_site_xmat_shape = trajectory_data.site_xmat.shape

    new_site_xpos_value = 1.0
    updated_trajectory_data: TrajectoryData = trajectory_data.add_site(
        new_site_xpos_value, backend_type
    )

    updated_site_xpos_shape = updated_trajectory_data.site_xpos.shape
    updated_site_xmat_shape = updated_trajectory_data.site_xmat.shape

    assert updated_site_xpos_shape == (
        original_site_xpos_shape[0],
        original_site_xpos_shape[1] + 1,
        original_site_xpos_shape[2],
    )

    assert updated_site_xmat_shape == (
        original_site_xmat_shape[0],
        original_site_xmat_shape[1] + 1,
        original_site_xmat_shape[2],
    )

    np.testing.assert_allclose(
        updated_trajectory_data.site_xpos[:, -1, :],
        backend_type.full((original_site_xpos_shape[0], 3), new_site_xpos_value),
        atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL,
    )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_trajectory_data_remove_joint(backend, input_trajectory_data):
    trajectory_data: TrajectoryData = input_trajectory_data(backend)

    backend_type = jnp if backend == "jax" else np

    original_qpos_shape = trajectory_data.qpos.shape
    original_qvel_shape = trajectory_data.qvel.shape

    if original_qpos_shape[1] < 1 or original_qvel_shape[1] < 1:
        pytest.skip("Not enough joints to remove")

    joint_qpos_ids = backend_type.array([original_qpos_shape[1] - 1])
    joint_qvel_ids = backend_type.array([original_qvel_shape[1] - 1])

    updated_trajectory_data = trajectory_data.remove_joints(
        joint_qpos_ids, joint_qvel_ids, backend_type
    )

    updated_qpos_shape = updated_trajectory_data.qpos.shape
    updated_qvel_shape = updated_trajectory_data.qvel.shape

    assert updated_qpos_shape == (original_qpos_shape[0], original_qpos_shape[1] - 1)
    assert updated_qvel_shape == (original_qvel_shape[0], original_qvel_shape[1] - 1)

    if original_qpos_shape[1] < 2 or original_qvel_shape[1] < 2:
        pytest.skip("Not enough joints to remove")
    # Case 2: Remove two joints (e.g., the last two joints)
    joint_qpos_ids = backend_type.array(
        [original_qpos_shape[1] - 2, original_qpos_shape[1] - 1]
    )
    joint_qvel_ids = backend_type.array(
        [original_qvel_shape[1] - 2, original_qvel_shape[1] - 1]
    )

    updated_trajectory_data = trajectory_data.remove_joints(
        joint_qpos_ids, joint_qvel_ids, backend_type
    )

    updated_qpos_shape = updated_trajectory_data.qpos.shape
    updated_qvel_shape = updated_trajectory_data.qvel.shape

    assert updated_qpos_shape == (original_qpos_shape[0], original_qpos_shape[1] - 2)
    assert updated_qvel_shape == (original_qvel_shape[0], original_qvel_shape[1] - 2)


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_trajectory_data_remove_body(backend, input_trajectory_data):
    trajectory_data: TrajectoryData = input_trajectory_data(backend)

    backend_type = jnp if backend == "jax" else np

    original_xpos_shape = trajectory_data.xpos.shape
    original_xquat_shape = trajectory_data.xquat.shape
    original_cvel_shape = trajectory_data.cvel.shape
    original_subtree_com_shape = trajectory_data.subtree_com.shape

    body_ids = backend_type.array([original_xpos_shape[1] - 1])

    updated_trajectory_data = trajectory_data.remove_bodies(body_ids, backend_type)

    updated_xpos_shape = updated_trajectory_data.xpos.shape
    updated_xquat_shape = updated_trajectory_data.xquat.shape
    updated_cvel_shape = updated_trajectory_data.cvel.shape
    updated_subtree_com_shape = updated_trajectory_data.subtree_com.shape

    assert updated_xpos_shape == (
        original_xpos_shape[0],
        original_xpos_shape[1] - 1,
        original_xpos_shape[2],
    )
    assert updated_xquat_shape == (
        original_xquat_shape[0],
        original_xquat_shape[1] - 1,
        original_xquat_shape[2],
    )
    assert updated_cvel_shape == (
        original_cvel_shape[0],
        original_cvel_shape[1] - 1,
        original_cvel_shape[2],
    )
    assert updated_subtree_com_shape == (
        original_subtree_com_shape[0],
        original_subtree_com_shape[1] - 1,
        original_subtree_com_shape[2],
    )

    # Case 2: Remove two bodies (e.g., the last two bodies)
    if (
        original_xpos_shape[1] < 2
        or original_xquat_shape[1] < 2
        or original_cvel_shape[1] < 2
        or original_subtree_com_shape[1] < 2
    ):
        pytest.skip("Not enough bodies to remove")

    body_ids = backend_type.array(
        [original_xpos_shape[1] - 2, original_xpos_shape[1] - 1]
    )

    updated_trajectory_data = trajectory_data.remove_bodies(body_ids, backend_type)

    updated_xpos_shape = updated_trajectory_data.xpos.shape
    updated_xquat_shape = updated_trajectory_data.xquat.shape
    updated_cvel_shape = updated_trajectory_data.cvel.shape
    updated_subtree_com_shape = updated_trajectory_data.subtree_com.shape

    assert updated_xpos_shape == (
        original_xpos_shape[0],
        original_xpos_shape[1] - 2,
        original_xpos_shape[2],
    )
    assert updated_xquat_shape == (
        original_xquat_shape[0],
        original_xquat_shape[1] - 2,
        original_xquat_shape[2],
    )
    assert updated_cvel_shape == (
        original_cvel_shape[0],
        original_cvel_shape[1] - 2,
        original_cvel_shape[2],
    )
    assert updated_subtree_com_shape == (
        original_subtree_com_shape[0],
        original_subtree_com_shape[1] - 2,
        original_subtree_com_shape[2],
    )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_trajectory_data_remove_site(backend, input_trajectory_data):
    trajectory_data: TrajectoryData = input_trajectory_data(backend)

    backend_type = jnp if backend == "jax" else np

    original_site_xpos_shape = trajectory_data.site_xpos.shape
    original_site_xmat_shape = trajectory_data.site_xmat.shape

    if original_site_xpos_shape[1] < 1 or original_site_xmat_shape[1] < 1:
        pytest.skip("Not enough sites to remove")

    # Case 1: Remove the last site
    site_ids = backend_type.array([original_site_xpos_shape[1] - 1])

    updated_trajectory_data = trajectory_data.remove_sites(site_ids, backend_type)

    updated_site_xpos_shape = updated_trajectory_data.site_xpos.shape
    updated_site_xmat_shape = updated_trajectory_data.site_xmat.shape

    assert updated_site_xpos_shape == (
        original_site_xpos_shape[0],
        original_site_xpos_shape[1] - 1,
        original_site_xpos_shape[2],
    )
    assert updated_site_xmat_shape == (
        original_site_xmat_shape[0],
        original_site_xmat_shape[1] - 1,
        original_site_xmat_shape[2],
    )

    # Case 2: Remove two sites
    if original_site_xpos_shape[1] < 2 or original_site_xmat_shape[1] < 2:
        pytest.skip("Not enough sites to remove")

    site_ids = backend_type.array(
        [original_site_xpos_shape[1] - 2, original_site_xpos_shape[1] - 1]
    )

    updated_trajectory_data = trajectory_data.remove_sites(site_ids, backend_type)

    updated_site_xpos_shape = updated_trajectory_data.site_xpos.shape
    updated_site_xmat_shape = updated_trajectory_data.site_xmat.shape

    assert updated_site_xpos_shape == (
        original_site_xpos_shape[0],
        original_site_xpos_shape[1] - 2,
        original_site_xpos_shape[2],
    )
    assert updated_site_xmat_shape == (
        original_site_xmat_shape[0],
        original_site_xmat_shape[1] - 2,
        original_site_xmat_shape[2],
    )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_trajectory_data_reorder_joints(backend, input_trajectory_data):
    trajectory_data: TrajectoryData = input_trajectory_data(backend)

    backend_type = jnp if backend == "jax" else np

    original_qpos_shape = trajectory_data.qpos.shape
    original_qvel_shape = trajectory_data.qvel.shape

    new_order_qpos = backend_type.array(list(reversed(range(original_qpos_shape[1]))))
    new_order_qvel = backend_type.array(list(reversed(range(original_qvel_shape[1]))))

    updated_trajectory_data: TrajectoryData = trajectory_data.reorder_joints(
        new_order_qpos, new_order_qvel
    )

    qpos_ind = np.arange(original_qpos_shape[1])
    qvel_ind = np.arange(original_qvel_shape[1])
    np.testing.assert_allclose(
        updated_trajectory_data.qpos[:, qpos_ind],
        trajectory_data.qpos[:, new_order_qpos[qpos_ind]],
        atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL,
    )
    np.testing.assert_allclose(
        updated_trajectory_data.qvel[:, qvel_ind],
        trajectory_data.qvel[:, new_order_qvel[qvel_ind]],
        atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL,
    )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_trajectory_data_reorder_bodies(backend, input_trajectory_data):
    trajectory_data: TrajectoryData = input_trajectory_data(backend)
    backend_type = jnp if backend == "jax" else np

    original_xpos_shape = trajectory_data.xpos.shape

    new_order = backend_type.array(list(reversed(range(original_xpos_shape[1]))))
    updated_trajectory_data = trajectory_data.reorder_bodies(new_order)

    body_ind = np.arange(original_xpos_shape[1])
    np.testing.assert_allclose(
        updated_trajectory_data.xpos[:, body_ind],
        trajectory_data.xpos[:, new_order[body_ind]],
        atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL,
    )
    np.testing.assert_allclose(
        updated_trajectory_data.xquat[:, body_ind],
        trajectory_data.xquat[:, new_order[body_ind]],
        atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL,
    )
    np.testing.assert_allclose(
        updated_trajectory_data.cvel[:, body_ind],
        trajectory_data.cvel[:, new_order[body_ind]],
        atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL,
    )
    np.testing.assert_allclose(
        updated_trajectory_data.subtree_com[:, body_ind],
        trajectory_data.subtree_com[:, new_order[body_ind]],
        atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL,
    )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_trajectory_data_reorder_sites(backend, input_trajectory_data):
    trajectory_data: TrajectoryData = input_trajectory_data(backend)

    backend_type = jnp if backend == "jax" else np

    original_site_xpos_shape = trajectory_data.site_xpos.shape

    new_order = backend_type.array(list(reversed(range(original_site_xpos_shape[1]))))

    updated_trajectory_data = trajectory_data.reorder_sites(new_order)

    site_ind = np.arange(original_site_xpos_shape[1])
    np.testing.assert_allclose(
        updated_trajectory_data.site_xpos[:, site_ind],
        trajectory_data.site_xpos[:, new_order[site_ind]],
        atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL,
    )

    np.testing.assert_allclose(
        updated_trajectory_data.site_xmat[:, site_ind],
        trajectory_data.site_xmat[:, new_order[site_ind]],
        atol=_TRAJ_ATOL, rtol=_TRAJ_RTOL,
    )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_trajectory_data_concatenate(
    backend, input_trajectory_data, input_trajectory_info_data
):
    trajectory_data_factory = input_trajectory_data(backend)
    trajectory_info_factory = input_trajectory_info_data(backend)

    backend_type = jnp if backend == "jax" else np

    traj_data1 = trajectory_data_factory
    traj_data2 = trajectory_data_factory
    traj_info1 = trajectory_info_factory
    traj_info2 = trajectory_info_factory

    concatenated_traj_data, concatenated_traj_info = TrajectoryData.concatenate(
        [traj_data1, traj_data2], [traj_info1, traj_info2], backend_type
    )

    assert concatenated_traj_data.qpos.shape == (
        traj_data1.qpos.shape[0] + traj_data2.qpos.shape[0],
        traj_data1.qpos.shape[1],
    )
    assert concatenated_traj_data.qvel.shape == (
        traj_data1.qvel.shape[0] + traj_data2.qvel.shape[0],
        traj_data1.qvel.shape[1],
    )
    assert concatenated_traj_data.xpos.shape == (
        traj_data1.xpos.shape[0] + traj_data2.xpos.shape[0],
        traj_data1.xpos.shape[1],
        traj_data1.xpos.shape[2],
    )
    assert concatenated_traj_data.xquat.shape == (
        traj_data1.xquat.shape[0] + traj_data2.xquat.shape[0],
        traj_data1.xquat.shape[1],
        traj_data1.xquat.shape[2],
    )
    assert concatenated_traj_data.cvel.shape == (
        traj_data1.cvel.shape[0] + traj_data2.cvel.shape[0],
        traj_data1.cvel.shape[1],
        traj_data1.cvel.shape[2],
    )
    assert concatenated_traj_data.subtree_com.shape == (
        traj_data1.subtree_com.shape[0] + traj_data2.subtree_com.shape[0],
        traj_data1.subtree_com.shape[1],
        traj_data1.subtree_com.shape[2],
    )
    assert concatenated_traj_data.site_xpos.shape == (
        traj_data1.site_xpos.shape[0] + traj_data2.site_xpos.shape[0],
        traj_data1.site_xpos.shape[1],
        traj_data1.site_xpos.shape[2],
    )
    assert concatenated_traj_data.site_xmat.shape == (
        traj_data1.site_xmat.shape[0] + traj_data2.site_xmat.shape[0],
        traj_data1.site_xmat.shape[1],
        traj_data1.site_xmat.shape[2],
    )

    expected_split_points = backend_type.array(
        [
            0,
            traj_data1.qpos.shape[0],
            traj_data1.qpos.shape[0] + traj_data2.qpos.shape[0],
        ]
    )

    assert backend_type.allclose(
        concatenated_traj_data.split_points, expected_split_points
    )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_trajectory_data_len_trajectory(backend, input_trajectory_data):
    trajectory_data: TrajectoryData = input_trajectory_data(backend)

    traj_len_0 = trajectory_data.len_trajectory(0)
    assert traj_len_0 == 1000, f"Expected 1000, got {traj_len_0}"


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_trajectory_data_n_samples(backend, input_trajectory_data):
    trajectory_data: TrajectoryData = input_trajectory_data(backend)

    n_samples = trajectory_data.n_samples

    assert n_samples == 1000, f"Expected 1000, got {n_samples}"


def test_trajectory_data_get_attribute_names():
    attribute_names = TrajectoryData.get_attribute_names()

    expected_attributes = [
        "qpos",
        "qvel",
        "xpos",
        "xquat",
        "cvel",
        "subtree_com",
        "site_xpos",
        "site_xmat",
        "split_points",
    ]

    assert set(attribute_names) == set(
        expected_attributes
    ), f"Attributes mismatch. Got {attribute_names}"


@pytest.mark.parametrize("backend", ["jax"])
def test_trajectory_data_to_numpy(backend, input_trajectory_data):
    trajectory_data: TrajectoryData = input_trajectory_data(backend)

    numpy_trajectory_data = trajectory_data.to_numpy()

    # Verify all fields are numpy arrays
    for field_name in TrajectoryData.get_attribute_names():
        value = getattr(numpy_trajectory_data, field_name)
        assert isinstance(value, np.ndarray), f"{field_name} is not a numpy array"


@pytest.mark.parametrize("backend", ["numpy"])
def test_trajectory_data_to_jax(backend, input_trajectory_data):
    trajectory_data: TrajectoryData = input_trajectory_data(backend)

    jax_trajectory_data = trajectory_data.to_jax()

    for field_name in TrajectoryData.get_attribute_names():
        value = getattr(jax_trajectory_data, field_name)
        assert isinstance(value, jax.Array), f"{field_name} is not a JAX array"


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_interpolate_trajectories_basic(
    input_trajectory_data, input_trajectory_info_data, backend
):
    traj_data: TrajectoryData = input_trajectory_data(backend)
    traj_info: TrajectoryInfo = input_trajectory_info_data(backend)

    new_frequency = traj_info.frequency * 2

    new_traj_data, new_traj_info = interpolate_trajectories(
        traj_data, traj_info, new_frequency
    )

    assert new_traj_info.frequency == new_frequency

    scaling_factor = new_frequency / traj_info.frequency
    expected_n_samples = round(traj_data.n_samples * scaling_factor)
    assert new_traj_data.n_samples == expected_n_samples

    assert new_traj_data.qpos.shape[0] == expected_n_samples
    assert new_traj_data.qvel.shape[0] == expected_n_samples
    assert new_traj_data.xquat.shape[0] == expected_n_samples


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_interpolate_trajectories_quaternions(
    input_trajectory_data, input_trajectory_info_data, backend
):
    traj_data: TrajectoryData = input_trajectory_data(backend)
    traj_info: TrajectoryInfo = input_trajectory_info_data(backend)

    backend_type = jnp if backend == "jax" else np

    new_frequency = traj_info.frequency * 2

    new_traj_data, _ = interpolate_trajectories(
        traj_data, traj_info, new_frequency, backend_type
    )

    norms = backend_type.linalg.norm(new_traj_data.xquat, axis=-1)
    np.testing.assert_allclose(
        norms, 1.0, atol=1e-6
    ), "Interpolated quaternions are not normalized"


@pytest.mark.parametrize("backend", ["numpy"])
def test_trajectory_transitions_to_jnp(backend, input_trajectory_transitions):
    trajectory_transitions: TrajectoryTransitions = input_trajectory_transitions(
        backend
    )

    jnp_transitions = trajectory_transitions.to_jnp()

    assert isinstance(jnp_transitions.observations, jax.Array)
    assert isinstance(jnp_transitions.next_observations, jax.Array)
    assert isinstance(jnp_transitions.absorbings, jax.Array)
    assert isinstance(jnp_transitions.dones, jax.Array)
    assert isinstance(jnp_transitions.actions, jax.Array)
    assert isinstance(jnp_transitions.rewards, jax.Array)


@pytest.mark.parametrize("backend", ["jax"])
def test_trajectory_transitions_to_np(backend, input_trajectory_transitions):
    trajectory_transitions: TrajectoryTransitions = input_trajectory_transitions(
        backend
    )

    np_transitions = trajectory_transitions.to_np()

    assert isinstance(np_transitions.observations, np.ndarray)
    assert isinstance(np_transitions.next_observations, np.ndarray)
    assert isinstance(np_transitions.absorbings, np.ndarray)
    assert isinstance(np_transitions.dones, np.ndarray)
    assert isinstance(np_transitions.actions, np.ndarray)
    assert isinstance(np_transitions.rewards, np.ndarray)


def test_trajectory_transitions_get_attribute_names():
    attribute_names = TrajectoryTransitions.get_attribute_names()
    expected_names = [
        "observations",
        "next_observations",
        "absorbings",
        "dones",
        "actions",
        "rewards",
    ]
    assert attribute_names == expected_names
