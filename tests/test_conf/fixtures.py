import pytest
import jax
import jax.numpy as jnp
import numpy as np
import mujoco
from mujoco import MjModel, MjData
from pathlib import Path
from omegaconf import OmegaConf

import jax.random as jr
import numpy.random as nr

from loco_mujoco.core.trajectory import (
    Trajectory,
    TrajectoryInfo,
    TrajectoryModel,
    TrajectoryData,
    TrajectoryTransitions,
)

from .dummy_humanoid_env import DummyHumamoidEnv

DEFAULTS = {"horizon": 1000, "gamma": 0.99, "n_envs": 1}


@pytest.fixture
def input_trajectory_info_data() -> TrajectoryInfo:
    def factory(backend):
        backend_type = jnp if backend == "jax" else np
        model_path = (Path(__file__).resolve().parent / "humanoid_test.xml").as_posix()
        mujoco_model = MjModel.from_xml_path(model_path)

        njnt = mujoco_model.njnt
        nbody = mujoco_model.nbody
        nsite = mujoco_model.nsite
        frequency = 100.0

        joint_names = [
            mujoco.mj_id2name(mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(njnt)
        ]
        jnt_type = backend_type.array(mujoco_model.jnt_type)

        body_names = [
            mujoco.mj_id2name(mujoco_model, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(nbody)
        ]
        body_pos = backend_type.array(mujoco_model.body_pos)
        body_quat = backend_type.array(mujoco_model.body_quat)
        body_ipos = backend_type.array(mujoco_model.body_ipos)
        body_iquat = backend_type.array(mujoco_model.body_iquat)
        body_rootid = backend_type.array(mujoco_model.body_rootid)
        body_weldid = backend_type.array(mujoco_model.body_weldid)
        body_mocapid = backend_type.array(mujoco_model.body_mocapid)

        site_names = [
            mujoco.mj_id2name(mujoco_model, mujoco.mjtObj.mjOBJ_SITE, i)
            for i in range(nsite)
        ]
        site_bodyid = backend_type.array(mujoco_model.site_bodyid)
        site_pos = backend_type.array(mujoco_model.site_pos)
        site_quat = backend_type.array(mujoco_model.site_quat)

        metadata = {
            "robot_name": "minimal bipedal robot",
            "description": "Test trajectory for a bipedal locomotion robot",
        }

        model = TrajectoryModel(
            njnt,
            jnt_type,
            nbody,
            body_rootid,
            body_weldid,
            body_mocapid,
            body_pos,
            body_quat,
            body_ipos,
            body_iquat,
            nsite,
            site_bodyid,
            site_pos,
            site_quat,
        )

        return TrajectoryInfo(
            joint_names, model, frequency, body_names, site_names, metadata
        )

    return factory


@pytest.fixture
def input_expected_joint_name2ind_qpos():
    def factory(backend):
        backend_type = jnp if backend == "jax" else np
        return {
            "root": backend_type.array([0, 1, 2, 3, 4, 5, 6]),
            "abdomen_z": backend_type.array([7]),
            "abdomen_y": backend_type.array([8]),
            "abdomen_x": backend_type.array([9]),
            "right_hip_x": backend_type.array([10]),
            "right_hip_z": backend_type.array([11]),
            "right_hip_y": backend_type.array([12]),
            "right_knee": backend_type.array([13]),
            "left_hip_x": backend_type.array([14]),
            "left_hip_z": backend_type.array([15]),
            "left_hip_y": backend_type.array([16]),
            "left_knee": backend_type.array([17]),
        }

    return factory


@pytest.fixture
def input_body_name2ind():
    def factory(backend):
        backend_type = jnp if backend == "jax" else np
        return {
            "world": backend_type.array([0]),
            "torso": backend_type.array([1]),
            "lwaist": backend_type.array([2]),
            "pelvis": backend_type.array([3]),
            "right_thigh": backend_type.array([4]),
            "right_shin": backend_type.array([5]),
            "right_foot": backend_type.array([6]),
            "left_thigh": backend_type.array([7]),
            "left_shin": backend_type.array([8]),
            "left_foot": backend_type.array([9]),
        }

    return factory


@pytest.fixture
def input_site_name2ind():
    def factory(backend):
        backend_type = jnp if backend == "jax" else np
        return {
            "torso_site": backend_type.array([0]),
            "pelvis_site": backend_type.array([1]),
            "right_thigh_site": backend_type.array([2]),
            "right_foot_site": backend_type.array([3]),
            "left_thigh_site": backend_type.array([4]),
            "left_foot_site": backend_type.array([5]),
        }

    return factory


@pytest.fixture
def input_trajectory_info_field_names():
    return ["joint_names", "model", "frequency", "body_names", "site_names", "metadata"]


@pytest.fixture
def input_trajectory_data() -> TrajectoryData:
    def factory(backend):
        backend_type = jnp if backend == "jax" else np
        model_path = (Path(__file__).resolve().parent / "humanoid_test.xml").as_posix()
        mujoco_model = MjModel.from_xml_path(model_path)

        data = MjData(mujoco_model)
        mujoco.mj_forward(mujoco_model, data)
        N_steps = 1000

        qpos = data.qpos
        qvel = data.qvel
        qpos = backend_type.tile(qpos, (N_steps, 1))
        qvel = backend_type.tile(qvel, (N_steps, 1))

        cvel = data.cvel
        cvel = backend_type.tile(cvel, (N_steps, 1, 1))

        xpos = data.xpos
        xpos = backend_type.tile(xpos, (N_steps, 1, 1))

        xquat = data.xquat
        xquat_norms = backend_type.linalg.norm(xquat, axis=-1)
        xquat = backend_type.where(
            xquat_norms[:, None] == 0, backend_type.array([1, 0, 0, 0]), xquat
        )
        xquat = backend_type.tile(xquat, (N_steps, 1, 1))

        subtree_com = data.subtree_com
        subtree_com = backend_type.tile(subtree_com, (N_steps, 1, 1))

        site_xpos = data.site_xpos
        site_xpos = backend_type.tile(site_xpos, (N_steps, 1, 1))
        site_xmat = data.site_xmat
        site_xmat = backend_type.tile(site_xmat, (N_steps, 1, 1))

        trajectoryData = TrajectoryData(
            qpos,
            qvel,
            xpos,
            xquat,
            cvel,
            subtree_com,
            site_xpos,
            site_xmat,
            split_points=backend_type.array([0, N_steps]),
        )

        return trajectoryData

    return factory


@pytest.fixture
def input_trajectory_data_2() -> TrajectoryData:
    def factory(backend):
        backend_type = jnp if backend == "jax" else np
        model_path = (Path(__file__).resolve().parent / "humanoid_test.xml").as_posix()
        mujoco_model = MjModel.from_xml_path(model_path)

        data = MjData(mujoco_model)
        mujoco.mj_forward(mujoco_model, data)
        N_steps = 1000

        # Generate data for trajectory 1
        qpos_1 = backend_type.tile(data.qpos, (N_steps, 1))
        qvel_1 = backend_type.tile(data.qvel, (N_steps, 1))
        cvel_1 = backend_type.tile(data.cvel, (N_steps, 1, 1))
        xpos_1 = backend_type.tile(data.xpos, (N_steps, 1, 1))
        xquat_1 = backend_type.tile(data.xquat, (N_steps, 1, 1))
        subtree_com_1 = backend_type.tile(data.subtree_com, (N_steps, 1, 1))
        site_xpos_1 = backend_type.tile(data.site_xpos, (N_steps, 1, 1))
        site_xmat_1 = backend_type.tile(data.site_xmat, (N_steps, 1, 1))

        # Generate data for trajectory 2 (same as trajectory 1 for simplicity)
        qpos_2 = qpos_1.copy()
        qvel_2 = qvel_1.copy()
        cvel_2 = cvel_1.copy()
        xpos_2 = xpos_1.copy()
        xquat_2 = xquat_1.copy()
        subtree_com_2 = subtree_com_1.copy()
        site_xpos_2 = site_xpos_1.copy()
        site_xmat_2 = site_xmat_1.copy()

        qpos = backend_type.concatenate([qpos_1, qpos_2], axis=0)
        qvel = backend_type.concatenate([qvel_1, qvel_2], axis=0)
        cvel = backend_type.concatenate([cvel_1, cvel_2], axis=0)
        xpos = backend_type.concatenate([xpos_1, xpos_2], axis=0)
        xquat = backend_type.concatenate([xquat_1, xquat_2], axis=0)
        subtree_com = backend_type.concatenate([subtree_com_1, subtree_com_2], axis=0)
        site_xpos = backend_type.concatenate([site_xpos_1, site_xpos_2], axis=0)
        site_xmat = backend_type.concatenate([site_xmat_1, site_xmat_2], axis=0)

        split_points = backend_type.array([0, N_steps, 2 * N_steps])

        trajectory_data = TrajectoryData(
            qpos=qpos,
            qvel=qvel,
            xpos=xpos,
            xquat=xquat,
            cvel=cvel,
            subtree_com=subtree_com,
            site_xpos=site_xpos,
            site_xmat=site_xmat,
            split_points=split_points,
        )

        return trajectory_data

    return factory


@pytest.fixture
def input_trajectory_transitions() -> TrajectoryTransitions:
    def factory(backend):
        backend_type = jnp if backend == "jax" else np

        observations = backend_type.array([[1.0, 2.0], [3.0, 4.0]])
        next_observations = backend_type.array([[5.0, 6.0], [7.0, 8.0]])
        absorbings = backend_type.array([True, False])
        dones = backend_type.array([False, True])
        actions = backend_type.array([1, 2])
        rewards = backend_type.array([0.5, 1.0])

        return TrajectoryTransitions(
            observations=observations,
            next_observations=next_observations,
            absorbings=absorbings,
            dones=dones,
            actions=actions,
            rewards=rewards,
        )

    return factory


@pytest.fixture
def input_trajectory(input_trajectory_info_data, input_trajectory_data) -> Trajectory:
    def factory(backend):
        # Get the TrajectoryInfo and TrajectoryData using their factory methods
        trajectory_info = input_trajectory_info_data(backend)
        trajectory_data = input_trajectory_data(backend)

        # Combine them into a Trajectory object
        trajectory = Trajectory(
            info=trajectory_info,
            data=trajectory_data,
        )

        return trajectory

    return factory


@pytest.fixture
def standing_trajectory() -> Trajectory:

    N_steps = 1000

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(enable_mjx=False,
                               terminal_state_type="RootPoseTrajTerminalStateHandler",
                               **DEFAULTS)

    # reset the env
    key = jax.random.PRNGKey(0)
    mjx_env.reset(key)

    # get the model and data of the environment
    model = mjx_env.model
    data = mjx_env.data

    # get the initial qpos and qvel of the environment
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()
    xpos = data.xpos.copy()
    xquat = data.xquat.copy()
    cvel = data.cvel.copy()
    subtree_com = data.subtree_com.copy()
    site_xpos = data.site_xpos.copy()
    site_xmat = data.site_xmat.copy()

    # stack qpos and qvel to a trajectory
    qpos = np.tile(qpos, (N_steps, 1))
    qvel = np.tile(qvel, (N_steps, 1))
    xpos = np.tile(xpos, (N_steps, 1, 1))
    xquat = np.tile(xquat, (N_steps, 1, 1))
    cvel = np.tile(cvel, (N_steps, 1, 1))
    subtree_com = np.tile(subtree_com, (N_steps, 1, 1))
    site_xpos = np.tile(site_xpos, (N_steps, 1, 1))
    site_xmat = np.tile(site_xmat, (N_steps, 1, 1))

    # create a trajectory info -- this stores basic information about the trajectory
    njnt = model.njnt
    jnt_type = model.jnt_type.copy()
    jnt_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(njnt)]
    traj_info = TrajectoryInfo(jnt_names, model=TrajectoryModel(njnt, jnp.array(jnt_type)), frequency=1 / mjx_env.dt)

    # create a trajectory data -- this stores the actual trajectory data
    traj_data = TrajectoryData(jnp.array(qpos), jnp.array(qvel), jnp.array(xpos), jnp.array(xquat), jnp.array(cvel),
                               jnp.array(subtree_com), jnp.array(site_xpos), jnp.array(site_xmat),
                               split_points=jnp.array([0, N_steps]))

    # combine them to a trajectory
    traj = Trajectory(traj_info, traj_data)

    return traj


@pytest.fixture
def falling_trajectory() -> Trajectory:

    N_steps = 1000

    # define a simple Mujoco environment
    mjx_env = DummyHumamoidEnv(enable_mjx=False,
                               **DEFAULTS)

    # reset the env
    key = jax.random.PRNGKey(0)
    mjx_env.reset(key)
    action_dim = mjx_env.info.action_space.shape[0]

    qpos = []
    qvel = []
    xpos = []
    xquat = []
    cvel = []
    subtree_com = []
    site_xpos = []
    site_xmat = []
    for i in range(N_steps):
        action = np.zeros(action_dim)
        mjx_env.step(action)
        data = mjx_env.get_data()
        qpos.append(data.qpos)
        qvel.append(data.qvel)
        xpos.append(data.xpos)
        xquat.append(data.xquat)
        cvel.append(data.cvel)
        subtree_com.append(data.subtree_com)
        site_xpos.append(data.site_xpos)
        site_xmat.append(data.site_xmat)

    # get the model and data of the environment
    model = mjx_env.get_model()

    # get the initial qpos and qvel of the environment
    qpos = np.stack(qpos)
    qvel = np.stack(qvel)
    xpos = np.stack(xpos)
    xquat = np.stack(xquat)
    cvel = np.stack(cvel)
    subtree_com = np.stack(subtree_com)
    site_xpos = np.stack(site_xpos)
    site_xmat = np.stack(site_xmat)

    # create a trajectory info -- this stores basic information about the trajectory
    njnt = model.njnt
    jnt_type = model.jnt_type
    jnt_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(njnt)]
    traj_info = TrajectoryInfo(jnt_names, model=TrajectoryModel(njnt, jnp.array(jnt_type)), frequency=1 / mjx_env.dt)

    # create a trajectory data -- this stores the actual trajectory data
    traj_data = TrajectoryData(jnp.array(qpos), jnp.array(qvel), jnp.array(xpos), jnp.array(xquat), jnp.array(cvel),
                               jnp.array(subtree_com), jnp.array(site_xpos), jnp.array(site_xmat),
                               split_points=jnp.array([0, N_steps]))

    # combine them to a trajectory
    traj = Trajectory(traj_info, traj_data)

    return traj


@pytest.fixture
def mock_random(monkeypatch):
    monkeypatch.setattr(
        jr,
        "normal",
        lambda key, shape=(), dtype=jnp.float32: jnp.full(shape, 0.5, dtype=dtype),
    )
    monkeypatch.setattr(
        jr,
        "uniform",
        lambda key, shape=(), dtype=jnp.float32, minval=0.0, maxval=1.0: jnp.full(
            shape, minval + (maxval - minval) * 0.3, dtype=dtype
        ),
    )
    monkeypatch.setattr(
        jr,
        "randint",
        lambda key, shape, minval, maxval, dtype=jnp.int32: jnp.full(
            shape, minval + (maxval - minval) // 2, dtype=dtype  # Middle value
        ),
    )
    monkeypatch.setattr(
        nr,
        "normal",
        lambda loc=0.0, scale=1.0, size=None: np.full(
            size if size is not None else (), loc + 0.5 * scale
        ),
    )
    monkeypatch.setattr(
        nr,
        "uniform",
        lambda low=0.0, high=1.0, size=None: np.full(
            (size if size is not None else ()) if (np.isscalar(low) and np.isscalar(high))
            else np.broadcast(low, high).size,
            np.asarray(low) + (np.asarray(high) - np.asarray(low)) * 0.3,
        ),
    )
    monkeypatch.setattr(
        nr,
        "randint",
        lambda low, high=None, size=None, dtype=int: np.full(
            size if size else (), 
            (low + (high - 1)) // 2 if high is not None else low,  # Middle of range or just 'low'
            dtype=dtype
        ),
    )
    monkeypatch.setattr(
        nr,
        "randn",
        lambda *args: np.full(args if args else (), 0.2),
    )


@pytest.fixture
def ppo_rl_config():
    return OmegaConf.load(Path(__file__).resolve().parent / 'algorithm_confs/ppo.yaml')


@pytest.fixture
def imitation_config():
    return OmegaConf.load(Path(__file__).resolve().parent / 'algorithm_confs/gail_amp.yaml')


@pytest.fixture
def s2pg_ppo_config():
    return OmegaConf.load(Path(__file__).resolve().parent / 'algorithm_confs/s2pg_ppo.yaml')


@pytest.fixture
def bptt_ppo_config():
    return OmegaConf.load(Path(__file__).resolve().parent / 'algorithm_confs/bptt_ppo.yaml')


@pytest.fixture
def history_ppo_config():
    return OmegaConf.load(Path(__file__).resolve().parent / 'algorithm_confs/history_ppo.yaml')
