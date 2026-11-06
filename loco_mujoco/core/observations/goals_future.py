from types import ModuleType
from typing import Any

import numpy as np
import mujoco

from loco_mujoco.core.observations.goals import Goal


def _quat_to_mat(q, backend):
    """Rotation matrix of a scalar-first quaternion."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return backend.stack([
        backend.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
        backend.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
        backend.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])])


def _heading_mat(R, backend):
    """Rotation about z by the heading of R."""
    n = backend.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2 + 1e-12)
    c, s = R[0, 0] / n, R[1, 0] / n
    zero, one = backend.zeros_like(c), backend.ones_like(c)
    return backend.stack([backend.stack([c, -s, zero]), backend.stack([s, c, zero]), backend.stack([zero, zero, one])])


def _rot6d(R, backend):
    return backend.ravel(R[:, :2])


class _FreeRootGoal(Goal):
    """Goal of a robot whose first joint is the free root joint, read from the trajectory's qpos and qvel."""

    def __init__(self, **kwargs):
        kwargs.pop("obs_name", None)
        kwargs.setdefault("visualize_goal", False)
        kwargs.setdefault("allow_randomization", False)
        super().__init__({}, n_visual_geoms=0, **kwargs)
        self._dim = None

    def _obs_dim(self, model):
        raise NotImplementedError

    def _init_from_mj(self, env, model, data, current_obs_size):
        assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE, f"{type(self).__name__} needs a free root joint."
        self._dim = self._obs_dim(model)
        self.min = [-np.inf] * self._dim
        self.max = [np.inf] * self._dim
        self.data_type_ind = np.array(list(range(data.userdata.size)))
        self.obs_ind = np.array(list(range(current_obs_size, current_obs_size + self._dim)))
        self._initialized_from_mj = True

    def init_from_traj(self, traj_handler: Any):
        assert traj_handler is not None, f"{type(self).__name__} requires a trajectory, but none was loaded."
        self._initialized_from_traj = True

    @property
    def requires_trajectory(self) -> bool:
        return True

    @property
    def has_visual(self) -> bool:
        return False

    @property
    def dim(self) -> int:
        return self._dim


class TrajFutureFrames(_FreeRootGoal):
    """
    The next ``n_future`` reference frames, from the reference only: per frame the root position and orientation
    in the heading frame of the current reference frame, the root velocities in the root frame, and the joint
    positions and scaled joint velocities. Frames past the end of the clip repeat its last frame.

    Args:
        n_future (int): Number of future frames.
        joint_vel_scale (float): Scale of the joint velocities.

    """

    def __init__(self, n_future: int = 10, joint_vel_scale: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self._n_future = int(n_future)
        self._joint_vel_scale = float(joint_vel_scale)

    def _obs_dim(self, model):
        return self._n_future * (15 + 2 * (model.nv - 6))

    def get_obs_and_update_state(self, env: Any, model: Any, data: Any, carry: Any, backend: ModuleType,
                                 traj: Any = None):
        ts, td = carry.traj_state, traj.data
        start = td.split_points[ts.traj_no]
        last = td.split_points[ts.traj_no + 1] - start - 1
        # all frames in one gather, this runs in every imagined step
        steps = backend.minimum(ts.subtraj_step_no + backend.arange(self._n_future + 1), last)
        qpos, qvel = td.qpos[start + steps], td.qvel[start + steps]
        ref_p, fut_p, fut_v = qpos[0, :3], qpos[1:], qvel[1:]
        heading_inv = _heading_mat(_quat_to_mat(qpos[0, 3:7], backend), backend).T
        R = _quat_to_mat(fut_p[:, 3:7].T, backend)  # (3, 3, K)
        R = backend.moveaxis(R, -1, 0)
        blocks = [(fut_p[:, :3] - ref_p) @ heading_inv.T,
                  (heading_inv @ R)[:, :, :2].reshape(self._n_future, 6),
                  (backend.swapaxes(R, 1, 2) @ fut_v[:, :3, None])[..., 0], fut_v[:, 3:6],
                  fut_p[:, 7:], self._joint_vel_scale * fut_v[:, 6:]]
        return backend.concatenate(blocks, axis=1).reshape(-1), carry


class RootOriRelTraj(_FreeRootGoal):
    """Orientation of the current reference root in the frame of the robot's root (6D rotation)."""

    def _obs_dim(self, model):
        return 6

    def get_obs_and_update_state(self, env: Any, model: Any, data: Any, carry: Any, backend: ModuleType,
                                 traj: Any = None):
        ref = traj.data.get(carry.traj_state.traj_no, carry.traj_state.subtraj_step_no, backend)
        R, R_ref = _quat_to_mat(data.qpos[3:7], backend), _quat_to_mat(ref.qpos[3:7], backend)
        return _rot6d(R.T @ R_ref, backend), carry


class TrackingError(_FreeRootGoal):
    """
    Robot-to-reference errors for a critic: joint positions, body positions with the reference moved to the robot's
    xy and heading, in the robot's heading frame, and the root position in the same frame.

    Args:
        body_names (list): Tracked bodies. Defaults to the env's ``tracked_body_names``.

    """

    def __init__(self, body_names=None, **kwargs):
        super().__init__(**kwargs)
        self._body_names = body_names
        self._body_ids = None

    def _obs_dim(self, model):
        return model.nv - 6 + 3 * len(self._body_ids) + 3

    def _init_from_mj(self, env, model, data, current_obs_size):
        names = self._body_names or env._get_all_info_properties()["tracked_body_names"]
        self._body_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in names])
        super()._init_from_mj(env, model, data, current_obs_size)

    def get_obs_and_update_state(self, env: Any, model: Any, data: Any, carry: Any, backend: ModuleType,
                                 traj: Any = None):
        ref = traj.data.get(carry.traj_state.traj_no, carry.traj_state.subtraj_step_no, backend)
        p, p_ref = data.qpos[:3], ref.qpos[:3]
        R, R_ref = _quat_to_mat(data.qpos[3:7], backend), _quat_to_mat(ref.qpos[3:7], backend)
        heading_inv = _heading_mat(R, backend).T
        yaw = _heading_mat(R @ R_ref.T, backend)
        anchor = backend.concatenate([p[:2], p_ref[2:3]])
        ref_bodies = anchor + (ref.xpos[self._body_ids] - p_ref) @ yaw.T
        body_error = (ref_bodies - data.xpos[self._body_ids]) @ heading_inv.T
        return backend.concatenate([ref.qpos[7:] - data.qpos[7:], backend.ravel(body_error),
                                    heading_inv @ (p_ref - p)]), carry
