from typing import Dict, Any, Union, Tuple
from types import ModuleType

import jax.numpy as jnp
import numpy as np
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.terminal_state_handler.base import TerminalStateHandler
from loco_mujoco.core.utils.math import quat_scalarfirst2scalarlast, safe_norm
from loco_mujoco.core.utils.backend import assert_backend_is_supported


class RootPoseTrajTerminalStateHandler(TerminalStateHandler):

    # widths of the smooth termination (mjx_absorbing_prob) around the limits
    soft_temperature_m = 0.02
    soft_temperature_rad = 0.05

    def __init__(self, env: Any, 
                 root_height_margin: float = 0.3, 
                 root_rot_margin_degrees: float = 30.0,
                 max_root_pos_deviation: float = 1e6):
        """
        Initialize the TerminalStateHandler.

        Args:
            env (Any): The environment instance.
            root_height_margin (float): Margin added to the minimum and maximum root 
                height before being terminal.
            root_rot_margin_degrees (float): Margin added to the minimum and maximum root
                orientation before being terminal.
            max_root_pos_deviation (float): Maximum deviation of the root position from the reference trajectory.
        """
        super(RootPoseTrajTerminalStateHandler, self).__init__(env)

        self.root_joint_name = self._info_props["root_free_joint_xml_name"]

        self.root_height_margin = root_height_margin
        self.root_rot_margin_degrees = root_rot_margin_degrees
        self.max_root_pos_deviation = max_root_pos_deviation

    def reset(self, env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType,
              traj=None) -> Tuple[Union[MjData, Data], Any]:
        """
        Reset the terminal state handler.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for computation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[MjData, Data], Any]: The updated simulation data and carry.

        """
        assert_backend_is_supported(backend)
        return data, carry
    
    def is_absorbing(self,
                     env: Any,
                     obs: np.ndarray,
                     info: Dict[str, Any],
                     data: MjData,
                     carry: Any,
                     traj=None) -> Union[bool, Any]:
        """
        Check if the current state is terminal. The state is terminal if the root height is outside the range or the
        root rotation is outside the valid threshold. Function for CPU Mujoco.

        Args:
            env (Any): The environment instance.
            obs (np.ndarray): shape (n_samples, n_obs), the observations
            info (dict): the info dictionary
            data (MjData): Mujoco data structure
            carry (Any): additional carry.
            traj: Trajectory to use (optional).

        Returns:
            Union[bool, Any]: Whether the current state is terminal, and the carry.

        """
        if traj is not None:
            return self._is_absorbing_compat(env, obs, info, data, carry, backend=np, traj=traj)
        else:
            return False, carry

    def mjx_is_absorbing(self,
                         env: Any,
                         obs: jnp.ndarray,
                         info: Dict[str, Any],
                         data: Data,
                         carry: Any,
                         traj=None) -> Union[bool, Any]:
        """
        Check if the current state is terminal. The state is terminal if the root height is outside the range or the
        root rotation is outside the valid threshold. Function for Mjx.

        Args:
            obs (jnp.ndarray): shape (n_samples, n_obs), the observations
            info (dict): the info dictionary
            data (Data): Mjx data structure
            carry (Any): additional carry.
            traj: Trajectory to use (optional).

        Returns:
            Union[bool, Any]: Whether the current state is terminal, and the carry.

        """
        if traj is not None:
            return self._is_absorbing_compat(env, obs, info, data, carry, backend=jnp, traj=traj)
        else:
            return False, carry

    def _is_absorbing_compat(self,
                             env: Any,
                             obs: Union[np.ndarray, jnp.ndarray],
                             info: Dict[str, Any],
                             data: Union[MjData, Data],
                             carry: Any,
                             backend: ModuleType,
                             traj=None) -> Union[bool, Any]:
        """
        Check if the current state is terminal. The state is terminal if the root height is outside the range or the
        root rotation is outside the valid threshold.

        Args:
            obs (Union[np.ndarray, jnp.ndarray]): shape (n_samples, n_obs), the observations
            info (dict): the info dictionary
            data (Union[MjData, Data]): Mujoco data structure
            carry (Any): additional carry.
            backend (ModuleType): the backend to use (np or jnp)
            traj: Trajectory to use (optional).

        Returns:
            Boolean indicating whether the current state is terminal or not.

        """
        return self.hard_any(self._violations(env, data, carry, backend, traj), backend), carry

    def mjx_absorbing_prob(self, env: Any, obs: jnp.ndarray, info: Dict[str, Any], data: Data, carry: Any,
                           traj=None) -> jnp.ndarray:
        """
        Smooth termination probability of the same conditions, see ``TerminalStateHandler.soft_any``.
        """
        return self.soft_any(self._violations(env, data, carry, jnp, traj))

    def _violations(self, env: Any, data: Union[MjData, Data], carry: Any, backend: ModuleType, traj) -> list:
        """
        (excess, temperature) pairs of the conditions: the root position too far from the reference, the root
        height outside the dataset's range, and the root rotation too far from the dataset's centroid.
        """
        # get indices from traj.info (static pytree aux data — available at trace time)
        root_ind = traj.info.joint_name2ind_qpos[self.root_joint_name]
        root_xy = np.array(root_ind[:2])
        root_height_ind = int(root_ind[2])
        root_quat_ind = np.array(root_ind[3:7])

        # get position, height and rotation of the root joint
        pos = data.qpos[root_xy]
        height = data.qpos[root_height_ind]
        root_quat = quat_scalarfirst2scalarlast(data.qpos[root_quat_ind])

        # deviation of the root position from the reference
        traj_data_cur = env.th.get_current_traj_data(traj.data, carry, backend)
        traj_data_init = env.th.get_init_traj_data(traj.data, carry, backend)
        traj_root_pos = traj_data_cur.qpos[root_xy] - traj_data_init.qpos[root_xy]
        norm = np.linalg.norm if backend == np else safe_norm
        pos_deviation = norm(pos - traj_root_pos)

        # mask out padded zeros using split_points (split_points may have -1 sentinels after padding)
        n_valid = traj.data.n_samples
        valid_mask = backend.arange(traj.data.qpos.shape[0]) < n_valid

        # compute height range from trajectory data (valid rows only)
        traj_heights = traj.data.qpos[:, root_height_ind]
        h_min = backend.min(backend.where(valid_mask, traj_heights, float('inf'))) - self.root_height_margin
        h_max = backend.max(backend.where(valid_mask, traj_heights, float('-inf'))) + self.root_height_margin

        # compute centroid quaternion and threshold from trajectory data (valid rows only)
        root_quat_curr = root_quat / backend.linalg.norm(root_quat)
        centroid_quat, valid_threshold = self._calc_root_rot_centroid_and_margin(
            traj.data.qpos[:, root_quat_ind], valid_mask, backend)
        if backend == np:
            angular_distance = 2 * np.arccos(np.clip(np.dot(centroid_quat, root_quat_curr), -1, 1))
        else:
            # 2 * arccos(dot), with a finite gradient at equal rotations
            v1, w1, v2, w2 = centroid_quat[:3], centroid_quat[3], root_quat_curr[:3], root_quat_curr[3]
            angular_distance = 2 * jnp.arctan2(safe_norm(w1 * v2 - w2 * v1 - jnp.cross(v1, v2)),
                                               jnp.dot(centroid_quat, root_quat_curr))

        return [(pos_deviation - self.max_root_pos_deviation, self.soft_temperature_m),
                (h_min - height, self.soft_temperature_m),
                (height - h_max, self.soft_temperature_m),
                (angular_distance - valid_threshold, self.soft_temperature_rad)]

    def _calc_root_rot_centroid_and_margin(self, root_quats_scalarfirst, valid_mask, backend) -> Tuple[Any, Any]:
        """
        Calculate the centroid quaternion and max angular distance threshold using the eigenvector method.
        Works with both numpy and jax.numpy backends. Ignores padded (invalid) rows via valid_mask.

        Args:
            root_quats_scalarfirst: shape (n_samples, 4), scalar-first quaternions from traj.data.qpos.
            valid_mask: shape (n_samples,), boolean mask of valid (non-padded) rows.
            backend: numpy or jax.numpy.

        Returns:
            centroid_quat: shape (4,), scalar-last.
            valid_threshold: scalar.
        """
        root_quats = quat_scalarfirst2scalarlast(root_quats_scalarfirst)
        # replace the zero padded rows by a unit quaternion before normalizing: guarding only the norm still
        # gives 0/0 = NaN in M when XLA constant-folds a trajectory captured as a constant
        root_quats = backend.where(valid_mask[:, None], root_quats, backend.array([0.0, 0.0, 0.0, 1.0]))
        norm_quats = root_quats / backend.linalg.norm(root_quats, axis=1, keepdims=True)

        # centroid = eigenvector of Q^T Q with largest eigenvalue (sum over valid rows only)
        masked_quats = norm_quats * valid_mask[:, None]
        M = backend.einsum('ni,nj->ij', masked_quats, norm_quats)
        _, vecs = backend.linalg.eigh(M)
        centroid_quat = vecs[:, -1]
        # eigenvectors have arbitrary sign; flip so majority of valid quats have positive dot product
        mean_dot = backend.mean(backend.where(valid_mask,
                                              backend.einsum('ij,j->i', norm_quats, centroid_quat),
                                              0.0))
        centroid_quat = backend.where(mean_dot < 0, -centroid_quat, centroid_quat)

        # angular distances over valid rows only
        dot_products = backend.clip(backend.einsum('ij,j->i', norm_quats, centroid_quat), -1, 1)
        all_distances = 2 * backend.arccos(dot_products)
        max_distance = backend.max(backend.where(valid_mask, all_distances, 0.0))
        valid_threshold = max_distance + np.radians(self.root_rot_margin_degrees)

        return centroid_quat, valid_threshold


class RootTrackingTerminalStateHandler(RootPoseTrajTerminalStateHandler):
    """
    Terminates when the root leaves the *current* reference frame, rather than the range of the whole dataset.
    Unlike RootPoseTrajTerminalStateHandler this stays meaningful for datasets containing motions on the
    ground: with a fall-and-get-up clip in the dataset, lying on the floor is inside the dataset's range.
    """

    soft_temperature_tilt = 0.04  # width of the smooth termination around the tilt limit

    def __init__(self, env: Any, max_height_error: float = 0.25, max_tilt_error: float = 0.8):
        """
        Initialize the TerminalStateHandler.

        Args:
            env (Any): The environment instance.
            max_height_error (float): Maximum deviation of the root height from the reference in meters.
            max_tilt_error (float): Maximum deviation of the z-component of the root's up axis from the reference.
        """
        super(RootTrackingTerminalStateHandler, self).__init__(env)
        self.max_height_error = max_height_error
        self.max_tilt_error = max_tilt_error

    def _is_absorbing_compat(self,
                             env: Any,
                             obs: Union[np.ndarray, jnp.ndarray],
                             info: Dict[str, Any],
                             data: Union[MjData, Data],
                             carry: Any,
                             backend: ModuleType,
                             traj=None) -> Union[bool, Any]:
        """
        Check if the current state is terminal. The state is terminal if the root height or tilt deviate
        too far from the current frame of the reference trajectory.

        Args:
            obs (Union[np.ndarray, jnp.ndarray]): shape (n_samples, n_obs), the observations
            info (dict): the info dictionary
            data (Union[MjData, Data]): Mujoco data structure
            carry (Any): additional carry.
            backend (ModuleType): the backend to use (np or jnp)
            traj: Trajectory to use (optional).

        Returns:
            Boolean indicating whether the current state is terminal or not.

        """
        return self.hard_any(self._violations(env, data, carry, backend, traj), backend), carry

    def _violations(self, env: Any, data: Union[MjData, Data], carry: Any, backend: ModuleType, traj) -> list:
        """
        (excess, temperature) pairs of the conditions: the root height and the z-component of the root's up axis
        too far from the current reference frame.
        """
        root_ind = traj.info.joint_name2ind_qpos[self.root_joint_name]
        root_height_ind = int(root_ind[2])
        root_quat_ind = np.array(root_ind[3:7])
        traj_qpos = env.th.get_current_traj_data(traj.data, carry, backend).qpos

        def up_z(quat):  # scalar-first
            quat = quat / backend.linalg.norm(quat)
            return 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)

        height_error = backend.abs(data.qpos[root_height_ind] - traj_qpos[root_height_ind])
        tilt_error = backend.abs(up_z(data.qpos[root_quat_ind]) - up_z(traj_qpos[root_quat_ind]))
        return [(height_error - self.max_height_error, self.soft_temperature_m),
                (tilt_error - self.max_tilt_error, self.soft_temperature_tilt)]
