from typing import Dict, Any, Union, Tuple
from types import ModuleType

import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation as np_R
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.terminal_state_handler.base import TerminalStateHandler
from loco_mujoco.core.utils.math import quat_scalarfirst2scalarlast
from loco_mujoco.core.utils.backend import assert_backend_is_supported


class RootPoseTrajTerminalStateHandler(TerminalStateHandler):

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
        
        self._initialized = False

        self.root_joint_name = self._info_props["root_free_joint_xml_name"]

        self.root_height_margin = root_height_margin
        self.root_rot_margin_degrees = root_rot_margin_degrees
        self.max_root_pos_deviation = max_root_pos_deviation

        # to be determined in init_from_traj
        self.root_xy = None
        self.root_height_ind = None
        self.root_quat_ind = None
        self.root_height_range = None
        self._centroid_quat = None
        self._valid_threshold = None

    def reset(self, env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType,
              traj_model=None,
              traj_data=None) -> Tuple[Union[MjData, Data], Any]:
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
    
    def init_from_traj(self, traj) -> None:
        """
        Initialize the TerminalStateHandler from a Trajectory.

        Args:
            traj: The trajectory containing the trajectory data.

        """
        assert traj is not None, f"{self.__class__.__name__} requires a Trajectory to be initialized."

        root_ind = traj.info.joint_name2ind_qpos[self.root_joint_name]
        self.root_xy = root_ind[:2]
        self.root_height_ind = root_ind[2]
        self.root_quat_ind = root_ind[3:7]
        assert len(self.root_quat_ind) == 4

        # get the root quaternions
        root_quats = traj.data.qpos[:, self.root_quat_ind]

        # calculate the centroid of the root quaternions and the maximum angular distance from the centroid
        self._centroid_quat, self._valid_threshold = self._calc_root_rot_centroid_and_margin(
            quat_scalarfirst2scalarlast(root_quats))

        # calculate the range of the root height
        root_height_min = np.min(traj.data.qpos[:, self.root_height_ind])
        root_height_max = np.max(traj.data.qpos[:, self.root_height_ind])
        self.root_height_range = (root_height_min - self.root_height_margin, root_height_max + self.root_height_margin)

        self._initialized = True

    def is_absorbing(self,
                     env: Any,
                     obs: np.ndarray,
                     info: Dict[str, Any],
                     data: MjData,
                     carry: Any,
                     traj_model=None,
                     traj_data=None) -> Union[bool, Any]:
        """
        Check if the current state is terminal. The state is terminal if the root height is outside the range or the
        root rotation is outside the valid threshold. Function for CPU Mujoco.

        Args:
            env (Any): The environment instance.
            obs (np.ndarray): shape (n_samples, n_obs), the observations
            info (dict): the info dictionary
            data (MjData): Mujoco data structure
            carry (Any): additional carry.

        Returns:
            Union[bool, Any]: Whether the current state is terminal, and the carry.

        """
        if self.initialized:
            return self._is_absorbing_compat(env, obs, info, data, carry, backend=np,
                                             traj_model=traj_model, traj_data=traj_data)
        else:
            return False, carry

    def mjx_is_absorbing(self,
                         env: Any,
                         obs: jnp.ndarray,
                         info: Dict[str, Any],
                         data: Data,
                         carry: Any,
                         traj_model=None,
                         traj_data=None) -> Union[bool, Any]:
        """
        Check if the current state is terminal. The state is terminal if the root height is outside the range or the
        root rotation is outside the valid threshold. Function for Mjx.

        Args:
            obs (jnp.ndarray): shape (n_samples, n_obs), the observations
            info (dict): the info dictionary
            data (Data): Mjx data structure
            carry (Any): additional carry.

        Returns:
            Union[bool, Any]: Whether the current state is terminal, and the carry.

        """
        if self.initialized:
            return self._is_absorbing_compat(env, obs, info, data, carry, backend=jnp,
                                             traj_model=traj_model, traj_data=traj_data)
        else:
            return False, carry

    def _is_absorbing_compat(self,
                             env: Any,
                             obs: Union[np.ndarray, jnp.ndarray],
                             info: Dict[str, Any],
                             data: Union[MjData, Data],
                             carry: Any,
                             backend: ModuleType,
                             traj_model=None,
                             traj_data=None) -> Union[bool, Any]:
        """
        Check if the current state is terminal. The state is terminal if the root height is outside the range or the
        root rotation is outside the valid threshold.

        Args:
            obs (Union[np.ndarray, jnp.ndarray]): shape (n_samples, n_obs), the observations
            info (dict): the info dictionary
            data (Union[MjData, Data]): Mujoco data structure
            carry (Any): additional carry.
            backend (ModuleType): the backend to use (np or jnp)

        Returns:
            Boolean indicating whether the current state is terminal or not.

        """
        # get position, height and rotation of the root joint
        pos = data.qpos[self.root_xy]
        height = data.qpos[self.root_height_ind]
        root_quat = quat_scalarfirst2scalarlast(data.qpos[self.root_quat_ind])

        # check if the root position is outside the maximum deviation
        traj_data_cur = env.th.get_current_traj_data(traj_data, carry, backend)
        traj_data_init = env.th.get_init_traj_data(traj_data, carry, backend)
        traj_root_pos = traj_data_cur.qpos[self.root_xy] - traj_data_init.qpos[self.root_xy]
        pos_deviation = backend.linalg.norm(pos - traj_root_pos)
        pos_cond = backend.greater(pos_deviation, self.max_root_pos_deviation)

        # check if the root height is outside the range
        height_cond = backend.logical_or(backend.less(height, self.root_height_range[0]),
                                         backend.greater(height, self.root_height_range[1]))

        # check if the root rotation is outside the valid threshold
        root_quat = root_quat / backend.linalg.norm(root_quat)
        angular_distance = 2 * backend.arccos(backend.clip(backend.dot(self._centroid_quat, root_quat), -1, 1))
        root_rot_cond = backend.greater(angular_distance, self._valid_threshold)

        is_absorbing = backend.logical_or(pos_cond, backend.logical_or(height_cond, root_rot_cond))

        return is_absorbing, carry

    def _calc_root_rot_centroid_and_margin(self, root_quats: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Calculate the centroid of the root quaternions and the maximum angular distance from the centroid.

        Args:
            root_quats (np.ndarray): shape (n_samples, 4), the root quaternions.
                (quaternions is expected to be scalar last)

        Returns:
            centroid_quat (np.ndarray): shape (4,), the centroid of the quaternions,
                where the quaternions are scalar last.
            valid_threshold (float): the maximum angular distance from the centroid.

        """

        # normalize them
        norm_root_quats = root_quats / np.linalg.norm(root_quats, axis=1, keepdims=True)

        # compute centroid of the quaternions
        r = np_R.from_quat(norm_root_quats)
        centroid_quat = r.mean().as_quat()

        # Compute maximum deviation in angular distance
        dot_products = np.clip(np.einsum('ij,j->i', norm_root_quats,
                                         centroid_quat), -1, 1)
        angular_distances = 2 * np.arccos(dot_products)

        max_distance = np.max(angular_distances)

        # Add margin
        valid_threshold = max_distance + np.radians(self.root_rot_margin_degrees)

        return centroid_quat, valid_threshold

    @property
    def initialized(self) -> bool:
        """
        Returns whether the current state is initialized.

        Returns:
            Boolean indicating whether the current state is initialized.

        """
        return self._initialized
