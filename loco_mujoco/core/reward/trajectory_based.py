from types import ModuleType
from typing import Any, Dict, Tuple, Union

import mujoco
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model
from flax import struct
import numpy as np
import jax.numpy as jnp
from jax._src.scipy.spatial.transform import Rotation as jnp_R
from scipy.spatial.transform import Rotation as np_R

from loco_mujoco.core.reward.base import Reward
from loco_mujoco.core.observations.base import ObservationType
from loco_mujoco.core.utils import mj_jntname2qposid, mj_jntname2qvelid, mj_jntid2qposid, mj_jntid2qvelid
from loco_mujoco.core.utils.math import calculate_relative_site_quatities, quaternion_angular_distance
from loco_mujoco.core.utils.math import quat_scalarfirst2scalarlast
from loco_mujoco.core.utils.math import calc_site_velocities, heading_mat, rotation_angle
from loco_mujoco.core.reward.utils import out_of_bounds_action_cost


def check_traj_provided(method):
    """
    Decorator to check if trajectory handler is None. Raises ValueError if not provided.
    """
    def wrapper(self, *args, **kwargs):
        env = kwargs.get('env', None) if 'env' in kwargs else args[5]  # Assumes 'env' is the 6th positional argument
        if getattr(env, "th") is None:
            raise ValueError("TrajectoryHandler not provided, but required for trajectory-based rewards.")
        return method(self, *args, **kwargs)
    return wrapper


class TrajectoryBasedReward(Reward):

    """
    Base class for trajectory-based reward functions. These reward functions require a
    trajectory handler to compute the reward.

    """

    @property
    def requires_trajectory(self) -> bool:
        return True


class TargetVelocityTrajReward(TrajectoryBasedReward):
    """
    Reward function that computes the reward based on the deviation from the trajectory velocity. The trajectory
    velocity is provided as an observation in the environment. The reward is computed as the negative exponential
    of the squared difference between the current velocity and the goal velocity. The reward is computed for the
    x, y, and yaw velocities of the root.

    """

    def __init__(self, env: Any,
                 w_exp=10.0,
                 **kwargs):
        """
        Initialize the reward function.

        Args:
            env (Any): Environment instance.
            w_exp (float, optional): Exponential weight for the reward. Defaults to 10.0.
            **kwargs (Any): Additional keyword arguments.
        """

        super().__init__(env, **kwargs)
        self._free_jnt_name = self._info_props["root_free_joint_xml_name"]
        self._free_joint_qpos_idx = np.array(mj_jntname2qposid(self._free_jnt_name, env._model))
        self._free_joint_qvel_idx = np.array(mj_jntname2qvelid(self._free_jnt_name, env._model))
        self._w_exp = w_exp

    @check_traj_provided
    def __call__(self,
                 state: Union[np.ndarray, jnp.ndarray],
                 action: Union[np.ndarray, jnp.ndarray],
                 next_state: Union[np.ndarray, jnp.ndarray],
                 absorbing: bool,
                 info: Dict[str, Any],
                 env: Any,
                 model: Union[MjModel, Model],
                 data: Union[MjData, Data],
                 carry: Any,
                 backend: ModuleType,
                 traj=None) -> Tuple[float, Any]:
        """
        Computes a tracking reward based on the deviation from the trajectory velocity.
        Tracking is done on the x, y, and yaw velocities of the root.

        Args:
            state (Union[np.ndarray, jnp.ndarray]): Last state.
            action (Union[np.ndarray, jnp.ndarray]): Applied action.
            next_state (Union[np.ndarray, jnp.ndarray]): Current state.
            absorbing (bool): Whether the state is absorbing.
            info (Dict[str, Any]): Additional information.
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            Tuple[float, Any]: The reward for the current transition and the updated carry.

        Raises:
            ValueError: If trajectory handler is not provided.

        """
        if backend == np:
            R = np_R
        else:
            R = jnp_R

        def calc_local_vel(_d):
            _lin_vel_global = backend.squeeze(_d.qvel[self._free_joint_qvel_idx])[:3]
            _ang_vel_global = backend.squeeze(_d.qvel[self._free_joint_qvel_idx])[3:]
            _root_quat = R.from_quat(quat_scalarfirst2scalarlast(backend.squeeze(_d.qpos[self._free_joint_qpos_idx])[3:7]))
            _lin_vel_local = _root_quat.as_matrix().T @ _lin_vel_global
            # construct vel, x, y and yaw
            return backend.concatenate([_lin_vel_local[:2], backend.atleast_1d(_ang_vel_global[2])])

        # get root velocity from data
        vel_local = calc_local_vel(data)

        # calculate the same for the trajectory
        # use passed `traj` so JIT-traced swaps are visible
        _traj = traj if traj is not None else env._traj
        traj_data = _traj.data.get(carry.traj_state.traj_no, carry.traj_state.subtraj_step_no, backend)
        traj_vel_local = calc_local_vel(traj_data)

        # calculate tracking reward
        tracking_reward = backend.exp(-self._w_exp*backend.mean(backend.square(vel_local - traj_vel_local)))

        # set nan values to 0
        tracking_reward = backend.nan_to_num(tracking_reward, nan=0.0)

        return tracking_reward, carry


@struct.dataclass
class MimicRewardState:
    """
    State of MimicReward.
    """
    last_qvel: Union[np.ndarray, jnp.ndarray]
    last_action: Union[np.ndarray, jnp.ndarray]


class MimicReward(TrajectoryBasedReward):
    """
    DeepMimic-style reward (Peng et al., 2018) and the original LocoMuJoCo imitation reward; see
    :class:`SiteTrackingReward` for a BeyondMimic-style alternative. The reward is based on the deviation from the
    trajectory, computed as the negative exponential of the squared difference between the current state and the
    trajectory state.
    The reward is computed for the joint positions, joint velocities, relative site positions,
    relative site orientations, and relative site velocities. These sites are specified in the environment properties
    and are placed at key points on the body to mimic the motion of the body.

    """

    def __init__(self, env: Any,
                 sites_for_mimic=None,
                 joints_for_mimic=None,
                 main_site_name=None,
                 **kwargs):
        """
        Initialize the DeepMimic reward function.

        Args:
            env (Any): Environment instance.
            sites_for_mimic (List[str], optional): List of site names to mimic. Defaults to None, taking all.
            joints_for_mimic (List[str], optional): List of joint names to mimic. Defaults to None, taking all.
            main_site_name (str, optional): Name of the site (within the resolved
                site list) used as the reference frame for relative quantities.
                Defaults to None → first site (index 0).
            **kwargs (Any): Additional keyword arguments.

        """

        super().__init__(env, **kwargs)

        # reward coefficients
        self._qpos_w_exp = kwargs.get("qpos_w_exp", 10.0)
        self._qvel_w_exp = kwargs.get("qvel_w_exp", 2.0)
        self._rpos_w_exp = kwargs.get("rpos_w_exp", 100.0)
        self._rquat_w_exp = kwargs.get("rquat_w_exp", 10.0)
        self._rvel_w_exp = kwargs.get("rvel_w_exp", 0.1)
        self._qpos_w_sum = kwargs.get("qpos_w_sum", 0.0)
        self._qvel_w_sum = kwargs.get("qvel_w_sum", 0.0)
        self._rpos_w_sum = kwargs.get("rpos_w_sum", 0.5)
        self._rquat_w_sum = kwargs.get("rquat_w_sum", 0.3)
        self._rvel_w_sum = kwargs.get("rvel_w_sum", 0.0)
        self._action_out_of_bounds_coeff = kwargs.get("action_out_of_bounds_coeff", 0.01)
        self._joint_acc_coeff = kwargs.get("joint_acc_coeff", 0.0)
        self._joint_torque_coeff = kwargs.get("joint_torque_coeff", 0.0)
        self._action_rate_coeff = kwargs.get("action_rate_coeff", 0.0)

        # get main body name of the environment
        self.main_body_name = self._info_props["upper_body_xml_name"]
        model = env._model
        self.main_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.main_body_name)
        rel_site_names = self._info_props["sites_for_mimic"] if sites_for_mimic is None else sites_for_mimic
        self._rel_site_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
                                       for name in rel_site_names])
        self._rel_body_ids = np.array([model.site_bodyid[site_id] for site_id in self._rel_site_ids])

        # Resolve the index of the reference site within rel_site_names. Default 0.
        if main_site_name is not None:
            if main_site_name not in rel_site_names:
                raise ValueError(
                    f"main_site_name={main_site_name!r} not found in "
                    f"sites_for_mimic={list(rel_site_names)!r}"
                )
            self._main_site_id = list(rel_site_names).index(main_site_name)
        else:
            self._main_site_id = 0

        # determine qpos and qvel indices
        quat_in_qpos = []
        qpos_ind = []
        qvel_ind = []
        for i in range(model.njnt):
            jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joints_for_mimic is None or jnt_name in joints_for_mimic:
                qposid = mj_jntid2qposid(i, model)
                qvelid = mj_jntid2qvelid(i, model)
                qpos_ind.append(qposid)
                qvel_ind.append(qvelid)
                if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                    quat_in_qpos.append(qposid[3:])
        self._qpos_ind = np.concatenate(qpos_ind)
        self._qvel_ind = np.concatenate(qvel_ind)
        quat_in_qpos = np.concatenate(quat_in_qpos)
        self._quat_in_qpos = np.array([True if q in quat_in_qpos else False for q in self._qpos_ind])

        # calc mask for the root free joint velocities
        self._free_joint_qvel_ind = np.array(mj_jntname2qvelid(self._info_props["root_free_joint_xml_name"], model))
        self._free_joint_qvel_mask = np.zeros(model.nv, dtype=bool)
        self._free_joint_qvel_mask[self._free_joint_qvel_ind] = True

    def init_state(self, env: Any,
                   key: Any,
                   model: Union[MjModel, Model],
                   data: Union[MjData, Data],
                   backend: ModuleType,
                   traj=None):
        """
        Initialize the reward state.

        Args:
            env (Any): The environment instance.
            key (Any): Key for the reward state.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            MimicRewardState: The initialized reward state.

        """
        return MimicRewardState(last_qvel=data.qvel, last_action=backend.zeros(env.info.action_space.shape[0]))

    def reset(self,
              env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType,
              traj=None):
        """
        Reset the reward state.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).
            traj: Trajectory to use (optional).

        Returns:
            Tuple[Union[MjData, Data], Any]: The updated data and carry.

        """
        reward_state = self.init_state(env, None, model, data, backend, traj)
        carry = carry.replace(reward_state=reward_state)
        return data, carry

    @check_traj_provided
    def __call__(self,
                 state: Union[np.ndarray, jnp.ndarray],
                 action: Union[np.ndarray, jnp.ndarray],
                 next_state: Union[np.ndarray, jnp.ndarray],
                 absorbing: bool,
                 info: Dict[str, Any],
                 env: Any,
                 model: Union[MjModel, Model],
                 data: Union[MjData, Data],
                 carry: Any,
                 backend: ModuleType,
                 traj=None) -> Tuple[float, Any]:
        """
        Computes a deep mimic tracking reward based on the deviation from the trajectory. The reward is computed as the
        negative exponential of the squared difference between the current state and the trajectory state. The reward
        is computed for the joint positions, joint velocities, relative site positions, relative site orientations, and
        relative site velocities.

        Args:
            state (Union[np.ndarray, jnp.ndarray]): Last state.
            action (Union[np.ndarray, jnp.ndarray]): Applied action.
            next_state (Union[np.ndarray, jnp.ndarray]): Current state.
            absorbing (bool): Whether the state is absorbing.
            info (Dict[str, Any]): Additional information.
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            Tuple[float, Any]: The reward for the current transition and the updated carry.

        Raises:
            ValueError: If trajectory handler is not provided.

        """
        # get current reward state
        reward_state = carry.reward_state

        # get trajectory data (use passed `traj` so JIT-traced swaps are visible)
        traj_data = (traj if traj is not None else env._traj).data

        # get all quantities from trajectory
        traj_data_single = traj_data.get(carry.traj_state.traj_no, carry.traj_state.subtraj_step_no, backend)
        qpos_traj, qvel_traj = traj_data_single.qpos[self._qpos_ind], traj_data_single.qvel[self._qvel_ind]
        qpos_quat_traj = qpos_traj[self._quat_in_qpos].reshape(-1, 4)
        if len(self._rel_site_ids) > 1:
            site_rpos_traj, site_rangles_traj, site_rvel_traj =\
                calculate_relative_site_quatities(traj_data_single, self._rel_site_ids,
                                                self._rel_body_ids, model.body_rootid, backend,
                                                main_site_id=self._main_site_id)

        # get all quantities from the current data
        qpos, qvel = data.qpos[self._qpos_ind], data.qvel[self._qvel_ind]
        qpos_quat = qpos[self._quat_in_qpos].reshape(-1, 4)
        if len(self._rel_site_ids) > 1:
            site_rpos, site_rangles, site_rvel = (
                calculate_relative_site_quatities(data, self._rel_site_ids, self._rel_body_ids,
                                                model.body_rootid, backend,
                                                main_site_id=self._main_site_id))

        # calculate distances
        qpos_dist = backend.mean(backend.square(qpos[~self._quat_in_qpos] - qpos_traj[~self._quat_in_qpos]))
        qpos_dist += backend.mean(quaternion_angular_distance(qpos_quat, qpos_quat_traj, backend))
        qvel_dist = backend.mean(backend.square(qvel - qvel_traj))
        if len(self._rel_site_ids) > 1:
            rpos_dist = backend.mean(backend.square(site_rpos - site_rpos_traj))
            rangles_dist = backend.mean(backend.square(site_rangles - site_rangles_traj))
            rvel_rot_dist = backend.mean(backend.square(site_rvel[:,:3] - site_rvel_traj[:,:3]))
            rvel_lin_dist = backend.mean(backend.square(site_rvel[:,3:] - site_rvel_traj[:,3:]))

        # calculate rewards
        qpos_reward = backend.exp(-self._qpos_w_exp*qpos_dist)
        qvel_reward = backend.exp(-self._qvel_w_exp*qvel_dist)
        if len(self._rel_site_ids) > 1:
            rpos_reward = backend.exp(-self._rpos_w_exp*rpos_dist)
            rangles_reward = backend.exp(-self._rquat_w_exp*rangles_dist)
            rvel_rot_reward = backend.exp(-self._rvel_w_exp*rvel_rot_dist)
            rvel_lin_reward = backend.exp(-self._rvel_w_exp*rvel_lin_dist)

        # calculate costs
        # out of bounds action cost
        if self._action_out_of_bounds_coeff > 0.0:
            out_of_bound_reward = -out_of_bounds_action_cost(action, lower_bound=env.mdp_info.action_space.low,
                                                             upper_bound=env.mdp_info.action_space.high, backend=backend)
        else:
            out_of_bound_reward = 0.0

        # joint acceleration reward
        if self._joint_acc_coeff > 0.0:
            last_joint_vel = reward_state.last_qvel[~self._free_joint_qvel_mask]
            joint_vel = data.qvel[~self._free_joint_qvel_mask]
            acceleration_norm = backend.sum(backend.square(joint_vel - last_joint_vel) / env.dt)
            acceleration_reward = self._joint_acc_coeff * -acceleration_norm
        else:
            acceleration_reward = 0.0

        # joint torque reward
        if self._joint_torque_coeff > 0.0:
            torque_norm = backend.sum(backend.square(data.qfrc_actuator[~self._free_joint_qvel_mask]))
            torque_reward = self._joint_torque_coeff * -torque_norm
        else:
            torque_reward = 0.0

        # action rate reward
        if self._action_rate_coeff > 0.0:
            action_rate_norm = backend.sum(backend.square(action - reward_state.last_action))
            action_rate_reward = self._action_rate_coeff * -action_rate_norm
        else:
            action_rate_reward = 0.0

        # total penality rewards
        total_penalities = (self._action_out_of_bounds_coeff * out_of_bound_reward
                            + self._joint_acc_coeff * acceleration_reward
                            + self._joint_torque_coeff * torque_reward
                            + self._action_rate_coeff * action_rate_reward)
        total_penalities = backend.maximum(total_penalities, -1.0)

        # calculate total reward
        total_reward = (self._qpos_w_sum * qpos_reward + self._qvel_w_sum * qvel_reward)
        if len(self._rel_site_ids) > 1:
            total_reward = (total_reward
                        + self._rpos_w_sum * rpos_reward + self._rquat_w_sum * rangles_reward
                        + self._rvel_w_sum * rvel_rot_reward + self._rvel_w_sum * rvel_lin_reward)

        total_reward = total_reward + total_penalities

        # clip to positive values
        total_reward = backend.maximum(total_reward, 0.0)

        # set nan values to 0
        total_reward = backend.nan_to_num(total_reward, nan=0.0)

        # update reward state
        reward_state = reward_state.replace(last_qvel=data.qvel, last_action=action)
        carry = carry.replace(reward_state=reward_state)

        return total_reward, carry


@struct.dataclass
class SiteTrackingRewardState:
    """
    State of SiteTrackingReward.
    """
    last_action: Union[np.ndarray, jnp.ndarray]


class SiteTrackingReward(TrajectoryBasedReward):
    """
    BeyondMimic-style tracking reward (Liao et al., 2025), computed on the mimic sites instead of on bodies, so
    that it applies to every robot without a robot-specific body list. Compared to :class:`MimicReward`, it has
    no joint-space terms, tracks the main site in the world, and tracks the other sites against a reference that
    is moved to the robot's position and heading. Every tracking term is exp(-error / std^2):

    - the position and the orientation of the main site in the world,
    - the positions and orientations of all mimic sites, against the reference moved to the robot's horizontal
      position and heading, so that drift only lowers the two main site terms,
    - the linear and angular velocities of all mimic sites in the world.

    The action rate and joint positions outside the soft joint limits are penalized. The main site is the first
    mimic site.

    """

    def __init__(self, env: Any,
                 sites_for_mimic=None,
                 main_pos_std: float = 0.3, main_ori_std: float = 0.4,
                 site_pos_std: float = 0.3, site_ori_std: float = 0.4,
                 site_lin_vel_std: float = 1.0, site_ang_vel_std: float = 3.14,
                 w_main_pos: float = 0.5, w_main_ori: float = 0.5,
                 w_site_pos: float = 1.0, w_site_ori: float = 1.0,
                 w_site_lin_vel: float = 1.0, w_site_ang_vel: float = 1.0,
                 w_action_rate: float = -0.1, w_joint_limit: float = -2.0,
                 soft_joint_pos_limit_factor: float = 0.9,
                 scale_by_dt: bool = True,
                 **kwargs):
        """
        Initialize the site tracking reward function.

        Args:
            env (Any): Environment instance.
            sites_for_mimic (List[str], optional): Sites to track, the main site first. Defaults to None, taking all.
            main_pos_std, main_ori_std, site_pos_std, site_ori_std, site_lin_vel_std, site_ang_vel_std (float):
                Width of each tracking term.
            w_main_pos, w_main_ori, w_site_pos, w_site_ori, w_site_lin_vel, w_site_ang_vel (float): Weight of each
                tracking term.
            w_action_rate (float): Weight of the squared action difference.
            w_joint_limit (float): Weight of the distance of the joint positions to their soft limits.
            soft_joint_pos_limit_factor (float): Part of each joint range, around its center, that is not penalized.
            scale_by_dt (bool): Multiply the reward by the control timestep, as BeyondMimic does.
            **kwargs (Any): Additional keyword arguments.

        """

        super().__init__(env, **kwargs)

        self._std_sq = np.square([main_pos_std, main_ori_std, site_pos_std, site_ori_std,
                                  site_lin_vel_std, site_ang_vel_std])
        self._w = np.array([w_main_pos, w_main_ori, w_site_pos, w_site_ori, w_site_lin_vel, w_site_ang_vel])
        self._w_action_rate = w_action_rate
        self._w_joint_limit = w_joint_limit
        self._scale = env.dt if scale_by_dt else 1.0

        # the mimic sites, the main site first
        model = env._model
        site_names = self._info_props["sites_for_mimic"] if sites_for_mimic is None else sites_for_mimic
        self._site_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in site_names])
        self._site_body_ids = model.site_bodyid[self._site_ids]
        self._site_root_ids = model.body_rootid[self._site_body_ids]

        # soft limits of the limited joints
        limited = [i for i in range(model.njnt)
                   if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE and model.jnt_limited[i]]
        self._limited_qpos_ind = model.jnt_qposadr[limited]
        center, half_range = model.jnt_range[limited].mean(axis=1), 0.5 * np.ptp(model.jnt_range[limited], axis=1)
        self._soft_low = center - soft_joint_pos_limit_factor * half_range
        self._soft_high = center + soft_joint_pos_limit_factor * half_range

    @property
    def max_reward(self) -> float:
        """Reward of a step on the reference, without penalties."""
        return float(self._scale * np.sum(self._w))

    def init_state(self, env: Any,
                   key: Any,
                   model: Union[MjModel, Model],
                   data: Union[MjData, Data],
                   backend: ModuleType,
                   traj=None):
        """
        Initialize the reward state.

        Args:
            env (Any): The environment instance.
            key (Any): Key for the reward state.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            SiteTrackingRewardState: The initialized reward state.

        """
        return SiteTrackingRewardState(last_action=backend.zeros(env.info.action_space.shape[0]))

    def reset(self,
              env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType,
              traj=None):
        """
        Reset the reward state.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).
            traj: Trajectory to use (optional).

        Returns:
            Tuple[Union[MjData, Data], Any]: The updated data and carry.

        """
        carry = carry.replace(reward_state=self.init_state(env, None, model, data, backend, traj))
        return data, carry

    @check_traj_provided
    def __call__(self,
                 state: Union[np.ndarray, jnp.ndarray],
                 action: Union[np.ndarray, jnp.ndarray],
                 next_state: Union[np.ndarray, jnp.ndarray],
                 absorbing: bool,
                 info: Dict[str, Any],
                 env: Any,
                 model: Union[MjModel, Model],
                 data: Union[MjData, Data],
                 carry: Any,
                 backend: ModuleType,
                 traj=None) -> Tuple[float, Any]:
        """
        Computes the site tracking reward of the current state against the current frame of the trajectory.

        Args:
            state (Union[np.ndarray, jnp.ndarray]): Last state.
            action (Union[np.ndarray, jnp.ndarray]): Applied action.
            next_state (Union[np.ndarray, jnp.ndarray]): Current state.
            absorbing (bool): Whether the state is absorbing.
            info (Dict[str, Any]): Additional information.
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            Tuple[float, Any]: The reward for the current transition and the updated carry.

        Raises:
            ValueError: If trajectory handler is not provided.

        """
        reward_state = carry.reward_state
        traj_data = (traj if traj is not None else env._traj).data
        ref = traj_data.get(carry.traj_state.traj_no, carry.traj_state.subtraj_step_no, backend)

        # poses and velocities (angular, then linear) of the mimic sites in the world; the main site is entry 0
        pos, ref_pos = data.site_xpos[self._site_ids], ref.site_xpos[self._site_ids]
        mat = data.site_xmat[self._site_ids].reshape(-1, 3, 3)
        ref_mat = ref.site_xmat[self._site_ids].reshape(-1, 3, 3)
        vel = calc_site_velocities(self._site_ids, data, self._site_body_ids, self._site_root_ids, backend)
        ref_vel = calc_site_velocities(self._site_ids, ref, self._site_body_ids, self._site_root_ids, backend)

        # the reference at the robot's horizontal position and heading, with its own height, roll and pitch
        turn = heading_mat(mat[0] @ ref_mat[0].T, backend)
        ref_pos_moved = (backend.concatenate([pos[0, :2], ref_pos[0, 2:]])
                         + (ref_pos - ref_pos[0]) @ turn.T)
        ref_mat_moved = backend.einsum("ij,bjk->bik", turn, ref_mat)

        dist = backend.stack([
            backend.sum(backend.square(ref_pos[0] - pos[0])),
            backend.square(rotation_angle(ref_mat[0], mat[0], backend)),
            backend.mean(backend.sum(backend.square(ref_pos_moved - pos), axis=-1)),
            backend.mean(backend.square(rotation_angle(ref_mat_moved, mat, backend))),
            backend.mean(backend.sum(backend.square(ref_vel[:, 3:] - vel[:, 3:]), axis=-1)),
            backend.mean(backend.sum(backend.square(ref_vel[:, :3] - vel[:, :3]), axis=-1))])
        tracking_reward = backend.sum(self._w * backend.exp(-dist / self._std_sq))

        # penalties
        action_rate = backend.sum(backend.square(action - reward_state.last_action))
        qpos = data.qpos[self._limited_qpos_ind]
        out_of_limits = backend.sum(backend.maximum(self._soft_low - qpos, 0.0)
                                    + backend.maximum(qpos - self._soft_high, 0.0))

        total_reward = self._scale * (tracking_reward + self._w_action_rate * action_rate
                                      + self._w_joint_limit * out_of_limits)

        # set nan values to 0
        total_reward = backend.nan_to_num(total_reward, nan=0.0)

        # update reward state
        carry = carry.replace(reward_state=reward_state.replace(last_action=action))

        return total_reward, carry
