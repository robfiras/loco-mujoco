"""Initial state handler that starts every episode from a sample of the reference trajectory."""

from typing import Any, Union, List, Tuple
from types import ModuleType

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.utils import assert_backend_is_supported, mj_jntname2qposid, mj_jntname2qvelid
from loco_mujoco.core.initial_state_handler.base import InitialStateHandler


class TrajInitialStateHandler(InitialStateHandler):
    """
    Initial state handler setting the initial joint positions and velocities from a trajectory.
    At reset, the handler sets the initial state of the simulation to the state of the trajectory.
    To control the state of the trajectory, the trajectory handler params need to be set in the environment.

    """

    def __init__(self, env: Any,
                 root_pos_noise: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 root_rot_noise: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 root_lin_vel_noise: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 root_ang_vel_noise: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 joint_pos_noise: float = 0.0,
                 **kwargs):
        """
        Args:
            env (Any): The environment instance.
            root_pos_noise (Tuple): Half-widths of the uniform root position noise (m). Mjx only, like all noise here.
            root_rot_noise (Tuple): Half-widths of the uniform root rotation noise, as a rotation vector (rad).
            root_lin_vel_noise (Tuple): Half-widths of the uniform root linear velocity noise (m/s).
            root_ang_vel_noise (Tuple): Half-widths of the uniform root angular velocity noise (rad/s).
            joint_pos_noise (float): Half-width of the uniform joint position noise (rad), clipped to the joint range.
            **kwargs (Dict): Additional keyword arguments.
        """
        super().__init__(env, **kwargs)
        self._root_noise = np.array([root_pos_noise, root_rot_noise, root_lin_vel_noise, root_ang_vel_noise], dtype=float)
        self._joint_pos_noise = float(joint_pos_noise)
        self._has_noise = bool(self._root_noise.any()) or self._joint_pos_noise > 0.0

        model = env._model
        self._root_qpos = np.array(mj_jntname2qposid(env.root_free_joint_xml_name, model))
        self._root_qvel = np.array(mj_jntname2qvelid(env.root_free_joint_xml_name, model))
        # compared as ints: from MuJoCo 3.16 on, `enum == numpy int` is always False, so `in (enum, ...)` is too
        hinges = [i for i in range(model.njnt) if int(model.jnt_type[i]) in (int(mujoco.mjtJoint.mjJNT_HINGE),
                                                                             int(mujoco.mjtJoint.mjJNT_SLIDE))]
        self._joint_qpos = model.jnt_qposadr[hinges]
        limited = model.jnt_limited[hinges].astype(bool)
        self._joint_low = np.where(limited, model.jnt_range[hinges, 0], -np.inf)
        self._joint_high = np.where(limited, model.jnt_range[hinges, 1], np.inf)

    def _add_noise(self, traj_data_sample: Any, carry: Any) -> Tuple[Any, Any]:
        """
        Adds the configured noise to a trajectory sample.

        Args:
            traj_data_sample (Any): The sample of the trajectory.
            carry (Any): Carry instance with additional state information.

        Returns:
            Tuple[Any, Any]: The noisy sample and the carry.
        """
        key, k_root, k_joint = jax.random.split(carry.key, 3)
        pos, rot, lin_vel, ang_vel = jax.random.uniform(k_root, (4, 3), minval=-1.0, maxval=1.0) * self._root_noise
        angle = jnp.linalg.norm(rot)
        axis = rot / jnp.maximum(angle, 1e-9)
        dw, (dx, dy, dz) = jnp.cos(angle / 2), axis * jnp.sin(angle / 2)

        qpos, qvel = traj_data_sample.qpos, traj_data_sample.qvel
        w, x, y, z = qpos[self._root_qpos[3:7]]
        quat = jnp.stack([dw * w - dx * x - dy * y - dz * z, dw * x + dx * w + dy * z - dz * y,
                          dw * y - dx * z + dy * w + dz * x, dw * z + dx * y - dy * x + dz * w])
        joints = qpos[self._joint_qpos] + jax.random.uniform(k_joint, self._joint_qpos.shape, minval=-1.0, maxval=1.0) \
            * self._joint_pos_noise
        qpos = qpos.at[self._root_qpos[:3]].add(pos).at[self._root_qpos[3:7]].set(quat)
        qpos = qpos.at[self._joint_qpos].set(jnp.clip(joints, self._joint_low, self._joint_high))
        qvel = qvel.at[self._root_qvel[:3]].add(lin_vel).at[self._root_qvel[3:6]].add(ang_vel)
        return traj_data_sample.replace(qpos=qpos, qvel=qvel), carry.replace(key=key)

    def reset(self, env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType,
              traj=None) -> Tuple[Union[MjData, Data], Any]:
        """
        Reset the init state handler with its state.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).
            traj (Trajectory, optional): The trajectory of the environment, if any.

        Returns:
            Tuple[Union[MjData, Data], Any]: The simulation data and carry.

        Raises:
            ValueError: If the backend module is not supported.
        """
        assert_backend_is_supported(backend)

        assert traj is not None, "If TrajInitialStateHandler is used, traj_data must be passed."
        assert env.th is not None, "If TrajInitialStateHandler is used, a trajectory has to be loaded."

        # Get the current trajectory data sample
        traj_data_sample = env.th.get_current_traj_data(traj.data, carry, backend)

        if self._has_noise:
            assert backend != np, "Reset noise is only available in Mjx."
            traj_data_sample, carry = self._add_noise(traj_data_sample, carry)

        if backend == np:
            data = env.set_sim_state_from_traj_data(data, traj_data_sample, carry)
        else:
            data = env.mjx_set_sim_state_from_traj_data(data, traj_data_sample, carry, traj=traj)

        return data, carry
