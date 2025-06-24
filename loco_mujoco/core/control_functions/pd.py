from typing import Any, Dict, List, Union, Tuple
from types import ModuleType

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from flax import struct
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.utils.backend import assert_backend_is_supported
from loco_mujoco.core.control_functions import ControlFunction
from loco_mujoco.core.utils import mj_jntname2qposid, mj_jntname2qvelid


@struct.dataclass
class PDControlState:
    """
    This state can be used by the domain randomizer to add noise to the PD-controller.
    """
    p_gain_noise: Union[np.ndarray, jax.Array]
    d_gain_noise: Union[np.ndarray, jax.Array]
    pos_offset: Union[np.ndarray, jax.Array]
    ctrl_mult: Union[np.ndarray, jax.Array]


class PDControl(ControlFunction):

    """
    PD controller function setting positions. This controller internally normalizes the action space to [-1, 1]
    for the agent but uses the joint position limits for the environment.

    """

    def __init__(self,
                 env: Any,
                 p_gain: Union[float, np.ndarray],
                 d_gain: Union[float, np.ndarray],
                 nominal_joint_positions: np.ndarray = None,
                 scale_action_to_jnt_limits: bool = True,
                 **kwargs: Any):
        """
        Initialize the PDControl class.

        Args:
            env (Any): The environment instance containing model and specifications.
            p_gain (Union[float, np.ndarray]): Proportional gain for the PD controller.
            d_gain (Union[float, np.ndarray]): Derivative gain for the PD controller.
            nominal_joint_positions (np.ndarray, optional): Default joint positions. If not provided, uses qpos0.
            scale_action_to_jnt_limits (bool): If true, the actions are scaled to the joint limits.
            **kwargs (Any): Additional keyword arguments for the parent class.
        """
        self._init_p_gain = np.array(p_gain)
        self._init_d_gain = np.array(d_gain)
        self._ctrl_ranges = []
        self._jnt_ranges = []
        self._jnt_names = []
        self._scale_action_to_jnt_limits = scale_action_to_jnt_limits
        for actuator in env.mjspec.actuators:
            jnt_name = actuator.target
            ctrl_range = actuator.ctrlrange if actuator.ctrllimited else np.array([-np.inf, np.inf])
            self._ctrl_ranges.append(ctrl_range)
            for j in env.mjspec.joints:
                if j.name == jnt_name:
                    assert j.type == mujoco.mjtJoint.mjJNT_HINGE or j.type == mujoco.mjtJoint.mjJNT_SLIDE, \
                        "Only Hinge and Slide joints are supported for PDControl."
                    self._jnt_names.append(jnt_name)
                    self._jnt_ranges.append(j.range)

        # sort according to action ind
        self._ctrl_ranges = np.concatenate([self._ctrl_ranges[i].reshape(1, 2) for i in env._action_indices])
        self._jnt_ranges = np.concatenate([self._jnt_ranges[i].reshape(1, 2) for i in env._action_indices])
        self._jnt_names = [self._jnt_names[i] for i in env._action_indices]

        # get qpos and qvel ids
        self._qpos_ids = np.concatenate([mj_jntname2qposid(name, env._model) for name in self._jnt_names])
        self._qvel_ids = np.concatenate([mj_jntname2qvelid(name, env._model) for name in self._jnt_names])

        if nominal_joint_positions is None:
            self._nominal_joint_positions = env._model.qpos0[self._qpos_ids]
        else:
            self._nominal_joint_positions = np.array(nominal_joint_positions)

        self._high_pos_target = self._jnt_ranges[:, 1] - self._nominal_joint_positions
        self._low_pos_target = self._jnt_ranges[:, 0] - self._nominal_joint_positions

        # calculate mean and delta
        self.norm_act_mean = (self._high_pos_target + self._low_pos_target) / 2.0
        self.norm_act_delta = (self._high_pos_target - self._low_pos_target) / 2.0

        # set the action space limits for the agent to -1 and 1
        low = -np.ones_like(self.norm_act_mean)
        high = np.ones_like(self.norm_act_mean)

        super(PDControl, self).__init__(env, low, high, **kwargs)

    def reset(self, env: Any,
                model: Union[MjModel, Model],
                data: Union[MjData, Data],
                carry: Any,
                backend: ModuleType) -> Tuple[Union[MjData, Data], Any]:
        """
        Reset the PD controller state.
        """
        assert_backend_is_supported(backend)
        return data, carry

    def init_state(self,
                   env: Any,
                   key: Union[jax.random.PRNGKey, Any],
                   model: Union[MjModel, Model],
                   data: Union[MjData, Data],
                   backend: ModuleType) -> PDControlState:
        """
        Initialize the state for PDControl.

        Args:
            env (Any): The environment instance.
            key (Union[jax.random.PRNGKey, Any]): Random key for noise generation, specific to JAX.
            model (Union[MjModel, Model]): The simulation model instance.
            data (Union[MjData, Data]): The simulation data instance.
            backend (ModuleType): Backend module (e.g., numpy or jax.numpy) for computation.

        Returns:
            PDControlState: The initialized PD control state containing zeroed noise values, offsets, and multipliers.

        Raises:
            ValueError: If the backend module is not supported.
        """
        assert_backend_is_supported(backend)
        return PDControlState(p_gain_noise=backend.zeros_like(self._nominal_joint_positions),
                              d_gain_noise=backend.zeros_like(self._nominal_joint_positions),
                              pos_offset=backend.zeros_like(self._nominal_joint_positions),
                              ctrl_mult=backend.ones_like(self._nominal_joint_positions))

    def generate_action(self, env: Any,
                        action: Union[np.ndarray, jax.Array],
                        model: Union[MjModel, Model],
                        data: Union[MjData, Data],
                        carry: Any,
                        backend: ModuleType) -> Tuple[Union[np.ndarray, jax.Array], Any]:
        """
        Generate the action using the PD controller. This function expects the action to be in the range [-1, 1].

        Args:
            env (Any): The environment instance.
            action (Union[np.ndarray, jax.Array]): The action to be updated.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jax.Array], Any]: The updated action and carry.

        Raises:
            ValueError: If the backend module is not supported.
        """
        assert_backend_is_supported(backend)

        if self._scale_action_to_jnt_limits:
            unnormalized_action = self._unnormalize_action(action)
        else:
            unnormalized_action = action

        pd_state = carry.control_func_state

        p_gain = pd_state.p_gain_noise + self._init_p_gain
        d_gain = pd_state.d_gain_noise + self._init_d_gain
        offsets = pd_state.pos_offset

        target_joint_pos = backend.clip(self._nominal_joint_positions + unnormalized_action + offsets,
                                        self._jnt_ranges[:, 0], self._jnt_ranges[:, 1])

        ctrl = p_gain * (target_joint_pos - data.qpos[self._qpos_ids]) - d_gain * data.qvel[self._qvel_ids]
        ctrl = backend.clip(ctrl * pd_state.ctrl_mult, self._ctrl_ranges[:, 0], self._ctrl_ranges[:, 1])

        return ctrl, carry

    def _unnormalize_action(self, action: Union[np.ndarray, jax.Array]) -> Union[np.ndarray, jax.Array]:
        """
        Rescale the action from [-1, 1] to the desired action space.

        Args:
            action (Union[np.ndarray, jax.Array]): The action to be unnormalized.

        Returns:
            Union[np.ndarray, jax.Array]: The unnormalized action

        """
        unnormalized_action = ((action * self.norm_act_delta) + self.norm_act_mean)
        return unnormalized_action

    @property
    def run_with_simulation_frequency(self):
        """
        If true, the control function is called with the simulation frequency.
        """
        return True
