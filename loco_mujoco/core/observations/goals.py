from copy import deepcopy
from typing import Union, Any, Tuple, Dict, List
from types import ModuleType
import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from mujoco import MjSpec, MjModel, MjData
from mujoco.mjx import Model, Data
from jax.scipy.spatial.transform import Rotation as jnp_R
from scipy.spatial.transform import Rotation as np_R
from flax import struct

from loco_mujoco.core.observations.base import StatefulObservation, Observation
from loco_mujoco.core.observations.visualizer import RootVelocityArrowVisualizer
from loco_mujoco.core.stateful_object import StatefulObject
from loco_mujoco.core.utils.math import (
    calculate_relative_site_quatities,
    quat_scalarfirst2scalarlast,
    quat_scalarlast2scalarfirst
)
from loco_mujoco.core.utils.mujoco import (
    mj_jntid2qposid, mj_jntid2qvelid,
    mj_jntname2qposid, mj_jntname2qvelid
)


class Goal(StatefulObservation):
    """
    Base class representing a goal in the environment.

    Args:
        info_props (Dict): Information properties required for initialization.
        visualize_goal (bool): Whether to visualize the goal.
        n_visual_geoms (int): Number of visual geometries for visualization.
    """

    def __init__(self, info_props: Dict, visualize_goal: bool = False, n_visual_geoms: int = 0, **kwargs):
        self._info_props = info_props
        if visualize_goal:
            assert self.has_visual, (
                f"{self.__class__.__name__} does not support visualization. "
                f"Please set visualize_goal to False."
            )
        self.visualize_goal = visualize_goal
        Observation.__init__(self, obs_name=self.__class__.__name__, **kwargs)
        StatefulObject.__init__(self, n_visual_geoms)

    @property
    def has_visual(self) -> bool:
        """Check if the goal supports visualization. Needs to be implemented in subclasses."""
        raise NotImplementedError

    @property
    def requires_trajectory(self) -> bool:
        """Check if the goal requires a trajectory."""
        return False

    @classmethod
    def data_type(cls) -> Any:
        """Return the data type used by this goal."""
        return None

    def reset_state(self,
                    env: Any,
                    model: Union[MjModel, Model],
                    data: Union[MjData, Data],
                    carry: Any,
                    backend: ModuleType, traj=None) -> Tuple[Union[MjData, Any], Any]:
        """
        Reset the state of the goal.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The Mujoco model.
            data (Union[MjData, Data]): The Mujoco data.
            carry (Any): Carry object.
            backend (ModuleType): The backend (numpy or jax).

        Returns:
            Tuple[Union[MjData, Any], Any]: Updated data and carry.
        """
        assert self.initialized
        return data, carry

    def is_done(self,
                env: Any,
                model: Union[MjModel, Model],
                data: Union[MjData, Data],
                carry: Any,
                backend: ModuleType,
                traj=None) -> bool:
        """
        Check if the goal is completed.
        """
        return False

    def mjx_is_done(self,
                    env: Any,
                    model: Union[MjModel, Model],
                    data: Union[MjData, Data],
                    carry: Any,
                    backend: ModuleType,
                    traj=None) -> bool:
        """
        Check if the goal is done (jax-compatible version).
        """
        return False

    def apply_spec_modifications(self,
                                 spec: MjSpec,
                                 info_props: Dict) -> MjSpec:
        """
        Apply modifications to the Mujoco XML specification to include the goal.

        Args:
            spec (MjSpec): Mujoco specification.
            info_props (Dict): Information properties.

        Returns:
            MjSpec: Modified Mujoco specification.
        """
        return spec

    def set_attr_compat(self,
                        data: Union[MjData, Data],
                        backend: ModuleType,
                        attr: str,
                        arr: Union[np.ndarray, jnp.ndarray],
                        ind: Union[np.ndarray, jnp.ndarray, None] = None) -> Union[MjData, Any]:
        """
        Set attributes in a backend-compatible manner.

        Args:
            data (Union[MjData, Data]): Data object to modify.
            backend (ModuleType): Backend to use (numpy or jax).
            attr (str): Attribute name to modify.
            arr (Union[np.ndarray, jnp.ndarray]): Array to set.
            ind (Union[np.ndarray, jnp.ndarray, None]): Indices to modify.

        Returns:
            Union[MjData, Any]: Modified data.
        """
        if ind is None:
            ind = backend.arange(len(arr))

        if backend == np:
            getattr(data, attr)[ind] = arr
        elif backend == jnp:
            data = data.replace(**{attr: getattr(data, attr).at[ind].set(arr)})
        else:
            raise NotImplementedError
        return data

    @property
    def initialized(self) -> bool:
        """Check if the goal is initialized."""
        return self._initialized_from_mj

    @property
    def dim(self) -> int:
        """Get the dimension of the goal."""
        raise NotImplementedError

    @property
    def requires_spec_modification(self) -> bool:
        """Check if the goal requires specification modification."""
        return self.__class__.apply_spec_modifications != Goal.apply_spec_modifications

    @classmethod
    def list_goals(cls) -> list:
        """List all subclasses of Goal."""
        return [goal for goal in Goal.__subclasses__()]


class NoGoal(Goal):
    """
    Empty goal class.
    """

    def _init_from_mj(self,
                      env: Any,
                      model: Union[MjModel, Any],
                      data: Union[MjData, Any],
                      current_obs_size: int):
        """
        Initialize the class from Mujoco model and data.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Any]): The Mujoco model.
            data (Union[MjData, Any]): The Mujoco data.
            current_obs_size (int): Current observation size.
        """
        self.min = [-np.inf] * self.dim
        self.max = [np.inf] * self.dim
        self.data_type_ind = np.array([])
        self.obs_ind = np.array([])
        self._initialized_from_mj = True

    def get_obs_and_update_state(self,
                                 env: Any,
                                 model: Union[MjModel, Model],
                                 data: Union[MjData, Data],
                                 carry: Any,
                                 backend: ModuleType, traj=None) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Get the observation and update the state. Always returns an empty array for NoGoal.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Any]): The Mujoco model.
            data (Union[MjData, Any]): The Mujoco data.
            carry (Any): Carry object.
            backend (ModuleType): The backend (numpy or jax).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: Empty observation and unchanged carry.
        """
        return backend.array([]), carry

    @property
    def has_visual(self) -> bool:
        """Check if the goal supports visualization. Always False for NoGoal."""
        return False

    @property
    def dim(self) -> int:
        """Get the dimension of the goal. Always 0 for NoGoal."""
        return 0


@struct.dataclass
class GoalRandomRootVelocityState:
    """
    State class for random root velocity goal.

    Attributes:
        goal_vel_x (float): Goal velocity in the x direction.
        goal_vel_y (float): Goal velocity in the y direction.
        goal_vel_yaw (float): Goal yaw velocity.
    """
    goal_vel_x: float
    goal_vel_y: float
    goal_vel_yaw: float


class GoalRandomRootVelocity(Goal, RootVelocityArrowVisualizer):
    """
    A class representing a random root velocity goal.

    This class defines a goal that specifies random velocities for the root body in
    the x, y, and yaw directions.

    Args:
        info_props (Dict): Information properties required for initialization.
        max_x_vel (float): Maximum velocity in the x direction.
        max_y_vel (float): Maximum velocity in the y direction.
        max_yaw_vel (float): Maximum yaw velocity.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self,
                 info_props: Dict,
                 max_x_vel: float = 1.0,
                 max_y_vel: float = 1.0,
                 max_yaw_vel: float = 1.0, **kwargs):

        self._traj_goal_ind = None
        self.max_x_vel = max_x_vel
        self.max_y_vel = max_y_vel
        self.max_yaw_vel = max_yaw_vel
        self.upper_body_xml_name = info_props["upper_body_xml_name"]
        self.free_jnt_name = info_props["root_free_joint_xml_name"]

        # To be initialized from Mujoco
        self._root_body_id = None
        self._root_jnt_qpos_start_id = None

        # call visualizer init
        RootVelocityArrowVisualizer.__init__(self, info_props)

        # call goal init
        n_visual_geoms = self._arrow_n_visual_geoms \
            if "visualize_goal" in kwargs.keys() and kwargs["visualize_goal"] else 0
        super().__init__(info_props, n_visual_geoms=n_visual_geoms, **kwargs)

    def _init_from_mj(self,
                      env: Any,
                      model: Union[MjModel, Model],
                      data: Union[MjData, Data],
                      current_obs_size: int):
        """
        Initialize the goal from Mujoco model and data.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The Mujoco model.
            data (Union[MjData, Data]): The Mujoco data.
            current_obs_size (int): Current observation size.
        """
        self.min = [-np.inf] * self.dim
        self.max = [np.inf] * self.dim
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + self.dim)])
        self._root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.upper_body_xml_name)
        self._free_jnt_qpos_id = np.array(mj_jntname2qposid(self.free_jnt_name, model))
        self._initialized_from_mj = True

    @property
    def has_visual(self) -> bool:
        """Check if the goal supports visualization."""
        return True

    def init_state(self,
                   env: Any,
                   key: jax.random.PRNGKey,
                   model: Union[MjModel, Model],
                   data: Union[MjData, Data],
                   backend: ModuleType, traj=None) -> GoalRandomRootVelocityState:
        """
        Initialize the goal state.

        Args:
            env (Any): The environment instance.
            key (jax.random.PRNGKey): Random key for sampling.
            model (Union[MjModel, Any]): The Mujoco model.
            data (Union[MjData, Any]): The Mujoco data.
            backend (ModuleType): The backend (numpy or jax).

        Returns:
            GoalRandomRootVelocityState: Initialized state.
        """
        return GoalRandomRootVelocityState(0.0, 0.0, 0.0)

    def reset_state(self,
                    env: Any,
                    model: Union[MjModel, Model],
                    data: Union[MjData, Data],
                    carry: Any,
                    backend: ModuleType, traj=None) -> Tuple[Union[MjData, Any], Any]:
        """
        Reset the goal state with random velocities.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Any]): The Mujoco model.
            data (Union[MjData, Any]): The Mujoco data.
            carry (Any): Carry object.
            backend (ModuleType): The backend (numpy or jax).

        Returns:
            Tuple[Union[MjData, Any], Any]: Updated data and carry.
        """
        key = carry.key
        if backend == np:
            goal_vel = np.random.uniform(
                [-self.max_x_vel, -self.max_y_vel, -self.max_yaw_vel],
                [self.max_x_vel, self.max_y_vel, self.max_yaw_vel]
            )
        else:
            key, subkey = jax.random.split(key)
            goal_vel = jax.random.uniform(
                subkey,
                shape=(3,),
                minval=jnp.array([-self.max_x_vel, -self.max_y_vel, -self.max_yaw_vel]),
                maxval=jnp.array([self.max_x_vel, self.max_y_vel, self.max_yaw_vel])
            )

        goal_state = GoalRandomRootVelocityState(goal_vel[0], goal_vel[1], goal_vel[2])
        observation_states = carry.observation_states.replace(**{self.name: goal_state})
        return data, carry.replace(key=key, observation_states=observation_states)

    def get_obs_and_update_state(self,
                                 env: Any,
                                 model: Union[MjModel, Model],
                                 data: Union[MjData, Data],
                                 carry: Any,
                                 backend: ModuleType, traj=None) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Get the current goal observation and update the state.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The Mujoco model.
            data (Union[MjData, Data]): The Mujoco data.
            carry (Any): Carry object.
            backend (ModuleType): The backend (numpy or jax).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: Goal observation and updated carry.
        """
        goal_vel_x = getattr(carry.observation_states, self.name).goal_vel_x
        goal_vel_y = getattr(carry.observation_states, self.name).goal_vel_y
        goal_vel_yaw = getattr(carry.observation_states, self.name).goal_vel_yaw
        goal = backend.array([goal_vel_x, goal_vel_y, goal_vel_yaw])
        goal_visual = backend.array([goal_vel_x, goal_vel_y, 0.0, 0.0, 0.0, goal_vel_yaw])

        if self.visualize_goal:
            carry = self.set_visuals(
                goal_visual, env, model, data, carry, self._root_body_id,
                self._free_jnt_qpos_id, self.visual_geoms_idx, backend
            )

        return goal, carry

    @property
    def dim(self) -> int:
        """Get the dimension of the goal."""
        return 3


@struct.dataclass
class GoalTrajRootVelocityState:
    """
    State class for trajectory root velocity goal.

    Attributes:
        goal_vel (Union[np.ndarray, jnp.ndarray]): Velocity goal for the root.
    """
    goal_vel: Union[np.ndarray, jnp.ndarray]


class GoalTrajRootVelocity(Goal, RootVelocityArrowVisualizer):
    """
    A class representing a trajectory-based root velocity goal.

    This class defines a goal that computes the root velocity based on trajectory data
    and averages over a specified number of future steps.

    Args:
        info_props (Dict): Information properties required for initialization.
        n_steps_average (int): Number of future steps to average over for the velocity goal.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self,
                 info_props: Dict,
                 n_steps_average: int = 3,
                 **kwargs):
        self._traj_goal_ind = None
        self.upper_body_xml_name = info_props["upper_body_xml_name"]
        self.free_jnt_name = info_props["root_free_joint_xml_name"]

        # To be initialized from Mujoco
        self._root_body_id = None
        self._root_jnt_qpos_start_id = None
        self._free_jnt_qvelid = None
        self._free_jnt_qposid = None

        # Number of future steps in the trajectory to average the goal over
        self._n_steps_average = n_steps_average

        # Call visualizer initialization (if applicable)
        RootVelocityArrowVisualizer.__init__(self, info_props)

        # Call goal initialization
        n_visual_geoms = self._arrow_n_visual_geoms \
            if "visualize_goal" in kwargs.keys() and kwargs["visualize_goal"] else 0

        super().__init__(info_props, n_visual_geoms=n_visual_geoms, **kwargs)

    def _init_from_mj(self,
                      env: Any,
                      model: Union[MjModel, Model],
                      data: Union[MjData, Data],
                      current_obs_size: int):
        """
        Initialize the goal from Mujoco model and data.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The Mujoco model.
            data (Union[MjData, Data]): The Mujoco data.
            current_obs_size (int): Current observation size.
        """
        self.min = [-np.inf] * self.dim
        self.max = [np.inf] * self.dim
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + self.dim)])
        self._initialized_from_mj = True
        self._root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.upper_body_xml_name)
        self._free_jnt_qposid = np.array(mj_jntname2qposid(self.free_jnt_name, model))
        self._free_jnt_qvelid = np.array(mj_jntname2qvelid(self.free_jnt_name, model))

    def init_state(self,
                   env: Any,
                   key: jax.random.PRNGKey,
                   model: Union[MjModel, Any],
                   data: Union[MjData, Any],
                   backend: ModuleType, traj=None) -> GoalTrajRootVelocityState:
        """
        Initialize the goal state.

        Args:
            env (Any): The environment instance.
            key (jax.random.PRNGKey): Random key for sampling.
            model (Union[MjModel, Any]): The Mujoco model.
            data (Union[MjData, Any]): The Mujoco data.
            backend (ModuleType): The backend (numpy or jax).

        Returns:
            GoalTrajRootVelocityState: Initialized state with zero velocity.
        """
        return GoalTrajRootVelocityState(backend.zeros(self.dim))

    def get_obs_and_update_state(self,
                                 env: Any,
                                 model: Union[MjModel, Model],
                                 data: Union[MjData, Data],
                                 carry: Any,
                                 backend: ModuleType, traj=None) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Get the current goal observation and update the state.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Any]): The Mujoco model.
            data (Union[MjData, Any]): The Mujoco data.
            carry (Any): Carry object.
            backend (ModuleType): The backend (numpy or jax).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: Goal observation and updated carry.
        """
        if backend == np:
            R = np_R
        else:
            R = jnp_R

        # Get trajectory data and state (use passed `traj` so JIT-traced swaps
        # are visible; fall back to self._traj for non-JIT callers).
        traj_data = (traj if traj is not None else env._traj).data
        traj_state = carry.traj_state

        # Get a slice of the trajectory data
        traj_qpos = backend.atleast_2d(traj_data.get_qpos_slice(traj_state.traj_no, traj_state.subtraj_step_no,
                                                                self._n_steps_average, backend))
        traj_qvel = backend.atleast_2d(traj_data.get_qvel_slice(traj_state.traj_no, traj_state.subtraj_step_no,
                                                                self._n_steps_average, backend))

        # Get the average goal over the slice
        traj_free_jnt_qpos = traj_qpos[0, self._free_jnt_qposid]
        traj_free_jnt_qvel = traj_qvel[:, self._free_jnt_qvelid]
        traj_free_jnt_lin_vel = backend.mean(traj_free_jnt_qvel[:, :3], axis=0)
        traj_free_jnt_rot_vel = backend.mean(traj_free_jnt_qvel[:, 3:], axis=0)
        traj_free_jnt_quat = traj_free_jnt_qpos[3:]
        traj_free_jnt_mat = R.from_quat(quat_scalarfirst2scalarlast(traj_free_jnt_quat)).as_matrix()

        # Transform lin and rot vel to local frame
        traj_free_jnt_lin_vel = traj_free_jnt_mat.T @ traj_free_jnt_lin_vel
        traj_free_jnt_rot_vel = traj_free_jnt_mat.T @ traj_free_jnt_rot_vel

        goal = backend.concatenate([traj_free_jnt_lin_vel, traj_free_jnt_rot_vel])

        if self.visualize_goal:
            carry = self.set_visuals(goal, env, model, data, carry, self._root_body_id,
                                     self._free_jnt_qposid, self.visual_geoms_idx, backend)

        return goal, carry

    def is_done(self,
                env: Any,
                model: Union[MjModel, Model],
                data: Union[MjData, Data],
                carry: Any,
                backend: ModuleType,
                traj=None) -> bool:
        """
        Check if the goal is completed.

        Terminates the episode if the number of steps till the end of the trajectory is
        less than the number of steps to average over.
        """
        traj_data = (traj if traj is not None else env._traj).data
        steps_till_end = self._steps_till_end(traj_data, carry.traj_state)
        return steps_till_end < self._n_steps_average

    def mjx_is_done(self,
                    env: Any,
                    model: Union[MjModel, Model],
                    data: Union[MjData, Data],
                    carry: Any,
                    backend: ModuleType,
                    traj=None) -> bool:
        """
        Check if the goal is done (JAX-compatible).
        """
        traj_data = (traj if traj is not None else env._traj).data
        steps_till_end = self._steps_till_end(traj_data, carry.traj_state)
        return jax.lax.cond(steps_till_end < self._n_steps_average, lambda: True, lambda: False)

    def _steps_till_end(self,
                        traj_data: Any,
                        traj_state: Any) -> int:
        """
        Calculate the number of steps till the end of the trajectory.

        Args:
            traj_data (Any): Trajectory data.
            traj_state (Any): Current trajectory state.

        Returns:
            int: Number of steps till the end of the trajectory.
        """
        traj_no = traj_state.traj_no
        subtraj_step_no = traj_state.subtraj_step_no
        current_idx = traj_data.split_points[traj_no] + subtraj_step_no
        idx_of_next_traj = traj_data.split_points[traj_no + 1]
        return idx_of_next_traj - current_idx

    @classmethod
    def get_all_obs_of_type(cls,
                            model: Union[MjModel, Model],
                            data: Union[MjData, Data],
                            ind: Any, backend: ModuleType) -> Any:
        """
        Retrieve all observations of this type.

        Args:
            model (Union[MjModel, Any]): The Mujoco model.
            data (Union[MjData, Any]): The Mujoco data.
            ind (Any): Index for observations.
            backend (ModuleType): The backend (numpy or jax).

        Returns:
            Any: Flattened observations.
        """
        return backend.ravel(data.userdata[ind.GoalTrajRootVelocity])

    @property
    def has_visual(self) -> bool:
        """Check if the goal supports visualization."""
        return True

    @property
    def requires_trajectory(self) -> bool:
        """Check if the goal requires a trajectory."""
        return True

    def set_visual_data(self,
                        data: Union[MjData, Data],
                        backend: ModuleType,
                        traj_goal: Union[np.ndarray, jnp.ndarray]) -> Union[MjData, Any]:
        """
        Set visualization data for the goal.

        Args:
            data (Union[MjData, Data]): The Mujoco data.
            backend (ModuleType): The backend (numpy or jax).
            traj_goal (Union[np.ndarray, jnp.ndarray]): The trajectory goal.

        Returns:
            Union[MjData, Any]: Updated Mujoco data with visualization settings.
        """
        rel_target_arrow_pos = backend.concatenate([traj_goal * self._arrow_to_goal_ratio, jnp.ones(1)])
        abs_target_arrow_pos = ((data.body(self.upper_body_xml_name).xmat.reshape(3, 3) @ rel_target_arrow_pos) +
                                data.body(self.upper_body_xml_name).xpos)
        data.site(self._site_name_keypoint_2).xpos = abs_target_arrow_pos
        return data

    @property
    def dim(self) -> int:
        """Get the dimension of the goal."""
        return 6


class GoalTrajMimic(Goal):
    """
    A class representing a trajectory goal in keypoint space (defined by sites) and joint properties.
    All entities are relative to the root body. This is the typical goal to be used with a DeepMimic-style reward.

    Args:
        info_props (Dict): Information properties required for initialization.
        rel_body_names (List[str]): List of relevant body names. Defaults to None.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, info_props: Dict,
                 rel_body_names: List[str] = None,
                 sites_for_mimic: List[str] = None,
                 main_site_name: str = None,
                 **kwargs):

        self.n_step_lookahead = 1   # todo: implement n_step_lookahead
        # Allow caller to override the env's default sites_for_mimic — passing
        # an empty list disables all site quantities in the goal observation.
        # Mirrors the MimicReward kwarg of the same name.
        self._sites_for_mimic_override = sites_for_mimic
        # Name of the site (within the resolved site list) used as the reference
        # frame for relative quantities. None → use the first site (index 0).
        self._main_site_name = main_site_name
        sites = sites_for_mimic if sites_for_mimic is not None else info_props["sites_for_mimic"]
        n_visual_geoms = len(sites) if \
            ("visualize_goal" in kwargs.keys() and kwargs["visualize_goal"]) else 0
        super().__init__(info_props, n_visual_geoms=n_visual_geoms, **kwargs)

        self.main_body_name = self._info_props["upper_body_xml_name"]
        self._qpos_ind = None
        self._qvel_ind = None
        self._size_additional_observation = None
        self._main_site_id = 0   # index into self._rel_site_ids; resolved in _init_from_mj

        # To be initialized
        self._relevant_body_names = [] if rel_body_names is None else rel_body_names
        self._relevant_body_ids = []
        self._rel_site_ids = []
        self._body_rootid = None
        self._site_bodyid = None
        self._dim = None

    def _resolve_sites_for_mimic(self):
        return (self._sites_for_mimic_override
                if self._sites_for_mimic_override is not None
                else self._info_props["sites_for_mimic"])

    def _init_from_mj(self,
                      env: Any,
                      model: Union[MjModel, Model],
                      data: Union[MjData, Data],
                      current_obs_size: int):
        """
        Initialize the goal from Mujoco model and data.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The Mujoco model.
            data (Union[MjData, Data]): The Mujoco data.
            current_obs_size (int): Current observation size.
        """

        for i in range(model.nbody):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name in self._relevant_body_names and body_name != self.main_body_name and body_name != "world":
                self._relevant_body_ids.append(i)

        sites = self._resolve_sites_for_mimic()
        for name in sites:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            self._rel_site_ids.append(site_id)

        # Resolve the index of the reference site within the list. Defaults to 0.
        if self._main_site_name is not None:
            if self._main_site_name not in sites:
                raise ValueError(
                    f"main_site_name={self._main_site_name!r} not found in "
                    f"sites_for_mimic={sites!r}"
                )
            self._main_site_id = sites.index(self._main_site_name)
        else:
            self._main_site_id = 0

        n_joints = model.njnt
        # Empty list = no reference site, no relative quantities. With ≥1 site,
        # one is the reference and (len−1) others are relative to it.
        n_sites = max(0, len(sites) - 1)
        size_for_joint_pos = (5 + (n_joints - 1)) * self.n_step_lookahead
        size_for_joint_vel = (6 + (n_joints - 1)) * self.n_step_lookahead
        size_for_sites = (3 + 3 + 6) * n_sites * self.n_step_lookahead

        self._dim = (size_for_joint_pos + size_for_joint_vel + size_for_sites) * self.n_step_lookahead
        self._size_additional_observation = size_for_sites

        self._rel_site_ids = np.array(self._rel_site_ids)
        self._body_rootid = model.body_rootid
        self._site_bodyid = model.site_bodyid

        root_free_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, env.root_free_joint_xml_name)
        self._qpos_ind = np.concatenate(
            [mj_jntid2qposid(i, model)[2:] for i in range(model.njnt) if i == root_free_joint_id] +
            [mj_jntid2qposid(i, model) for i in range(model.njnt) if i != root_free_joint_id]
        )
        self._qvel_ind = np.concatenate([mj_jntid2qvelid(i, model) for i in range(model.njnt)])

        self.min = [-np.inf] * self.dim
        self.max = [np.inf] * self.dim
        self.data_type_ind = np.array([i for i in range(data.userdata.size)])
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + self.dim)])

        self._initialized_from_mj = True

    def get_obs_and_update_state(self,
                                 env: Any,
                                 model: Union[MjModel, Model],
                                 data: Union[MjData, Data],
                                 carry: Any,
                                 backend: ModuleType, traj=None) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Get the current goal observation and update the state.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The Mujoco model.
            data (Union[MjData, Data]): The Mujoco data.
            carry (Any): Carry object.
            backend (ModuleType): The backend (numpy or jax).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: Goal observation and updated carry.
        """
        # use passed `traj` so JIT-traced swaps are visible; fall back to self._traj
        traj_data = (traj if traj is not None else env._traj).data
        traj_state = carry.traj_state

        traj_data_single = traj_data.get(traj_state.traj_no, traj_state.subtraj_step_no, backend)

        qpos_traj = traj_data_single.qpos
        qvel_traj = traj_data_single.qvel

        if len(self._rel_site_ids) > 0:
            rel_site_ids = self._rel_site_ids
            rel_body_ids = self._site_bodyid[rel_site_ids]
            site_rpos, site_rangles, site_rvel = calculate_relative_site_quatities(
                traj_data_single, rel_site_ids, rel_body_ids, self._body_rootid, backend,
                main_site_id=self._main_site_id)
            traj_goal_obs = backend.concatenate([
                qpos_traj[self._qpos_ind],
                qvel_traj[self._qvel_ind],
                backend.ravel(site_rpos),
                backend.ravel(site_rangles),
                backend.ravel(site_rvel)
            ])
        else:
            traj_goal_obs = backend.concatenate([
                qpos_traj[self._qpos_ind],
                qvel_traj[self._qvel_ind],
            ])

        if self.visualize_goal:
            carry = self.set_visuals(env, model, data, carry, backend, traj)

        if len(self._rel_site_ids) > 0:
            rel_site_ids = self._rel_site_ids
            rel_body_ids = self._site_bodyid[rel_site_ids]
            site_rpos, site_rangles, site_rvel = calculate_relative_site_quatities(
                data, rel_site_ids, rel_body_ids, self._body_rootid, backend,
                main_site_id=self._main_site_id)

            goal = backend.concatenate([
                backend.ravel(site_rpos),
                backend.ravel(site_rangles),
                backend.ravel(site_rvel),
                backend.ravel(traj_goal_obs)
            ])

            return goal, carry
        else:
            return traj_goal_obs, carry

    @property
    def has_visual(self) -> bool:
        """Check if the goal supports visualization."""
        return True

    @property
    def requires_trajectory(self) -> bool:
        """Check if the goal requires a trajectory."""
        return True

    def set_visuals(self,
                    env: Any,
                    model: Union[MjModel, Model],
                    data: Union[MjData, Data],
                    carry: Any,
                    backend: ModuleType,
                    traj=None) -> Any:
        """
        Set the visualizations for the goal.
        """
        if backend == np:
            R = np_R
        else:
            R = jnp_R

        traj_data = (traj if traj is not None else env._traj).data
        traj_state = carry.traj_state
        user_scene = carry.user_scene
        goal_geoms = user_scene.geoms

        site_xpos = traj_data.get_site_xpos(traj_state.traj_no, traj_state.subtraj_step_no, backend)
        site_xmat = traj_data.get_site_xmat(traj_state.traj_no, traj_state.subtraj_step_no, backend)
        s_ids = jnp.array(self._rel_site_ids)

        qpos_init = traj_data.get_qpos(traj_state.traj_no, traj_state.subtraj_step_no_init, backend)
        type = backend.full(self.n_visual_geoms, int(mujoco.mjtGeom.mjGEOM_BOX), dtype=backend.int32).reshape((-1, 1))
        size = backend.tile(backend.array([0.075, 0.05, 0.025]), (self.n_visual_geoms, 1))
        color = backend.tile(backend.array([0.0, 1.0, 0.0, 1.0]), (self.n_visual_geoms, 1))
        if backend == jnp:
            geom_pos = user_scene.geoms.pos.at[self.visual_geoms_idx].set(site_xpos[s_ids])
            geom_mat = user_scene.geoms.mat.at[self.visual_geoms_idx].set(site_xmat[s_ids])
            geom_type = user_scene.geoms.type.at[self.visual_geoms_idx].set(type)
            geom_size = user_scene.geoms.size.at[self.visual_geoms_idx].set(size)
            geom_rgba = user_scene.geoms.rgba.at[self.visual_geoms_idx].set(color)
            geom_pos = geom_pos.at[:, :2].add(-qpos_init[:2])
        else:
            user_scene.geoms.pos[self.visual_geoms_idx] = site_xpos[s_ids]
            user_scene.geoms.mat[self.visual_geoms_idx] = site_xmat[s_ids]
            user_scene.geoms.type[self.visual_geoms_idx] = type
            user_scene.geoms.size[self.visual_geoms_idx] = size
            user_scene.geoms.rgba[self.visual_geoms_idx] = color
            geom_pos = user_scene.geoms.pos[self.visual_geoms_idx]
            geom_mat = user_scene.geoms.mat[self.visual_geoms_idx]
            geom_type = user_scene.geoms.type[self.visual_geoms_idx]
            geom_rgba = user_scene.geoms.rgba[self.visual_geoms_idx]
            geom_size = user_scene.geoms.size[self.visual_geoms_idx]
            geom_pos[:, :2] -= qpos_init[:2]

        # update carry
        new_user_scene = user_scene.replace(geoms=user_scene.geoms.replace(
            pos=geom_pos, mat=geom_mat, size=geom_size, type=geom_type, rgba=geom_rgba))
        carry = carry.replace(user_scene=new_user_scene)

        return carry

    @property
    def dim(self) -> int:
        """Get the dimension of the goal."""
        return self._dim + self._size_additional_observation


class GoalTrajMimicv2(GoalTrajMimic):

    """
    Equivalent to GoalTrajMimic but with the ability to visualize the goal with the robot's geoms/body.

    ..note:: This class might slows down the simulation. Use it for visualization purposes only.

    Args:
        info_props (Dict): Information properties required for initialization.
        rel_body_names (List[str]): List of relevant body names. Defaults to None.
        target_geom_rgba (Tuple[float, float, float, float]): RGBA values for the target geom.
        Defaults to (0.471, 0.38, 0.812, 0.5).

    """

    def __init__(self, info_props: Dict,
                 rel_body_names: List[str] = None,
                 target_geom_rgba: Tuple[float, float, float, float] = (0.471, 0.38, 0.812, 0.5),
                 **kwargs):

        self.n_step_lookahead = 1
        self._geom_group_to_include = 0
        self._geom_ids_to_exclude = (0,)  # worldbody
        self._target_geom_rgba = target_geom_rgba
        # NOTE: this deliberately calls Goal.__init__ (grandparent), skipping
        # GoalTrajMimic.__init__, so the attributes GoalTrajMimic sets up must be
        # replicated here. v2 does not expose a sites_for_mimic / main_site_name
        # override, so both default to None (i.e. use the env's defaults).
        self._sites_for_mimic_override = None
        self._main_site_name = None
        self._main_site_id = 0   # index into self._rel_site_ids; resolved in _init_from_mj
        super(GoalTrajMimic, self).__init__(info_props, **kwargs)

        self.main_body_name = self._info_props["upper_body_xml_name"]
        self._qpos_ind = None
        self._qvel_ind = None
        self._size_additional_observation = None

        # To be initialized
        self._relevant_body_names = [] if rel_body_names is None else rel_body_names
        self._relevant_body_ids = []
        self._rel_site_ids = []
        self._body_rootid = None
        self._site_bodyid = None
        self._dim = None

    def _init_from_mj(self,
                      env: Any,
                      model: Union[MjModel, Model],
                      data: Union[MjData, Data],
                      current_obs_size: int):
        """
        Initialize the goal from Mujoco model and data.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The Mujoco model.
            data (Union[MjData, Data]): The Mujoco data.
            current_obs_size (int): Current observation size.
        """

        super()._init_from_mj(env, model, data, current_obs_size)

        geom_ids = []
        geom_bodyid = []
        geom_type = []
        geom_size = []
        geom_rgba = []
        geom_dataid = []
        geom_group = []

        for i in range(model.ngeom):
            if i not in self._geom_ids_to_exclude and model.geom_group[i] == self._geom_group_to_include:
                geom_ids.append(i)
                geom_bodyid.append(model.geom_bodyid[i])
                geom_type.append(model.geom_type[i])
                geom_size.append(model.geom_size[i])
                geom_rgba.append(self._target_geom_rgba)
                geom_dataid.append(model.geom_dataid[i])
                geom_group.append(model.geom_group[i])

        self._geom_ids = np.array(geom_ids)
        self._geom_bodyid = np.array(geom_bodyid)
        self._geom_type = np.array(geom_type).reshape(-1, 1)
        self._geom_size = np.array(geom_size)
        self._geom_rgba = np.array(geom_rgba)
        self._geom_dataid = np.array(geom_dataid).reshape(-1, 1)
        self._geom_group = np.array(geom_group).reshape(-1, 1)

        for i, geom in enumerate(env.mjspec.geoms):
            if i not in self._geom_ids_to_exclude and geom.group == self._geom_group_to_include:
                self.n_visual_geoms += 1

    def set_visuals(self,
                    env: Any,
                    model: Union[MjModel, Model],
                    data: Union[MjData, Data],
                    carry: Any,
                    backend: ModuleType,
                    traj=None) -> Any:
        """
        Set the visualizations for the goal.
        """
        if backend == np:
            R = np_R
        else:
            R = jnp_R

        traj_data = (traj if traj is not None else env._traj).data
        traj_state = carry.traj_state
        user_scene = carry.user_scene
        goal_geoms = user_scene.geoms

        qpos_init = traj_data.get_qpos(traj_state.traj_no, traj_state.subtraj_step_no_init, backend)
        qpos = traj_data.get_qpos(traj_state.traj_no, traj_state.subtraj_step_no, backend)
        qvel = traj_data.get_qvel(traj_state.traj_no, traj_state.subtraj_step_no, backend)

        if backend == jnp:

            # calculate geom positions and orientations usinf FK
            qpos = qpos.at[:2].add(-qpos_init[:2])
            data = data.replace(qpos=qpos, qvel=qvel)
            data = mujoco.mjx.kinematics(env.sys, data)
            geom_pos = data.geom_xpos[self._geom_ids]
            geom_mat = data.geom_xmat[self._geom_ids]

            geom_pos = user_scene.geoms.pos.at[self.visual_geoms_idx].set(geom_pos)
            geom_mat = user_scene.geoms.mat.at[self.visual_geoms_idx].set(geom_mat.reshape(-1, 9))
            geom_type = user_scene.geoms.type.at[self.visual_geoms_idx].set(self._geom_type)
            geom_size = user_scene.geoms.size.at[self.visual_geoms_idx].set(self._geom_size)
            geom_rgba = user_scene.geoms.rgba.at[self.visual_geoms_idx].set(self._geom_rgba)
            geom_data = user_scene.geoms.dataid.at[self.visual_geoms_idx].set(self._geom_dataid)
        else:

            # calculate geom positions and orientations usinf FK
            data = deepcopy(data)
            data.qpos = qpos
            data.qpos[:2] -= qpos_init[:2]
            data.qvel = qvel
            mujoco.mj_kinematics(model, data)
            geom_pos = data.geom_xpos[self._geom_ids]
            geom_mat = data.geom_xmat[self._geom_ids]

            user_scene.geoms.pos[self.visual_geoms_idx] = geom_pos
            user_scene.geoms.mat[self.visual_geoms_idx] = geom_mat.reshape(-1, 9)
            user_scene.geoms.type[self.visual_geoms_idx] = self._geom_type
            user_scene.geoms.size[self.visual_geoms_idx] = self._geom_size
            user_scene.geoms.rgba[self.visual_geoms_idx] = self._geom_rgba
            user_scene.geoms.dataid[self.visual_geoms_idx] = self._geom_dataid
            geom_pos = user_scene.geoms.pos[self.visual_geoms_idx]
            geom_mat = user_scene.geoms.mat[self.visual_geoms_idx]
            geom_type = user_scene.geoms.type[self.visual_geoms_idx]
            geom_rgba = user_scene.geoms.rgba[self.visual_geoms_idx]
            geom_size = user_scene.geoms.size[self.visual_geoms_idx]
            geom_data = user_scene.geoms.dataid[self.visual_geoms_idx]

        # update carry
        new_user_scene = user_scene.replace(geoms=user_scene.geoms.replace(
            pos=geom_pos, mat=geom_mat, size=geom_size, type=geom_type, rgba=geom_rgba, dataid=geom_data))
        carry = carry.replace(user_scene=new_user_scene)

        return carry