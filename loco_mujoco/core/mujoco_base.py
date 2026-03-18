import atexit
from copy import deepcopy
from typing import Union, List, Dict, Tuple, Optional
from types import ModuleType
import requests
import webbrowser

from functools import partial
import mujoco
from mujoco import MjSpec, MjModel, MjData
import numpy as np
from flax import struct
import jax
import jax.numpy as jnp

from loco_mujoco.core.utils import Box, MDPInfo, info_property
from loco_mujoco.core.reward.base import Reward
from loco_mujoco.core.observations import (Observation, ObservationType, Goal,
                                           ObservationIndexContainer, ObservationContainer)
from loco_mujoco.core.control_functions import ControlFunction
from loco_mujoco.core.domain_randomizer import DomainRandomizer
from loco_mujoco.core.terrain import Terrain
from loco_mujoco.core.initial_state_handler import InitialStateHandler
from loco_mujoco.core.terminal_state_handler.base import TerminalStateHandler
from loco_mujoco.core.visuals import MjvScene, MujocoViewer
from importlib.resources import files
from loco_mujoco.core.utils.mujoco import mj_jntid2qposid, mj_jntid2qvelid
from loco_mujoco.core.trajectory import TrajectoryModel, TrajectoryData, Trajectory


@struct.dataclass
class AdditionalCarry:
    key: jax.Array
    cur_step_in_episode: int
    last_action: Union[np.ndarray, jax.Array]
    observation_states: struct.PyTreeNode
    reward_state: struct.PyTreeNode
    domain_randomizer_state: struct.PyTreeNode
    terrain_state: struct.PyTreeNode
    init_state_handler_state: struct.PyTreeNode
    terminal_state_handler_state: struct.PyTreeNode
    control_func_state: struct.PyTreeNode
    user_scene: MjvScene


class Mujoco:
    """
    Base class for all Mujoco environments, supporting both CPU-based Mujoco and MjX.

    Attributes:
        registered_envs (dict): A registry of Mujoco environments.

    Args:
        spec (Union[MjSpec, str]): Mujoco Specification. Either a MjSpec object or a path to a Mujoco XML file.
        actuation_spec (List[str]): List of actuator names from the Mujoco XML.
        observation_spec (List[ObservationType]): Specification of the observation space.
        gamma (float): Discount factor for reinforcement learning.
        horizon (int): Maximum number of steps per episode.
        timestep (int, optional): Simulation timestep. If None, it's read from the model.
        n_substeps (int, optional): Number of substeps per simulation step. Defaults to 1.
        model_option_conf (Dict, optional): Changes to apply to the Mujoco option config.
        reward_type (str, optional): The type of reward function. Defaults to "NoReward".
        reward_params (Dict, optional): Parameters for the reward function. Defaults to None.
        goal_type (str, optional): The type of goal specification. Defaults to "NoGoal".
        goal_params (Dict, optional): Parameters for the goal specification. Defaults to None.
        terminal_state_type (str, optional): The type of terminal state handler. Defaults to "NoTerminalStateHandler".
        terminal_state_params (Dict, optional): Parameters for the terminal state handler. Defaults to None.
        domain_randomization_type (str, optional): Type of domain randomization. Defaults to "NoDomainRandomization".
        domain_randomization_params (Dict, optional): Parameters for domain randomization. Defaults to None.
        terrain_type (str, optional): The type of terrain used. Defaults to "StaticTerrain".
        terrain_params (Dict, optional): Parameters for the terrain configuration. Defaults to None.
        init_state_type (str, optional): Initial state handler type. Defaults to "DefaultInitialStateHandler".
        init_state_params (Dict, optional): Parameters for the initial state handler. Defaults to None.
        control_type (str, optional): Type of control function. Defaults to "DefaultControl".
        control_params (Dict, optional): Parameters for the control function. Defaults to None.
        **viewer_params: Additional parameters for the Mujoco viewer.

        """

    registered_envs = dict()

    @staticmethod
    def get_model_path(*relative_parts: str) -> str:
        """
        Return absolute path to a model file inside loco-mujoco-models.
        """
        return str(files("loco_mujoco_models").joinpath(*relative_parts))

    def __init__(self,
                 spec: Union[MjSpec, str],
                 actuation_spec: List[str],
                 observation_spec: List[ObservationType],
                 gamma: float = 0.99,
                 horizon: int = 1000,
                 timestep: int = None,
                 n_substeps: int = 1,
                 model_option_conf: Dict = None,
                 reward_type: str = "NoReward",
                 reward_params: Dict = None,
                 goal_type: str = "NoGoal",
                 goal_params: Dict = None,
                 terminal_state_type: str = "NoTerminalStateHandler",
                 terminal_state_params: Dict = None,
                 domain_randomization_type: str = "NoDomainRandomization",
                 domain_randomization_params: Dict = None,
                 terrain_type: str = "StaticTerrain",
                 terrain_params: Dict = None,
                 init_state_type: str = "DefaultInitialStateHandler",
                 init_state_params: Dict = None,
                 control_type: str = "DefaultControl",
                 control_params: Dict = None,
                 max_num_samples_traj: int = 10_000,
                 **viewer_params):

        # set the timestep if provided, else read it from model
        if timestep is not None:
            if model_option_conf is None:
                model_option_conf = {"timestep": timestep}
            else:
                model_option_conf["timestep"] = timestep

        # load the model, spec and data
        self._init_model, self._model, self._data, self._mjspec = self.load_mujoco(spec, model_option_conf)

        # set some attributes
        self._n_substeps = n_substeps
        self._n_intermediate_steps = 1
        self._viewer_params = viewer_params
        self._viewer = None
        self._obs = None
        self._info = None
        self._additional_carry = None
        self._cur_step_in_episode = 0
        self.action_dim = len(actuation_spec)

        # setup goal
        spec, self._goal = self._setup_goal(spec, goal_type, goal_params)
        if self._goal.requires_spec_modification:
            self._init_model, self._model, self._data, self._mjspec = self.load_mujoco(spec)
        observation_spec.append(self._goal)

        # read the observation space, create a dictionary of observations and goals containing information
        # about each observation's type, indices, min and max values, etc. Additionally, create two dataclasses
        # containing the indices in the datastructure for each observation type (data_indices) and the indices for
        # each observation type in the observation array (obs_indices).
        self.obs_container, self._data_indices, self._obs_indices = (
            self._setup_observations(observation_spec, self._model, self._data))

        # define observation space bounding box
        observation_space = Box(*self._get_obs_limits())

        # read the actuation spec and build the mapping between actions and ids
        self._action_indices = self.get_action_indices(self._model, self._data, actuation_spec)

        # setup control function
        if control_params is None:
            control_params = {}
        self._control_func = ControlFunction.registered[control_type](self, **control_params)
        if self._control_func.run_with_simulation_frequency:
            self._n_intermediate_steps = n_substeps
            self._n_substeps = 1

        # define action space bounding box
        action_space = Box(*self._control_func.action_limits)

        # create the MDP information
        self._mdp_info = MDPInfo(observation_space, action_space, gamma, horizon, self.dt)

        # setup reward function
        reward_cls = Reward.registered[reward_type]
        self._reward_function = reward_cls(self) if reward_params is None else reward_cls(self, **reward_params)

        # setup terrain
        terrain_params = {} if terrain_params is None else terrain_params
        self._terrain = Terrain.registered[terrain_type](self, **terrain_params)
        if self._terrain.requires_spec_modification:
            spec = self._terrain.modify_spec(spec)
            self._init_model, self._model, self._data, self._mjspec = self.load_mujoco(spec)

        # setup domain randomization
        domain_randomization_params = {} if domain_randomization_params is None else domain_randomization_params
        self._domain_randomizer = DomainRandomizer.registered[domain_randomization_type](self, **domain_randomization_params)

        # setup initial state handler
        if init_state_params is None:
            init_state_params = {}
        self._init_state_handler = InitialStateHandler.registered[init_state_type](self, **init_state_params)

        # setup terminal state handler
        if terminal_state_params is None:
            terminal_state_params = {}
        self._terminal_state_handler = TerminalStateHandler.registered[terminal_state_type](self,
                                                                                            **terminal_state_params)

        # set the warning callback to stop the simulation when a mujoco warning occurs
        mujoco.set_mju_user_warning(self.user_warning_raise_exception)

        # path to the video file if one is recorded
        self._video_file_path = None
        self._added_carry_visual_to_user_scene = False
        self._added_carry_visual_start_idx = None
        self._max_num_samples_traj = max_num_samples_traj
        self._traj = None

    def reset(self, key=None, traj: Optional[Trajectory] = None) -> np.ndarray:
        """
        Resets the environment to the initial state.

        Args:
            key: Random key. For now, not used in the Mujoco environment.
                Could be used in future to set the numpy seed.
            traj (Trajectory, optional): Trajectory to use. Falls back to self._traj if not provided.

        Returns:
            The initial observation as a numpy array.

        """
        # resolve trajectory from self._traj if not explicitly provided
        if traj is None:
            traj = self._traj
        # ensure numpy format for CPU reset
        if traj is not None and not isinstance(traj.data.qpos, np.ndarray):
            traj = traj.replace(data=traj.data.to_numpy())

        if key is None:
            key = jax.random.key(0)
        key, subkey = jax.random.split(key)
        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)
        # todo: replace all cur_step_in_episode to use additional info!
        self._additional_carry = self._init_additional_carry(key, self._model, self._data, np, traj)
        self._data, self._additional_carry =\
            self._reset_carry(self._model, self._data, self._additional_carry, traj)

        # reset all stateful entities
        self._data, self._additional_carry = self.obs_container.reset_state(self, self._model, self._data,
                                                                            self._additional_carry, jnp,
                                                                            traj)
        self._obs, self._additional_carry = self._create_observation(self._model, self._data, self._additional_carry)
        self._info = self._reset_info_dictionary(self._obs, self._data, subkey)
        self._cur_step_in_episode = 0
        return self._obs

    def step(self, action: np.ndarray, traj: Optional[Trajectory] = None) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Takes a step in the environment.

        Args:
            action (np.ndarray): The action to take in the environment.
            traj (Trajectory, optional): Trajectory to use. Falls back to self._traj if not provided.

        Returns:
            A tuple containing the next observation, the reward, a flag indicating whether the state is absorbing,
            a flag indicating whether the episode is done, and a dictionary containing additional information.

        """
        # resolve trajectory from self._traj if not explicitly provided
        if traj is None:
            traj = self._traj

        cur_info = self._info.copy()
        carry = self._additional_carry
        carry = carry.replace(last_action=action)

        # preprocess action
        processed_action, carry = self._preprocess_action(action, self._model, self._data, carry, traj)

        # modify data and model during simulation, before main step
        self._model, self._data, carry = self._simulation_pre_step(self._model, self._data, carry, traj)

        for i in range(self._n_intermediate_steps):

            # compute the action at every intermediate step
            ctrl_action, carry = self._compute_action(processed_action, self._model, self._data, carry, traj)

            # main mujoco step, runs the sim for n_substeps
            self._data.ctrl[self._action_indices] = ctrl_action
            mujoco.mj_step(self._model, self._data, self._n_substeps)

        # modify data during simulation, after main step (does nothing by default)
        self._data, carry = self._simulation_post_step(self._model, self._data, carry, traj)

        # create the final observation
        cur_obs, carry = self._create_observation(self._model, self._data, carry)

        # modify obs and data, before stepping in the env (does nothing by default)
        cur_obs, self._data, cur_info, carry = self._step_finalize(cur_obs, self._model, self._data, cur_info, carry, traj)

        # update info (does nothing by default)
        cur_info = self._update_info_dictionary(cur_info, cur_obs, self._data, carry)

        # check if the current state is an absorbing state
        absorbing, carry = self._is_absorbing(cur_obs, cur_info, self._data, carry, traj)

        # calculate the reward
        reward, carry = self._reward(self._obs, action, cur_obs, absorbing, cur_info, self._model, self._data, carry, traj)

        # calculate flag indicating whether this is the last obs before resetting
        done = self._is_done(cur_obs, absorbing, cur_info, self._data, carry, traj)

        self._obs = cur_obs
        self._cur_step_in_episode += 1
        self._additional_carry = carry

        return np.asarray(cur_obs), reward, absorbing, done, cur_info

    def render(self, record: bool = False) -> np.ndarray:
        """
        Renders the current state of the environment.

        Args:
            record (bool): A flag indicating whether to record the video or not.

        Returns:
            The rendered image as a numpy array.

        """

        if self._viewer is None:
            self._viewer = MujocoViewer(self._model, self.dt, record=record, **self._viewer_params)

            headless = self._viewer_params.get("headless", False)

            if not headless:
                # register stop function to be called at exit
                atexit.register(self.stop)

        if self._terrain.is_dynamic:
            terrain_state = self._additional_carry.terrain_state
            assert hasattr(terrain_state, "height_field_raw"), "Terrain state does not have height_field_raw."
            assert self._terrain.hfield_id is not None, "Terrain hfield id is not set."
            # todo: updating hfield buffer at every render is not efficient, rendering will be slow
            hfield_data = np.array(terrain_state.height_field_raw)
            self._model.hfield_data = hfield_data
            self._viewer.upload_hfield(self._model, hfield_id=self._terrain.hfield_id)

        return self._viewer.render(self._data, self._additional_carry, record)

    def stop(self) -> None:
        """
        Stops the rendering of the environment. Disable glfw and delete the viewer.

        """
        if self._viewer is not None:
            self._video_file_path = self._viewer.stop()
            del self._viewer
            self._viewer = None

    @partial(jax.jit, static_argnums=(0,))
    def sample_action_space(self, key: jax.Array) -> jax.Array:
        """
        Samples an action from the action space using jax.

        Args:
            key: Random key.

        Returns:
            The sampled action.

        """

        action_dim = self.info.action_space.shape[0]
        action = jax.random.uniform(key, minval=self.info.action_space.low, maxval=self.info.action_space.high,
                                    shape=(action_dim,))
        return action

    def _is_absorbing(self, obs: np.ndarray,
                      info: Dict,
                      data: MjData,
                      carry: AdditionalCarry,
                      traj: Optional[Trajectory] = None) -> Tuple[bool, AdditionalCarry]:
        """
        Check whether the given state is an absorbing state or not.

        Args:
            obs (np.array): the state of the system.
            info (dict): additional information.
            data (MjData): Mujoco data structure.
            carry (AdditionalCarry): Additional carry information.
            traj (Trajectory, optional): Trajectory to use.

        Returns:
            A boolean flag indicating whether this state is absorbing or not.

        """
        return self._terminal_state_handler.is_absorbing(self, obs, info, data, carry, traj)

    def _is_done(self, obs: np.ndarray,
                 absorbing: bool,
                 info: Dict,
                 data: MjData,
                 carry: AdditionalCarry,
                 traj: Optional[Trajectory] = None) -> bool:
        """
        Check whether the episode is done or not.

        Args:
            obs (np.array): the state of the system.
            absorbing (bool): flag indicating whether the state is absorbing or not.
            info (dict): additional information.
            data (MjData): Mujoco data structure.
            carry (AdditionalCarry): Additional carry information.
            traj (Trajectory, optional): Trajectory to use.

        Returns:
            A boolean flag indicating whether the episode is done or not.

        """
        done = absorbing or (self._cur_step_in_episode >= self.info.horizon)
        return done

    def _reset_carry(self, model: MjModel,
                     data: MjData,
                     carry: AdditionalCarry,
                     traj: Optional[Trajectory] = None) -> Tuple[MjData, AdditionalCarry]:
        """
        Resets the additional carry. Also allows modification to the MjData.

        Args:
            model (MjModel): Mujoco model.
            data (MjData): Mujoco data structure.
            carry (AdditionalCarry): Additional carry information.
            traj (Trajectory, optional): Trajectory to use.

        Returns:
            The updated carry and data.

        """
        data, carry = self._terminal_state_handler.reset(self, model, data, carry, np, traj)
        data, carry = self._terrain.reset(self, model, data, carry, np, traj)
        data, carry = self._init_state_handler.reset(self, model, data, carry, np, traj)
        data, carry = self._domain_randomizer.reset(self, model, data, carry, np, traj)
        data, carry = self._reward_function.reset(self, model, data, carry, np, traj)
        data, carry = self._control_func.reset(self, model, data, carry, np, traj)

        return data, carry

    def _step_finalize(self, obs: np.ndarray,
                       model: MjModel,
                       data: MjData,
                       info: Dict,
                       carry: AdditionalCarry,
                       traj: Optional[Trajectory] = None) -> Tuple[np.ndarray, MjData, Dict, AdditionalCarry]:
        """
        Allows information to be accessed at the end of the step function.

        Args:
            obs (np.array): the state of the system.
            model (MjModel): Mujoco model.
            data (MjData): Mujoco data structure.
            info (dict): additional information.
            carry (AdditionalCarry): Additional carry information.
            traj (Trajectory, optional): Trajectory to use.

        Returns:
            The updated observation, data, info, and carry.

        """

        obs, carry = self._domain_randomizer.update_observation(self, obs, model, data, carry, np, traj)

        return obs, data, info, carry

    def _reset_info_dictionary(self, obs: np.ndarray,
                               data: MjData,
                               key: jax.Array) -> Dict:
        """
        Resets the info dictionary.

        Args:
            obs (np.array): the state of the system.
            data (MjData): Mujoco data structure.
            key: Random key.

        Returns:
            A dictionary containing the updated information.

        """
        return {}

    def _update_info_dictionary(self, info: Dict,
                                obs: np.ndarray,
                                data: MjData,
                                carry: AdditionalCarry) -> Dict:
        """
        Updates the info dictionary.

        Args:
            info (Dict): the current information dictionary.
            obs (np.ndarray): the current state.
            data (MjData): Mujoco data structure.
            carry (AdditionalCarry): Additional carry information.

        Returns:
            The updated information dictionary.

        """
        return info

    def _preprocess_action(self, action: np.ndarray,
                           model: MjModel,
                           data: MjData,
                           carry: AdditionalCarry,
                           traj: Optional[Trajectory] = None) -> Tuple[np.ndarray, AdditionalCarry]:
        """
        Compute a transformation of the action provided to the
        environment. This is done once in the beginning of the step function.

        Args:
            action (np.ndarray): numpy array with the actions
                provided to the environment.
            model (MjModel): Mujoco model.
            data (MjData): Mujoco data structure.
            carry (AdditionalCarry): Additional carry information.
            traj (Trajectory, optional): Trajectory to use.

        Returns:
            The action to be used for the current step and the updated carry.

        """
        action, carry = self._domain_randomizer.update_action(self, action, model, data, carry, np, traj)
        return action, carry

    def _compute_action(self, action: np.ndarray,
                        model: MjModel,
                        data: MjData,
                        carry: AdditionalCarry,
                        traj: Optional[Trajectory] = None) -> Tuple[np.ndarray, AdditionalCarry]:
        """
        Compute a transformation of the action at every intermediate step.
        Useful to add control signals simulated directly in python.

        Args:
            action (np.ndarray): numpy array with the actions
                provided to the environment.
            model (MjModel): Mujoco model.
            data (MjData): Mujoco data structure.
            carry (AdditionalCarry): Additional carry information.
            traj (Trajectory, optional): Trajectory to use.

        Returns:
            The action to be used for the current step and the updated carry.

        """
        action, carry = self._control_func.generate_action(self, action, model, data, carry, np, traj)
        return action, carry

    def _simulation_pre_step(self, model: MjModel,
                             data: MjData,
                             carry: AdditionalCarry,
                             traj: Optional[Trajectory] = None) -> Tuple[MjModel, MjData, AdditionalCarry]:
        """
        Allows to access and modify the model, data and carry to be modified before the main simulation step.
        Here, this function is used to modify the model and data before the simulation step using domain randomization.

        Args:
            model (MjModel): Mujoco model.
            data (MjData): Mujoco data structure.
            carry (AdditionalCarry): Additional carry information.
            traj (Trajectory, optional): Trajectory to use.

        Returns:
            The updated model, data and carry.

        """
        model, data, carry = self._terrain.update(self, model, data, carry, np, traj)
        model, data, carry = self._domain_randomizer.update(self, model, data, carry, np, traj)
        return model, data, carry

    def _simulation_post_step(self, model: MjModel,
                              data: MjData,
                              carry: AdditionalCarry,
                              traj: Optional[Trajectory] = None) -> Tuple[MjData, AdditionalCarry]:
        """
        Allows to access and modify the model, data and carry to be modified after the main simulation step.

        Args:
            model (MjModel): Mujoco model.
            data (MjData): Mujoco data structure.
            carry (AdditionalCarry): Additional carry information.
            traj (Trajectory, optional): Trajectory to use.

        Returns:
            The updated model, data and carry.

        """
        return data, carry

    def set_actuation_spec(self, actuation_spec: List[str]) -> None:
        """
        Updates the action space of the environment.

        Args:
            actuation_spec (List[str]): A list of actuator names.

        """
        self._action_indices = self.get_action_indices(self._model, self._data, actuation_spec)
        self._mdp_info.action_space = Box(*self._control_func.action_limits)
        self.action_dim = len(actuation_spec)

    def set_observation_spec(self, observation_spec: List[ObservationType]) -> None:
        """
        Sets the observation space of the environment including the obs_container.

        Args:
            observation_spec (List[ObservationType]): A list of observation types.

        """
        # update the obs_container and the data_indices and obs_indices
        self.obs_container, self._data_indices, self._obs_indices = (
            self._setup_observations(observation_spec, self._model, self._data))

        # update the observation space
        self._mdp_info.observation_space = Box(*self._get_obs_limits())

    def _setup_observations(self, observation_spec: List[ObservationType],
                            model: MjModel,
                            data: MjData)\
            -> Tuple[ObservationContainer, ObservationIndexContainer, ObservationIndexContainer]:
        """
        Sets up the observation space for the environment. It generates a dictionary containing all the observation
        types and their corresponding information, as well as two dataclasses containing the indices in the
        Mujoco datastructure for each observation type (data_indices) and the indices for each observation type
        in the observation array (obs_indices).

        Args:
            observation_spec (List[ObservationType]): A list of observation types.
            model (MjModel): Mujoco model.
            data (MjData): Mujoco data structure.

        Returns:
            A dictionary containing all the observation types and their corresponding information, as well as two
            dataclasses containing the indices in the Mujoco datastructure for each observation type (data_indices)
            and the indices for each observation type in the observation array (obs_indices).

        """

        # this dict will contain all the observation types and their corresponding information
        obs_container = ObservationContainer()

        # these containers will be used to store the indices of the different observation
        # types in the data structure and in the observation array.
        data_ind = ObservationIndexContainer()
        obs_ind = ObservationIndexContainer()

        i = 0
        # calculate the indices for the different observation types
        for obs in observation_spec:
            # initialize the observation type and get all relevant data indices
            obs.init_from_mj(self, model, data, i, data_ind, obs_ind)
            i += obs.dim
            obs_container[obs.name] = obs

        # lock container to avoid unwanted modifications
        obs_container.lock()

        # convert all lists to numpy arrays
        data_ind.convert_to_numpy()
        obs_ind.convert_to_numpy()

        return obs_container, data_ind, obs_ind

    def _setup_goal(self, spec: MjSpec,
                    goal_type: str,
                    goal_params: Dict) -> Tuple[MjSpec, Goal]:
        """
        Setup the goal.

        Args:
            spec (MjSpec): Specification of the environment.
            goal_type (str): Type of the goal.
            goal_params (Dict): Parameters of the goal.

        Returns:
            MjSpec: Modified specification.
            Goal: Goal

        """
        # collect all info properties of the env (dict all @info_properties decorated function returns)
        info_props = self._get_all_info_properties()

        # get the goal
        goal_cls = Goal.registered[goal_type]
        goal = goal_cls(info_props=info_props) if goal_params is None \
            else goal_cls(info_props=info_props, **goal_params)

        # apply the modification to the spec if needed
        spec = goal.apply_spec_modifications(spec, info_props)

        return spec, goal

    def _reward(self,
                obs: np.ndarray,
                action: np.ndarray,
                next_obs: np.ndarray,
                absorbing: bool,
                info: Dict,
                model: MjModel,
                data: MjData,
                carry: AdditionalCarry,
                traj: Optional[Trajectory] = None) -> Tuple[float, AdditionalCarry]:
        """
        Computes the reward for the current transition.

        Args:
            obs (np.ndarray): The current state of the environment.
            action (np.ndarray): The action taken.
            next_obs (np.ndarray): The resulting next state.
            absorbing (bool): Whether the state is absorbing.
            info (Dict): Additional information dictionary.
            model (MjModel): The Mujoco model.
            data (MjData): The Mujoco data structure.
            carry (AdditionalCarry): Additional carry information.
            traj (Trajectory, optional): Trajectory to use.

        Returns:
            Tuple[float, AdditionalCarry]: The computed reward and updated carry.
        """
        return self._reward_function(obs, action, next_obs, absorbing, info, self, model, data, carry, np, traj)

    def _create_observation(self,
                            model: MjModel,
                            data: MjData,
                            carry: AdditionalCarry) -> Tuple[np.ndarray, AdditionalCarry]:
        """
        Creates the observation array based on the current Mujoco state.

        Args:
            model (MjModel): The Mujoco model.
            data (MjData): The Mujoco data structure.
            carry (AdditionalCarry): Additional carry information.

        Returns:
            Tuple[np.ndarray, AdditionalCarry]: The observation array and updated carry.
        """
        return self._create_observation_compat(model, data, carry, np)

    def _create_observation_compat(self,
                                   model: MjModel,
                                   data: MjData,
                                   carry: AdditionalCarry,
                                   backend: ModuleType,
                                   traj: Optional[Trajectory] = None) -> Tuple[np.ndarray, AdditionalCarry]:
        """
        Creates the observation array by concatenating extracted observations from all types.

        Args:
            model (MjModel): The Mujoco model.
            data (MjData): The Mujoco data structure.
            carry (AdditionalCarry): Additional carry information.
            backend (ModuleType): The numerical backend to use (NumPy or JAX NumPy).
            traj (Trajectory, optional): Trajectory to use.

        Returns:
            Tuple[np.ndarray, AdditionalCarry]: The observation array and updated carry.
        """
        obs_not_stateful = backend.concatenate([
            obs_type.get_all_obs_of_type(self, model, data, self._data_indices, backend)
            for obs_type in ObservationType.list_all_non_stateful()
        ])

        obs_not_stateful = obs_not_stateful[self._obs_indices.concatenated_indices]

        obs_stateful = []
        for obs in self.obs_container.list_all_stateful():
            obs_s, carry = obs.get_obs_and_update_state(self, model, data, carry, backend, traj)
            obs_stateful.append(obs_s)

        obs_stateful = backend.concatenate(obs_stateful)
        stateful_obs_ind = self.obs_container.get_all_stateful_indices()

        # merge the non-stateful and stateful observations and bring them in the order of the observation spec
        merged = backend.empty(len(obs_not_stateful) + len(stateful_obs_ind), dtype=float)
        mask = backend.ones(len(merged), dtype=bool)

        if backend == np:
            mask[stateful_obs_ind] = False
            merged[mask] = obs_not_stateful
            merged[stateful_obs_ind] = obs_stateful
        elif backend == jnp:
            mask = mask.at[stateful_obs_ind].set(False)
            indices = jnp.where(mask, size=len(obs_not_stateful))[0]  # Convert boolean mask to indices
            merged = merged.at[indices].set(obs_not_stateful)
            merged = merged.at[stateful_obs_ind].set(obs_stateful)

        return merged, carry

    @staticmethod
    def set_sim_state_from_traj_data(data: MjData,
                                     traj_data,
                                     carry: AdditionalCarry) -> MjData:
        """
        Sets the Mujoco datastructure to the state specified in the trajectory data.

        Args:
            data (MjData): The Mujoco data structure.
            traj_data: The trajectory data containing state information.
            carry (AdditionalCarry): Additional carry information.

        Returns:
            MjData: The updated Mujoco data structure.
        """
        if traj_data.xpos.size > 0:
            data.xpos = traj_data.xpos
        if traj_data.xquat.size > 0:
            data.xquat = traj_data.xquat
        if traj_data.cvel.size > 0:
            data.cvel = traj_data.cvel
        if traj_data.qpos.size > 0:
            data.qpos = traj_data.qpos
        if traj_data.qvel.size > 0:
            data.qvel = traj_data.qvel

        return data

    def _set_sim_state_from_obs(self,
                                data: MjData,
                                obs: np.ndarray) -> MjData:
        """
        Sets the Mujoco datastructure to the state specified in the observation.

        Args:
            data (MjData): The Mujoco data structure.
            obs (np.ndarray): The observation array.

        Returns:
            MjData: The updated Mujoco data structure.
        """
        data.xpos[self._data_indices.body_xpos, :] = obs[self._obs_indices.body_xpos].reshape(-1, 3)
        data.xquat[self._data_indices.body_xquat, :] = obs[self._obs_indices.body_xquat].reshape(-1, 4)
        data.cvel[self._data_indices.body_cvel, :] = obs[self._obs_indices.body_cvel].reshape(-1, 6)
        data.qpos[self._data_indices.free_joint_qpos] = obs[self._obs_indices.free_joint_qpos]
        data.qvel[self._data_indices.free_joint_qvel] = obs[self._obs_indices.free_joint_qvel]
        data.qpos[self._data_indices.joint_qpos] = obs[self._obs_indices.joint_qpos]
        data.qvel[self._data_indices.joint_qvel] = obs[self._obs_indices.joint_qvel]
        data.site_xpos[self._data_indices.site_xpos, :] = obs[self._obs_indices.site_xpos].reshape(-1, 3)
        data.site_xmat[self._data_indices.site_xmat, :] = obs[self._obs_indices.site_xmat].reshape(-1, 9)

        return data

    def _get_obs_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the minimum and maximum limits for the observation space.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The minimum and maximum values of the observation space.
        """
        obs_min = np.concatenate([np.array(entry.min) for entry in self.obs_container.values()])
        obs_max = np.concatenate([np.array(entry.max) for entry in self.obs_container.values()])
        return obs_min, obs_max

    def _init_additional_carry(self,
                               key: jax.Array,
                               model: MjModel,
                               data: MjData,
                               backend: ModuleType,
                               traj: Optional[Trajectory] = None) -> AdditionalCarry:
        """
        Initializes the additional carry structure.

        Args:
            key (jax.Array): Random key for JAX operations.
            model (MjModel): The Mujoco model.
            data (MjData): The Mujoco data structure.
            backend (ModuleType): The numerical backend to use (NumPy or JAX NumPy).
            traj (Trajectory, optional): Trajectory to use.

        Returns:
            AdditionalCarry: The initialized additional carry structure.
        """
        key, _k1, _k2, _k3, _k4, _k5, _k6, _k7 = jax.random.split(key, 8)

        carry = AdditionalCarry(
            key=key,
            cur_step_in_episode=1,
            last_action=backend.zeros(self.info.action_space.shape),
            observation_states=self.obs_container.init_state(self, _k1, model, data, backend, traj),
            reward_state=self._reward_function.init_state(self, _k2, model, data, backend, traj),
            domain_randomizer_state=self._domain_randomizer.init_state(self, _k3, model, data, backend, traj),
            terrain_state=self._terrain.init_state(self, _k4, model, data, backend, traj),
            init_state_handler_state=self._init_state_handler.init_state(self, _k5, model, data, backend, traj),
            control_func_state=self._control_func.init_state(self, _k6, model, data, backend, traj),
            terminal_state_handler_state=self._terminal_state_handler.init_state(self, _k7, model, data, backend, traj),
            user_scene=MjvScene.init_for_all_stateful_objects(backend))

        return carry

    def get_model(self) -> MjModel:
        """
        Returns a deep copy of the Mujoco model.

        Returns:
            MjModel: A deep copy of the Mujoco model.
        """
        return deepcopy(self._model)

    def get_data(self) -> MjData:
        """
        Returns a deep copy of the Mujoco data structure.

        Returns:
            MjData: A deep copy of the Mujoco data structure.
        """
        return deepcopy(self._data)

    def load_mujoco(self,
                    xml_file: Union[str, MjSpec],
                    model_option_conf: Dict = None) -> Tuple[MjModel, MjModel, MjData, MjSpec]:
        """
        Loads and compiles the Mujoco model from an XML file or MjSpec object.

        Args:
            xml_file (Union[str, MjSpec]): Path to the XML file or a Mujoco specification object.
            model_option_conf (Dict, optional): Configuration options for the model. Defaults to None.

        Returns:
            Tuple[MjModel, MjModel, MjData, MjSpec]: The compiled Mujoco model, duplicate model, data, and spec.
        """
        if isinstance(xml_file, MjSpec):
            if model_option_conf is not None:
                xml_file = self._modify_option_spec(xml_file, model_option_conf)
            model = xml_file.compile()
            spec = xml_file
        elif isinstance(xml_file, str):
            spec = mujoco.MjSpec.from_file(xml_file)
            if model_option_conf is not None:
                spec = self._modify_option_spec(spec, model_option_conf)
            model = spec.compile()
        else:
            raise ValueError(f"Unsupported type for xml_file {type(xml_file)}.")

        data = mujoco.MjData(model)
        return model, model, data, spec

    def reload_mujoco(self, xml_file: Union[str, MjSpec]) -> None:
        """
        Reloads the Mujoco model from the XML file or MjSpec object.

        Args:
            xml_file (Union[str, MjSpec]): Path to the XML file or a Mujoco specification object.
        """
        self._init_model, self._model, self._data, self._mjspec = self.load_mujoco(xml_file)

    @staticmethod
    def _modify_option_spec(spec: MjSpec, option_config: Dict) -> MjSpec:
        """
        Modifies the Mujoco specification options.

        Args:
            spec (MjSpec): The Mujoco specification.
            option_config (Dict): Dictionary of options to modify.

        Returns:
            MjSpec: The modified Mujoco specification.
        """
        if option_config is not None:
            for key, value in option_config.items():
                setattr(spec.option, key, value)
        return spec

    @staticmethod
    def get_action_indices(model: MjModel, data: MjData, actuation_spec: List[str]) -> List[int]:
        """
        Returns the action indices given the MuJoCo model, data, and actuation specification.

        Args:
            model (MjModel): The MuJoCo model.
            data (MjData): The MuJoCo data structure.
            actuation_spec (List[str]): A list specifying the names of the joints
                which should be controllable by the agent. Can be left empty
                when all actuators should be used.

        Returns:
            List[int]: A list of actuator indices.
        """
        if len(actuation_spec) == 0:
            action_indices = [i for i in range(0, len(data.actuator_force))]
        else:
            action_indices = []
            for name in actuation_spec:
                action_indices.append(model.actuator(name).id)
        return action_indices

    def _get_all_info_properties(self) -> Dict:
        """
        Returns all info properties of the environment. (decorated with @info_property)

        Returns:
            Dict: A dictionary containing all info properties of the environment.
        """
        info_props = {}
        for attr_name in dir(self):
            attr_value = getattr(self.__class__, attr_name, None)
            if isinstance(attr_value, property) and getattr(attr_value.fget, '_is_info_property', False):
                info_props[attr_name] = deepcopy(getattr(self, attr_name))
        return info_props

    def create_observation_summary(
            self,
            filename: Optional[str] = None,
            open_in_browser: bool = True
    ) -> str:
        """Generates and uploads an HTML summary of a LocoMuJoCo environment's observations.

        This function creates a main table with all observations and additional group-specific
        tables. Each table shows observation indices, names, types, min/max values, group membership,
        and randomizability flags. Arrays longer than 6 elements are summarized by showing the
        first 3 elements, an ellipsis (`...`), and the last 3 elements.
        Observation indices and filtered indices are shown as `range(...)` if longer than 1.

        The generated HTML is uploaded to https://0x0.st for easy sharing. If `filename` is provided,
        it is also saved locally.

        Args:
            filename (str | None, optional): The filename to save the HTML locally. If None, nothing is saved.
                Defaults to "obs_table.html".
            open_in_browser (bool, optional): Whether to open the uploaded URL in the default browser.
                Defaults to True.

        Returns:
            str: A public URL to the uploaded HTML summary hosted on 0x0.st.

        Raises:
            Exception: If the upload to 0x0.st fails.
        """
        obs_container = self.obs_container
        env_name = self.__class__.__name__

        def summarize_array(arr, max_len=6, force_range: bool = False):
            if arr is None:
                return "None"
            arr_list = list(arr) if isinstance(arr, (np.ndarray, list)) else [arr]
            n = len(arr_list)

            if force_range and n > 1:
                return f"range({arr_list[0]}, {arr_list[0] + n})"

            if n > max_len:
                first_part = ", ".join(map(str, arr_list[:3]))
                last_part = ", ".join(map(str, arr_list[-3:]))
                return f"{first_part}, ..., {last_part}"
            return ", ".join(map(str, arr_list))

        html = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <title>LocoMuJoCo Observation Summary - {env_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; background: #f8f9fa; padding: 2rem; }}
                table {{ border-collapse: collapse; width: 90%; margin: auto; background: white; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                th, td {{ padding: 12px 18px; border: 1px solid #ddd; text-align: left; }}
                th {{ background-color: rgb(9, 16, 61); color: white; }}
                tr:hover {{ background-color: #f1f1f1; }}
                h2 {{ text-align: center; }}
            </style>
        </head>
        <body>
            <h2>LocoMuJoCo Observation Summary – {env_name}</h2>
            <table>
                <tr>
                    <th>Obs Indices</th>
                    <th>Obs Name</th>
                    <th>Obs Type</th>
                    <th>Min / Max</th>
                    <th>Group(s)</th>
                    <th>Randomizable</th>
                </tr>
        """

        for obs in obs_container.values():
            obs_ind = summarize_array(obs.obs_ind, force_range=True)
            name = obs.name
            obs_type = obs.__class__.__name__

            if obs.min is not None and obs.max is not None:
                min_vals = summarize_array(np.round(obs.min, 3))
                max_vals = summarize_array(np.round(obs.max, 3))
                minmax = f"{min_vals} / {max_vals}"
            else:
                minmax = "None"

            groups = "None"
            if obs.group:
                filtered = [g for g in obs.group if g is not None]
                groups = ", ".join(filtered) if filtered else "None"

            rand = str(obs.allow_randomization)

            html += f"""
                <tr>
                    <td>{obs_ind}</td>
                    <td>{name}</td>
                    <td>{obs_type}</td>
                    <td>{minmax}</td>
                    <td>{groups}</td>
                    <td>{rand}</td>
                </tr>
            """

        html += "</table>"

        # Additional tables per group
        all_groups = {g for obs in obs_container.values() if obs.group for g in obs.group if g is not None}

        for group in sorted(all_groups):
            filtered_obs = [obs for obs in obs_container.values() if group in (obs.group or [])]

            html += f"""
                <h2 style="text-align:center;">Group: {group}</h2>
                <table>
                    <tr>
                        <th>Obs Indices</th>
                        <th>Filtered Obs Indices</th>
                        <th>Obs Name</th>
                        <th>Obs Type</th>
                        <th>Min / Max</th>
                        <th>Group(s)</th>
                        <th>Randomizable</th>
                    </tr>
            """

            offset = 0
            for obs in filtered_obs:
                obs_ind = summarize_array(obs.obs_ind, force_range=True)
                name = obs.name
                obs_type = obs.__class__.__name__

                if obs.min is not None and obs.max is not None:
                    min_vals = summarize_array(np.round(obs.min, 3))
                    max_vals = summarize_array(np.round(obs.max, 3))
                    minmax = f"{min_vals} / {max_vals}"
                else:
                    minmax = "None"

                groups = "None"
                if obs.group:
                    filtered_group = [g for g in obs.group if g is not None]
                    groups = ", ".join(filtered_group) if filtered_group else "None"

                rand = str(obs.allow_randomization)

                dim = len(obs.obs_ind) if obs.obs_ind is not None else 0
                filtered_indices = summarize_array(range(offset, offset + dim), force_range=True)
                offset += dim

                html += f"""
                    <tr>
                        <td>{obs_ind}</td>
                        <td>{filtered_indices}</td>
                        <td>{name}</td>
                        <td>{obs_type}</td>
                        <td>{minmax}</td>
                        <td>{groups}</td>
                        <td>{rand}</td>
                    </tr>
                """

            html += "</table>"

        html += "</body></html>"

        # Save to file if specified
        if filename:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(html)

        # Upload HTML
        files = {"file": ("file.html", html, "text/html")}
        headers = {"User-Agent": "curl/7.68.0"}
        res = requests.post("https://0x0.st", files=files, headers=headers)

        if res.status_code == 200:
            url = res.text.strip()
            print(f"✅ Uploaded Observation summary to: {url}")
            if open_in_browser:
                webbrowser.open(url)
            return url
        else:
            raise Exception(f"❌ Upload failed: {res.status_code} - {res.text}")

    @staticmethod
    def parse_observation_spec(obs_spec: List[Dict]) -> List[ObservationType]:
        """
        Parse the observation specification.

        Args:
            obs_spec (List[Dict]): List of observation specifications. Each observation specification
                consists of an ObservationType object.

        Returns:
            List[ObservationType]: List of observation types.

        """
        observation_spec = []
        for obs in obs_spec:
            if isinstance(obs, Observation):
                observation_spec.append(obs)
            else:
                obs_type = ObservationType.get(obs["type"])
                # all other elements in dict are params
                obs_params = {k: v for k, v in obs.items() if k != "type"}
                observation_spec.append(obs_type(**obs_params))
        return observation_spec

    @property
    def model(self) -> MjModel:
        """
        Returns the Mujoco model.

        Returns:
            MjModel: The Mujoco model.
        """
        return self._model

    @property
    def data(self) -> MjData:
        """
        Returns the Mujoco data structure.

        Returns:
            MjData: The Mujoco data structure.
        """
        return self._data

    @property
    def mjspec(self) -> MjSpec:
        """
        Returns the Mujoco specification.

        Returns:
            MjSpec: The Mujoco specification.
        """
        return self._mjspec

    @property
    def free_jnt_qpos_id(self):
        """
        Get the qpos index of free joints

        Returns:
            np.ndarray: The qpos index of free joints with shape (n_free_joints, 7)

        """
        free_jnt_qpos_id = np.concatenate([mj_jntid2qposid(i, self._model)
                                           for i in range(self._model.njnt)
                                           if self._model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]).reshape(-1, 7)
        return free_jnt_qpos_id

    @property
    def free_jnt_qvel_id(self):
        """
        Get the qvel index of free joints

        Returns:
            np.ndarray: The qvel index of free joints with shape (n_free_joints, 6)

        """
        free_jnt_qvel_id = np.concatenate([mj_jntid2qvelid(i, self._model)
                                           for i in range(self._model.njnt)
                                           if self._model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]).reshape(-1, 6)
        return free_jnt_qvel_id

    @property
    def cur_step_in_episode(self) -> int:
        """
        Returns the current step in the episode.

        Returns:
            int: The current step in the episode.
        """
        return self._cur_step_in_episode

    @property
    def mdp_info(self) -> MDPInfo:
        """
        Returns the MDP information.

        Returns:
            MDPInfo: The MDP information.
        """
        return self._mdp_info

    @staticmethod
    def user_warning_raise_exception(warning: str) -> None:
        """
        Detects warnings in Mujoco and raises the respective exception.

        Args:
            warning (str): Mujoco warning message.

        Raises:
            RuntimeError: If a Mujoco warning is detected.
        """
        if 'Pre-allocated constraint buffer is full' in warning:
            raise RuntimeError(warning + ' Increase njmax in mujoco XML')
        elif 'Pre-allocated contact buffer is full' in warning:
            raise RuntimeError(warning + ' Increase njconmax in mujoco XML')
        elif 'Unknown warning type' in warning:
            raise RuntimeError(warning + ' Check for NaN in simulation.')
        else:
            raise RuntimeError('Got MuJoCo Warning: ' + warning)

    @property
    def video_file_path(self) -> str:
        """
        Returns the path to the recorded video file if it exists.

        Returns:
            str: The path to the recorded video file.
        """
        return self._video_file_path

    @property
    def info(self) -> MDPInfo:
        """
        Returns an object containing the info of the environment.

        Returns:
            MDPInfo: The info of the environment.
        """
        return self._mdp_info

    @property
    def viewer(self) -> MujocoViewer:
        """
        Returns the Mujoco viewer.

        Returns:
            MujocoViewer: The Mujoco viewer.
        """
        return self._viewer

    @info_property
    def simulation_dt(self) -> float:
        """
        Returns the simulation timestep.

        Returns:
            float: The simulation timestep.
        """
        return self._model.opt.timestep

    @info_property
    def dt(self) -> float:
        """
        Returns the effective timestep considering intermediate steps and substeps.

        Returns:
            float: The effective timestep.
        """
        return self.simulation_dt * self._n_intermediate_steps * self._n_substeps

    @classmethod
    def register(cls) -> None:
        """
        Register an environment in the environment list.
        """
        env_name = cls.__name__

        if env_name not in Mujoco.registered_envs:
            Mujoco.registered_envs[env_name] = cls

    @staticmethod
    def list_registered() -> List[str]:
        """
        List registered environments.

        Returns:
            List[str]: The list of the registered environments.
        """
        return list(Mujoco.registered_envs.keys())

    @staticmethod
    def make(env_name: str, *args, **kwargs) -> 'Mujoco':
        """
        Generate an environment given an environment name and parameters.
        The environment is created using the generate method, if available. Otherwise, the constructor is used.
        The generate method has a simpler interface than the constructor, making it easier to generate a standard
        version of the environment. If the environment name contains a '.' separator, the string is split, the first
        element is used to select the environment and the other elements are passed as positional parameters.

        Args:
            env_name (str): Name of the environment.
            *args: Positional arguments to be provided to the environment generator.
            **kwargs: Keyword arguments to be provided to the environment generator.

        Returns:
            Mujoco: An instance of the constructed environment.
        """
        if '.' in env_name:
            env_data = env_name.split('.')
            env_name = env_data[0]
            args = env_data[1:] + list(args)

        env = Mujoco.registered_envs[env_name]

        if hasattr(env, 'generate'):
            return env.generate(*args, **kwargs)
        else:
            return env(*args, **kwargs)

