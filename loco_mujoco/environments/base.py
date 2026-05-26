from typing import List, Tuple, Dict, Any, Union
from types import ModuleType

import warnings
from copy import deepcopy
from tqdm import tqdm
from dataclasses import replace

import jax.random
import jax.numpy as jnp
import numpy as np
from flax import struct
import mujoco
from mujoco import MjSpec, MjModel, MjData
from mujoco.mjx import Model, Data
from scipy.spatial.transform import Rotation as np_R

from loco_mujoco.core.stateful_object import EmptyState
from loco_mujoco.core.observations import Observation
from loco_mujoco.core.mujoco_mjx import Mjx, MjxAdditionalCarry, MjxState
from loco_mujoco.core.visuals import VideoRecorder
from loco_mujoco.trajectory import TrajectoryHandler
from loco_mujoco.core.utils import info_property
from loco_mujoco.core.utils.mujoco import mj_jntname2qposid
from loco_mujoco.trajectory import Trajectory, TrajState, TrajectoryTransitions


@struct.dataclass
class LocoCarry(MjxAdditionalCarry):
    """
    Additional carry for the LocoEnv.

    Args:
        traj_state (TrajState): Trajectory state.
    """
    traj_state: TrajState


class LocoEnv(Mjx):
    """
    Base class for all kinds of environments.

    Args:
        mjx_enabled (bool): If True, the environment is enabled for the MJX backend.

    """

    mjx_enabled = False

    def __init__(self,
                 n_substeps: int = 10,
                 timestep: float = 0.001,
                 default_camera_mode: str = "follow",
                 th_params: Dict = None,
                 traj_params: Dict = None,
                 **core_params):
        """
        Constructor.

        Args:
            th_params (Dict): Dictionary of parameters for the trajectory handler.
            traj_params (Dict): Dictionary of parameters to load trajectories.
            core_params (Dict): Dictionary of other core parameters.

        """

        if "geom_group_visualization_on_startup" not in core_params.keys():
            core_params["geom_group_visualization_on_startup"] = [0, 2]   # enable robot geom [0] and floor visual [2]

        # take over default values
        core_params["n_substeps"] = n_substeps
        core_params["timestep"] = timestep
        core_params["default_camera_mode"] = default_camera_mode

        if self.mjx_enabled:
            # call parent (Mjx) constructor
            super(LocoEnv, self).__init__(**core_params)
        else:
            # call grandparent constructor (Mujoco (CPU) environment)
            super(Mjx, self).__init__(**core_params)

        # dataset dummy
        self._dataset = None

        # setup trajectory
        if traj_params:
            self.th = None
            self.load_trajectory(**traj_params)
        else:
            self.th = None

        self._th_params = th_params

    def load_trajectory(self, traj: Trajectory = None,
                        traj_path: str = None,
                        warn: bool = True) -> None:
        """
        Loads trajectories. If there were trajectories loaded already, this function overrides the latter.

        Args:
            traj (Trajectory): Datastructure containing all trajectory files. If traj_path is specified, this
                should be None.
            traj_path (string): path with the trajectory for the model to follow. Should be a numpy zipped file (.npz)
                with a 'traj_data' array and possibly a 'split_points' array inside. The 'traj_data'
                should be in the shape (joints x observations). If traj_files is specified, this should be None.
            warn (bool): If True, a warning will be raised.

        """

        if self.th is not None and warn:
            warnings.warn("New trajectories loaded, which overrides the old ones.", RuntimeWarning)

        th_params = self._th_params if self._th_params is not None else {}
        self.th = TrajectoryHandler(model=self._model, warn=warn, traj_path=traj_path,
                                    traj=traj, control_dt=self.dt, **th_params)

        if self.th.traj.obs_container is not None:
            assert self.obs_container == self.th.traj.obs_container, \
                ("Observation containers of trajectory and environment do not match. \n"
                 "Please, either load a trajectory with the same observation container or "
                 "set the observation container of the environment to the one of the trajectory.")

        # setup trajectory information in observation_dict, goal and reward if needed
        for obs_entry in self.obs_container.entries():
            obs_entry.init_from_traj(self.th)
        self._goal.init_from_traj(self.th)
        self._terminal_state_handler.init_from_traj(self.th)

    def _is_done(self, obs: np.ndarray,
                 absorbing: bool,
                 info: Dict,
                 data: MjData,
                 carry: LocoCarry) -> bool:
        """
        Check whether the episode is done or not.

        Args:
            obs (np.array): the state of the system.
            absorbing (bool): flag indicating whether the state is absorbing or not.
            info (dict): additional information.
            data (MjData): Mujoco data structure.
            carry (LocoCarry): Additional carry information.

        Returns:
            A boolean flag indicating whether the episode is done or not.

        """
        done = super()._is_done(obs, absorbing, info, data, carry)

        if self._goal.requires_trajectory or self._reward_function.requires_trajectory:
            # either the goal or the reward function requires the trajectory at each step, so we need to check
            # if the end of the trajectory is reached, if so, we set done to True
            traj_state = carry.traj_state
            if traj_state.subtraj_step_no >= self.th.len_trajectory(traj_state.traj_no) - 1:
                done |= True
            else:
                done |= False

            # goals can terminate an episode
            done |= self._goal.is_done(self, self._model, data, carry, np)

        return done

    def _mjx_is_done(self, obs: jnp.ndarray,
                     absorbing: bool,
                     info: Dict,
                     data: Data,
                     carry: LocoCarry) -> bool:
        """
        Determines if the episode is done.

        Args:
            obs (jnp.ndarray): Current observation.
            absorbing (bool): Whether the next state is absorbing.
            info (Dict): Information dictionary.
            data (Data): Mujoco data structure.
            carry (LocoCarry): Additional carry information.

        Returns:
            bool: True if the episode is done, False otherwise.
        """
        done = super()._mjx_is_done(obs, absorbing, info, data, carry)

        if self._goal.requires_trajectory or self._reward_function.requires_trajectory:
            # either the goal or the reward function requires the trajectory at each step, so we need to check
            # if the end of the trajectory is reached, if so, we set done to True
            traj_state = carry.traj_state
            len_traj = self.th.len_trajectory(traj_state.traj_no)
            reached_end_of_traj = jax.lax.cond(jnp.greater_equal(traj_state.subtraj_step_no, len_traj - 1),
                                               lambda: True, lambda: False)
            done = jnp.logical_or(done, reached_end_of_traj)
            # goals can terminate an episode
            done = jnp.logical_or(done, self._goal.mjx_is_done(self, self._model, data, carry, jnp))

        return done

    def _simulation_post_step(self, model: MjModel,
                              data: MjData,
                              carry: LocoCarry) -> Tuple[MjData, LocoCarry]:
        """
        Allows to access and modify the model, data and carry to be modified after the main simulation step.

        Args:
            model (MjModel): Mujoco model.
            data (MjData): Mujoco data structure.
            carry (AdditionalCarry): Additional carry information.

        Returns:
            The updated model, data and carry.

        """
        # call parent to update domain randomization and terrain
        data, carry = super()._simulation_post_step(model, data, carry)

        # update trajectory state
        if self.th is not None:
            carry = self.th.update_state(self, model, data, carry, np)

        return data, carry

    def _mjx_simulation_post_step(self, model: Model,
                                  data: Data,
                                  carry: LocoCarry) -> Tuple[Data, LocoCarry]:
        """
        Applies post-step modifications to the data and carry.

        Args:
            model (Model): Mujoco model.
            data (Data): Mujoco data structure.
            carry (LocoCarry): Additional carry information.

        Returns:
            Tuple[Data, LocoCarry]: Updated data and carry.
        """
        # call parent to update domain randomization and terrain
        data, carry = super()._mjx_simulation_post_step(model, data, carry)

        # update trajectory state
        if self.th is not None:
            carry = self.th.update_state(self, self._model, data, carry, jnp)

        return data, carry

    def create_dataset(self, rng_key=None) -> TrajectoryTransitions:
        """
        Generates a dataset from the specified trajectories without including actions.

        Notes:
        - Observations are created by following steps similar to the `reset()` and `step()` methods.
        - TrajectoryData is used instead of MjData to reduce memory usage. Forward dynamics are applied
          to compute additional entities.
        - Since TrajectoryData only contains very few kinematic properties (to save memory), Mujoco's
          forward dynamics are used to calculate other entities.
        - Kinematic entities derived from mocap data are generally reliable, while dynamics-related
          properties may be less accurate.
        - Observations based on kinematic entities are recommended to ensure realistic datasets.
        - The dataset is built iteratively to compute stateful observations consistently with the
          `reset()` and `step()` methods.

        Args:
            rng_key (jax.random.PRNGKey, optional): A random key for reproducibility. Defaults to None.

        Returns:
        TrajectoryTransitions: A dataclass containing the following fields:
            - observations (array): An array of shape (N_traj x (N_samples_per_traj-1), dim_state).
            - next_observations (array): An array of shape (N_traj x (N_samples_per_traj-1), dim_state).
            - absorbing (array): A flag array of shape (N_traj x (N_samples_per_traj-1)), indicating absorbing states.
            - For non-mocap datasets, actions may also be included.

        Raises:
        ValueError: If no trajectory is provided to the environment.

        """

        if self.th is not None:

            if self.th.traj.transitions is None:

                # create new trajectory and trajectory handler
                info, data = deepcopy(self.th.traj.info), deepcopy(self.th.traj.data)
                info.model = info.model.to_numpy()
                data = data.to_numpy()
                traj = Trajectory(info, data)
                th = TrajectoryHandler(model=self._model, traj=traj, control_dt=self.dt,
                                       random_start=False, fixed_start_conf=(0, 0))

                # set trajectory handler and store old one for later
                orig_th = self.th
                self.th = th

                # get a new model and data
                model = self.mjspec.compile()
                data = mujoco.MjData(model)
                mujoco.mj_resetData(model, data)

                # setup containers for the dataset
                all_observations, all_next_observations, all_dones = [], [], []

                if rng_key is None:
                    rng_key = jax.random.key(0)

                for i in tqdm(range(self.th.n_trajectories), desc="Creating Transition Dataset"):

                    # set configuration to the first state of the current trajectory
                    self.th.fixed_start_conf = (i, 0)

                    # do a reset
                    key, subkey = jax.random.split(rng_key)
                    traj_data_single = self.th.traj.data.get(i, 0, np)  # get first sample
                    carry = self._init_additional_carry(key, model, data, np)

                    # set data from traj_data (qpos and qvel) and forward to calculate other kinematic entities.
                    mujoco.mj_resetData(model, data)
                    data = self.set_sim_state_from_traj_data(data, traj_data_single, carry)
                    mujoco.mj_forward(model, data)

                    data, carry = self._reset_carry(model, data, carry)
                    data, carry = (
                        self.obs_container.reset_state(self, model, data, carry, np))
                    obs, carry = self._create_observation(model, data, carry)
                    info = self._reset_info_dictionary(obs, data, subkey)

                    # initiate obs container
                    observations = [obs]
                    for j in range(1, self.th.len_trajectory(i)):
                        # get next sample and calculate forward dynamics
                        traj_data_single = self.th.traj.data.get(i, j, np)  # get next sample
                        data = self.set_sim_state_from_traj_data(data, traj_data_single, carry)
                        mujoco.mj_forward(model, data)

                        data, carry = self._simulation_post_step(model, data, carry)
                        obs, carry = self._create_observation(model, data, carry)
                        obs, data, info, carry = (
                            self._step_finalize(obs, model, data, info, carry))
                        observations.append(obs)

                        # check if the current state is an absorbing state
                        is_absorbing, carry = self._is_absorbing(obs, info, data, carry)
                        if is_absorbing:
                            warnings.warn("Some of the states in the created dataset are terminal states. "
                                          "This should not happen.")

                    observations = np.vstack(observations)
                    all_observations.append(observations[:-1])
                    all_next_observations.append(observations[1:])
                    dones = np.zeros(observations.shape[0]-1)
                    dones[-1] = 1
                    all_dones.append(dones)

                all_observations = np.concatenate(all_observations).astype(np.float32)
                all_next_observations = np.concatenate(all_next_observations).astype(np.float32)
                all_dones = np.concatenate(all_dones).astype(np.float32)
                all_absorbing = np.zeros_like(all_dones).astype(np.float32)    # assume no absorbing states

                if orig_th.is_numpy:
                    backend = np
                else:
                    backend = jnp

                transitions = TrajectoryTransitions(backend.array(all_observations),
                                                    backend.array(all_next_observations),
                                                    backend.array(all_absorbing),
                                                    backend.array(all_dones))

                self.th = orig_th
                self.th.traj = replace(self.th.traj, transitions=transitions)

            return self.th.traj.transitions

        else:
            raise ValueError("No trajectory was passed to the environment. "
                             "To create a dataset pass a trajectory first.")

    def play_trajectory(self, n_episodes: int = None,
                        n_steps_per_episode: int = None,
                        from_velocity: bool = False,
                        render: bool = True,
                        record: bool = False,
                        recorder_params: Dict = None,
                        callback_class=None,
                        quiet: bool = False,
                        key: jax.random.PRNGKey = None) -> None:
        """
        Plays a demo of the loaded trajectory by forcing the model
        positions to the ones in the trajectories at every step.

        Args:
            n_episodes (int): Number of episode to replay.
            n_steps_per_episode (int): Number of steps to replay per episode.
            from_velocity (bool): If True, the joint positions are calculated from the joint
                velocities in the trajectory.
            render (bool): If True, trajectory will be rendered.
            record (bool): If True, the rendered trajectory will be recorded.
            recorder_params (dict): Dictionary containing the recorder parameters.
            callback_class: Object to be called at each step of the simulation.
            quiet (bool): If True, disable tqdm.
            key (jax.random.PRNGKey): Random key to use for the simulation.

        """

        assert self.th is not None

        if not self.th.is_numpy:
            was_jax = True
            self.th.to_numpy()
        else:
            was_jax = False

        if key is None:
            key = jax.random.key(0)

        if record:
            assert render
            fps = 1/self.dt
            recorder = VideoRecorder(fps=fps, **recorder_params) if recorder_params is not None else\
                VideoRecorder(fps=fps)
        else:
            recorder = None

        is_free_joint_qpos_quat, is_free_joint_qvel_rotvec = [], []
        for i in range(self._model.njnt):
            if self._model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                is_free_joint_qpos_quat.extend([False, False, False, True, True, True, True])
                is_free_joint_qvel_rotvec.extend([False, False, False, True, True, True])
            else:
                is_free_joint_qpos_quat.append(False)
                is_free_joint_qvel_rotvec.append(False)

        is_free_joint_qpos_quat = np.array(is_free_joint_qpos_quat)
        is_free_joint_qvel_rotvec = np.array(is_free_joint_qvel_rotvec)

        key, subkey = jax.random.split(key)
        self.reset(subkey)
        subtraj_step_no = 0
        traj_data_sample = self.th.get_current_traj_data(self._additional_carry, np)

        if render:
            frame = self.render(record)
        else:
            frame = None

        if record:
            recorder(frame)

        highest_int = np.iinfo(np.int32).max
        if n_episodes is None:
            n_episodes = highest_int
        for i in range(n_episodes):
            if n_steps_per_episode is None:
                nspe = (self.th.len_trajectory(self._additional_carry.traj_state.traj_no) -
                        self._additional_carry.traj_state.subtraj_step_no)
            else:
                nspe = n_steps_per_episode

            for j in tqdm(range(nspe), disable=quiet):
                if callback_class is None:
                    self._data = self.set_sim_state_from_traj_data(self._data, traj_data_sample, self._additional_carry)
                    self._model, self._data, self._additional_carry = (
                        self._simulation_pre_step(self._model, self._data, self._additional_carry))
                    mujoco.mj_forward(self._model, self._data)
                    self._data, self._additional_carry = (
                        self._simulation_post_step(self._model, self._data, self._additional_carry))
                else:
                    self._model, self._data, self._additional_carry = (
                        callback_class(self, self._model, self._data, traj_data_sample, self._additional_carry))

                traj_data_sample = self.th.get_current_traj_data(self._additional_carry, np)

                if from_velocity and subtraj_step_no != 0:
                    qpos = self._data.qpos
                    qvel = np.array(traj_data_sample.qvel)

                    qpos_quat = self._data.qpos[is_free_joint_qpos_quat]

                    # Integrate angular velocity using rotation vector approach
                    delta_q = np_R.from_rotvec(self.dt * qvel[is_free_joint_qvel_rotvec])

                    # Apply the incremental rotation to the current quaternion orientation
                    new_qpos = delta_q * np_R.from_quat(qpos_quat)

                    qpos_quat = new_qpos.as_quat()

                    # todo: implement for more than one free joint
                    assert len(qpos_quat) <= 4, "currently only one free joints per scene is supported for replay."

                    qpos[~is_free_joint_qpos_quat] = [qp + self.dt * qv
                                                      for qp, qv in zip(self._data.qpos[~is_free_joint_qpos_quat],
                                                                        qvel[~is_free_joint_qvel_rotvec])]
                    qpos[is_free_joint_qpos_quat] = qpos_quat
                    traj_data_sample = traj_data_sample.replace(qpos=jnp.array(qpos))

                obs, self._additional_carry = self._create_observation(self._model, self._data, self._additional_carry)

                if render:
                    frame = self.render(record)
                else:
                    frame = None

                if record:
                    recorder(frame)

            key, subkey = jax.random.split(key)
            self.reset(subkey)

        self.stop()
        if record:
            recorder.stop()

        if was_jax:
            self.th.to_jax()

    def play_trajectory_from_velocity(self, n_episodes: int = None,
                                      n_steps_per_episode: int = None,
                                      render: bool = True,
                                      record: bool = False,
                                      recorder_params: Dict = None,
                                      callback_class=None,
                                      quiet: bool = False,
                                      key: jax.random.PRNGKey = None) -> None:
        """
        Plays a demo of the loaded trajectory by forcing the model
        positions to the ones calculated from the joint velocities
        in the trajectories at every step. Therefore, the joint positions
        are set from the trajectory in the first step. Afterward, numerical
        integration is used to calculate the next joint positions using
        the joint velocities in the trajectory.

        Args:
            n_episodes (int): Number of episode to replay.
            n_steps_per_episode (int): Number of steps to replay per episode.
            render (bool): If True, trajectory will be rendered.
            record (bool): If True, the rendered trajectory will be recorded.
            recorder_params (dict): Dictionary containing the recorder parameters.
            callback_class (class): Class to be called at each step of the simulation.
            quiet (bool): If True, disable tqdm.
            key (jax.random.PRNGKey): Random key to use for the simulation.

        """
        warnings.warn(
            "play_trajectory_from_velocity() is deprecated and will be removed in future. "
            "Use play_trajectory() and set from_velocity=True instead.",
            category=DeprecationWarning,
            stacklevel=3
        )
        self.play_trajectory(n_episodes, n_steps_per_episode, True, render,
                             record, recorder_params, callback_class, quiet, key)

    def set_sim_state_from_traj_data(self, data, traj_data, carry) -> MjData:
        """
        Sets the Mujoco datastructure to the state specified in the trajectory data.

        Args:
            data (MjData): The Mujoco data structure.
            traj_data: The trajectory data containing state information.
            carry (AdditionalCarry): Additional carry information.

        Returns:
            MjData: The updated Mujoco data structure.
        """
        robot_free_jnt_name = self.root_free_joint_xml_name
        robot_free_jnt_qpos_id_xy = np.array(mj_jntname2qposid(robot_free_jnt_name, self._model))[:2]
        all_free_jnt_qpos_id_xy = self.free_jnt_qpos_id[:, :2].reshape(-1)
        traj_state = carry.traj_state
        # get the initial state of the current trajectory
        traj_data_init = self.th.traj.data.get(traj_state.traj_no, traj_state.subtraj_step_no_init, np)
        # subtract the initial state from the current state
        traj_data.qpos[all_free_jnt_qpos_id_xy] -= traj_data_init.qpos[all_free_jnt_qpos_id_xy]
        return Mjx.set_sim_state_from_traj_data(data, traj_data, carry)

    def mjx_set_sim_state_from_traj_data(self, data, traj_data, carry) -> Data:
        """
        Sets the simulation state from the trajectory data.

        Args:
            data (Data): Current Mujoco data.
            traj_data (TrajectoryData): Data from the trajectory.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            Data: Updated Mujoco data.
        """
        robot_free_jnt_name = self.root_free_joint_xml_name
        robot_free_jnt_qpos_id_xy = np.array(mj_jntname2qposid(robot_free_jnt_name, self._model))[:2]
        all_free_jnt_qpos_id_xy = self.free_jnt_qpos_id[:, :2].reshape(-1)
        traj_state = carry.traj_state
        # get the initial state of the current trajectory
        traj_data_init = self.th.traj.data.get(traj_state.traj_no, traj_state.subtraj_step_no_init, jnp)
        # subtract the initial state from the current state
        traj_data = traj_data.replace(
            qpos=traj_data.qpos.at[all_free_jnt_qpos_id_xy].add(-traj_data_init.qpos[all_free_jnt_qpos_id_xy]))
        return Mjx.mjx_set_sim_state_from_traj_data(data, traj_data, carry)

    def _init_additional_carry(self,
                               key: jax.Array,
                               model: MjModel,
                               data: MjData,
                               backend: ModuleType) -> LocoCarry:
        """
        Initializes the additional carry structure.

        Args:
            key (jax.Array): Random key for JAX operations.
            model (MjModel): The Mujoco model.
            data (MjData): The Mujoco data structure.
            backend (ModuleType): The numerical backend to use (NumPy or JAX NumPy).

        Returns:
            AdditionalCarry: The initialized additional carry structure.
        """

        carry = super()._init_additional_carry(key, model, data, backend)

        key = carry.key
        key, _k = jax.random.split(key)

        # create additional carry
        carry = LocoCarry(
            traj_state=self.th.init_state(self, _k, model, data, backend) if self.th is not None else EmptyState(),
            **vars(carry.replace(key=key)))

        return carry

    def reset(self, key=None) -> np.ndarray:
        """
        Resets the environment to the initial state.

        Args:
            key: Random key. For now, not used in the Mujoco environment.
                Could be used in future to set the numpy seed.

        Returns:
            The initial observation as a numpy array.

        """
        if self.th is not None and not self.th.is_numpy:
            self.th.to_numpy()
        return super().reset(key)

    def mjx_reset(self, key: jax.random.PRNGKey) -> MjxState:
        """
        Resets the environment.

        Args:
            key (jax.random.PRNGKey): Random key for the reset.

        Returns:
            MjxState: The reset state of the environment.

        """
        if self.th is not None and self.th.is_numpy:
            raise ValueError("Trajectory is in numpy format, but your attempting to run the MJX backend. "
                             "Please call the <your_env_name>.th.to_jax() function on your environment first.")
        return super().mjx_reset(key)

    def _reset_carry(self, model: MjModel,
                     data: MjData,
                     carry: LocoCarry) -> Tuple[MjData, LocoCarry]:
        """
        Resets the additional carry. Also allows modification to the MjData.

        Args:
            model (MjModel): Mujoco model.
            data (MjData): Mujoco data structure.
            carry (AdditionalCarry): Additional carry information.

        Returns:
            The updated carry and data.

        """

        # reset trajectory state
        if self.th is not None:
            data, carry = self.th.reset_state(self, self._model, data, carry, np) \
                if self.th is not None else (data, carry)

        # call parent to apply domain randomization and terrain
        data, carry = super()._reset_carry(model, data, carry)

        return data, carry

    def _mjx_reset_carry(self, model: Model,
                         data: Data,
                         carry: MjxAdditionalCarry) -> Tuple[Data, MjxAdditionalCarry]:
        """
        Resets the additional carry and allows modification to the Mujoco data.

        Args:
            model (Model): Mujoco model.
            data (Data): Mujoco data structure.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            Tuple[Data, MjxAdditionalCarry]: Updated data and carry.
        """

        # reset trajectory state
        if self.th is not None:
            data, carry = self.th.reset_state(self, self._model, data, carry, jnp) \
                if self.th is not None else (data, carry)

        # call parent to apply domain randomization and terrain
        data, carry = super()._mjx_reset_carry(model, data, carry)

        return data, carry

    def _get_from_obs(self, obs: Union[np.ndarray, jnp.ndarray], obs_name: str) -> Union[np.ndarray, jnp.ndarray]:
        """
        Returns a part of the observation based on the specified keys.

        Args:
            obs (Union[np.ndarray, jnp.ndarray]): Observation array.
            key str: Key which are used to extract entries from the observation.

        Returns:
            np or jnp array including the parts of the original observation whose
            keys were specified.

        """

        idx = self.obs_container[obs_name].obs_ind
        return obs[idx]

    def _modify_spec_for_mjx(self, spec: MjSpec) -> MjSpec:
        """
        Modifies the specification for the MJX backend if needed.

        Args:
            spec: The specification to modify.

        Returns:
            MjSpec: The modified specification.

        """
        raise NotImplementedError

    @classmethod
    def generate(cls, task: str = None,
                 dataset_type: str = "mocap",
                 debug: bool = False,
                 clip_trajectory_to_joint_ranges: bool = False,
                 **kwargs) -> Any:
        """
        Returns an environment corresponding to the specified task.

        Args:
            task (str): Main task to solve.
            dataset_type (str): "mocap" or "pretrained". "real" uses real motion capture data as the
            reference trajectory. This data does not perfectly match the kinematics
            and dynamics of this environment, hence it is more challenging. "perfect" uses
            a perfect dataset.
            debug (bool): If True, the smaller test datasets are used for debugging purposes.
            clip_trajectory_to_joint_ranges (bool): If True, trajectory is clipped to joint ranges.

        Returns:
            An environment of specified class and task.

        """

        warnings.warn(
            "The methods `LocoEnv.make()` and `LocoEnv.generate()` are deprecated and will be "
            "removed in a future release.\nPlease use the task factory classes instead: "
            "`ImitationFactory` or `RLFactory`.",
            category=DeprecationWarning,
            stacklevel=3
        )

        # import here to avoid circular dependency
        from loco_mujoco.task_factories import ImitationFactory, DefaultDatasetConf
        from loco_mujoco.task_factories import RLFactory

        if task is None:
            return RLFactory.make(cls.__name__, **kwargs)
        else:
            return ImitationFactory.make(cls.__name__, DefaultDatasetConf(task, dataset_type, debug), **kwargs)

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        """
        Returns the default path to the xml file of the environment.
        """
        raise NotImplementedError(f"Please implement the default_xml_file_path property "
                                  f"in the {type(cls).__name__} environment.")

    @info_property
    def root_body_name(self) -> str:
        """
        Returns the name of the root body.

        """
        raise NotImplementedError(f"Please implement the root_body_name "
                                  f"info property in the {type(self).__name__} environment.")

    @info_property
    def root_free_joint_xml_name(self) -> str:
        """
        Returns the name of the root free joint.

        """
        return "root"

    @info_property
    def upper_body_xml_name(self) -> str:
        """
        Returns the name of the upper body.

        """
        raise NotImplementedError(f"Please implement the upper_body_xml_name property "
                                  f"in the {type(self).__name__} environment.")

    @info_property
    def root_height_healthy_range(self) -> Tuple[float, float]:
        """
        Returns the healthy range of the root height. This is only used when HeightBasedTerminalStateHandler is used.

        """
        raise NotImplementedError(f"Please implement the root_height_healthy_range property "
                                  f"in the {type(self).__name__} environment.")

    @info_property
    def foot_geom_names(self) -> List[str]:
        """
        Returns the names of the foot geometries.

        """
        # todo: raise NotImplementedError, once added to all envs
        return []

    @info_property
    def goal_visualization_arrow_offset(self) -> List[float]:
        """
        Returns the offset of the goal visualization arrow. Needs to be a 3D vector for the x, y, z offset.

        """
        return [0.0, 0.0, 0.0]

    @staticmethod
    def _get_observation_specification(spec: MjSpec) -> List[Observation]:
        """
        Returns the observation specification of the environment.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            List[Observation]: A list of observations.
        """
        raise NotImplementedError(f"Please implement the _get_observation_specification method "
                                  f"in the {type(spec).__name__} environment.")

    @staticmethod
    def _get_action_specification(spec: MjSpec) -> List[str]:
        """
        Getter for the action space specification.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            A list of actuator names.

        """
        raise NotImplementedError(f"Please implement the _get_action_specification method "
                                  f"in the {type(spec).__name__} environment.")

    @staticmethod
    def list_registered_loco_mujoco() -> List[str]:
        """
        List registered loco_mujoco environments.

        Returns:
             The list of the registered loco_mujoco environments.

        """
        return list(LocoEnv.registered_envs.keys())

    @staticmethod
    def _delete_from_spec(spec: MjSpec,
                          joints_to_remove: List[str],
                          actuators_to_remove: List[str],
                          equ_constraints_to_remove: List[str]):
        """
        Deletes certain joints, motors and equality constraints from a Mujoco specification.

        Args:
            spec (MjSpec): Mujoco specification.
            joints_to_remove (List[str]): List of joint names to remove.
            actuators_to_remove (List[str]): List of actuator names to remove.
            equ_constraints_to_remove (List[str]): List of equality constraint names to remove.

        Returns:
            Modified Mujoco specification.

        """

        # delete actuators before joints so no actuator references a removed joint
        for actuator in list(spec.actuators):
            if actuator.name in actuators_to_remove:
                spec.delete(actuator)
        for joint in list(spec.joints):
            if joint.name in joints_to_remove:
                spec.delete(joint)
        for equality in list(spec.equalities):
            if equality.name in equ_constraints_to_remove:
                spec.delete(equality)

        return spec
