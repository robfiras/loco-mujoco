"""
Base class of all LocoMuJoCo environments.

:class:`LocoEnv` extends the simulation classes in :mod:`loco_mujoco.core` with everything related to reference
trajectories: loading and processing a trajectory, tracking the position within it during an episode, replaying
it, and converting it to a dataset of transitions.
"""

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
from loco_mujoco.core.trajectory import TrajectoryHandler, FixedStartTrajectoryHandler
from loco_mujoco.core.utils import info_property
from loco_mujoco.core.utils.mujoco import mj_jntname2qposid
from loco_mujoco.core.trajectory import Trajectory, TrajState, TrajectoryTransitions


@struct.dataclass
class LocoCarry(MjxAdditionalCarry):
    """
    Additional carry of :class:`LocoEnv`.

    Extends :class:`~loco_mujoco.core.mujoco_mjx.MjxAdditionalCarry` by the state of the trajectory handler.

    Attributes:
        traj_state (TrajState): State of the trajectory handler (current trajectory and step within it), or an
            empty state if the environment has no trajectory handler.

    """
    traj_state: TrajState


class LocoEnv(Mjx):
    """
    Base class of all LocoMuJoCo environments.

    The class adds reference trajectories to the simulation: a trajectory can be loaded with
    :meth:`process_trajectory`, which also creates the trajectory handler ``th`` that selects the trajectory and the
    starting point of an episode and advances the trajectory state stored in the carry (:class:`LocoCarry`) at every
    step. On top of that, a loaded trajectory can be replayed (:meth:`play_trajectory`) and converted to a dataset of
    transitions (:meth:`create_dataset`). Episodes end when the end of the trajectory is reached, if the goal or the
    reward depends on the trajectory.

    Every environment exists for both simulation backends. A robot class deriving from this class (e.g.
    ``UnitreeH1``) runs on the CPU with MuJoCo; its counterpart with the prefix ``Mjx`` sets ``mjx_enabled = True``
    and additionally provides the functional, JAX-based interface of :class:`~loco_mujoco.core.mujoco_mjx.Mjx`.
    Subclasses provide the model through the observation and action specification and describe the robot through
    info properties such as :attr:`root_body_name` or :attr:`root_height_healthy_range`.

    Environments are usually not constructed directly, but through
    :class:`~loco_mujoco.task_factories.imitation_factory.ImitationFactory` or
    :class:`~loco_mujoco.task_factories.rl_factory.RLFactory`.

    Attributes:
        mjx_enabled (bool): If True, the environment is enabled for the MJX backend.
        th (TrajectoryHandler): The trajectory handler, or ``None`` as long as no trajectory has been loaded.

    """

    mjx_enabled = False

    def __init__(self,
                 n_substeps: int = 10,
                 timestep: float = 0.001,
                 default_camera_mode: str = "follow",
                 th_params: Dict = None,
                 traj_params: Dict = None,
                 recenter_root_xy: Union[bool, None] = None,
                 **core_params):
        """
        Constructor.

        Args:
            n_substeps (int): Number of simulation steps per environment (control) step.
            timestep (float): Simulation time step in seconds. The control time step is ``n_substeps * timestep``.
            default_camera_mode (str): Camera mode the viewer starts with.
            th_params (Dict): Parameters for the trajectory handler: ``name``, a registered TrajectoryHandler
                (default "RandomStartTrajectoryHandler"), and the keyword arguments of that class.
            traj_params (Dict): Dictionary of parameters to load a trajectory at construction: either ``traj``
                (a :class:`~loco_mujoco.core.trajectory.dataclasses.Trajectory`) or ``traj_path`` (path to a saved
                trajectory, which takes precedence).
            recenter_root_xy (Union[bool, None]): If True, episodes start at the origin in xy (or at the env's
                origin on terrains that place envs). If False, they start at the xy of the
                trajectory, so that the robot and the trajectory share a world frame. None (default): False on
                terrains built around the dataset's clips (``trajectories_in_world_frame``, e.g. HeightmapTerrain),
                True otherwise.
            **core_params: Other parameters passed to the constructor of the simulation base class
                (:class:`~loco_mujoco.core.mujoco_mjx.Mjx` if ``mjx_enabled``, otherwise
                :class:`~loco_mujoco.core.mujoco_base.Mujoco`).

        Raises:
            ValueError: If ``recenter_root_xy`` contradicts the way the terrain places the robot.

        """

        # resolved against the terrain once it exists (see _resolve_recenter_root_xy)
        self._recenter_root_xy = True if recenter_root_xy is None else recenter_root_xy

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

        self._recenter_root_xy = self._resolve_recenter_root_xy(recenter_root_xy)

        # dataset dummy
        self._dataset = None

        self._th_params = th_params
        self.th = None

        # setup trajectory
        if traj_params:
            traj = traj_params.get("traj")
            traj_path = traj_params.get("traj_path")
            if traj_path is not None:
                traj = Trajectory.load(traj_path)
            if traj is not None:
                self.process_trajectory(traj)

    def _resolve_recenter_root_xy(self, recenter_root_xy: Union[bool, None]) -> bool:
        """
        Whether episodes are recentered in xy, given the constructor argument and the terrain.

        Args:
            recenter_root_xy (Union[bool, None]): The constructor argument.

        Returns:
            bool: True if trajectories are moved to the origin (plus the env's terrain origin) on reset.

        Raises:
            ValueError: If the argument contradicts the terrain's placement of the robot.
        """
        terrain = self._terrain
        if recenter_root_xy is None:
            return not terrain.trajectories_in_world_frame
        if recenter_root_xy and terrain.trajectories_in_world_frame:
            raise ValueError(f"{terrain.get_name()} is built around the dataset's clips, so episodes have to start "
                             f"where their clip is: use recenter_root_xy=False (or leave it at None).")
        if not recenter_root_xy and terrain.has_env_origins:
            raise ValueError(f"{terrain.get_name()} places every env at its own origin, which needs recentered "
                             f"trajectories: use recenter_root_xy=True (or leave it at None).")
        return recenter_root_xy

    def get_traj_offset_xy(self, carry: Any, backend: ModuleType, traj: Trajectory = None):
        """
        Get the offset from the current trajectory's xy to the world xy of the env.

        The reference root is at ``trajectory xy + offset`` in the world. The offset is the env's terrain origin minus
        the trajectory's start when episodes are recentered, and zero otherwise.

        Args:
            carry (Any): Carry instance with the trajectory state.
            backend (ModuleType): Backend module used for computation (e.g., numpy or jax.numpy).
            traj (Trajectory): Trajectory to use. Defaults to the env's trajectory.

        Returns:
            Union[np.ndarray, jnp.ndarray]: The xy offset, shape ``(2,)``.

        """
        if not self._recenter_root_xy:
            return backend.zeros(2)
        traj_data = (traj if traj is not None else self._traj).data
        traj_state = carry.traj_state
        qpos_init = traj_data.get_qpos(traj_state.traj_no, traj_state.subtraj_step_no_init, backend)
        root_xy = np.array(mj_jntname2qposid(self.root_free_joint_xml_name, self._model))[:2]
        return self._terrain.get_env_origin_xy(carry, backend) - qpos_init[root_xy]

    def process_trajectory(self, traj: Trajectory) -> Trajectory:
        """
        Process a trajectory and load it into the environment.

        The trajectory is filtered and extended to match the model of the environment and interpolated to the control
        frequency of the environment if needed. On the first call, the trajectory handler is created from the
        ``th_params`` given at construction; on later calls, the new trajectory has to have the same structure (joints,
        bodies, sites) as the one the handler was created with. The trajectory data is padded to the fixed shapes of the
        handler, so that trajectories can be swapped without recompilation, and stored in the environment.

        Args:
            traj (Trajectory): The trajectory to process.

        Returns:
            Trajectory: The processed (and padded) trajectory.

        Raises:
            ValueError: If the trajectory handler named in ``th_params`` is not registered.

        """
        from loco_mujoco.core.trajectory import interpolate_trajectories

        # filter and extend to match current model
        traj_data, traj_info = TrajectoryHandler.filter_and_extend(traj.data, traj.info, self._model)

        # interpolate if needed
        traj_dt = 1 / traj_info.frequency
        if traj_dt != self.dt:
            traj_data, traj_info = interpolate_trajectories(traj_data, traj_info, 1.0 / self.dt)

        processed_traj = traj.replace(data=traj_data, info=traj_info)

        # create or validate TrajectoryHandler using _th_params
        th_params = _resolve_legacy_th_params(dict(self._th_params) if self._th_params is not None else {})
        th_name = th_params.pop("name", "RandomStartTrajectoryHandler")

        if self.th is None:
            if th_name not in TrajectoryHandler.registered:
                raise ValueError(
                    f"Unknown TrajectoryHandler '{th_name}'. "
                    f"Registered: {TrajectoryHandler.list_registered()}"
                )
            th_cls = TrajectoryHandler.registered[th_name]
            self.th = th_cls(traj_info, control_dt=self.dt, **th_params)
        else:
            assert self.th.traj_info == traj_info, (
                "New trajectory has incompatible structure with existing trajectory handler. "
                "traj_info must match (same joints, bodies, sites, etc.).")

        # pad traj_data to fixed shapes for JIT shape stability when swapping trajectories
        padded_data = traj_data.pad(self.th.max_n_samples, self.th.max_n_trajs)
        processed_traj = processed_traj.replace(data=padded_data)

        self._traj = processed_traj

        if processed_traj.obs_container is not None:
            assert self.obs_container == processed_traj.obs_container, \
                ("Observation containers of trajectory and environment do not match. \n"
                 "Please, either load a trajectory with the same observation container or "
                 "set the observation container of the environment to the one of the trajectory.")

        return processed_traj

    def _is_done(self, obs: np.ndarray,
                 absorbing: bool,
                 info: Dict,
                 data: MjData,
                 carry: LocoCarry,
                 traj=None) -> bool:
        """
        Check whether the episode is done or not.

        Args:
            obs (np.array): the state of the system.
            absorbing (bool): flag indicating whether the state is absorbing or not.
            info (dict): additional information.
            data (MjData): Mujoco data structure.
            carry (LocoCarry): Additional carry information.
            traj: Trajectory to use (optional).

        Returns:
            A boolean flag indicating whether the episode is done or not.

        """
        done = super()._is_done(obs, absorbing, info, data, carry, traj)

        if self._goal.requires_trajectory or self._reward_function.requires_trajectory:
            # either the goal or the reward function requires the trajectory at each step, so we need to check
            # if the end of the trajectory is reached, if so, we set done to True
            traj_state = carry.traj_state
            if traj_state.subtraj_step_no >= self.th.len_trajectory(traj_state.traj_no, traj.data) - 1:
                done |= True
            else:
                done |= False

            # goals can terminate an episode
            done |= self._goal.is_done(self, self._model, data, carry, np, traj)

        return done

    def _mjx_is_done(self, obs: jnp.ndarray,
                     absorbing: bool,
                     info: Dict,
                     data: Data,
                     carry: LocoCarry,
                     traj=None) -> bool:
        """
        Determines if the episode is done.

        Args:
            obs (jnp.ndarray): Current observation.
            absorbing (bool): Whether the next state is absorbing.
            info (Dict): Information dictionary.
            data (Data): Mujoco data structure.
            carry (LocoCarry): Additional carry information.
            traj: Trajectory to use (optional).

        Returns:
            bool: True if the episode is done, False otherwise.
        """
        done = super()._mjx_is_done(obs, absorbing, info, data, carry, traj)

        if self._goal.requires_trajectory or self._reward_function.requires_trajectory:
            # either the goal or the reward function requires the trajectory at each step, so we need to check
            # if the end of the trajectory is reached, if so, we set done to True
            traj_state = carry.traj_state
            len_traj = self.th.len_trajectory(traj_state.traj_no, traj.data)
            reached_end_of_traj = jax.lax.cond(jnp.greater_equal(traj_state.subtraj_step_no, len_traj - 1),
                                               lambda: True, lambda: False)
            done = jnp.logical_or(done, reached_end_of_traj)
            # goals can terminate an episode
            done = jnp.logical_or(done, self._goal.mjx_is_done(self, self._model, data, carry, jnp, traj))

        return done

    def _simulation_post_step(self, model: MjModel,
                              data: MjData,
                              carry: LocoCarry,
                              traj=None) -> Tuple[MjData, LocoCarry]:
        """
        Update the data and the carry after the main simulation step, including the trajectory state.

        Args:
            model (MjModel): Mujoco model.
            data (MjData): Mujoco data structure.
            carry (LocoCarry): Additional carry information.
            traj: Trajectory to use (optional).

        Returns:
            Tuple[MjData, LocoCarry]: The updated data and carry.

        """
        # call parent to update domain randomization and terrain
        data, carry = super()._simulation_post_step(model, data, carry, traj)

        # update trajectory state
        if self.th is not None:
            _traj = traj if traj is not None else self._traj
            carry = self.th.update_state(self, model, data, carry, np,
                                         traj_model=_traj.info.model if _traj is not None else None,
                                         traj_data=_traj.data if _traj is not None else None)

        return data, carry

    def _mjx_simulation_post_step(self, model: Model,
                                  data: Data,
                                  carry: LocoCarry,
                                  traj=None) -> Tuple[Data, LocoCarry]:
        """
        Applies post-step modifications to the data and carry.

        Args:
            model (Model): Mujoco model.
            data (Data): Mujoco data structure.
            carry (LocoCarry): Additional carry information.
            traj: Trajectory to use (optional).

        Returns:
            Tuple[Data, LocoCarry]: Updated data and carry.
        """
        # call parent to update domain randomization and terrain
        data, carry = super()._mjx_simulation_post_step(model, data, carry, traj)

        # update trajectory state
        if self.th is not None:
            _traj = traj if traj is not None else self._traj
            carry = self.th.update_state(self, self._model, data, carry, jnp,
                                         traj_model=_traj.info.model if _traj is not None else None,
                                         traj_data=_traj.data if _traj is not None else None)

        return data, carry

    def create_dataset(self, rng_key=None) -> TrajectoryTransitions:
        """
        Generate a dataset of transitions from the loaded trajectories, without actions.

        The result is cached in the trajectory, so the dataset is only built on the first call.

        Notes:

        - Observations are created by following steps similar to the ``reset()`` and ``step()`` methods.
        - The trajectory only contains a few kinematic quantities (to save memory), so Mujoco's forward kinematics and
          dynamics are used to compute all other quantities.
        - Kinematic quantities derived from mocap data are generally reliable, while dynamics-related
          quantities may be less accurate. Observations based on kinematic quantities are therefore recommended.
        - The dataset is built iteratively to compute stateful observations consistently with the
          ``reset()`` and ``step()`` methods.

        Args:
            rng_key (jax.random.PRNGKey, optional): A random key for reproducibility. Defaults to ``jax.random.key(0)``.

        Returns:
            TrajectoryTransitions: A dataclass with ``N`` transitions, where ``N`` is the sum of ``len - 1`` over all
            trajectories, containing the fields:

            - ``observations``: array of shape ``(N, obs_dim)``.
            - ``next_observations``: array of shape ``(N, obs_dim)``.
            - ``absorbings``: array of shape ``(N,)``, all zero, since trajectories are assumed to contain no
              absorbing states.
            - ``dones``: array of shape ``(N,)``, one for the last transition of each trajectory.

        Raises:
            ValueError: If no trajectory is provided to the environment.

        """

        if self.th is not None:

            if self._traj.transitions is None:

                # create new trajectory and trajectory handler
                info = deepcopy(self._traj.info)
                data = deepcopy(self._traj.data)
                data = data.to_numpy()
                tmp_traj = Trajectory(info=info, data=data)
                tmp_th = FixedStartTrajectoryHandler(info, control_dt=self.dt, start_conf=(0, 0))

                # set trajectory handler and store old ones for later
                orig_th = self.th
                orig_traj = self._traj
                self.th = tmp_th
                self._traj = tmp_traj

                # get a new model and data
                model = self.mjspec.compile()
                data = mujoco.MjData(model)
                mujoco.mj_resetData(model, data)

                # setup containers for the dataset
                all_observations, all_next_observations, all_dones = [], [], []

                if rng_key is None:
                    rng_key = jax.random.key(0)

                for i in tqdm(range(self.th.n_trajectories(self._traj.data)), desc="Creating Transition Dataset"):

                    # set configuration to the first state of the current trajectory
                    self.th.start_conf = (i, 0)

                    # do a reset
                    key, subkey = jax.random.split(rng_key)
                    traj_data_single = self._traj.data.get(i, 0, np)  # get first sample
                    carry = self._init_additional_carry(key, model, data, np)

                    # set data from traj_data (qpos and qvel) and forward to calculate other kinematic entities.
                    mujoco.mj_resetData(model, data)
                    data = self.set_sim_state_from_traj_data(data, traj_data_single, carry)
                    mujoco.mj_forward(model, data)

                    data, carry = self._reset_carry(model, data, carry, self._traj)
                    data, carry = (
                        self.obs_container.reset_state(self, model, data, carry, np))
                    obs, carry = self._create_observation(model, data, carry)
                    info = self._reset_info_dictionary(obs, data, subkey)

                    # initiate obs container
                    observations = [obs]
                    for j in range(1, self.th.len_trajectory(i, self._traj.data)):
                        # get next sample and calculate forward dynamics
                        traj_data_single = self._traj.data.get(i, j, np)  # get next sample
                        data = self.set_sim_state_from_traj_data(data, traj_data_single, carry)
                        mujoco.mj_forward(model, data)

                        data, carry = self._simulation_post_step(model, data, carry, self._traj)
                        obs, carry = self._create_observation(model, data, carry)
                        obs, data, info, carry = (
                            self._step_finalize(obs, model, data, info, carry))
                        observations.append(obs)

                        # check if the current state is an absorbing state
                        is_absorbing, carry = self._is_absorbing(obs, info, data, carry, self._traj)
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

                if TrajectoryHandler.is_numpy(orig_traj.data):
                    backend = np
                else:
                    backend = jnp

                transitions = TrajectoryTransitions(backend.array(all_observations),
                                                    backend.array(all_next_observations),
                                                    backend.array(all_absorbing),
                                                    backend.array(all_dones))

                self.th = orig_th
                self._traj = orig_traj
                self._traj = self._traj.replace(transitions=transitions)

            return self._traj.transitions

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
                        key: jax.random.PRNGKey = None,
                        viser: bool = False,
                        usd: bool = False,
                        usd_params: Dict = None) -> None:
        """
        Play a demo of the loaded trajectory by forcing the model
        positions to the ones in the trajectories at every step.

        Args:
            n_episodes (int): Number of episodes to replay. If None, the replay runs until it is interrupted.
            n_steps_per_episode (int): Number of steps to replay per episode. If None, each episode runs to the end
                of its trajectory.
            from_velocity (bool): If True, the joint positions are calculated from the joint
                velocities in the trajectory.
            render (bool): If True, trajectory will be rendered.
            record (bool): If True, the rendered trajectory will be recorded.
            recorder_params (dict): Dictionary containing the recorder parameters.
            callback_class: Object to be called at each step of the simulation, in place of the default update. It is
                called with ``(env, model, data, traj_sample, carry)`` and returns ``(model, data, carry)``; see
                :class:`~loco_mujoco.datasets.data_generation.utils.ReplayCallback`.
            quiet (bool): If True, disable tqdm.
            key (jax.random.PRNGKey): Random key to use for the simulation.
            viser (bool): If True, the browser-based viser viewer is used instead of the
                OpenGL viewer. Recording then requires a connected browser. Keyword only in
                practice: it is last so that existing positional callers keep working.
            usd (bool): If True, collect the trajectory into an animated USD stage instead of
                opening an interactive viewer.
            usd_params (dict): Parameters passed to ``render_usd`` on its first frame.

        Raises:
            ValueError: If ``viser`` and ``usd``, or ``usd`` and ``record``, are enabled together.

        """

        assert self.th is not None

        if viser and usd:
            raise ValueError("viser and USD export cannot both be enabled.")
        if usd and record:
            raise ValueError(
                "record=True is for framebuffer viewers and cannot be combined with USD export."
            )
        pending_usd_params = {} if usd_params is None else dict(usd_params)

        def render_usd_frame(_record=False):
            nonlocal pending_usd_params
            self.render_usd(**pending_usd_params)
            pending_usd_params = {}
            return None

        render_fn = render_usd_frame if usd else (self.render_viser if viser else self.render)

        if not TrajectoryHandler.is_numpy(self._traj.data):
            was_jax = True
            self._traj = self._traj.replace(data=TrajectoryHandler.to_numpy(self._traj.data))
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
        traj_data_sample = self.th.get_current_traj_data(self._traj.data, self._additional_carry, np)

        # Offline USD export starts inside the loop after the trajectory
        # sample has been written to MjData. Rendering here would otherwise
        # prepend the reset pose, which may be far from the trajectory origin.
        if render and not usd:
            frame = render_fn(record)
        else:
            frame = None

        if record:
            recorder(frame)

        highest_int = np.iinfo(np.int32).max
        if n_episodes is None:
            n_episodes = highest_int
        for i in range(n_episodes):
            if n_steps_per_episode is None:
                nspe = (self.th.len_trajectory(self._additional_carry.traj_state.traj_no, self._traj.data) -
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
                        self._simulation_post_step(self._model, self._data, self._additional_carry, self._traj))
                else:
                    self._model, self._data, self._additional_carry = (
                        callback_class(self, self._model, self._data, traj_data_sample, self._additional_carry))

                traj_data_sample = self.th.get_current_traj_data(self._traj.data, self._additional_carry, np)

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
                    frame = render_fn(record)
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
            self._traj = self._traj.replace(data=TrajectoryHandler.to_jax(self._traj.data))

    def play_trajectory_from_velocity(self, n_episodes: int = None,
                                      n_steps_per_episode: int = None,
                                      render: bool = True,
                                      record: bool = False,
                                      recorder_params: Dict = None,
                                      callback_class=None,
                                      quiet: bool = False,
                                      key: jax.random.PRNGKey = None) -> None:
        """
        Play a demo of the loaded trajectory by integrating the joint velocities (deprecated).

        The joint positions are set from the trajectory in the first step. Afterward, numerical
        integration is used to calculate the next joint positions using
        the joint velocities in the trajectory. Deprecated: use :meth:`play_trajectory` with ``from_velocity=True``.

        Args:
            n_episodes (int): Number of episodes to replay.
            n_steps_per_episode (int): Number of steps to replay per episode.
            render (bool): If True, trajectory will be rendered.
            record (bool): If True, the rendered trajectory will be recorded.
            recorder_params (dict): Dictionary containing the recorder parameters.
            callback_class: Object to be called at each step of the simulation.
            quiet (bool): If True, disable tqdm.
            key (jax.random.PRNGKey): Random key to use for the simulation.

        """
        warnings.warn(
            "play_trajectory_from_velocity() is deprecated and will be removed in future. "
            "Use play_trajectory() and set from_velocity=True instead.",
            category=DeprecationWarning,
            stacklevel=3
        )
        self.play_trajectory(n_episodes=n_episodes, n_steps_per_episode=n_steps_per_episode,
                             from_velocity=True, render=render, record=record,
                             recorder_params=recorder_params, callback_class=callback_class,
                             quiet=quiet, key=key)

    def set_sim_state_from_traj_data(self, data, traj_data, carry) -> MjData:
        """
        Set the Mujoco data structure to the state specified in the trajectory data.

        If episodes are recentered (``recenter_root_xy``), the xy position of the robot's root at the start of the
        current trajectory segment is subtracted from the xy position of all free joints first; this changes
        ``traj_data.qpos`` in place.

        Args:
            data (MjData): The Mujoco data structure.
            traj_data: The trajectory data of a single sample.
            carry (LocoCarry): Additional carry information.

        Returns:
            MjData: The updated Mujoco data structure.

        """
        if not self._recenter_root_xy:
            return Mjx.set_sim_state_from_traj_data(data, traj_data, carry)
        robot_free_jnt_name = self.root_free_joint_xml_name
        robot_free_jnt_qpos_id_xy = np.array(mj_jntname2qposid(robot_free_jnt_name, self._model))[:2]
        all_free_jnt_qpos_id_xy = self.free_jnt_qpos_id[:, :2].reshape(-1)
        traj_state = carry.traj_state
        # get the initial state of the current trajectory
        traj_data_init = self._traj.data.get(traj_state.traj_no, traj_state.subtraj_step_no_init, np)
        # subtract the initial state from the current state
        traj_data.qpos[all_free_jnt_qpos_id_xy] -= traj_data_init.qpos[robot_free_jnt_qpos_id_xy]
        return Mjx.set_sim_state_from_traj_data(data, traj_data, carry)

    def mjx_set_sim_state_from_traj_data(self, data, traj_data, carry, traj=None) -> Data:
        """
        Set the MJX data structure to the state specified in the trajectory data.

        If episodes are recentered (``recenter_root_xy``), the xy position of the robot's root at the start of the
        current trajectory segment is subtracted from the xy position of all free joints first.

        Args:
            data (Data): Current Mujoco data.
            traj_data (TrajectoryData): The trajectory data of a single sample.
            carry (MjxAdditionalCarry): Additional carry information.
            traj (Trajectory, optional): Full trajectory for initial state lookup. Falls back to the env's trajectory.

        Returns:
            Data: Updated Mujoco data.

        """
        if not self._recenter_root_xy:
            return Mjx.mjx_set_sim_state_from_traj_data(data, traj_data, carry)
        if traj is None:
            traj = self._traj
        robot_free_jnt_name = self.root_free_joint_xml_name
        robot_free_jnt_qpos_id_xy = np.array(mj_jntname2qposid(robot_free_jnt_name, self._model))[:2]
        all_free_jnt_qpos_id_xy = self.free_jnt_qpos_id[:, :2].reshape(-1)
        traj_state = carry.traj_state
        # get the initial state of the current trajectory
        traj_data_init = traj.data.get(traj_state.traj_no, traj_state.subtraj_step_no_init, jnp)
        # subtract the initial state from the current state
        traj_data = traj_data.replace(
            qpos=traj_data.qpos.at[all_free_jnt_qpos_id_xy].add(-traj_data_init.qpos[robot_free_jnt_qpos_id_xy]))
        return Mjx.mjx_set_sim_state_from_traj_data(data, traj_data, carry)

    def _init_additional_carry(self,
                               key: jax.Array,
                               model: MjModel,
                               data: MjData,
                               backend: ModuleType,
                               traj=None) -> LocoCarry:
        """
        Initialize the additional carry structure, including the trajectory state.

        Args:
            key (jax.Array): Random key for JAX operations.
            model (MjModel): The Mujoco model.
            data (MjData): The Mujoco data structure.
            backend (ModuleType): The numerical backend to use (NumPy or JAX NumPy).
            traj: Trajectory to use (optional).

        Returns:
            LocoCarry: The initialized additional carry structure.

        """

        carry = super()._init_additional_carry(key, model, data, backend, traj)

        key = carry.key
        key, _k = jax.random.split(key)

        # create additional carry
        _traj = traj if traj is not None else self._traj
        carry = LocoCarry(
            traj_state=self.th.init_state(self, _k, model, data, backend,
                                          _traj.info.model if _traj is not None else None,
                                          _traj.data if _traj is not None else None) if self.th is not None else EmptyState(),
            **vars(carry.replace(key=key)))

        return carry

    def _reset_carry(self, model: MjModel,
                     data: MjData,
                     carry: LocoCarry,
                     traj=None) -> Tuple[MjData, LocoCarry]:
        """
        Reset the additional carry, including the trajectory state. Also allows modification to the MjData.

        Args:
            model (MjModel): Mujoco model.
            data (MjData): Mujoco data structure.
            carry (LocoCarry): Additional carry information.
            traj: Trajectory to use (optional).

        Returns:
            Tuple[MjData, LocoCarry]: The updated data and carry.

        """

        # reset trajectory state
        if self.th is not None:
            data, carry = self.th.reset_state(self, self._model, data, carry, np,
                                              traj_data=traj.data if traj is not None else None)

        # call parent to apply domain randomization and terrain
        data, carry = super()._reset_carry(model, data, carry, traj)

        return data, carry

    def _mjx_reset_carry(self, model: Model,
                         data: Data,
                         carry: MjxAdditionalCarry,
                         traj=None) -> Tuple[Data, MjxAdditionalCarry]:
        """
        Resets the additional carry and allows modification to the Mujoco data.

        Args:
            model (Model): Mujoco model.
            data (Data): Mujoco data structure.
            carry (MjxAdditionalCarry): Additional carry information.
            traj: Trajectory to use (optional).

        Returns:
            Tuple[Data, MjxAdditionalCarry]: Updated data and carry.
        """

        # reset trajectory state
        if self.th is not None:
            data, carry = self.th.reset_state(self, self._model, data, carry, jnp,
                                              traj_data=traj.data if traj is not None else None)

        # call parent to apply domain randomization and terrain
        data, carry = super()._mjx_reset_carry(model, data, carry, traj)

        return data, carry

    def _get_from_obs(self, obs: Union[np.ndarray, jnp.ndarray], obs_name: str) -> Union[np.ndarray, jnp.ndarray]:
        """
        Return the part of an observation that belongs to one named observation.

        Args:
            obs (Union[np.ndarray, jnp.ndarray]): Observation array.
            obs_name (str): Name of the observation in the observation container.

        Returns:
            Union[np.ndarray, jnp.ndarray]: The entries of ``obs`` that belong to ``obs_name``.

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
        Create an environment of this class for the specified task (deprecated).

        Deprecated: use :class:`~loco_mujoco.task_factories.imitation_factory.ImitationFactory` or
        :class:`~loco_mujoco.task_factories.rl_factory.RLFactory` instead.

        Args:
            task (str): Task of the default dataset to load (e.g. ``"walk"``). If None, a plain RL environment
                without trajectory is created.
            dataset_type (str): Only "mocap" is available: motion capture data retargeted to the
                robot. This data does not perfectly match the kinematics and dynamics of the
                environment.
            debug (bool): Passed to :class:`~loco_mujoco.task_factories.dataset_confs.DefaultDatasetConf`.
            clip_trajectory_to_joint_ranges (bool): Not used.
            **kwargs: Passed to the task factory and from there to the constructor of the environment.

        Returns:
            Tuple[LocoEnv, Trajectory]: The return value of the task factory: the environment and its processed
            trajectory (``None`` if ``task`` is None).

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
        Return the default path to the xml file of the environment.

        Returns:
            str: Path to the xml file. Has to be implemented by the subclass.

        """
        raise NotImplementedError(f"Please implement the default_xml_file_path property "
                                  f"in the {type(cls).__name__} environment.")

    @info_property
    def root_body_name(self) -> str:
        """
        Return the name of the root body.

        Returns:
            str: Name of the root body in the Mujoco model. Has to be implemented by the subclass.

        """
        raise NotImplementedError(f"Please implement the root_body_name "
                                  f"info property in the {type(self).__name__} environment.")

    @info_property
    def root_free_joint_xml_name(self) -> str:
        """
        Return the name of the root free joint.

        Returns:
            str: Name of the free joint of the root in the Mujoco model, ``"root"`` by default.

        """
        return "root"

    @info_property
    def upper_body_xml_name(self) -> str:
        """
        Return the name of the upper body.

        Returns:
            str: Name of the upper body in the Mujoco model. Has to be implemented by the subclass.

        """
        raise NotImplementedError(f"Please implement the upper_body_xml_name property "
                                  f"in the {type(self).__name__} environment.")

    @info_property
    def root_height_healthy_range(self) -> Tuple[float, float]:
        """
        Return the healthy range of the root height. This is only used when HeightBasedTerminalStateHandler is used.

        Returns:
            Tuple[float, float]: Lower and upper bound of the root height in meters. Has to be implemented by the
            subclass.

        """
        raise NotImplementedError(f"Please implement the root_height_healthy_range property "
                                  f"in the {type(self).__name__} environment.")

    @info_property
    def foot_geom_names(self) -> List[str]:
        """
        Return the names of the foot geometries.

        Returns:
            List[str]: Names of the foot geoms in the Mujoco model; empty by default.

        """
        # todo: raise NotImplementedError, once added to all envs
        return []

    @info_property
    def goal_visualization_arrow_offset(self) -> List[float]:
        """
        Return the offset of the goal visualization arrow.

        Returns:
            List[float]: The x, y and z offset; zero by default.

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
            List[str]: The names of the registered loco_mujoco environments.

        """
        return list(LocoEnv.registered_envs.keys())

    @staticmethod
    def _delete_from_spec(spec: MjSpec,
                          joints_to_remove: List[str],
                          actuators_to_remove: List[str],
                          equ_constraints_to_remove: List[str]):
        """
        Delete certain joints, motors and equality constraints from a Mujoco specification.

        Names that do not exist in the specification are ignored.

        Args:
            spec (MjSpec): Mujoco specification.
            joints_to_remove (List[str]): List of joint names to remove.
            actuators_to_remove (List[str]): List of actuator names to remove.
            equ_constraints_to_remove (List[str]): List of equality constraint names to remove.

        Returns:
            MjSpec: Modified Mujoco specification.

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


def _resolve_legacy_th_params(th_params: Dict) -> Dict:
    """
    Map the trajectory handler parameters of earlier versions (``random_start``, ``fixed_start_conf``) to the
    handler classes, with a DeprecationWarning.

    Args:
        th_params (Dict): The ``th_params`` given to the environment (a copy, changed in place).

    Returns:
        Dict: Parameters with ``name`` and the class's keyword arguments.
    """
    if "name" in th_params or not {"random_start", "fixed_start_conf"} & th_params.keys():
        return th_params
    random_start = th_params.pop("random_start", True)
    fixed_start_conf = th_params.pop("fixed_start_conf", None)
    if fixed_start_conf is not None:
        th_params.update(name="FixedStartTrajectoryHandler", start_conf=fixed_start_conf)
        new = 'name="FixedStartTrajectoryHandler", start_conf=...'
    elif random_start:
        th_params.update(name="RandomStartTrajectoryHandler")
        new = 'name="RandomStartTrajectoryHandler" (the default)'
    else:
        raise ValueError("th_params: random_start=False needs a fixed_start_conf (use "
                         "name=\"FixedStartTrajectoryHandler\", start_conf=...).")
    warnings.warn(f"th_params random_start / fixed_start_conf are deprecated, use {new}.", DeprecationWarning,
                  stacklevel=3)
    return th_params
