"""
Trajectory handlers.

A trajectory handler keeps track of which trajectory and which sample of it an environment currently follows
(:class:`TrajState`), advances that position at every step and chooses the start of every episode. It also
provides the preprocessing that makes a loaded trajectory match a MuJoCo model and a control frequency.

"""

import warnings
from dataclasses import replace
from typing import Dict, List, Tuple
import mujoco
import numpy as np
import jax
import jax.numpy as jnp
from flax import struct

from loco_mujoco.core.stateful_object import StatefulObject
from loco_mujoco.core.trajectory.dataclasses import Trajectory, interpolate_trajectories


@struct.dataclass
class TrajState:
    """
    Position of an environment in its trajectory data, stored in the carry as ``traj_state``.

    Attributes:
        traj_no (int): Index of the trajectory that is currently followed.
        subtraj_step_no (int): Index of the current sample within that trajectory.
        subtraj_step_no_init (int): Index of the sample within the trajectory at which the episode started. It is
            set to 0 when the handler moves on to the next trajectory within an episode.
        next_traj_no (int): If non-negative, the trajectory with which the next episode starts. Set by algorithms
            that choose their own resets; ``-1`` lets the handler sample the start.
        next_subtraj_step_no (int): Sample within ``next_traj_no`` at which the next episode starts. Only used
            if ``next_traj_no`` is non-negative.

    """

    traj_no: int
    subtraj_step_no: int
    subtraj_step_no_init: int
    # If non-negative, the start of the next episode: set by algorithms that choose their own resets.
    next_traj_no: int = -1
    next_subtraj_step_no: int = -1


class TrajectoryHandler(StatefulObject):
    """
    Base class of all trajectory handlers.

    The handler stores only the :class:`~loco_mujoco.core.trajectory.dataclasses.TrajectoryInfo`; the trajectory
    data itself is owned by the environment (``LocoEnv``) and passed to the methods as ``traj_data``. The position
    in the data is kept in a :class:`TrajState` in the carry. At every step :meth:`update_state` advances it by one
    sample, and at every reset :meth:`reset_state` sets it to the start chosen by :meth:`sample_start`, which the
    subclasses implement.

    Attributes:
        registered (Dict[str, type]): Registry of handler classes by name.
        control_dt (float): Control timestep of the environment in seconds.
        max_n_samples (int): Number of samples the trajectory data is padded to.
        max_n_trajs (int): Number of trajectories the split points are padded to.

    """

    registered: Dict[str, type] = dict()

    def __init__(self, traj_info, control_dt=0.01, max_n_samples=100_000, max_n_trajs=100):
        """
        Constructor.

        Args:
            traj_info (TrajectoryInfo): Information about the trajectory.
            control_dt (float): Control timestep of the environment in seconds.
            max_n_samples (int): Maximum number of samples for padding (for JIT shape stability).
            max_n_trajs (int): Maximum number of sub-trajectories for padding (for JIT shape stability).

        """
        self._traj_info = traj_info
        self.control_dt = control_dt
        self.max_n_samples = max_n_samples
        self.max_n_trajs = max_n_trajs

    @classmethod
    def get_name(cls) -> str:
        """
        Return the registry key of this class, which defaults to the class name.

        Returns:
            str: Name under which the class is registered.

        """
        return cls.__name__

    @classmethod
    def register(cls) -> None:
        """Add this class to the registry of trajectory handlers under its name, unless that name is taken."""
        name = cls.get_name()
        if name not in TrajectoryHandler.registered:
            TrajectoryHandler.registered[name] = cls

    @staticmethod
    def list_registered() -> List[str]:
        """
        Return the names of all registered trajectory handlers.

        Returns:
            List[str]: Names of the registered classes.

        """
        return list(TrajectoryHandler.registered.keys())

    def len_trajectory(self, traj_ind, traj_data):
        """
        Return the number of samples of one trajectory.

        Args:
            traj_ind (Union[int, jax.Array]): Index of the trajectory.
            traj_data (TrajectoryData): The trajectory data.

        Returns:
            Union[int, jax.Array]: Number of samples of trajectory ``traj_ind``.

        """
        return traj_data.split_points[traj_ind + 1] - traj_data.split_points[traj_ind]

    def n_trajectories(self, traj_data):
        """
        Return the number of trajectories in the data, not counting padding.

        Args:
            traj_data (TrajectoryData): The trajectory data.

        Returns:
            Union[int, jax.Array]: Number of trajectories.

        """
        return traj_data.n_trajectories

    @property
    def traj_info(self):
        """
        Static information about the trajectories this handler was created for.

        Returns:
            TrajectoryInfo: The trajectory info.

        """
        return self._traj_info

    @staticmethod
    def is_numpy(traj_data):
        """
        Check whether the trajectory data is stored in NumPy arrays.

        Args:
            traj_data (TrajectoryData): The trajectory data.

        Returns:
            bool: True if ``traj_data.qpos`` is a ``np.ndarray``.

        """
        return isinstance(traj_data.qpos, np.ndarray)

    @staticmethod
    def to_numpy(traj_data):
        """
        Convert the trajectory data to NumPy arrays.

        Args:
            traj_data (TrajectoryData): The trajectory data.

        Returns:
            TrajectoryData: A new instance holding NumPy arrays.

        """
        return traj_data.to_numpy()

    @staticmethod
    def to_jax(traj_data):
        """
        Convert the trajectory data to JAX arrays.

        Args:
            traj_data (TrajectoryData): The trajectory data.

        Returns:
            TrajectoryData: A new instance holding JAX arrays.

        """
        return traj_data.to_jax()

    @staticmethod
    def process(traj, model, control_dt, backend=None):
        """
        Filter, extend, and interpolate a trajectory to match the given model and control frequency.

        Calls :meth:`filter_and_extend` and, if the trajectory frequency differs from ``1 / control_dt``,
        :func:`~loco_mujoco.core.trajectory.dataclasses.interpolate_trajectories`.

        Args:
            traj (Trajectory): Raw trajectory to process.
            model (MjModel): Current model.
            control_dt (float): Desired control timestep in seconds.
            backend (ModuleType): Array backend (``jax.numpy`` or ``numpy``). Defaults to ``jax.numpy``.

        Returns:
            Trajectory: Copy of ``traj`` with processed ``data`` and ``info``.

        """
        if backend is None:
            backend = jnp
        from loco_mujoco.core.trajectory import interpolate_trajectories
        traj_data, traj_info = TrajectoryHandler.filter_and_extend(traj.data, traj.info, model, backend=backend)
        traj_dt = 1 / traj_info.frequency
        if traj_dt != control_dt:
            traj_data, traj_info = interpolate_trajectories(traj_data, traj_info, 1.0 / control_dt, backend=backend)
        return traj.replace(data=traj_data, info=traj_info)

    @staticmethod
    def filter_and_extend(traj_data, traj_info, model, backend=None):
        """
        Make the joints, bodies and sites of a trajectory match those of a model.

        To ensure that the trajectory data and the current model have the same dimensionality and order for all
        supported attributes, this function

        - removes the joints, bodies and sites that are present in the trajectory but not in the model,
        - adds dummy joints, bodies and sites for elements that are present in the model but not in the trajectory
          (zero positions and velocities, identity rotations), and
        - reorders the joints, bodies and sites to the order of the model.

        Bodies and sites are only processed if the trajectory info contains body and site names, respectively.

        Args:
            traj_data (TrajectoryData): Trajectory data to be filtered and extended.
            traj_info (TrajectoryInfo): Trajectory info to be filtered and extended.
            model (MjModel): Current model.
            backend (ModuleType): Array backend (``jax.numpy`` or ``numpy``). Defaults to ``jax.numpy``.

        Returns:
            Tuple[TrajectoryData, TrajectoryInfo]: Filtered and extended trajectory data and trajectory info.

        """
        if backend is None:
            backend = jnp

        # --- filter the trajectory based on the model and data ---
        # get the joint names from current model
        joint_names = []
        joint_ids = []
        joint_name2id_qpos = dict()
        joint_name2id_qvel = dict()
        j_qpos, j_qvel = 0, 0
        for i in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            j_type = model.jnt_type[i]
            joint_names.append(name)

            if j_type == mujoco.mjtJoint.mjJNT_FREE:
                joint_name2id_qpos[name] = backend.arange(j_qpos, j_qpos + 7)
                joint_name2id_qvel[name] = backend.arange(j_qvel, j_qvel + 6)
                j_qpos += 7
                j_qvel += 6
            elif j_type == mujoco.mjtJoint.mjJNT_SLIDE or j_type == mujoco.mjtJoint.mjJNT_HINGE:
                joint_name2id_qpos[name] = backend.array([j_qpos])
                joint_name2id_qvel[name] = backend.array([j_qvel])
                j_qpos += 1
                j_qvel += 1

            joint_ids.append(i)

        # get the body names from current model
        body_names = set()
        body_name2id = dict()
        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            body_names.add(name)
            body_name2id[name] = i

        # get the site names from current model
        site_names = set()
        site_name2id = dict()
        for i in range(model.nsite):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
            site_names.add(name)
            site_name2id[name] = i

        joint_to_be_removed_qpos = dict()
        joint_to_be_removed_qvel = dict()
        for i, j_name in enumerate(traj_info.joint_names):
            if j_name not in joint_names:
                joint_to_be_removed_qpos[j_name] = traj_info.joint_name2ind_qpos[j_name]
                joint_to_be_removed_qvel[j_name] = traj_info.joint_name2ind_qvel[j_name]

        bodies_to_be_removed = dict()
        if traj_info.body_names is not None:
            for i, b_name in enumerate(traj_info.body_names):
                if b_name not in body_names:
                    bodies_to_be_removed[b_name] = i

        site_to_be_removed = dict()
        if traj_info.site_names is not None:
            for i, s_name in enumerate(traj_info.site_names):
                if s_name not in site_names:
                    site_to_be_removed[s_name] = i

        # create new traj_data and traj_info with removed joints, bodies and sites
        if joint_to_be_removed_qpos:
            qpos_ind = backend.concatenate(list(joint_to_be_removed_qpos.values()))
            qvel_ind = backend.concatenate(list(joint_to_be_removed_qvel.values()))
            traj_data = traj_data.remove_joints(qpos_ind, qvel_ind)
            traj_info = traj_info.remove_joints(list(joint_to_be_removed_qpos.keys()))
        if bodies_to_be_removed:
            traj_data = traj_data.remove_bodies(backend.array(list(bodies_to_be_removed.values())))
            traj_info = traj_info.remove_bodies(list(bodies_to_be_removed.keys()))
        if site_to_be_removed:
            traj_data = traj_data.remove_sites(backend.array(list(site_to_be_removed.values())))
            traj_info = traj_info.remove_sites(list(site_to_be_removed.keys()))

        # --- extend the trajectory data's joints, bodies and sites using the current model and data ---
        for j_name, j_id in zip(joint_names, joint_ids):
            j_type = model.jnt_type[j_id]
            if j_name not in traj_info.joint_names:
                traj_info = traj_info.add_joint(j_name, j_type)
                traj_data = traj_data.add_joint()

        if traj_info.body_names is not None:
            for b_name in body_names:
                if b_name not in traj_info.body_names:
                    b_id = body_name2id[b_name]
                    traj_info = traj_info.add_body(b_name, model.body_rootid[b_id], model.body_weldid[b_id],
                                                   model.body_mocapid[b_id], model.body_pos[b_id],
                                                   model.body_quat[b_id], model.body_ipos[b_id],
                                                   model.body_iquat[b_id])
                    traj_data = traj_data.add_body()

        if traj_info.site_names is not None:
            for s_name in site_names:
                if s_name not in traj_info.site_names:
                    s_id = site_name2id[s_name]
                    traj_info = traj_info.add_site(s_name, model.site_pos[s_id], model.site_quat[s_id],
                                                   model.site_bodyid[s_id])
                    traj_data = traj_data.add_site()

        # --- reorder the joints and bodies based on the model ---
        new_joint_order_names = []
        new_joint_order_ids_qpos = []
        new_joint_order_ids_qvel = []
        for j_name in joint_names:
            new_joint_order_names.append(traj_info.joint_names.index(j_name))
            new_joint_order_ids_qpos.append(traj_info.joint_name2ind_qpos[j_name])
            new_joint_order_ids_qvel.append(traj_info.joint_name2ind_qvel[j_name])

        if traj_info.body_names is not None:
            new_body_order = []
            for b_name in body_name2id.keys():
                new_body_order.append(traj_info.body_names.index(b_name))

        if traj_info.site_names is not None:
            new_site_order = []
            for s_name in site_name2id.keys():
                new_site_order.append(traj_info.site_names.index(s_name))

        traj_info = traj_info.reorder_joints(new_joint_order_names)
        traj_info = traj_info.reorder_bodies(new_body_order) if traj_info.body_names is not None else traj_info
        traj_info = traj_info.reorder_sites(new_site_order) if traj_info.site_names is not None else traj_info
        traj_data = traj_data.reorder_joints(backend.concatenate(new_joint_order_ids_qpos),
                                             backend.concatenate(new_joint_order_ids_qvel))
        traj_data = traj_data.reorder_bodies(backend.array(new_body_order)) \
            if traj_info.body_names is not None else traj_data
        traj_data = traj_data.reorder_sites(backend.array(new_site_order)) \
            if traj_info.site_names is not None else traj_data

        return traj_data, traj_info

    def init_state(self, env, key, model, data, backend, traj_model=None, traj_data=None):
        """
        Create the initial trajectory state, which points to the first sample of the first trajectory.

        Args:
            env (Any): The environment instance.
            key (Any): JAX random key (unused).
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            backend (ModuleType): Backend module, either ``numpy`` or ``jax.numpy``.
            traj_model (TrajectoryModel): Model of the trajectory (unused).
            traj_data (TrajectoryData): The trajectory data (unused).

        Returns:
            TrajState: The initial trajectory state.

        """
        return TrajState(0, 0, 0)

    def update_state(self, env, model, data, carry, backend, traj_model=None, traj_data=None):
        """
        Advance the trajectory state by one sample.

        When the end of the current trajectory is reached, the state moves on to the first sample of the next
        trajectory, and from the last trajectory back to the first.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry; its ``traj_state`` is updated.
            backend (ModuleType): Backend module, either ``numpy`` or ``jax.numpy``.
            traj_model (TrajectoryModel): Model of the trajectory (unused).
            traj_data (TrajectoryData): The trajectory data.

        Returns:
            Any: The carry with the updated ``traj_state``.

        """

        traj_state = carry.traj_state
        traj_no = traj_state.traj_no
        subtraj_step_no = traj_state.subtraj_step_no
        subtraj_step_no_init = traj_state.subtraj_step_no_init

        length_trajectory = self.len_trajectory(traj_no, traj_data)

        subtraj_step_no += 1

        # set to zero once exceeded
        next_subtraj_step_no = backend.mod(subtraj_step_no, length_trajectory)

        if backend == jnp:
            # check whether to go to the next trajectory
            next_traj_no = jax.lax.cond(next_subtraj_step_no == 0, lambda t, nt: jnp.mod(t+1, nt),
                                        lambda t, nt: t, traj_no, self.n_trajectories(traj_data))
            next_subtraj_step_no_init = jax.lax.cond(next_traj_no != traj_no, lambda: 0,
                                                     lambda: subtraj_step_no_init)
        else:
            next_traj_no = traj_no if next_subtraj_step_no != 0 else (traj_no + 1) % self.n_trajectories(traj_data)
            next_subtraj_step_no_init = 0 if traj_no != next_traj_no else subtraj_step_no_init

        traj_state = traj_state.replace(traj_no=next_traj_no, subtraj_step_no=next_subtraj_step_no,
                                        subtraj_step_no_init=next_subtraj_step_no_init)

        return carry.replace(traj_state=traj_state)

    def reset_state(self, env, model, data, carry, backend, traj_model=None, traj_data=None):
        """
        Set the trajectory state to the start of a new episode.

        The start is the one an algorithm chose (``next_traj_no`` / ``next_subtraj_step_no`` of the trajectory
        state, if non-negative), otherwise the one :meth:`sample_start` draws. The choice is cleared afterwards.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry; its ``key`` and ``traj_state`` are updated.
            backend (ModuleType): Backend module, either ``numpy`` or ``jax.numpy``.
            traj_model (TrajectoryModel): Model of the trajectory (unused).
            traj_data (TrajectoryData): The trajectory data.

        Returns:
            Tuple[Union[MjData, Data], Any]: The unchanged simulation data and the updated carry.

        """
        key, new_traj_no, new_subtraj_step_no = self.sample_start(carry.key, backend, traj_data)
        chosen = carry.traj_state.next_traj_no >= 0
        new_traj_no = backend.where(chosen, carry.traj_state.next_traj_no, new_traj_no)
        new_subtraj_step_no = backend.where(chosen, carry.traj_state.next_subtraj_step_no, new_subtraj_step_no)
        no_choice = new_traj_no * 0 - 1

        return data, carry.replace(key=key, traj_state=TrajState(new_traj_no, new_subtraj_step_no,
                                                                 new_subtraj_step_no, no_choice, no_choice))

    def sample_start(self, key, backend, traj_data) -> Tuple:
        """
        Draw the start of an episode. Subclasses implement their sampling policy (random, fixed, etc.).

        Args:
            key (jax.Array): The JAX random key of the carry.
            backend (ModuleType): Backend module, either ``numpy`` or ``jax.numpy``.
            traj_data (TrajectoryData): The trajectory data.

        Returns:
            Tuple: The (possibly split) key, the trajectory index and the step within that trajectory.

        Raises:
            NotImplementedError: If called on the base class.

        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement sample_start. "
            f"Use a concrete TrajectoryHandler subclass "
            f"(e.g. RandomStartTrajectoryHandler, FixedStartTrajectoryHandler, "
            f"RandomTrajFixedStepTrajectoryHandler)."
        )

    def get_current_traj_data(self, traj_data, carry, backend):
        """
        Return the trajectory sample the environment is currently at.

        Args:
            traj_data (TrajectoryData): The trajectory data.
            carry (Any): Additional carry holding the ``traj_state``.
            backend (ModuleType): Backend module, either ``numpy`` or ``jax.numpy``.

        Returns:
            SingleData: The sample at ``(traj_no, subtraj_step_no)``.

        """
        traj_no = carry.traj_state.traj_no
        subtraj_step_no = carry.traj_state.subtraj_step_no
        return traj_data.get(traj_no, subtraj_step_no, backend)

    def get_init_traj_data(self, traj_data, carry, backend):
        """
        Return the trajectory sample at which the current episode started.

        Args:
            traj_data (TrajectoryData): The trajectory data.
            carry (Any): Additional carry holding the ``traj_state``.
            backend (ModuleType): Backend module, either ``numpy`` or ``jax.numpy``.

        Returns:
            SingleData: The sample at ``(traj_no, subtraj_step_no_init)``.

        """
        traj_no = carry.traj_state.traj_no
        subtraj_step_no_init = carry.traj_state.subtraj_step_no_init
        return traj_data.get(traj_no, subtraj_step_no_init, backend)


class RandomStartTrajectoryHandler(TrajectoryHandler):
    """Trajectory handler that starts every episode at a uniformly random sample of a uniformly random trajectory."""

    def sample_start(self, key, backend, traj_data):
        """
        Draw a random trajectory and a random sample within it.

        With the JAX backend the random key is split and used; with NumPy the global ``np.random`` state is used and
        the key is returned unchanged.

        Args:
            key (jax.Array): The JAX random key of the carry.
            backend (ModuleType): Backend module, either ``numpy`` or ``jax.numpy``.
            traj_data (TrajectoryData): The trajectory data.

        Returns:
            Tuple: The key, the trajectory index and the step within that trajectory.

        """
        if backend == jnp:
            key, _k1, _k2 = jax.random.split(key, 3)
            traj_idx = jax.random.randint(_k1, shape=(1,), minval=0, maxval=self.n_trajectories(traj_data))
            subtraj_step_idx = jax.random.randint(_k2, shape=(1,), minval=0,
                                                  maxval=self.len_trajectory(traj_idx, traj_data))
            return key, traj_idx[0], subtraj_step_idx[0]
        traj_idx = np.random.randint(0, self.n_trajectories(traj_data))
        return key, traj_idx, np.random.randint(0, self.len_trajectory(traj_idx, traj_data))


class RandomTrajFixedStepTrajectoryHandler(TrajectoryHandler):
    """
    Trajectory handler that starts every episode at a fixed sample of a uniformly random trajectory.

    Attributes:
        fixed_step (int): Index of the sample within the trajectory at which every episode starts.

    """

    def __init__(self, traj_info, fixed_step: int = 0, control_dt: float = 0.01,
                 max_n_samples: int = 100_000, max_n_trajs: int = 100):
        """
        Constructor.

        Args:
            traj_info (TrajectoryInfo): Information about the trajectory.
            fixed_step (int): Index of the sample within the trajectory at which every episode starts.
            control_dt (float): Control timestep of the environment in seconds.
            max_n_samples (int): Maximum number of samples for padding (for JIT shape stability).
            max_n_trajs (int): Maximum number of sub-trajectories for padding (for JIT shape stability).

        """
        super().__init__(traj_info, control_dt=control_dt,
                         max_n_samples=max_n_samples, max_n_trajs=max_n_trajs)
        self.fixed_step = int(fixed_step)

    def sample_start(self, key, backend, traj_data):
        """
        Draw a random trajectory and return it together with the fixed step.

        With the JAX backend the random key is split and used; with NumPy the global ``np.random`` state is used and
        the key is returned unchanged.

        Args:
            key (jax.Array): The JAX random key of the carry.
            backend (ModuleType): Backend module, either ``numpy`` or ``jax.numpy``.
            traj_data (TrajectoryData): The trajectory data.

        Returns:
            Tuple: The key, the trajectory index and ``fixed_step``.

        """
        if backend == jnp:
            key, _k = jax.random.split(key)
            traj_idx = jax.random.randint(_k, shape=(1,), minval=0, maxval=self.n_trajectories(traj_data))
            return key, traj_idx[0], self.fixed_step
        return key, np.random.randint(0, self.n_trajectories(traj_data)), self.fixed_step


class FixedStartTrajectoryHandler(TrajectoryHandler):
    """
    Trajectory handler that starts every episode at the same sample of the same trajectory.

    Attributes:
        start_conf (Tuple[int, int]): Trajectory index and step within that trajectory at which every episode
            starts. It is a mutable attribute, so that test harnesses can sweep the starting frame between resets.

    """

    def __init__(self, traj_info, start_conf: Tuple[int, int] = (0, 0),
                 control_dt: float = 0.01, max_n_samples: int = 100_000, max_n_trajs: int = 100):
        """
        Constructor.

        Args:
            traj_info (TrajectoryInfo): Information about the trajectory.
            start_conf (Tuple[int, int]): Trajectory index and step within that trajectory at which every episode
                starts.
            control_dt (float): Control timestep of the environment in seconds.
            max_n_samples (int): Maximum number of samples for padding (for JIT shape stability).
            max_n_trajs (int): Maximum number of sub-trajectories for padding (for JIT shape stability).

        """
        super().__init__(traj_info, control_dt=control_dt,
                         max_n_samples=max_n_samples, max_n_trajs=max_n_trajs)
        self.start_conf = tuple(start_conf)

    def sample_start(self, key, backend, traj_data):
        """
        Return the fixed start.

        Args:
            key (jax.Array): The JAX random key of the carry; returned unchanged.
            backend (ModuleType): Backend module (unused).
            traj_data (TrajectoryData): The trajectory data (unused).

        Returns:
            Tuple: The key, the trajectory index and the step within that trajectory from ``start_conf``.

        """
        return key, *self.start_conf

    @property
    def fixed_start_conf(self) -> Tuple[int, int]:
        """Deprecated name of ``start_conf``; emits a ``DeprecationWarning``."""
        warnings.warn("TrajectoryHandler.fixed_start_conf is deprecated, use start_conf.", DeprecationWarning,
                      stacklevel=2)
        return self.start_conf

    @fixed_start_conf.setter
    def fixed_start_conf(self, value: Tuple[int, int]) -> None:
        """Deprecated setter of ``start_conf``; emits a ``DeprecationWarning``."""
        warnings.warn("TrajectoryHandler.fixed_start_conf is deprecated, use start_conf.", DeprecationWarning,
                      stacklevel=2)
        self.start_conf = tuple(value)
