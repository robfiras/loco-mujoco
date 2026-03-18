from dataclasses import replace
import mujoco
import numpy as np
import jax
import jax.numpy as jnp
from flax import struct

from loco_mujoco.core.stateful_object import StatefulObject
from loco_mujoco.core.trajectory.dataclasses import Trajectory, interpolate_trajectories


@struct.dataclass
class TrajState:
    traj_no: int
    subtraj_step_no: int
    subtraj_step_no_init: int


class TrajectoryHandler(StatefulObject):
    """
    General class to handle Trajectories. It stores only traj_info (not the full trajectory).
    The full trajectory data is stored on LocoEnv as self._traj.

    """
    def __init__(self, traj_info, control_dt=0.01, random_start=True, fixed_start_conf=None):
        """
        Constructor.

        Args:
            traj_info (TrajectoryInfo): Information about the trajectory.
            control_dt (float): Model control frequency.
            random_start (bool): If True, the trajectory is started at a random position.
            fixed_start_conf (tuple): If not None, the trajectory is started at the specified position.

        """
        assert (fixed_start_conf is not None) != random_start, "Please specify either fixed_start_conf or random_start."
        self._traj_info = traj_info
        self.random_start = random_start
        self.fixed_start_conf = fixed_start_conf
        self.use_fixed_start = True if fixed_start_conf is not None else False
        self.control_dt = control_dt

    def len_trajectory(self, traj_ind, traj_data):
        return traj_data.split_points[traj_ind + 1] - traj_data.split_points[traj_ind]

    def n_trajectories(self, traj_data):
        return traj_data.split_points.shape[0] - 1

    @property
    def traj_info(self):
        return self._traj_info

    @staticmethod
    def is_numpy(traj_data):
        return isinstance(traj_data.qpos, np.ndarray)

    @staticmethod
    def to_numpy(traj_data):
        return traj_data.to_numpy()

    @staticmethod
    def to_jax(traj_data):
        return traj_data.to_jax()

    @staticmethod
    def process(traj, model, control_dt):
        """
        Filter, extend, and interpolate a trajectory to match the given model and control frequency.

        Args:
            traj (Trajectory): Raw trajectory to process.
            model (mjModel): Current model.
            control_dt (float): Desired control timestep.

        Returns:
            Trajectory: Processed trajectory.
        """
        from loco_mujoco.core.trajectory import interpolate_trajectories
        traj_data, traj_info = TrajectoryHandler.filter_and_extend(traj.data, traj.info, model)
        traj_dt = 1 / traj_info.frequency
        if traj_dt != control_dt:
            traj_data, traj_info = interpolate_trajectories(traj_data, traj_info, 1.0 / control_dt)
        return traj.replace(data=traj_data, info=traj_info)

    @staticmethod
    def filter_and_extend(traj_data, traj_info, model):
        """
        To ensure that the data structure of the current model and the trajectory data have the same dimensionality
        and order for all supported attributes, this function filters the elements present in the trajectory but not
        the current model and extends the trajectory data's joints, bodies and sites with elements present in
        the current model but not the trajectory. It is doing so by adding dummy joints, bodies and sites to the
        trajectory data if they are not present in the trajectory data but in the model. It also reorders the
        joints, bodies and sites based on the model.

        Args:
            traj_data (TrajectoryData): Trajectory data to be filtered and extended.
            traj_info (TrajectoryInfo): Trajectory info to be filtered and extended.
            model (mjModel): Current model.

        Returns:
            TrajectoryData, TrajectoryInfo: Filtered and extended trajectory data and trajectory info.

        """

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
                joint_name2id_qpos[name] = jnp.arange(j_qpos, j_qpos + 7)
                joint_name2id_qvel[name] = jnp.arange(j_qvel, j_qvel + 6)
                j_qpos += 7
                j_qvel += 6
            elif j_type == mujoco.mjtJoint.mjJNT_SLIDE or j_type == mujoco.mjtJoint.mjJNT_HINGE:
                joint_name2id_qpos[name] = jnp.array([j_qpos])
                joint_name2id_qvel[name] = jnp.array([j_qvel])
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
            qpos_ind = jnp.concatenate(list(joint_to_be_removed_qpos.values()))
            qvel_ind = jnp.concatenate(list(joint_to_be_removed_qvel.values()))
            traj_data = traj_data.remove_joints(qpos_ind, qvel_ind)
            traj_info = traj_info.remove_joints(list(joint_to_be_removed_qpos.keys()))
        if bodies_to_be_removed:
            traj_data = traj_data.remove_bodies(jnp.array(list(bodies_to_be_removed.values())))
            traj_info = traj_info.remove_bodies(list(bodies_to_be_removed.keys()))
        if site_to_be_removed:
            traj_data = traj_data.remove_sites(jnp.array(list(site_to_be_removed.values())))
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
        traj_data = traj_data.reorder_joints(jnp.concatenate(new_joint_order_ids_qpos),
                                             jnp.concatenate(new_joint_order_ids_qvel))
        traj_data = traj_data.reorder_bodies(jnp.array(new_body_order)) \
            if traj_info.body_names is not None else traj_data
        traj_data = traj_data.reorder_sites(jnp.array(new_site_order)) \
            if traj_info.site_names is not None else traj_data

        return traj_data, traj_info

    def init_state(self, env, key, model, data, backend, traj_model=None, traj_data=None):
        return TrajState(0, 0, 0)

    def reset_state(self, env, model, data, carry, backend, traj_model=None, traj_data=None):

        key = carry.key

        if self.random_start:
            if backend == jnp:
                key, _k1, _k2 = jax.random.split(key, 3)
                traj_idx = jax.random.randint(_k1, shape=(1,), minval=0, maxval=self.n_trajectories(traj_data))
                subtraj_step_idx = jax.random.randint(_k2, shape=(1,), minval=0, maxval=self.len_trajectory(traj_idx, traj_data))
                idx = [traj_idx[0], subtraj_step_idx[0]]
            else:
                traj_idx = np.random.randint(0, self.n_trajectories(traj_data))
                subtraj_step_idx = np.random.randint(0, self.len_trajectory(traj_idx, traj_data))
                idx = [traj_idx, subtraj_step_idx]
        elif self.use_fixed_start:
            idx = self.fixed_start_conf
        else:
            idx = [0, 0]

        new_traj_no, new_subtraj_step_no = idx
        new_subtraj_step_no_init = new_subtraj_step_no

        return data, carry.replace(key=key, traj_state=TrajState(new_traj_no, new_subtraj_step_no,
                                                                 new_subtraj_step_no_init))

    def update_state(self, env, model, data, carry, backend, traj_model=None, traj_data=None):

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

    def get_current_traj_data(self, traj_data, carry, backend):
        traj_no = carry.traj_state.traj_no
        subtraj_step_no = carry.traj_state.subtraj_step_no
        return traj_data.get(traj_no, subtraj_step_no, backend)

    def get_init_traj_data(self, traj_data, carry, backend):
        traj_no = carry.traj_state.traj_no
        subtraj_step_no_init = carry.traj_state.subtraj_step_no_init
        return traj_data.get(traj_no, subtraj_step_no_init, backend)
