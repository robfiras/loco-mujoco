"""
Utilities for generating trajectory datasets: callbacks that record model-specific quantities while a trajectory
is replayed, mocap bodies for retargeting, finite-difference velocities and loaders for configuration files.
"""

import math
from copy import deepcopy
from dataclasses import replace
from typing import List

import jax.numpy as jnp
import mujoco
import numpy as np
import yaml
from mujoco import MjSpec
from scipy.spatial.transform import Rotation as sRot
from omegaconf import DictConfig

from loco_mujoco.core.utils.math import quat_scalarfirst2scalarlast
from loco_mujoco.core.trajectory import (
    TrajectoryInfo,
    TrajectoryModel,
    TrajectoryData)


class ReplayCallback:

    """
    Base class that can be used to do things while replaying a trajectory.

    An instance is passed as ``callback_class`` to
    :meth:`~loco_mujoco.environments.base.LocoEnv.play_trajectory` and is called once per replayed sample in place of
    the default update. The base implementation reproduces the default update: it sets the simulation state from the
    trajectory sample and runs the forward kinematics.

    """

    @staticmethod
    def __call__(env, model, data, traj_sample, carry):
        """
        Set the simulation to a trajectory sample and compute the forward kinematics.

        Args:
            env: The environment replaying the trajectory.
            model (MjModel): Mujoco model.
            data (MjData): Mujoco data structure.
            traj_sample (TrajectoryData): The current sample of the trajectory.
            carry: Additional carry of the environment.

        Returns:
            Tuple[MjModel, MjData, Any]: The model, the updated data and the updated carry.

        """
        data = env.set_sim_state_from_traj_data(data, traj_sample, carry)
        model, data, carry = env._simulation_pre_step(model, data, carry)
        mujoco.mj_forward(model, data)
        data, carry = env._simulation_post_step(model, data, carry)
        return model, data, carry


class ExtendTrajData(ReplayCallback):
    """
    Replay callback that records the model-specific quantities of every replayed sample.

    A trajectory may only contain joint positions and velocities. While such a trajectory is replayed, this callback
    records the quantities that follow from the kinematics of the model: body positions, orientations and velocities
    (``xpos``, ``xquat``, ``cvel``), subtree centers of mass (``subtree_com``), site positions and orientations
    (``site_xpos``, ``site_xmat``), together with ``qpos`` and ``qvel``. After the replay,
    :meth:`extend_trajectory_data` returns the trajectory completed with these quantities.

    """

    def __init__(self, env, n_samples, model, body_names=None, site_names=None):
        """
        Constructor.

        Args:
            env: The environment replaying the trajectory.
            n_samples (int): Total number of samples that will be replayed.
            model (MjModel): Mujoco model of the environment.
            body_names (List[str], optional): Names of bodies that have to exist in the model. All bodies of the
                model are recorded in any case.
            site_names (List[str], optional): Names of sites that have to exist in the model. All sites of the model
                are recorded in any case.

        """
        self.b_names, self.b_ids = self.get_body_names_and_ids(env._model, body_names)
        self.s_names, self.s_ids = self.get_site_names_and_ids(env._model, site_names)
        dim_qpos, dim_qvel = 0, 0
        for i in range(model.njnt):
            if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                dim_qpos += 7
                dim_qvel += 6
            else:
                dim_qpos += 1
                dim_qvel += 1
        self.recorder = dict(xpos=np.zeros((n_samples, model.nbody, 3)),
                             xquat=np.zeros((n_samples, model.nbody, 4)),
                             cvel=np.zeros((n_samples, model.nbody, 6)),
                             subtree_com=np.zeros((n_samples, model.nbody, 3)),
                             site_xpos=np.zeros((n_samples, model.nsite, 3)),
                             site_xmat=np.zeros((n_samples, model.nsite, 9)),
                             qpos=np.zeros((n_samples, dim_qpos)),
                             qvel=np.zeros((n_samples, dim_qvel)))
        self.traj_model = TrajectoryModel(njnt=model.njnt,
                                          jnt_type=jnp.array(model.jnt_type),
                                          nbody=model.nbody,
                                          body_rootid=jnp.array(model.body_rootid),
                                          body_weldid=jnp.array(model.body_weldid),
                                          body_mocapid=jnp.array(model.body_mocapid),
                                          body_pos=jnp.array(model.body_pos),
                                          body_quat=jnp.array(model.body_quat),
                                          body_ipos=jnp.array(model.body_ipos),
                                          body_iquat=jnp.array(model.body_iquat),
                                          nsite=model.nsite,
                                          site_bodyid=jnp.array(model.site_bodyid),
                                          site_pos=jnp.array(model.site_pos),
                                          site_quat=jnp.array(model.site_quat))
        self.current_length = 0

    def __call__(self, env, model, data, traj_sample, carry):
        """
        Apply the trajectory sample to the simulation and record the resulting quantities.

        Args:
            env: The environment replaying the trajectory.
            model (MjModel): Mujoco model.
            data (MjData): Mujoco data structure.
            traj_sample (TrajectoryData): The current sample of the trajectory.
            carry: Additional carry of the environment.

        Returns:
            Tuple[MjModel, MjData, Any]: The model, the updated data and the updated carry.

        """
        model, data, carry = super().__call__(env, model, data, traj_sample, carry)

        self.recorder["xpos"][self.current_length] = data.xpos[self.b_ids]
        self.recorder["xquat"][self.current_length] = data.xquat[self.b_ids]
        self.recorder["cvel"][self.current_length] = data.cvel[self.b_ids]
        self.recorder["subtree_com"][self.current_length] = data.subtree_com[self.b_ids]
        self.recorder["site_xpos"][self.current_length] = data.site_xpos[self.s_ids]
        self.recorder["site_xmat"][self.current_length] = data.site_xmat[self.s_ids]

        # add joint properties
        self.recorder["qpos"][self.current_length] = data.qpos
        self.recorder["qvel"][self.current_length] = data.qvel

        self.current_length += 1

        return model, data, carry

    def extend_trajectory_data(self, traj_data: TrajectoryData, traj_info: TrajectoryInfo):
        """
        Add the recorded quantities to a trajectory.

        Args:
            traj_data (TrajectoryData): Data of the replayed trajectory. Its number of samples has to equal the number
                of recorded samples.
            traj_info (TrajectoryInfo): Info of the replayed trajectory.

        Returns:
            Tuple[TrajectoryData, TrajectoryInfo]: The data with the recorded quantities, and the info with the body
            names, the site names and the model description (:class:`~loco_mujoco.core.trajectory.dataclasses.TrajectoryModel`)
            of the replaying model.

        """
        assert self.current_length == traj_data.qpos.shape[0]
        assert traj_info.model.njnt == self.traj_model.njnt
        converted_data = {}
        for key, value in self.recorder.items():
            converted_data[key] = jnp.array(value)
        return (traj_data.replace(**converted_data),
                replace(traj_info, body_names=self.b_names if len(self.b_names) > 0 else None,
                        site_names=self.s_names if len(self.s_names) > 0 else None,
                        model=self.traj_model))

    @staticmethod
    def get_body_names_and_ids(model, keys=None):
        """
        Get the names and ids of all bodies in the model.

        If ``keys`` is given, it is additionally checked that all of these body names exist in the model.

        Args:
            model (MjModel): Mujoco model.
            keys (List[str], optional): Body names that have to exist in the model.

        Returns:
            Tuple[List[str], List[int]]: Names and ids of all bodies in the model.

        Raises:
            AssertionError: If a name in ``keys`` is not found in the model.

        """
        keys = deepcopy(keys)
        body_names = []
        ids = range(model.nbody)
        for i in ids:
            b_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if (keys is not None) and (b_name in keys):
                keys.remove(b_name)
            body_names.append(b_name)
        assert keys is None or len(keys) == 0, f"Could not find the following body names: {keys}"
        return body_names, list(ids)

    @staticmethod
    def get_site_names_and_ids(model, keys=None):
        """
        Get the names and ids of all sites in the model.

        If ``keys`` is given, it is additionally checked that all of these site names exist in the model.

        Args:
            model (MjModel): Mujoco model.
            keys (List[str], optional): Site names that have to exist in the model.

        Returns:
            Tuple[List[str], List[int]]: Names and ids of all sites in the model.

        Raises:
            AssertionError: If a name in ``keys`` is not found in the model.

        """
        keys = deepcopy(keys)
        site_names = []
        ids = range(model.nsite)
        for i in ids:
            s_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
            if (keys is not None) and (s_name in keys):
                keys.remove(s_name)
            site_names.append(s_name)
        assert keys is None or len(keys) == 0, f"Could not find the following site names: {keys}"
        return site_names, list(ids)


def add_mocap_bodies(mjspec: MjSpec,
                     sites_for_mimic: List[str],
                     mocap_bodies: List[str],
                     robot_conf: DictConfig = None,
                     add_equality_constraint: bool = True,
                     height_adjustment_geom_names: List[str] = None,
                     max_height_adjustment: float = 1.0):
    """
    Add mocap bodies to the model specification and prepare the model for retargeting.

    One mocap body with a visual box site is added per name in ``mocap_bodies``. If requested, each mocap body is
    connected to its mimic site with an equality constraint, so that moving the mocap bodies pulls the robot into the
    corresponding pose. The robot configuration can additionally disable joint limits and collisions, set the
    stiffness of the joint limits, and select the type and the solver parameters of each constraint.

    Args:
        mjspec (MjSpec): The model specification. It is modified in place.
        sites_for_mimic (List[str]): The names of the mimic sites of the robot.
        mocap_bodies (List[str]): The names of the mocap bodies to be added to the model specification, one per
            mimic site and in the same order.
        robot_conf (DictConfig): Robot configuration for retargeting (see ``loco_mujoco/smpl/robot_confs``). If
            None, weld constraints with default parameters are used and the model is not changed otherwise.
        add_equality_constraint (bool): Whether to add equality constraints between the sites and the mocap bodies.
        height_adjustment_geom_names (List[str]): The names of the geoms to be used for height adjustment of the
            trajectory. Their contact margin and gap are set to ``max_height_adjustment``, so that their distance
            to the floor is available up to that value.
        max_height_adjustment (float): The maximum height adjustment for the geoms.

    Returns:
        MjSpec: The modified model specification.

    """

    if robot_conf is not None and robot_conf.optimization_params.disable_joint_limits:
        for j in mjspec.joints:
            j.limited = False

    if robot_conf is not None:
        # stiffness of the joint limits in the retargeting model only; None keeps the model's own
        limit_solref = robot_conf.optimization_params.get("joint_limit_solref")
        limit_solimp = robot_conf.optimization_params.get("joint_limit_solimp")
        for j in mjspec.joints:
            if limit_solref is not None:
                j.solref_limit = list(limit_solref)
            if limit_solimp is not None:
                j.solimp_limit = list(limit_solimp)

    if robot_conf is not None and robot_conf.optimization_params.disable_collisions:
        for g in mjspec.geoms:
            g.contype = 0
            g.conaffinity = 0

    for g in mjspec.geoms:
        if height_adjustment_geom_names is not None and g.name in height_adjustment_geom_names:
            g.margin = max_height_adjustment
            g.gap = max_height_adjustment

    for mb_name in mocap_bodies:
        b_handle = mjspec.worldbody.add_body(name=mb_name, mocap=True)
        b_handle.add_site(name=mb_name, type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.05, 0.01],
                          rgba=[0.0, 1.0, 0.0, 0.5], group=1)

    if add_equality_constraint:
        for b1, b2 in zip(sites_for_mimic, mocap_bodies):
            if robot_conf is not None:
                eq_type = getattr(mujoco.mjtEq, robot_conf.site_joint_matches[b1].equality_constraint_type)
                torque_scale = robot_conf.site_joint_matches[b1].torque_scale
            else:
                eq_type = mujoco.mjtEq.mjEQ_WELD
                torque_scale = 1.0

            # the length of the constraint data depends on the MuJoCo version (11 in 3.x, 12 in 4.x)
            constraint_data = np.zeros(mujoco.mjNEQDATA)
            constraint_data[10] = torque_scale
            e = mjspec.add_equality(type=eq_type, name1=b1, name2=b2,
                                objtype=mujoco.mjtObj.mjOBJ_SITE, data=constraint_data)

            if robot_conf is not None:
                if hasattr(robot_conf.site_joint_matches[b1], "solref"):
                    assert len(robot_conf.site_joint_matches[b1].solref) == 2, "solref must be a list of length 2"
                    e.solref = robot_conf.site_joint_matches[b1].solref
                if hasattr(robot_conf.site_joint_matches[b1], "solimp"):
                    assert len(robot_conf.site_joint_matches[b1].solimp) == 5, "solimp must be a list of length 5"
                    e.solimp = robot_conf.site_joint_matches[b1].solimp

    return mjspec


def calculate_qvel_with_finite_difference(qpos, frequency):
    """
    Calculate joint velocities (qvel) using finite differences for a given trajectory.

    This function computes joint velocities by applying a central finite difference
    method to the joint positions (``qpos``). It handles the free joint (the floating base
    with translational and rotational components) separately from the other joints. Since central
    differences are not defined for the first and the last sample, these two samples are dropped.

    Args:
        qpos (numpy.ndarray): A 2D array of shape ``(nsamples, qpos_dim)`` representing the joint positions.
            The first 7 columns correspond to the free joint (3 for position and 4 for orientation
            as a quaternion with the scalar part first), and the remaining columns correspond to joints with a
            single degree of freedom.
        frequency (float): The sampling frequency of the trajectory in Hz.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: The joint positions without the first and the last sample, shape
        ``(nsamples - 2, qpos_dim)``, and the joint velocities, shape ``(nsamples - 2, qpos_dim - 1)``. The
        velocities of the free joint consist of 3 translational components and 3 rotational components, the latter
        computed as the finite difference of the orientation expressed as a rotation vector.

    Notes:
        The central finite difference formula is ``(qpos[t+1] - qpos[t-1]) / (2 * delta_t)``.

    """

    free_joint_qpos_ids = np.arange(7)
    free_joint_qvel_ids = np.arange(6)
    njnt = qpos.shape[1] - 7
    joints_qpos_ids = np.arange(7, 7+njnt)
    joints_qvel_ids = np.arange(6, 6+njnt)

    # Calculate qvel using finite difference
    free_joint_pos = qpos[:, free_joint_qpos_ids[:3]]
    free_joint_quat_scalarfirst = qpos[:, free_joint_qpos_ids[3:]]
    free_joint_quat = quat_scalarfirst2scalarlast(free_joint_quat_scalarfirst)
    free_joint_rotvec = sRot.from_quat(free_joint_quat).as_rotvec()
    joint_qpos = qpos[:, joints_qpos_ids]

    delta_t = 1 / frequency
    free_joint_vel = (free_joint_pos[2:] - free_joint_pos[:-2]) / (2 * delta_t)
    free_joint_vel_rot = (free_joint_rotvec[2:] - free_joint_rotvec[:-2]) / (2 * delta_t)
    joints_vel = (joint_qpos[2:] - joint_qpos[:-2]) / (2 * delta_t)

    qvel = np.concatenate([free_joint_vel, free_joint_vel_rot, joints_vel], axis=1)
    qpos = qpos[1:-1]

    return qpos, qvel


def expression_constructor(loader, node):
    """
    YAML constructor for the tag ``!expr``, which evaluates a scalar as a Python expression.

    Only the names ``pi``, ``math`` and ``np`` (an alias of the ``math`` module) are available in the expression.

    Args:
        loader: The YAML loader.
        node: The scalar node holding the expression.

    Returns:
        The value of the expression.

    Raises:
        ValueError: If the expression cannot be evaluated.

    """
    # Get the scalar value (the expression in string form)
    value = loader.construct_scalar(node)
    try:
        # Safely evaluate the expression
        return eval(value, {"__builtins__": None}, {"pi": math.pi, "np": math, "math": math})
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {value}, {e}")


def load_robot_conf(file_path):
    """
    Load a robot configuration from a YAML file, with support for the tag ``!expr``.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: The content of the file.

    """
    # Register the custom constructor for the tag '!expr'
    yaml.SafeLoader.add_constructor('!expr', expression_constructor)
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def load_dataset_conf(file_path):
    """
    Load a dataset configuration from a YAML file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: The content of the file.

    """
    # simple yaml loader
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
