import os
import glob
from dataclasses import replace
import logging
from typing import Union, List, Dict
import hashlib

import yaml
from omegaconf import DictConfig, OmegaConf
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as sRot

try:
    from tqdm import tqdm
    import torch
    from torch.autograd import Variable
    from smplx.lbs import transform_mat
    import joblib
    from loco_mujoco.smpl import SMPLH_Parser
    _OPTIONAL_IMPORT_INSTALLED = True
except ImportError as e:
    _OPTIONAL_IMPORT_INSTALLED = False
    _OPTIONAL_IMPORT_EXCEPTION = e

import loco_mujoco
from loco_mujoco.smpl import SMPLH_BONE_ORDER_NAMES
from loco_mujoco.smpl.utils.smoothing import gaussian_filter_1d_batch
from loco_mujoco.environments import LocoEnv
from loco_mujoco.core.utils.math import quat_scalarlast2scalarfirst, quat_scalarfirst2scalarlast
from loco_mujoco.core.trajectory import (
    Trajectory,
    TrajectoryInfo,
    TrajectoryModel,
    TrajectoryData,
    interpolate_trajectories,
)
from loco_mujoco.datasets.data_generation import ExtendTrajData, optimize_for_collisions
from loco_mujoco.core.utils.mujoco import mj_get_collision_dist_and_normal
from loco_mujoco import PATH_TO_SMPL_ROBOT_CONF
from loco_mujoco.utils import setup_logger
from loco_mujoco.core.utils.mujoco import mj_jntname2qposid, mj_jntname2qvelid
from loco_mujoco.datasets.data_generation.utils import add_mocap_bodies


OPTIMIZED_SHAPE_FILE_NAME = "shape_optimized.pkl"


def check_optional_imports():
    if not _OPTIONAL_IMPORT_INSTALLED:
        raise ImportError(f"[LocoMuJoCo] Optional smpl depencies not installed. "
                          f"Checkout the README for installation instructions. {_OPTIONAL_IMPORT_EXCEPTION}")


def get_amass_dataset_path():
    path_to_conf = loco_mujoco.PATH_TO_VARIABLES
    with open(path_to_conf, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        path_to_amass_datasets = data["LOCOMUJOCO_AMASS_PATH"]

    assert path_to_amass_datasets, "Please set the environment variable LOCOMUJOCO_AMASS_PATH."

    return path_to_amass_datasets


def get_converted_amass_dataset_path():
    path_to_conf = loco_mujoco.PATH_TO_VARIABLES
    with open(path_to_conf, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        path_to_converted_amass_datasets = data["LOCOMUJOCO_CONVERTED_AMASS_PATH"]

    assert path_to_converted_amass_datasets, (
        "Please set the environment variable LOCOMUJOCO_CONVERTED_AMASS_PATH."
    )

    return path_to_converted_amass_datasets


def get_smpl_model_path():
    path_to_conf = loco_mujoco.PATH_TO_VARIABLES
    with open(path_to_conf, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        path_to_smpl_model = data["LOCOMUJOCO_SMPL_MODEL_PATH"]

    assert path_to_smpl_model, "Please set the environment variable LOCOMUJOCO_SMPL_MODEL_PATH."

    return path_to_smpl_model


def load_amass_data(data_path: str) -> dict:
    """Load AMASS data from a file.

    Args:
        data_path (str): Path to the AMASS data file.


    Returns:
        dict: Parsed AMASS data including poses, translations, and other attributes.

    """
    path_to_amass_datasets = get_amass_dataset_path()

    # get paths to all amass files
    path_to_all_amass_files = os.path.join(path_to_amass_datasets, "**/*.npz")
    all_pkls = glob.glob(path_to_all_amass_files, recursive=True)

    # get full dataset path
    key_names = [
        "/".join(data_path.replace(path_to_amass_datasets + "/", "").split("/")).replace(".npz", "")
        for data_path in all_pkls]
    if data_path.startswith("/"):
        data_path = data_path[1:]
    data_path = data_path.replace(".npz", "")
    data_path = all_pkls[key_names.index(data_path)]

    # load data
    entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))

    if 'mocap_framerate' in entry_data:
        framerate = entry_data['mocap_framerate']
    elif 'mocap_frame_rate' in entry_data:
        framerate = entry_data['mocap_frame_rate']
    else:
        raise ValueError("Framerate not found in the data file.")

    root_trans = entry_data['trans']
    pose_aa = np.concatenate([entry_data['poses'][:, :66], np.zeros((root_trans.shape[0], 6))], axis=-1)
    betas = entry_data['betas']
    gender = entry_data['gender']

    return {
        "pose_aa": pose_aa,
        "gender": gender,
        "trans": root_trans,
        "betas": betas,
        "fps": framerate,
    }


def load_robot_conf_file(env_name: str):
    """Load a robot configuration file."""
    if "Mjx" in env_name:
        conf_name = env_name.replace("Mjx", "")
    else:
        conf_name = env_name
    filename = f"{conf_name}.yaml"
    filepaths = [os.path.join(PATH_TO_SMPL_ROBOT_CONF, filename)]
    custom_smpl_robot_confs = loco_mujoco.get_variable("LOCOMUJOCO_CUSTOM_SMPL_CONF_PATH")
    custom_smpl_robot_confs = (
        [custom_smpl_robot_confs] if isinstance(custom_smpl_robot_confs, str)
        else (custom_smpl_robot_confs or [])
    )
    filepaths += [os.path.join(p, filename) for p in custom_smpl_robot_confs]
    for filepath in filepaths:
        if not os.path.exists(filepath):
            continue
        default_conf = OmegaConf.load(PATH_TO_SMPL_ROBOT_CONF / "defaults.yaml")
        robot_conf = OmegaConf.load(filepath)
        robot_conf = OmegaConf.merge(default_conf, robot_conf)
        return robot_conf

    raise FileNotFoundError(f"YAML file '{filename}' not found in paths: {filepaths}")


def to_t_pose(env, robot_conf):
    """
    Set the humanoid to a T-pose by modifying the Mujoco Data structure.

    Args:
        env: environment.
        robot_conf: robot configuration file including the joint positions for the T-pose.

    """
    data = env._data
    # apply init pose modifiers
    for modifier in robot_conf.robot_pose_modifier:
        name, val = list(modifier.items())[0]
        if name != "root":
            # convert string to numpy value
            val = np.array(eval(val))
            qpos_id = mj_jntname2qposid(name, env._model)
            data.qpos[qpos_id] += val
        else:
            # convert string to numpy value
            val = sRot.from_euler("xyz", eval(val), degrees=False).as_quat()
            val = quat_scalarlast2scalarfirst(val)
            data.qpos[3:7] += val


def fit_smpl_motion(
    env_name: str,
    robot_conf: DictConfig,
    path_to_smpl_model: str,
    motion_data: Union[str, Dict],
    path_to_optimized_smpl_shape: str,
    logger: logging.Logger,
    skip_steps: bool = True,
    visualize: bool = False
) -> Trajectory:
    """Fit SMPL motion data to a robot configuration.

    Args:
        env_name (str): Name of the environment.
        robot_conf (DictConfig): Configuration of the robot.
        path_to_smpl_model (str): Path to the SMPL model.
        motion_data (Dict): Dict containing the motion data to process.
        path_to_optimized_smpl_shape (str): Path to the optimized SMPL shape file for the robot.
        logger (logging.Logger): Logger for status updates.
        visualize (bool): Whether to visualize the optimization process.

    Returns:
        Trajectory: The fitted motion trajectory.

    """

    def get_xpos_and_xquat(smpl_positions, smpl_rot_mats, s2m_pos, s2m_rot_mat):

        # get rotations of mimic sites
        new_smpl_rot_mats = np.einsum("bij,bjk->bik",  smpl_rot_mats, s2m_rot_mat)
        new_smpl_quat = sRot.from_matrix(new_smpl_rot_mats).as_quat()
        new_smpl_quat = quat_scalarlast2scalarfirst(new_smpl_quat)
        pos_offset = np.einsum("bij,bj->bi", new_smpl_rot_mats, s2m_pos)
        new_smpl_pos = torch.squeeze(smpl_positions - pos_offset)

        return new_smpl_pos, new_smpl_quat

    check_optional_imports()

    # get environment
    env_cls = LocoEnv.registered_envs[env_name]
    env = env_cls(**robot_conf.env_params, th_params=dict(random_start=False, fixed_start_conf=(0, 0)))

    # add mocap bodies for all 'site_for_mimic' instances of an environment
    mjspec = env.mjspec
    sites_for_mimic = env.sites_for_mimic
    site_ids = [mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_SITE, s) for s in sites_for_mimic]
    target_mocap_bodies = ["target_mocap_body_" + s for s in sites_for_mimic]
    mjspec = add_mocap_bodies(mjspec, sites_for_mimic, target_mocap_bodies, robot_conf,
                              add_equality_constraint=True,
                              height_adjustment_geom_names=robot_conf.optimization_params.height_adjustment_geom_names,
                              max_height_adjustment=robot_conf.optimization_params.max_height_adjustment)
    env.reload_mujoco(mjspec)
    key = jax.random.key(0)
    env.reset(key)

    smpl2mimic_site_idx = []
    for s in sites_for_mimic:
        # find smpl name
        for site_name, conf in robot_conf.site_joint_matches.items():
            if site_name == s:
                smpl2mimic_site_idx.append(SMPLH_BONE_ORDER_NAMES.index(conf.smpl_joint))

    smpl_parser_n = SMPLH_Parser(model_path=path_to_smpl_model, gender="neutral")

    shape_new, scale, smpl2robot_pos, smpl2robot_rot_mat, offset_z, height_scale = (
        joblib.load(path_to_optimized_smpl_shape))

    skip = robot_conf.optimization_params.skip_frames if skip_steps else 1
    pose_aa = torch.from_numpy(motion_data['pose_aa'][::skip]).float()
    pose_aa = torch.cat(
        [pose_aa, torch.zeros((pose_aa.shape[0], 156 - pose_aa.shape[1]))], axis=-1
    )
    len_traj = pose_aa.shape[0]

    total_z_offet = offset_z + robot_conf.optimization_params.z_offset_feet
    trans = (torch.from_numpy(motion_data['trans'][::skip]) + torch.tensor([0.0, 0.0, total_z_offet]))

    # apply height scaling while perserving init height
    trans[:, :2] *= height_scale    # scale x and y
    trans[:, 2] = (trans[: , 2] - trans[0 , 2]) * height_scale + trans[0 , 2]

    with torch.no_grad():
        transformations_matrices = smpl_parser_n.get_joint_transformations(pose_aa.reshape(len_traj, -1, 3),
                                                                           shape_new.repeat(len_traj, 1), trans)
        global_pos = transformations_matrices[..., :3, 3]
        global_rot_mats = transformations_matrices[..., :3, :3].detach().numpy()
        root_pos = global_pos[:, 0:1]
        global_pos = (global_pos - global_pos[:, 0:1]) * scale.detach() + root_pos

    # calculate initial qpos from initial mocap pos
    init_mocap_pos, init_mocap_quat = get_xpos_and_xquat(global_pos[0, smpl2mimic_site_idx],
                                                         global_rot_mats[0, smpl2mimic_site_idx],
                                                         smpl2robot_pos, smpl2robot_rot_mat)
    qpos_init = get_init_qpos_for_motion_retargeting(env, init_mocap_pos, init_mocap_quat, robot_conf)
    env._data.qpos = qpos_init

    qpos = np.zeros((len_traj, env._model.nq))

    min_dist_feet_floor = np.inf if len(robot_conf.optimization_params.height_adjustment_geom_names) > 0 else 0.0
    for i in tqdm(range(len_traj)):

        mocap_pos, mocap_quat = get_xpos_and_xquat(global_pos[i, smpl2mimic_site_idx],
                                                   global_rot_mats[i, smpl2mimic_site_idx],
                                                   smpl2robot_pos, smpl2robot_rot_mat)
        env._data.mocap_pos = mocap_pos
        env._data.mocap_quat = mocap_quat

        mujoco.mj_step(env._model, env._data, robot_conf.optimization_params.motion_iterations)

        # save qpos
        qpos[i] = env._data.qpos.copy()

        # get min distance to floor from height adjustment geoms
        floor_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        for geom in robot_conf.optimization_params.height_adjustment_geom_names:
            geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom)
            dist, _ = mj_get_collision_dist_and_normal(geom_id, floor_id, env._data, np)
            dist, _ = mj_get_collision_dist_and_normal(geom_id, floor_id, env._data, np)
            min_dist_feet_floor = min(min_dist_feet_floor, dist)

        if visualize:
            env.render()

    fps = motion_data["fps"] // skip

    free_joint_qpos_ids = mj_jntname2qposid(env.root_free_joint_xml_name, env._model)
    free_joint_qpos_mask = np.zeros(env._model.nq, dtype=bool)
    free_joint_qpos_mask[free_joint_qpos_ids] = True
    free_joint_qpos = qpos[:, free_joint_qpos_ids]
    joint_pos = qpos[:, ~free_joint_qpos_mask]

    free_joint_qvel_ids = mj_jntname2qvelid(env.root_free_joint_xml_name, env._model)
    free_joint_qvel_mask = np.zeros(env._model.nv, dtype=bool)
    free_joint_qvel_mask[free_joint_qvel_ids] = True

    free_joint_pos = free_joint_qpos[:, :3]
    free_joint_quat = free_joint_qpos[:, 3:]
    free_joint_quat_scalar_last = quat_scalarfirst2scalarlast(free_joint_quat)
    free_joint_rotvec = sRot.from_quat(free_joint_quat_scalar_last).as_rotvec()

    free_joint_vel = (free_joint_pos[2:] - free_joint_pos[:-2]) / (2 * (1 / fps))
    free_joint_vel_rot = (free_joint_rotvec[2:] - free_joint_rotvec[:-2]) / (2 * (1 / fps))
    joints_vel = (joint_pos[2:] - joint_pos[:-2]) / (2 * (1 / fps))

    free_joint_pos = free_joint_pos[1:-1]
    free_joint_quat = free_joint_quat[1:-1]
    joint_pos = joint_pos[1:-1]

    # add min distance for geom based height adjustment
    free_joint_pos[:, 2] -= min_dist_feet_floor

    # concatenate free joint qpos and qvel
    free_joint_qpos = np.concatenate([free_joint_pos, free_joint_quat], axis=1)
    free_joint_qvel = np.concatenate([free_joint_vel, free_joint_vel_rot], axis=1)

    # create empty qpos and qvel arrays
    qpos = np.zeros((len_traj-2, env._model.nq))
    qvel = np.zeros((len_traj-2, env._model.nv))

    # set qpos and qvel
    qpos[:, free_joint_qpos_mask] = free_joint_qpos
    qpos[:, ~free_joint_qpos_mask] = joint_pos
    qvel[:, free_joint_qvel_mask] = free_joint_qvel
    qvel[:, ~free_joint_qvel_mask] = joints_vel

    njnt = env._model.njnt
    jnt_type = env._model.jnt_type
    jnt_names = [
        mujoco.mj_id2name(env._model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(njnt)
    ]

    traj_info = TrajectoryInfo(
        jnt_names, model=TrajectoryModel(njnt, jnp.array(jnt_type)), frequency=fps
    )

    traj_data = TrajectoryData(
        jnp.array(qpos), jnp.array(qvel), split_points=jnp.array([0, len(qpos)])
    )

    return Trajectory(traj_info, traj_data)


def get_init_qpos_for_motion_retargeting(env, init_mocap_pos, init_mocap_quat, robot_conf):
    """
    Get the initial qpos for motion retargeting by temporarily disabling joint limits and collisions and
    running the simulation to solve for the initial qpos. This avoids problems that could arise from bad initialzation
    from the default qpos (getting stuck in joint limits or collisions).

    Args:
        env: environment.
        init_mocap_pos: initial mocap positions.
        init_mocap_quat: initial mocap quaternions.

    Returns:
        np.ndarray: initial qpos.

    """

    old_mjspec = env.mjspec.copy()
    new_mjspec = env.mjspec

    # disable joint limits and collisions
    if robot_conf.optimization_params.disable_joint_limits_on_initialization:
        for j in new_mjspec.joints:
            j.limited = False
    if robot_conf.optimization_params.disable_collisions_on_initialization:
        for g in new_mjspec.geoms:
            g.contype = 0
            g.conaffinity = 0

    env.reload_mujoco(new_mjspec)

    env._data.mocap_pos = init_mocap_pos
    env._data.mocap_quat = init_mocap_quat
    mujoco.mj_step(env._model, env._data, robot_conf.optimization_params.init_motion_iterations)
    qpos = env._data.qpos.copy()

    # load old model to env
    env.reload_mujoco(old_mjspec)
    key = jax.random.key(0)
    env.reset(key)

    return qpos


def fit_smpl_shape(
    env_name: str,
    robot_conf: DictConfig,
    path_to_smpl_model: str,
    save_path_new_smpl_shape: str,
    logger: logging.Logger,
    visualize: bool = False,
) -> None:
    """Fit the SMPL shape to match the robot configuration.

    Args:
        env_name (str): Name of the environment.
        robot_conf (DictConfig): Configuration of the robot.
        path_to_smpl_model (str): Path to the SMPL model.
        save_path_new_smpl_shape (str): Path to save the optimized shape.
        logger (logging.Logger): Logger for status updates.
        visualize (bool): Whether to visualize the optimization process.

    """

    check_optional_imports()

    Z_OFFSET = 2.0 # for visualization only

    # get environment
    env_cls = LocoEnv.registered_envs[env_name]
    env = env_cls(**robot_conf.env_params, th_params=dict(random_start=False, fixed_start_conf=(0, 0)))

    # add mocap bodies for all 'site_for_mimic' instances of an environment
    mjspec = env.mjspec
    sites_for_mimic = env.sites_for_mimic
    target_mocap_bodies = ["target_mocap_body_" + s for s in sites_for_mimic]
    mjspec = add_mocap_bodies(mjspec, sites_for_mimic, target_mocap_bodies, robot_conf, add_equality_constraint=False)
    env.reload_mujoco(mjspec)
    key = jax.random.key(0)
    env.reset(key)

    smpl2mimic_site_idx = []
    for s in sites_for_mimic:
        # find smpl name
        for site_name, conf in robot_conf.site_joint_matches.items():
            if site_name == s:
                smpl2mimic_site_idx.append(SMPLH_BONE_ORDER_NAMES.index(conf.smpl_joint))

    # set humanoid to T-pose
    to_t_pose(env, robot_conf)

    # safe initial qpos
    qpos_init = env._data.qpos.copy()
    qpos_init[0:3] = 0
    qpos_init[2] = Z_OFFSET

    # set initial qpos and forward
    env._data.qpos = qpos_init
    mujoco.mj_forward(env._model, env._data)

    # get joint names
    robot_joint_pick = [i for i in robot_conf.site_joint_matches.keys()]

    device = torch.device(robot_conf.optimization_params.torch_device)

    # get initial pose
    pose_aa_stand = np.zeros((1, 156)).reshape(-1, 52, 3)
    pose_aa_stand[:, SMPLH_BONE_ORDER_NAMES.index("Pelvis")] = sRot.from_euler("xyz",
                                                                               [np.pi/2, 0.0 , np.pi/2],
                                                                               degrees=False).as_rotvec()
    pose_aa_stand = torch.from_numpy(pose_aa_stand.reshape(-1, 156)).requires_grad_(False)

    # setup parser
    smpl_parser_n = SMPLH_Parser(model_path=path_to_smpl_model, gender="neutral")

    # define optimization variables
    shape_new = Variable(torch.zeros([1, 16]).to(device), requires_grad=True)
    scale = Variable(torch.ones([1]).to(device), requires_grad=True)
    trans = torch.zeros([1, 3]).requires_grad_(False)
    optimizer_shape = torch.optim.Adam([shape_new, scale], lr=robot_conf.optimization_params.shape_lr)

    # get target site positions
    site_ids = [mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_SITE, s) for s in sites_for_mimic]
    target_site_pos = torch.from_numpy(env._data.site_xpos[site_ids])[None]
    target_site_mat = torch.from_numpy(env._data.site_xmat[site_ids])

    # get z offset
    z_offset = torch.tensor([0.0, 0.0, Z_OFFSET])[None]

    # get the transformation matrices
    transformations_matrices = smpl_parser_n.get_joint_transformations(pose_aa_stand, shape_new, trans)
    global_rot_mats = transformations_matrices.detach().numpy()[..., :3, :3]
    global_pos = transformations_matrices.detach().numpy()[..., :3, 3]
    global_rot_mats = sRot.from_matrix(global_rot_mats[0, smpl2mimic_site_idx])

    # get rotations of mimic sites
    target_site_mat = sRot.from_matrix(target_site_mat.detach().numpy().reshape(-1, 3, 3))

    # rel rotation smpl to robot
    smpl2robot_rot_mat = np.einsum("bij,bjk->bik",
                                   global_rot_mats.inv().as_matrix(), target_site_mat.as_matrix())

    # transform smpl rotations to match robot site rotations
    # (here done just for visualization, only used in motion_fit function)
    new_smpl_rot_mats = np.einsum("bij,bjk->bik",  global_rot_mats.as_matrix(), smpl2robot_rot_mat)
    new_smpl_quats = quat_scalarlast2scalarfirst(sRot.from_matrix(new_smpl_rot_mats).as_quat())

    pbar = tqdm(range(robot_conf.optimization_params.shape_iterations))
    init_feet_z_pos = None
    init_head_z_pos = None
    for iteration in pbar:

        transformations_matrices = smpl_parser_n.get_joint_transformations(pose_aa_stand, shape_new, trans)
        global_pos = transformations_matrices[..., :3, 3]

        if init_feet_z_pos is None:
            init_feet_z_pos = np.minimum(global_pos[0, SMPLH_BONE_ORDER_NAMES.index("R_Ankle"), 2].detach().numpy(),
                                         global_pos[0, SMPLH_BONE_ORDER_NAMES.index("L_Ankle"), 2].detach().numpy())
            init_head_z_pos = global_pos[0, SMPLH_BONE_ORDER_NAMES.index("Head"), 2].detach().numpy()

        if iteration == robot_conf.optimization_params.shape_iterations - 1:
            final_feet_z_pos = np.minimum(global_pos[0, SMPLH_BONE_ORDER_NAMES.index("R_Ankle"), 2].detach().numpy(),
                                          global_pos[0, SMPLH_BONE_ORDER_NAMES.index("L_Ankle"), 2].detach().numpy())
            final_head_z_pos = global_pos[0, SMPLH_BONE_ORDER_NAMES.index("Head"), 2].detach().numpy()

        root_pos = global_pos[:, 0]
        global_pos = (global_pos - global_pos[:, 0] + z_offset) * scale

        if visualize:
            env._data.qpos = qpos_init
            env._data.mocap_pos = torch.squeeze(global_pos[:, smpl2mimic_site_idx]).detach().numpy()
            env._data.mocap_quat = new_smpl_quats

            mujoco.mj_forward(env._model, env._data)
            env.render()

        # calculate loss
        diff = target_site_pos - global_pos[:, smpl2mimic_site_idx]
        loss = diff.norm(dim=-1).mean()

        pbar.set_description_str(f"{iteration} - Loss: {loss.item() * 1000}")

        optimizer_shape.zero_grad()
        loss.backward(retain_graph=True)
        optimizer_shape.step()

    # save positions offset
    smpl_pos = global_pos[0, smpl2mimic_site_idx].detach().numpy()
    smpl2robot_pos = smpl_pos - target_site_pos.detach().numpy()
    smpl2robot_pos = np.squeeze(smpl2robot_pos)

    # save new z-offset
    offset_z = init_feet_z_pos - final_feet_z_pos
    height_scale = (final_head_z_pos - final_feet_z_pos) / (init_head_z_pos - init_feet_z_pos)

    # Extract the directory path from the save path
    directory = os.path.dirname(save_path_new_smpl_shape)

    # Create the directory if it does not exist
    os.makedirs(directory, exist_ok=True)

    # save
    joblib.dump((shape_new.detach(), scale, smpl2robot_pos, smpl2robot_rot_mat,
                 offset_z, height_scale), save_path_new_smpl_shape)
    logger.info(f"Shape parameters saved at {save_path_new_smpl_shape}")


def motion_transfer_robot_to_robot(
    env_name_source: str,
    robot_conf_source: DictConfig,
    traj_source: Trajectory,
    path_source_robot_smpl_data: str,
    env_name_target: str,
    robot_conf_target: DictConfig,
    path_target_robot_smpl_data: str,
    path_to_smpl_model: str,
    logger: logging.Logger,
    path_to_fitted_motion_source: str = None,
    visualize: bool = False) -> Trajectory:

    def rotation_matrix_loss_geodesic(R1, R2):
        """
        Computes the geodesic distance loss between two rotation matrices with two batch dimensions.

        R1, R2: (B1, B2, 3, 3)
        Returns: Mean geodesic loss over (B1, B2)
        """
        R_diff = torch.matmul(R1.transpose(-2, -1), R2)  # R1^T * R2, supports (B1, B2, 3, 3)
        trace = torch.einsum("...ii->...", R_diff)  # Extract trace along last two dims (B1, B2)
        eps = 1e-6  # Small epsilon for numerical stability
        theta = torch.acos(torch.clamp((trace - 1) / 2, -1.0 + eps, 1.0 - eps))
        return theta.mean()

    check_optional_imports()

    path_to_target_robot_smpl_shape = os.path.join(path_target_robot_smpl_data, OPTIMIZED_SHAPE_FILE_NAME)

    if path_to_fitted_motion_source is not None and not os.path.exists(path_to_fitted_motion_source):

        device = torch.device("cuda")

        # get the source env
        env_cls = LocoEnv.registered_envs[env_name_source]
        env = env_cls(**robot_conf_source.env_params, th_params=dict(random_start=False, fixed_start_conf=(0, 0)))

        # add mocap bodies for all 'site_for_mimic' instances of an environment
        mjspec = env.mjspec
        sites_for_mimic = env.sites_for_mimic
        target_mocap_bodies = ["target_mocap_body_" + s for s in sites_for_mimic]
        mjspec = add_mocap_bodies(mjspec, sites_for_mimic, target_mocap_bodies, robot_conf_source, add_equality_constraint=False)
        env.reload_mujoco(mjspec)
        key = jax.random.key(0)
        env.reset(key)

        # extend the trajectory to include more model-specific entities
        traj_source = extend_motion(env_name_source, robot_conf_source.env_params, traj_source, logger)

        # load the source trajectory
        env.process_trajectory(traj_source)

        # convert traj to numpy
        env._traj = env._traj.replace(data=TrajectoryHandler.to_numpy(env._traj.data))

        # get the body_shape of the source robot
        path_to_source_robot_smpl_shape = os.path.join(path_source_robot_smpl_data, OPTIMIZED_SHAPE_FILE_NAME)
        if not os.path.exists(path_to_source_robot_smpl_shape):
            logger.info("Robot shape file not found, fitting new one ...")
            fit_smpl_shape(env_name_source, robot_conf_source, path_to_smpl_model, path_to_source_robot_smpl_shape, logger)
        else:
            logger.info(f"Found existing robot shape file at {path_to_source_robot_smpl_shape}")
        (shape_source, scale_source, smpl2robot_pos_source, smpl2robot_rot_mat_source,
         offset_z_source, height_scale_source) = joblib.load(path_to_source_robot_smpl_shape)

        # get the source site positions used as a target for optimization
        sites_for_mimic = env.sites_for_mimic
        site_ids = np.array([mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_SITE, s) for s in sites_for_mimic])
        target_site_pos = torch.from_numpy(env._traj.data.site_xpos[:, site_ids])
        target_site_mat = torch.from_numpy(env._traj.data.site_xmat[:, site_ids])
        len_dataset = env._traj.data.n_samples

        # define the optimization variables
        pose = np.zeros([len_dataset, 156]).reshape(-1, 52, 3)
        init_rot_mat = target_site_mat[:, sites_for_mimic.index("pelvis_mimic")].reshape(-1, 3, 3)
        init_rot_mat = np.einsum("nij,jk->nik", init_rot_mat, np.linalg.inv(smpl2robot_rot_mat_source[sites_for_mimic.index("pelvis_mimic")]))
        pose[:, SMPLH_BONE_ORDER_NAMES.index("Pelvis")] = sRot.from_matrix(init_rot_mat).as_rotvec()
        pose = torch.from_numpy(pose.reshape(-1, 156))
        pose = Variable(pose.float().to(device), requires_grad=True)
        trans = target_site_pos[:, sites_for_mimic.index("pelvis_mimic")].clone().to(device).requires_grad_(True)
        optimizer = torch.optim.Adam([pose, trans], lr=robot_conf_source.optimization_params.pose_lr)
        scale_source = torch.tensor(scale_source).float().to(device).detach()

        # setup parser
        smpl_parser_n = SMPLH_Parser(model_path=path_to_smpl_model, gender="neutral").to(device)

        smpl2mimic_site_idx = []
        for s in sites_for_mimic:
            # find smpl name
            for site_name, conf in robot_conf_source.site_joint_matches.items():
                if site_name == s:
                    smpl2mimic_site_idx.append(SMPLH_BONE_ORDER_NAMES.index(conf.smpl_joint))

        shape_source = shape_source.repeat(len_dataset, 1).detach().to(device)

        # convert target site poses from site frame to smpl frame
        robot2smpl_pos_source = -smpl2robot_pos_source
        robot2smpl_rot_mat_source = np.linalg.inv(smpl2robot_rot_mat_source)
        pos_offset = np.einsum("nbij,bj->nbi", target_site_mat.reshape(len_dataset, -1, 3, 3),
                               robot2smpl_pos_source)
        target_site_mat = np.einsum("nbij,bjk->nbik", target_site_mat.reshape(len_dataset, -1, 3, 3),
                                    robot2smpl_rot_mat_source)
        target_site_pos = target_site_pos - pos_offset

        # convert to torch
        target_site_pos = target_site_pos.float().to(device)
        target_site_mat = torch.from_numpy(target_site_mat).float().to(device)

        iterations = robot_conf_source.optimization_params.pose_iterations
        pos_loss_weight = robot_conf_source.optimization_params.pos_loss_weight
        rot_loss_weight = robot_conf_source.optimization_params.rot_loss_weight
        for i in tqdm(range(iterations)):

            # sample random indices
            transformations_matrices = smpl_parser_n.get_joint_transformations(pose,
                                                                               shape_source,
                                                                               trans)

            # get the global positions and rotations
            global_pos = transformations_matrices[..., :3, 3]
            global_rot_mats = transformations_matrices[..., :3, :3]

            # scale
            global_pos = (global_pos - global_pos[:, 0:1]) * scale_source + global_pos[:, 0:1]

            # calculate the loss
            pos_loss = (target_site_pos - global_pos[:, smpl2mimic_site_idx]).norm(dim=-1).mean()
            mat_loss = rotation_matrix_loss_geodesic(global_rot_mats[:, smpl2mimic_site_idx],
                                                     target_site_mat.reshape(len_dataset, -1, 3, 3))
            root_consistency_loss = torch.norm(pose[:1, :3] - pose[1:, :3], p=2).mean()

            if torch.any(torch.isnan(mat_loss)):
                raise ValueError("NaN in rotation matrix loss.")

            loss =  pos_loss_weight * pos_loss + rot_loss_weight * mat_loss + 0.0 * root_consistency_loss

            if visualize:
                index = 0

                # convert smpl frame to site frame for visualization
                new_global_pos = global_pos[index, smpl2mimic_site_idx].cpu().detach().numpy()
                new_smpl_rot_mats = np.einsum("bij,bjk->bik",
                                              global_rot_mats[index, smpl2mimic_site_idx].cpu().detach().numpy(),
                                              smpl2robot_rot_mat_source)
                pos_offset = np.einsum("bij,bj->bi", new_smpl_rot_mats, smpl2robot_pos_source)
                new_global_pos = np.squeeze(new_global_pos - pos_offset)

                env._data.mocap_pos = new_global_pos
                env._data.mocap_quat = quat_scalarlast2scalarfirst(
                    sRot.from_matrix(new_smpl_rot_mats).as_quat())
                env._data.qpos = env._traj.data.qpos[index]
                mujoco.mj_forward(env._model, env._data)
                env.render()

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # apply smoothing
            kernel_size = robot_conf_target.optimization_params.smoothing_kernel_size
            sigma = robot_conf_target.optimization_params.smoothing_sigma
            if sigma > 0:
                pose_non_filtered = pose[:, 3:].reshape(len_dataset, -1, 3)
                pose_non_filtered = pose_non_filtered.permute(1, 2, 0)
                pose_filtered = gaussian_filter_1d_batch(pose_non_filtered, kernel_size, sigma, device)
                pose_filtered = pose_filtered.permute(2, 0, 1)
                pose.data[:, 3:] = pose_filtered.reshape(-1, 153)

        motion_file = {"pose_aa": pose.cpu().detach().numpy().reshape(-1, 156),
                       "trans": trans.cpu().detach().numpy(),
                       "fps": env._traj.info.frequency}

        # account for scale
        motion_file["pose_aa"] /= scale_source.cpu().detach().numpy()
        motion_file["trans"][:, 2] -= offset_z_source

        # apply height scaling while perserving init height
        height_scale_source_inv = 1 / height_scale_source
        trans[:, :2].data *= height_scale_source_inv  # scale x and y
        trans[:, 2].data = (trans[:, 2] - trans[0, 2]) * height_scale_source_inv + trans[0, 2]

        if path_to_fitted_motion_source is not None:
            # create dir if it does not exist
            directory = os.path.dirname(path_to_fitted_motion_source)
            os.makedirs(directory, exist_ok=True)
            # if a file path is provided, save the fitted motion
            np.savez(path_to_fitted_motion_source, **motion_file)

    else:
        logger.info(f"Loading fitted motion from {path_to_fitted_motion_source}.")
        motion_file = np.load(path_to_fitted_motion_source)

    # generate the body_shape of the target robot if it does not exist
    if not os.path.exists(path_to_target_robot_smpl_shape):
        logger.info("Robot shape file not found, fitting new one ...")
        fit_smpl_shape(env_name_target, robot_conf_target, path_to_smpl_model, path_to_target_robot_smpl_shape,
                       logger, visualize)
    else:
        logger.info(f"Found existing robot shape file at {path_to_target_robot_smpl_shape}")

    traj_target = fit_smpl_motion(env_name_target, robot_conf_target, path_to_smpl_model, motion_file,
                                  path_to_target_robot_smpl_shape, logger, skip_steps=False, visualize=visualize)

    return traj_target


def extend_motion(
    env_name: str,
    env_params: DictConfig,
    traj: Trajectory,
    logger: logging.Logger = None
) -> Trajectory:
    """
    Extend a motion trajectory to include more model-specific entities
    like body xpos, site positions, etc. and to match the environment's frequency.

    Args:
        env_name (str): Name of the environment.
        env_params (DictConfig): Environment params.
        traj (Trajectory): The original trajectory data.
        logger (logging.Logger): Logger for status updates.

    Returns:
        Trajectory: The extended trajectory.

    """
    env_cls = LocoEnv.registered_envs[env_name]
    env = env_cls(**env_params, th_params=dict(random_start=False, fixed_start_conf=(0, 0)))

    traj_data, traj_info = interpolate_trajectories(traj.data, traj.info, 1.0 / env.dt)
    traj = Trajectory(info=traj_info, data=traj_data)

    env.process_trajectory(traj)
    traj_data, traj_info = env._traj.data, env._traj.info

    callback = ExtendTrajData(env, model=env._model, n_samples=traj_data.n_samples)
    env.play_trajectory(
        n_episodes=env.th.n_trajectories(env._traj.data),
        render=False,
        callback_class=callback
    )
    traj_data, traj_info = callback.extend_trajectory_data(traj_data, traj_info)
    traj = traj.replace(data=traj_data, info=traj_info)

    return traj


def create_multi_trajectory_hash(names: List[str]) -> str:
    """
    Generates a stable hash for a list of strings using SHA256 with incremental updates.

    Args:
        names (list[str]): The list of strings to hash.

    Returns:
        str: A hexadecimal hash string.
    """

    # Sort the list to ensure order invariance
    sorted_names = sorted(names)

    hash_obj = hashlib.sha256()
    for s in sorted_names:
        hash_obj.update(s.encode())
    return hash_obj.hexdigest()


def load_retargeted_amass_trajectory(
        env_name: str,
        dataset_name: Union[str, List[str]],
        robot_conf: DictConfig = None
) -> Trajectory:
    """
    Load a retargeted AMASS trajectory for a specific environment.

    Args:
        env_name (str): Name of the environment.
        dataset_name (Union[str, List[str]]): Name of the dataset or list of datasets to process.
        robot_conf (DictConfig): Configuration of the robot.

    Returns:
        Trajectory: The retargeted trajectories.

    """

    check_optional_imports()

    logger = setup_logger("amass", identifier="[LocoMuJoCo's AMASS Retargeting Pipeline]")

    path_to_smpl_model = get_smpl_model_path()
    path_to_converted_amass_datasets = get_converted_amass_dataset_path()

    # if robot_conf is not provided, load default one it from the YAML file
    if robot_conf is None:
        robot_conf = load_robot_conf_file(env_name)

    path_robot_smpl_data = os.path.join(path_to_converted_amass_datasets, env_name)
    if not os.path.exists(path_robot_smpl_data):
        os.makedirs(path_robot_smpl_data, exist_ok=True)

    path_to_robot_smpl_shape = os.path.join(path_robot_smpl_data, OPTIMIZED_SHAPE_FILE_NAME)
    if not os.path.exists(path_to_robot_smpl_shape):
        logger.info("Robot shape file not found, fitting new one ...")
        fit_smpl_shape(env_name, robot_conf, path_to_smpl_model, path_to_robot_smpl_shape, logger)
    else:
        logger.info(f"Found existing robot shape file at {path_to_robot_smpl_shape}")

    # load trajectory file(s)
    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]

    all_trajectories = []
    for i, d_name in enumerate(dataset_name):
        d_path = os.path.join(path_robot_smpl_data, f"{d_name}.npz")
        if not os.path.exists(d_path):
            logger.info(f"Dataset {i+1}/{len(dataset_name)}: "
                        f"Retargeting AMASS motion file using optimized body shape ...")
            motion_data = load_amass_data(d_name)
            path_converted_shape = os.path.join(
                path_to_converted_amass_datasets, f"{env_name}/{OPTIMIZED_SHAPE_FILE_NAME}")
            trajectory = fit_smpl_motion(
                env_name,
                robot_conf,
                path_to_smpl_model,
                motion_data,
                path_converted_shape,
                logger
            )
            logger.info("Using Mujoco to calculate other model-specific entities ...")
            trajectory = extend_motion(env_name, robot_conf.env_params, trajectory, logger)
            trajectory.save(d_path)
            all_trajectories.append(trajectory)
        else:
            logger.info(f"Dataset {i+1}/{len(dataset_name)}: "
                        f"Found existing retargeted motion file at {d_path}. Loading ...")
            trajectory = Trajectory.load(d_path)
            all_trajectories.append(trajectory)

    if len(all_trajectories) == 1:
        trajectory = all_trajectories[0]
    else:
        logger.info("Concatenating trajectories ...")
        traj_datas = [t.data for t in all_trajectories]
        traj_infos = [t.info for t in all_trajectories]
        traj_data, traj_info = TrajectoryData.concatenate(traj_datas, traj_infos)
        trajectory = Trajectory(traj_info, traj_data)

    logger.info("Trajectory data loaded!")

    return trajectory


def retarget_traj_from_robot_to_robot(
        env_name_source: str,
        traj_source: Trajectory,
        env_name_target: str,
        robot_conf_source: DictConfig = None,
        robot_conf_target: DictConfig = None,
        path_to_fitted_motion_source: str = None,
):

    check_optional_imports()

    logger = setup_logger("amass", identifier="[LocoMuJoCo's Robot2Robot Retargeting Pipeline]")

    path_to_smpl_model = get_smpl_model_path()
    path_to_converted_amass_datasets = get_converted_amass_dataset_path()

    # if robot_conf is not provided, load default one it from the YAML file
    if robot_conf_source is None:
        robot_conf_source = load_robot_conf_file(env_name_source)
    if robot_conf_target is None:
        robot_conf_target = load_robot_conf_file(env_name_target)

    path_source_robot_smpl_data = os.path.join(path_to_converted_amass_datasets, env_name_source)
    path_target_robot_smpl_data = os.path.join(path_to_converted_amass_datasets, env_name_target)

    traj_target = motion_transfer_robot_to_robot(env_name_source, robot_conf_source, traj_source,
                                                 path_source_robot_smpl_data, env_name_target,
                                                 robot_conf_target, path_target_robot_smpl_data,
                                                 path_to_smpl_model, logger, path_to_fitted_motion_source)

    return traj_target
