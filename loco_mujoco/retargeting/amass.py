from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from loco_mujoco.retargeting.bvh import BVHMotion, BVHSkeleton
from loco_mujoco.retargeting.soma import global_transforms

# The 22 body joints of SMPL-H, the hands are left out.
SMPL_JOINTS = ("Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2", "L_Ankle", "R_Ankle",
               "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder",
               "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist")
SMPL_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)

# SMPL's body is Y-up facing +z, AMASS clips are already in the Z-up world
WORLD_FROM_SMPL = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

# mimic site -> (SMPL joint, position weight, rotation weight)
SMPL_SITE_MAP = {
    "pelvis_mimic": ("Pelvis", 15.0, 10.0),
    "upper_body_mimic": ("Spine3", 0.7, 4.0),
    "head_mimic": ("Head", 0.5, 1.0),
    "left_hip_mimic": ("L_Hip", 1.5, 0.15),
    "left_knee_mimic": ("L_Knee", 10.0, 1.0),
    "left_foot_mimic": ("L_Ankle", 30.0, 2.0),
    "right_hip_mimic": ("R_Hip", 1.5, 0.15),
    "right_knee_mimic": ("R_Knee", 10.0, 1.0),
    "right_foot_mimic": ("R_Ankle", 30.0, 2.0),
    "left_shoulder_mimic": ("L_Shoulder", 1.5, 2.0),
    "left_elbow_mimic": ("L_Elbow", 1.0, 2.0),
    "left_hand_mimic": ("L_Wrist", 10.0, 1.5),
    "right_shoulder_mimic": ("R_Shoulder", 1.5, 2.0),
    "right_elbow_mimic": ("R_Elbow", 1.0, 2.0),
    "right_hand_mimic": ("R_Wrist", 10.0, 1.5),
}

# mimic site -> SMPL joints that reach the floor with it, and the height below which they touch it
SMPL_CONTACTS = {
    "left_foot_mimic": (("L_Ankle", 0.11), ("L_Foot", 0.04)),
    "right_foot_mimic": (("R_Ankle", 0.11), ("R_Foot", 0.04)),
    "left_hand_mimic": (("L_Wrist", 0.10),),
    "right_hand_mimic": (("R_Wrist", 0.10),),
    "left_knee_mimic": (("L_Knee", 0.12),),
    "right_knee_mimic": (("R_Knee", 0.12),),
    "pelvis_mimic": (("Pelvis", 0.15),),
}


def smpl_skeleton(model_path: str, betas: Optional[np.ndarray] = None) -> Tuple[BVHSkeleton, float]:
    """Build the joint hierarchy of an SMPL-H body.

    Only the joint regressor, template and shape blend shapes of the model are read, the mesh
    is not posed.

    Args:
        model_path: Path to an SMPL-H model.npz, such as smplh/neutral/model.npz.
        betas: Shape parameters, the mean shape of the model if None.

    Returns:
        The skeleton with the root offset at the rest pelvis, and the rest height of the pelvis
        above the sole.

    """
    with np.load(model_path, allow_pickle=False) as model:
        vertices = model["v_template"]
        if betas is not None:
            n = min(len(betas), model["shapedirs"].shape[-1])
            vertices = vertices + model["shapedirs"][..., :n] @ np.asarray(betas)[:n]
        joints = model["J_regressor"][:len(SMPL_JOINTS)] @ vertices
    parents = np.array(SMPL_PARENTS)
    offsets = joints - np.where(parents[:, None] >= 0, joints[parents], 0.0)
    return BVHSkeleton(SMPL_JOINTS, SMPL_PARENTS, offsets), float(joints[0, 1] - vertices[:, 1].min())


def smpl_rest_pose(skeleton: BVHSkeleton, pelvis_height: float) -> Tuple[np.ndarray, np.ndarray]:
    """Standing T-pose of an SMPL skeleton in the Z-up world, facing +x with the sole at z = 0.

    Returns:
        Local translations (J, 3) and rotations (J, 3, 3), for use with an identity world frame.

    """
    local_pos = np.array(skeleton.offsets)
    local_pos[0] = [0.0, 0.0, pelvis_height]
    local_rot = np.tile(np.eye(3), (len(skeleton.names), 1, 1))
    local_rot[0] = WORLD_FROM_SMPL
    return local_pos, local_rot


def _resample(quat: np.ndarray, positions: np.ndarray, fps: float, new_fps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Resample (T, J, 4) quaternions by slerp and (T, 3) positions linearly."""
    n_frames, n_joints = quat.shape[:2]
    times = np.arange(0.0, (n_frames - 1) / fps + 1e-9, 1.0 / new_fps) * fps
    lower = np.minimum(np.floor(times).astype(int), n_frames - 2)
    weight = times - lower
    start = Rotation.from_quat(quat[lower].reshape(-1, 4))
    end = Rotation.from_quat(quat[lower + 1].reshape(-1, 4))
    step = (start.inv() * end).as_rotvec() * np.repeat(weight, n_joints)[:, None]
    quat = (start * Rotation.from_rotvec(step)).as_quat().reshape(len(times), n_joints, 4)
    return quat, positions[lower] + weight[:, None] * (positions[lower + 1] - positions[lower])


def load_amass(path: str, skeleton: BVHSkeleton, fps: Optional[float] = None) -> BVHMotion:
    """Load an AMASS clip as local joint transforms of the given SMPL skeleton.

    The clip's own betas are not used: the motion is played on the skeleton as it is.

    Args:
        path: Path to an AMASS SMPL+H npz file.
        skeleton: Skeleton from smpl_skeleton.
        fps: Frame rate to resample to, the clip's own if None.

    Returns:
        The motion, in the Z-up world, so use an identity world frame with it.

    Raises:
        ValueError: If the file has no motion or contains non-finite values.

    """
    with np.load(path, allow_pickle=False) as clip:
        if "poses" not in clip.files:
            raise ValueError(f"No poses in {path}.")
        poses, trans = clip["poses"], clip["trans"]
        clip_fps = float(clip["mocap_framerate"] if "mocap_framerate" in clip.files else clip["mocap_frame_rate"])
    n_joints = len(skeleton.names)
    if len(poses) < 2 or not (np.isfinite(poses).all() and np.isfinite(trans).all()):
        raise ValueError(f"Expected a finite clip of at least two frames in {path}.")

    quat = Rotation.from_rotvec(poses[:, :3 * n_joints].reshape(-1, 3)).as_quat().reshape(len(poses), n_joints, 4)
    if fps is not None and abs(fps - clip_fps) > 1e-3:
        quat, trans = _resample(quat, trans, clip_fps, fps)
        clip_fps = fps
    n_frames = len(trans)

    local_rot = Rotation.from_quat(quat.reshape(-1, 4)).as_matrix().reshape(n_frames, n_joints, 3, 3)
    local_pos = np.tile(skeleton.offsets, (n_frames, 1, 1))
    local_pos[:, 0] += trans
    return BVHMotion(skeleton, local_pos, local_rot, clip_fps)


def place_on_floor(motion: BVHMotion, pelvis_height: float, percentile: float = 5.0) -> BVHMotion:
    """Shift a clip vertically so that its feet stand on z = 0.

    AMASS clips have no common floor height, and a skeleton with other bone lengths than the
    recorded subject also moves the feet. The floor is put at a low percentile of the height of
    the lowest foot joint, less that joint's height above the sole in the rest pose.

    Args:
        motion: Motion from load_amass.
        pelvis_height: Rest pelvis height from smpl_skeleton.
        percentile: Percentile of the frames taken as the floor, robust to a few bad frames.

    Returns:
        The shifted motion.

    """
    skeleton = motion.skeleton
    feet = [skeleton.index(name) for name in ("L_Ankle", "R_Ankle", "L_Foot", "R_Foot")]
    rest_pos, rest_rot = smpl_rest_pose(skeleton, pelvis_height)
    clearance = np.asarray(global_transforms(skeleton.parents, rest_pos, rest_rot, feet, np.eye(3))[0])[:, 2]
    heights = np.asarray(global_transforms(skeleton.parents, motion.local_pos, motion.local_rot, feet,
                                           np.eye(3))[0])[..., 2]
    floor = np.percentile((heights - clearance).min(axis=1), percentile)
    local_pos = np.array(motion.local_pos)
    local_pos[:, 0, 2] -= floor
    return BVHMotion(skeleton, local_pos, motion.local_rot, motion.fps)
