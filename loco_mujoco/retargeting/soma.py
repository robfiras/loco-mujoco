from typing import List, NamedTuple, Sequence, Tuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from loco_mujoco.retargeting.bvh import BVHSkeleton

# Y-up BVH facing +z -> Z-up world facing +x
WORLD_FROM_BVH = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

# mimic site -> (SOMA joint, position weight, rotation weight)
SOMA_SITE_MAP = {
    "pelvis_mimic": ("Hips", 15.0, 10.0),
    "upper_body_mimic": ("Chest", 0.7, 4.0),
    "head_mimic": ("Head", 0.5, 1.0),
    "left_hip_mimic": ("LeftLeg", 1.5, 0.15),
    "left_knee_mimic": ("LeftShin", 10.0, 1.0),
    "left_foot_mimic": ("LeftFoot", 30.0, 2.0),
    "right_hip_mimic": ("RightLeg", 1.5, 0.15),
    "right_knee_mimic": ("RightShin", 10.0, 1.0),
    "right_foot_mimic": ("RightFoot", 30.0, 2.0),
    "left_shoulder_mimic": ("LeftArm", 1.5, 2.0),
    "left_elbow_mimic": ("LeftForeArm", 1.0, 2.0),
    "left_hand_mimic": ("LeftHand", 10.0, 1.5),
    "right_shoulder_mimic": ("RightArm", 1.5, 2.0),
    "right_elbow_mimic": ("RightForeArm", 1.0, 2.0),
    "right_hand_mimic": ("RightHand", 10.0, 1.5),
}


# Limb bones as pairs of mimic sites, parents first
LIMB_BONES = (
    ("left_shoulder_mimic", "left_elbow_mimic"),
    ("left_elbow_mimic", "left_hand_mimic"),
    ("right_shoulder_mimic", "right_elbow_mimic"),
    ("right_elbow_mimic", "right_hand_mimic"),
    ("left_hip_mimic", "left_knee_mimic"),
    ("left_knee_mimic", "left_foot_mimic"),
    ("right_hip_mimic", "right_knee_mimic"),
    ("right_knee_mimic", "right_foot_mimic"),
)


# Sites placed relative to their parent site with the robot's own bones, parents first. The pelvis
# and the legs are left out to keep the human's global foot positions and thus its floor contact.
SITE_CHAIN = (
    ("upper_body_mimic", "pelvis_mimic"),
    ("head_mimic", "upper_body_mimic"),
    ("left_shoulder_mimic", "upper_body_mimic"),
    ("left_elbow_mimic", "left_shoulder_mimic"),
    ("left_hand_mimic", "left_elbow_mimic"),
    ("right_shoulder_mimic", "upper_body_mimic"),
    ("right_elbow_mimic", "right_shoulder_mimic"),
    ("right_hand_mimic", "right_elbow_mimic"),
)


# mimic site -> SOMA joints that reach the floor with it, and the height below which they touch it
SOMA_CONTACTS = {
    "left_foot_mimic": (("LeftFoot", 0.11), ("LeftToeBase", 0.04)),
    "right_foot_mimic": (("RightFoot", 0.11), ("RightToeBase", 0.04)),
    "left_hand_mimic": (("LeftHand", 0.10),),
    "right_hand_mimic": (("RightHand", 0.10),),
    "left_knee_mimic": (("LeftShin", 0.12),),
    "right_knee_mimic": (("RightShin", 0.12),),
    "pelvis_mimic": (("Hips", 0.15),),
}


class Calibration(NamedTuple):
    """Mapping from SOMA joint poses to robot site targets, computed once per robot.

    Attributes:
        scale: Ratio of robot to human pelvis height.
        joint_ids: SOMA joint of each site, shape (S,).
        root_id: Index of the root within joint_ids.
        offset_pos: Site position offset in the site frame, shape (S, 3).
        offset_rot: Rotation from the SOMA joint frame to the site frame, shape (S, 3, 3).
        chain: Tuples of site index, parent site index, and the robot's bone in the parent site
            frame, parents first.

    """
    scale: float
    joint_ids: np.ndarray
    root_id: int
    offset_pos: np.ndarray
    offset_rot: np.ndarray
    chain: Tuple[Tuple[int, int, np.ndarray], ...]


def global_transforms(
    parents: Sequence[int],
    local_pos: jax.Array,
    local_rot: jax.Array,
    joint_ids: Sequence[int],
) -> Tuple[jax.Array, jax.Array]:
    """Compute world poses of selected joints.

    Args:
        parents: Parent of each joint, parents before their children.
        local_pos: Local translations, shape (..., J, 3).
        local_rot: Local rotation matrices, shape (..., J, 3, 3).
        joint_ids: Joints to return.

    Returns:
        World positions (..., N, 3) and rotation matrices (..., N, 3, 3).

    """
    needed = set()
    for joint in joint_ids:
        joint = int(joint)
        while joint != -1:
            needed.add(joint)
            joint = parents[joint]

    world = jnp.asarray(WORLD_FROM_BVH, local_pos.dtype)
    pos, rot = {}, {}
    for joint in sorted(needed):
        parent = parents[joint]
        if parent == -1:
            pos[joint] = local_pos[..., joint, :] @ world.T
            rot[joint] = world @ local_rot[..., joint, :, :]
        else:
            pos[joint] = pos[parent] + jnp.einsum("...ij,...j->...i", rot[parent],
                                                  local_pos[..., joint, :])
            rot[joint] = rot[parent] @ local_rot[..., joint, :, :]

    return (jnp.stack([pos[int(j)] for j in joint_ids], axis=-2),
            jnp.stack([rot[int(j)] for j in joint_ids], axis=-3))


def site_targets(
    joint_pos: jax.Array,
    joint_rot: jax.Array,
    calibration: Calibration,
) -> Tuple[jax.Array, jax.Array]:
    """Map SOMA joint poses to robot site targets.

    Args:
        joint_pos: Joint positions from global_transforms for calibration.joint_ids, (..., S, 3).
        joint_rot: Joint rotations from global_transforms for calibration.joint_ids, (..., S, 3, 3).
        calibration: The robot's calibration.

    Returns:
        Target site positions (..., S, 3) and rotation matrices (..., S, 3, 3).

    """
    rot = joint_rot @ jnp.asarray(calibration.offset_rot, joint_rot.dtype)
    offset = jnp.einsum("...ij,...j->...i", rot, jnp.asarray(calibration.offset_pos, rot.dtype))
    pos = calibration.scale * joint_pos + offset
    for site, parent, bone in calibration.chain:
        child = pos[..., parent, :] + jnp.einsum("...ij,j->...i", rot[..., parent, :, :],
                                                 jnp.asarray(bone, rot.dtype))
        pos = pos.at[..., site, :].set(child)
    return pos, rot


def foot_contact_points(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    foot_sites: Sequence[str],
) -> Tuple[List[int], List[np.ndarray], float]:
    """Find the corners of each foot's sole from its collision geometry.

    Args:
        model: The MuJoCo model.
        data: Data holding a pose with flat feet, with kinematics computed.
        foot_sites: Names of the foot sites.

    Returns:
        Body id of each point, position of each point in its body frame, and the world height of
        the lowest point.

    """
    bodies, points, lowest = [], [], np.inf
    for site in foot_sites:
        body = int(model.site_bodyid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)])
        geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body]
        collision = [g for g in geoms if model.geom_contype[g] or model.geom_conaffinity[g]]
        corners = []
        for g in collision or geoms:
            center, half = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
            signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
            world = data.geom_xpos[g] + (center + signs * half) @ data.geom_xmat[g].reshape(3, 3).T
            corners.append(world)
        corners = np.concatenate(corners)
        body_rot = data.xmat[body].reshape(3, 3)
        local = (corners - data.xpos[body]) @ body_rot
        low, high = local.min(axis=0), local.max(axis=0)
        box = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1])
                        for z in (low[2], high[2])])
        heights = data.xpos[body][2] + box @ body_rot[2]
        sole = box[np.argsort(heights)[:4]]
        lowest = min(lowest, heights.min())
        bodies += [body] * 4
        points += list(sole)
    return bodies, points, float(lowest)


def _geom_points(model: mujoco.MjModel, geom: int) -> np.ndarray:
    """Points on the surface of a geom in the geom's frame."""
    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        first = model.mesh_vertadr[mesh]
        return model.mesh_vert[first:first + model.mesh_vertnum[mesh]].astype(np.float64)
    center, half = model.geom_aabb[geom, :3], model.geom_aabb[geom, 3:]
    signs = np.array([[x, y, z] for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if x or y or z], float)
    if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX:
        signs /= np.linalg.norm(signs, axis=1, keepdims=True)
    return center + signs * half


def body_contact_points(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    sites: Sequence[str],
) -> Tuple[List[int], List[np.ndarray], List[List[int]]]:
    """Find points on the visible surface of the bodies carrying the given sites.

    Each body gets the extreme points of its geoms along 14 directions. Hands and feet include
    their descendants, such as fingers and toes, held rigid in the pose of the given data.

    Args:
        model: The MuJoCo model.
        data: Data holding a pose, with kinematics computed.
        sites: Names of the sites.

    Returns:
        Body id of each point, position of each point in its body frame, and the indices of the
        points of each site.

    """
    directions = np.array([[x, y, z] for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)
                           if abs(x) + abs(y) + abs(z) in (1, 3)], float)
    bodies, points, groups = [], [], []
    for site in sites:
        body = _site_body(model, site)
        members = {body}
        if "hand" in site or "foot" in site:
            for child in range(body + 1, model.nbody):
                if model.body_parentid[child] in members:
                    members.add(child)
        surface = [data.geom_xpos[g] + _geom_points(model, g) @ data.geom_xmat[g].reshape(3, 3).T
                   for g in range(model.ngeom) if model.geom_bodyid[g] in members and model.geom_group[g] < 3]
        if not surface:
            groups.append([])
            continue
        local = (np.concatenate(surface) - data.xpos[body]) @ data.xmat[body].reshape(3, 3)
        extreme = np.unique((local @ directions.T).argmax(axis=0))
        groups.append(list(range(len(points), len(points) + len(extreme))))
        bodies += [body] * len(extreme)
        points += list(local[extreme])
    return bodies, points, groups


def contact_joints(skeleton: BVHSkeleton, sites: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Look up the SOMA joints that show when the given sites touch the floor.

    Args:
        skeleton: The SOMA skeleton.
        sites: Names of the sites, keys of SOMA_CONTACTS.

    Returns:
        The joints (J,), their contact heights (J,) and the mask of each site's joints (G, J).

    """
    entries = [(i, name, height) for i, site in enumerate(sites) for name, height in SOMA_CONTACTS[site]]
    joints = np.array([skeleton.index(name) for _, name, _ in entries])
    heights = np.array([height for _, _, height in entries])
    owner = np.array([i for i, _, _ in entries])
    return joints, heights, owner[None, :] == np.arange(len(sites))[:, None]


def detect_contacts(
    joint_pos: jax.Array,
    heights: jax.Array,
    groups: jax.Array,
    fps: float,
    max_speed: float = 0.5,
) -> jax.Array:
    """Find the frames of a clip in which the human touches the floor.

    Args:
        joint_pos: Joint positions in the human's own size, shape (T, J, 3).
        heights: Contact height of each joint, shape (J,).
        groups: Mask of the joints of each body, shape (G, J).
        fps: Frame rate of the clip.
        max_speed: Fastest motion in m/s that still counts as resting on the floor.

    Returns:
        The contact state of each body, shape (T, G).

    """
    velocity = jnp.diff(joint_pos, axis=0, prepend=joint_pos[:1]) * fps
    touching = (joint_pos[..., 2] < heights) & (jnp.linalg.norm(velocity, axis=-1) < max_speed)
    contact = (touching.astype(jnp.float32) @ jnp.asarray(groups, jnp.float32).T) > 0
    # majority of three frames, removes single frame flicker
    padded = jnp.pad(contact.astype(jnp.int32), ((1, 1), (0, 0)), mode="edge")
    return padded[:-2] + padded[1:-1] + padded[2:] >= 2


def _nearest_contact(touching: jax.Array, blend: int) -> Tuple[jax.Array, jax.Array]:
    """Nearest frame in contact of each frame, and a weight fading out with the distance to it."""
    frames = jnp.arange(touching.shape[0])
    far = 2 * touching.shape[0] + blend + 1
    before = jax.lax.cummax(jnp.where(touching, frames, -far))
    after = jax.lax.cummin(jnp.where(touching, frames, far), reverse=True)
    nearest = jnp.where(frames - before <= after - frames, before, after)
    weight = jnp.clip(1.0 - jnp.abs(frames - nearest) / (blend + 1), 0.0, 1.0)
    return jnp.clip(nearest, 0, touching.shape[0] - 1), weight


def soften_contacts(contact: jax.Array, blend: int = 8) -> jax.Array:
    """Fade contacts in and out in the frames around them, so that they do not snap the robot.

    Args:
        contact: The contact state of each body, shape (T, G).
        blend: Frames before and after each contact over which it fades.

    Returns:
        The contact weights, 1 during contact, shape (T, G).

    """
    return jax.vmap(lambda touching: _nearest_contact(touching, blend)[1], in_axes=1, out_axes=1)(contact)


def hold_contacts(
    target_pos: jax.Array,
    contact: jax.Array,
    site_ids: Sequence[int],
    blend: int = 8,
) -> jax.Array:
    """Hold the horizontal targets of bodies in contact at their mean over each contact.

    Args:
        target_pos: Target site positions of a clip, shape (T, S, 3).
        contact: The contact state of each body, shape (T, G).
        site_ids: Site of each body, shape (G,).
        blend: Frames before and after each contact over which the hold fades.

    Returns:
        The target positions.

    """
    frames = target_pos.shape[0]

    def hold(xy, touching):
        starts = touching & ~jnp.concatenate([jnp.zeros(1, bool), touching[:-1]])
        segment = jnp.cumsum(starts)
        weight = touching.astype(xy.dtype)
        total = jax.ops.segment_sum(xy * weight[:, None], segment, frames + 1)
        count = jax.ops.segment_sum(weight, segment, frames + 1)
        offset = weight[:, None] * ((total / jnp.maximum(count, 1.0)[:, None])[segment] - xy)
        nearest, fade = _nearest_contact(touching, blend)
        return xy + fade[:, None] * offset[nearest]

    site_ids = jnp.asarray(site_ids)
    held = jax.vmap(hold, in_axes=(1, 1), out_axes=1)(target_pos[:, site_ids, :2], contact)
    return target_pos.at[:, site_ids, :2].set(held)


def _site_body(model: mujoco.MjModel, name: str) -> int:
    return int(model.site_bodyid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)])


def _dofs_between(model: mujoco.MjModel, lower: int, upper: int) -> List[int]:
    """Degrees of freedom on the path from a body up to, but excluding, one of its ancestors."""
    dofs = []
    body = lower
    while body not in (upper, 0):
        first = model.body_dofadr[body]
        dofs += list(range(first, first + model.body_dofnum[body]))
        body = int(model.body_parentid[body])
    return dofs


def limb_dof_damping(
    model: mujoco.MjModel,
    shoulder_damping: float = 10.0,
    wrist_damping: float = 1.0,
) -> np.ndarray:
    """Damping per degree of freedom for the shoulders and wrists of a robot.

    The joints are found from the kinematic tree between the mimic sites, since joint names differ
    between robots.

    Args:
        model: The MuJoCo model.
        shoulder_damping: Damping of all joints between the torso and the elbows, excluding the
            elbows themselves.
        wrist_damping: Damping of all joints between the elbows and the hands.

    Returns:
        The damping of each degree of freedom, shape (nv,).

    """
    damping = np.zeros(model.nv)
    torso = _site_body(model, "upper_body_mimic")
    for side in ("left", "right"):
        elbow, hand = _site_body(model, f"{side}_elbow_mimic"), _site_body(model, f"{side}_hand_mimic")
        damping[_dofs_between(model, int(model.body_parentid[elbow]), torso)] = shoulder_damping
        damping[_dofs_between(model, hand, elbow)] = wrist_damping
    return damping


def limit_dofs(model: mujoco.MjModel) -> np.ndarray:
    """Mask of the degrees of freedom whose limits the solver should handle inside its solve.

    This covers everything except the elbows: an arm pinned at its limit with a straight elbow has
    no direction left to fold towards a target close to the body, and would stay there.

    Args:
        model: The MuJoCo model.

    Returns:
        The mask, shape (nv,).

    """
    mask = np.ones(model.nv, dtype=bool)
    for side in ("left", "right"):
        elbow = _site_body(model, f"{side}_elbow_mimic")
        mask[_dofs_between(model, elbow, int(model.body_parentid[elbow]))] = False
    return mask


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a onto unit vector b."""
    axis = np.cross(a, b)
    cos = float(a @ b)
    if cos < -1.0 + 1e-8:
        perpendicular = np.cross(a, np.eye(3)[int(np.abs(a).argmin())])
        perpendicular /= np.linalg.norm(perpendicular)
        return 2.0 * np.outer(perpendicular, perpendicular) - np.eye(3)
    skew = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + skew + skew @ skew / (1.0 + cos)


def align_rest_pose(
    skeleton: BVHSkeleton,
    rest_local_pos: np.ndarray,
    rest_local_rot: np.ndarray,
    site_names: Sequence[str],
    site_pos: np.ndarray,
) -> np.ndarray:
    """Rotate the limbs of a SOMA pose onto the bone directions of the robot's rest pose.

    SOMA's zero pose is not a T-pose, so without this the difference between the two poses would
    end up in large constant offsets that swing around with the human's limbs.

    Args:
        skeleton: The SOMA skeleton.
        rest_local_pos: Local translations of the SOMA pose, shape (J, 3).
        rest_local_rot: Local rotations of the SOMA pose, shape (J, 3, 3).
        site_names: Names of the robot sites.
        site_pos: Site positions in the robot's rest pose, shape (S, 3).

    Returns:
        Local rotations of the aligned pose, shape (J, 3, 3).

    """
    site_ids = {name: i for i, name in enumerate(site_names)}
    local_rot = np.array(rest_local_rot)

    for parent_site, child_site in LIMB_BONES:
        joint = skeleton.index(SOMA_SITE_MAP[parent_site][0])
        child = skeleton.index(SOMA_SITE_MAP[child_site][0])
        above = skeleton.parents[joint]
        pos, rot = global_transforms(skeleton.parents, jnp.asarray(rest_local_pos),
                                     jnp.asarray(local_rot), [above, joint, child])
        pos, rot = np.asarray(pos, dtype=np.float64), np.asarray(rot, dtype=np.float64)

        current = pos[2] - pos[1]
        target = site_pos[site_ids[child_site]] - site_pos[site_ids[parent_site]]
        align = _rotation_between(current / np.linalg.norm(current),
                                  target / np.linalg.norm(target))
        local_rot[joint] = rot[0].T @ align @ rot[1]

    return local_rot


def calibrate(
    skeleton: BVHSkeleton,
    rest_local_pos: np.ndarray,
    rest_local_rot: np.ndarray,
    site_names: Sequence[str],
    site_pos: np.ndarray,
    site_rot: np.ndarray,
    pelvis_height: float,
) -> Calibration:
    """Compute the mapping from SOMA joints to robot sites from matching rest poses.

    Args:
        skeleton: The SOMA skeleton.
        rest_local_pos: Local translations of the SOMA rest pose, shape (J, 3).
        rest_local_rot: Local rotations of the SOMA rest pose, shape (J, 3, 3).
        site_names: Names of the robot sites.
        site_pos: Site positions in the robot's rest pose with its feet on the floor, shape (S, 3).
        site_rot: Site rotation matrices in the robot's rest pose, shape (S, 3, 3).
        pelvis_height: Height of the robot's pelvis site above the floor in its rest pose.

    Returns:
        The calibration.

    Raises:
        KeyError: If a site has no SOMA joint assigned.

    """
    joint_ids = np.array([skeleton.index(SOMA_SITE_MAP[s][0]) for s in site_names])
    root_id = list(site_names).index("pelvis_mimic")

    aligned_rot = align_rest_pose(skeleton, rest_local_pos, rest_local_rot, site_names, site_pos)
    pos, rot = global_transforms(skeleton.parents, jnp.asarray(rest_local_pos),
                                 jnp.asarray(aligned_rot), joint_ids)
    pos, rot = np.asarray(pos, dtype=np.float64), np.asarray(rot, dtype=np.float64)

    scale = pelvis_height / pos[root_id, 2]
    offset_rot = np.einsum("nji,njk->nik", rot, site_rot)
    offset_pos = np.einsum("nji,nj->ni", site_rot, site_pos - scale * pos)

    site_ids = {name: i for i, name in enumerate(site_names)}
    chain = tuple((site_ids[child], site_ids[parent],
                   site_rot[site_ids[parent]].T @ (site_pos[site_ids[child]] - site_pos[site_ids[parent]]))
                  for child, parent in SITE_CHAIN)
    return Calibration(float(scale), joint_ids, root_id, offset_pos, offset_rot, chain)
