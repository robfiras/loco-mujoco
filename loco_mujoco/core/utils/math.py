import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation as jnp_R
from scipy.spatial.transform import Rotation as np_R
# from mushroom_rl.utils.angles import euler_to_mat, mat_to_euler


def rotate_obs(state, angle, idx_rot, idx_xvel, idx_yvel):
    """
    Function to rotate a state (or set of states) around the rotation axis.

    Args:
        state (list or np.array): Single state or multiple states to be rotated.
        angle (float): Angle of rotation in radians.
        idx_rot (int): Index of rotation angle entry in the state.
        idx_xvel (int): Index of x-velocity entry in the state.
        idx_yvel (int): Index of y-velocity entry in the state.

    Returns:
        np.array of rotated states.

    """

    state = np.array(state)
    rotated_state = state.copy()

    # add rotation to trunk rotation and transform to range [-pi, pi]
    rotated_state[idx_rot] = (state[idx_rot] + angle + np.pi) % (2 * np.pi) - np.pi
    # rotate x,y velocity
    rotated_state[idx_xvel] = np.cos(angle) * state[idx_xvel] - np.sin(angle) * state[idx_yvel]
    rotated_state[idx_yvel] = np.sin(angle) * state[idx_xvel] + np.cos(angle) * state[idx_yvel]

    return rotated_state


def mat2angle_xy(mat):
    """
    Converts a rotation matrix to an angle in the x-y-plane.

    Args:
        mat (np.array): np.array of dim 9.

    Returns:
        Float constituting the rotation angle in the x-y--plane (in radians).

    """

    angle = np_R.from_matrix(mat.reshape((3, 3))).as_euler("xyz")[-1]

    return angle


def angle2mat_xy(angle):
    """
    Converts a rotation angle in the x-y-plane to a rotation matrix.

    Args:
        angle (float): Angle to be converted.

    Returns:
        np.array of shape (3, 3)

    """

    mat = np_R.from_euler("xyz", np.array([0, 0, angle])).as_matrix()

    return mat


def transform_angle_2pi(angle):
    """
    Transforms an angle to be in [-pi, pi].

    Args:
        angle (float): Angle in radians.

    Returns:
        Angle in radians in [-pi, pi].

    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def calc_rel_positions(xpos, x_pos_main_body, backend):
    """
    Calculate the relative positions of the bodies in b_ids to the main body.

    Args:
        xpos (array): Data array containing the positions of the bodies.
        x_pos_main_body (array): Position of the main body.
        backend: Backend to use (either np or jnp).

    Returns:
        Array of relative positions.

    """

    rpos = xpos - x_pos_main_body

    return rpos


def calculate_relative_velocities(xvel, xvel_main_body, backend):
    """
    Calculate the relative velocity of two 6D velocities in world frame.

    Args:
        xvel (array): Data array containing the velocities of the bodies.
        xvel_main_body (array): Velocity of the main body.
        backend: Backend to use (either np or jnp).

    Returns:
        array: Relative 6D velocity vector. Shape (6,)
    """
    # Calculate the relative velocity
    relative_velocity = xvel - xvel_main_body  # Shape: (6,)

    return relative_velocity


def calc_rel_quaternions(xquat, xquat_main_body, backend):
    """
    Calculate the relative quaternions of the bodies in b_ids to the main body.

    Args:
        xquat (array): Data array containing the quaternions of the bodies (quaternions is expected to be scalar last).
        xquat_main_body (array): Quaternion of the main body (quaternions is expected to be scalar last).
        backend: Backend to use (either np or jnp).
    Returns:
        Array of relative quaternions, where the quaternions are scalar last.

    """
    if backend == np:
        R = np_R
    else:
        R = jnp_R

    rquat = (R.from_quat(xquat_main_body).inv() * R.from_quat(xquat)).as_quat()

    return rquat


def calculate_relative_rotation_matrices(main_rot, other_rots, backend):
    """
    Calculate the relative rotation matrices of N bodies with respect to a main rotation matrix.

    Args:
        main_rot (array): Rotation matrix of the main body in world frame. Shape (3, 3).
        other_rots (array): Rotation matrices of other bodies in world frame. Shape (N, 3, 3) where N is the number of other bodies.
        backend: Backend to use (either np or jnp).

    Returns:
        array: Relative rotation matrices of other bodies with respect to the main body. Shape (N, 3, 3).
    """
    # Ensure main_rot is a 2D array and other_rots is at least 3D
    main_rot = backend.atleast_2d(main_rot)  # Shape: (3, 3)
    other_rots = atleast_3d(other_rots, backend)  # Shape: (N, 3, 3)

    # Use the transpose of the main rotation matrix as the inverse
    main_rot_inv = main_rot.T  # Shape: (3, 3)

    # Calculate relative rotation matrices
    relative_rots = backend.einsum('ik,nkj->nij', main_rot_inv, other_rots)  # Shape: (N, 3, 3)

    return relative_rots


def calculate_global_rotation_matrices(main_rot, relative_rots, backend):
    """
    Calculate the global rotation matrices of N bodies in the world frame given a main rotation matrix.

    Args:
        main_rot (array): Rotation matrix of the main body in world frame. Shape (3, 3).
        relative_rots (array): Relative rotation matrices of other bodies with respect to the main body. Shape (N, 3, 3).
        backend: Backend to use (either np or jnp).

    Returns:
        array: Global rotation matrices of other bodies in the world frame. Shape (N, 3, 3).
    """
    # Ensure main_rot is a 2D array and relative_rots is at least 3D
    main_rot = backend.atleast_2d(main_rot)  # Shape: (3, 3)
    relative_rots = atleast_3d(relative_rots, backend)  # Shape: (N, 3, 3)

    # Calculate global rotation matrices
    global_rots = backend.einsum('ij,njk->nik', main_rot, relative_rots)  # Shape: (N, 3, 3)

    return global_rots


def calculate_relative_velocity_in_local_frame(vel_a, vel_b, rot_mat_w_a, rot_mat_a_b, backend):
    """
    Calculate the relative velocity vel_a-vel_b expressed in the local frame of vel_a.

    Args:
        vel_a (array): 6D velocity vector of body A in world frame. Shape (6,) or (batch_size, 6)
        vel_b (array): 6D velocity vector of body B in world frame. Shape (6,) or (batch_size, 6)
        rot_mat_w_a (array): Rotation matrix to transform from world frame to the frame of vel_a.
            Shape (3, 3) or (batch_size, 3, 3)
        rot_mat_a_b (array): Rotation matrix to transform from the frame of vel_a to the frame of vel_b.
            Shape (3, 3) or (batch_size, 3, 3)
        backend: Backend to use (either np or jnp).

    Returns:
        array: Relative 6D velocity vector of vel_a in the local frame of vel_b. Shape (batch_size, 6)
    """
    # Ensure the inputs are at least 2D arrays for batch processing
    vel_a = backend.atleast_1d(vel_a)  # Shape: (6,)
    vel_b = backend.atleast_2d(vel_b)  # Shape: (1, 6) or (batch_size, 6)
    rot_mat_w_a = backend.atleast_2d(rot_mat_w_a)  # Shape: (3, 3)
    rot_mat_a_b = atleast_3d(rot_mat_a_b, backend)  # Shape: (1, 3, 3) or (batch_size, 3, 3)

    # Extract linear and angular components (corrected based on your input structure)
    ang_a = vel_a[:3]  # Angular components of A (Shape: (3,))
    lin_a = vel_a[3:]  # Linear components of A (Shape: (3,))

    ang_b = vel_b[:, :3]  # Angular components of B (Shape: (batch_size, 3))
    lin_b = vel_b[:, 3:]  # Linear components of B (Shape: (batch_size, 3))

    # Transform the relativ linear velocity of vel_a-vel_b to the local frame of vel_a
    relative_lin_vel = backend.einsum('jk,ik->ij', rot_mat_w_a, lin_a - lin_b)  # Shape: (batch_size, 3)

    # Transform the angular velocity of vel_b to the local frame of vel_a (einsum is taking the transpose of rot_mat_a_b)
    ang_b_transformed_to_a = backend.einsum('ikj,ik->ij', rot_mat_a_b, ang_b)  # Shape: (batch_size, 3)

    # Calculate the relative velocities in the local frame of vel_b
    relative_ang_vel = ang_b_transformed_to_a - ang_a  # Shape: (batch_size, 3)

    # Combine into a 6D vector
    relative_velocity = backend.hstack([relative_ang_vel, relative_lin_vel])  # Shape: (batch_size, 6)

    return relative_velocity


def calc_rel_body_velocities(cvel, xmat_main_body, backend):
    """
    Calculate the relative velocities of the bodies in b_ids to the main body.

    Args:
        cvel (array): Data array containing the velocities of the bodies.
        xmat_main_body (array): Rotation matrix of the main body.
        backend: Backend to use (either np or jnp).

    Returns:
        Array of relative velocities.

    """

    rot_mat_main_body = xmat_main_body.reshape(3, 3)
    lin_vel = backend.einsum('ij,nj->ni', rot_mat_main_body, cvel[:, :3])
    rot_vel = backend.einsum('ij,nj->ni', rot_mat_main_body, cvel[:, 3:])
    rvel = backend.concatenate([lin_vel, rot_vel], axis=1)

    return rvel


def calc_site_velocities(site_ids, data, parent_body_id, root_body_id, backend, flg_local=False):
    """
    Calculate the velocities of a batch of sites in world frame.
    This function is implemented similarly to Mujoco's mj_objectVelocity function
    https://github.com/google-deepmind/mujoco/blob/1f9dca8bc4cfbfc23f68c6bda7cdb6abebfdeb98/src/engine/engine_support.c#L1294

    Args:
        site_ids (array): Site ids for which to calculate the velocities. Shape (batch_size,)
        data: Mujoco/Mjx Data structure.
        parent_body_id (array): Parent body ids.
        root_body_id (array): Root body ids.
        backend: Backend to use (either np or jnp).
        flg_local (bool, optional): Whether to return velocities in local coordinates. Defaults to False.

    Returns:
        array: Global 6D velocity vectors of the sites. Shape (batch_size, 6)

    """

    site_xpos = data.site_xpos[site_ids]
    site_xmat = data.site_xmat[site_ids].reshape(*site_ids.shape, 3, 3)
    parent_body_cvel = data.cvel[parent_body_id]
    root_subtree_com = data.subtree_com[root_body_id]

    return transform_motion(parent_body_cvel, site_xpos, root_subtree_com, site_xmat, backend, flg_local)


def calc_body_velocities(body_ids, data, root_body_id, backend, flg_local=False):
    """
    Calculate the velocities of a batch of bodies in world frame.
    This function is implemented similarly to Mujoco's mj_objectVelocity function
    https://github.com/google-deepmind/mujoco/blob/1f9dca8bc4cfbfc23f68c6bda7cdb6abebfdeb98/src/engine/engine_support.c#L1294

    Args:
        body_ids (array): Body ids for which to calculate the velocities. Shape (batch_size,)
        data: Mujoco/Mjx Data structure.
        root_body_id (array): Root body ids.
        backend: Backend to use (either np or jnp).
        flg_local (bool, optional): Whether to return velocities in local coordinates. Defaults to False.

    Returns:
        array: Global 6D velocity vectors of the bodies. Shape (batch_size, 6)

    """

    body_xpos = data.xpos[body_ids]
    body_xmat = data.xmat[body_ids].reshape(*body_ids.shape, 3, 3)
    root_subtree_com = data.subtree_com[root_body_id]
    body_cvel = data.cvel[body_ids]

    return transform_motion(body_cvel, body_xpos, root_subtree_com, body_xmat, backend, flg_local)


def transform_motion(vel, new_pos, old_pos, rot_mat_new2old, backend, flg_local=True):
    """
    Transforms a motion vector from one frame to another. This function is implemented similarly to Mujoco's
    mju_transformSpatial function, but is limited to velocities.
    (https://github.com/google-deepmind/mujoco/blob/1f9dca8bc4cfbfc23f68c6bda7cdb6abebfdeb98/src/engine/engine_util_spatial.c#L454).


    Args:
        vel (array): Motion vector to be transformed. Shape (6,) or (batch_size, 6)
        new_pos (array): Position of the new frame. Shape (3,) or (batch_size, 3)
        old_pos (array): Position of the old frame. Shape (3,) or (batch_size, 3)
        rot_mat_new2old (array): Rotation matrix from the new to the old frame. Shape (3, 3) or (batch_size, 3, 3)
        backend: Backend to use (either np or jnp).

    Returns:
        array: Transformed motion vector. Shape (batch_size, 6)

    """

    # Use atleast_2d to handle both single and batched inputs
    vel = backend.atleast_2d(vel)  # Shape: (batch_size, 6)
    new_pos = backend.atleast_2d(new_pos)  # Shape: (batch_size, 3)
    old_pos = backend.atleast_2d(old_pos)  # Shape: (batch_size, 3)
    assert rot_mat_new2old.shape[-2:] == (3, 3)
    rot_mat_new2old = atleast_3d(rot_mat_new2old, backend)  # Shape: (batch_size, 3, 3)

    # Step 1: Compute the relative position of the new frame w.r.t. the old frame
    lin_vel = vel[:, 3:]  # Shape: (batch_size, 3)
    rot_vel = vel[:, :3]  # Shape: (batch_size, 3)
    rpos = calc_rel_positions(new_pos, old_pos, backend)  # Shape: (batch_size, 3)

    # Step 2: Compute the linear velocity at the site
    lin_vel = lin_vel - backend.cross(rpos, rot_vel, axis=-1)
    if flg_local:
        lin_vel = backend.einsum('bij,bj->bi', rot_mat_new2old.transpose(0, 2, 1), lin_vel)

    # Step 3: Compute the angular velocity at the site
    if flg_local:
        rot_vel = backend.einsum('bij,bj->bi', rot_mat_new2old.transpose(0, 2, 1), rot_vel)
    else:
        rot_vel = rot_vel

    # Step 4: Combine linear and angular velocities into a 6D velocity vector
    xvel = backend.hstack([rot_vel, lin_vel])  # Shape: (batch_size, 6)

    return xvel


def rotvec_from_matrix_jax(mat):
    """
    Rotation vectors of rotation matrices (..., 3, 3), angle in [0, pi], with the same values as
    ``jax.scipy``'s ``Rotation.from_matrix(mat).as_rotvec()`` but finite gradients everywhere.

    ``from_matrix`` evaluates all four branches of Shepperd's method and selects one with ``where``;
    the unselected branches take the square root of non-positive numbers, and their NaN gradients
    survive the selection (0 * NaN = NaN), poisoning every gradient that passes through. Here every
    branch's square root is clamped, and the rotation angle uses a smoothed norm.

    Args:
        mat (jnp.ndarray): Rotation matrices of shape (..., 3, 3).

    Returns:
        jnp.ndarray: Rotation vectors of shape (..., 3).
    """
    m = mat
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    trace = m00 + m11 + m22

    def _half_sqrt(x):
        return 0.5 * jnp.sqrt(jnp.maximum(x, 1e-12))

    # each candidate is (w, x, y, z), built around the largest of w, x, y, z
    sw = _half_sqrt(1.0 + trace)
    qw = jnp.stack([sw, (m21 - m12) / (4 * sw), (m02 - m20) / (4 * sw), (m10 - m01) / (4 * sw)], axis=-1)
    sx = _half_sqrt(1.0 + m00 - m11 - m22)
    qx = jnp.stack([(m21 - m12) / (4 * sx), sx, (m01 + m10) / (4 * sx), (m02 + m20) / (4 * sx)], axis=-1)
    sy = _half_sqrt(1.0 - m00 + m11 - m22)
    qy = jnp.stack([(m02 - m20) / (4 * sy), (m01 + m10) / (4 * sy), sy, (m12 + m21) / (4 * sy)], axis=-1)
    sz = _half_sqrt(1.0 - m00 - m11 + m22)
    qz = jnp.stack([(m10 - m01) / (4 * sz), (m02 + m20) / (4 * sz), (m12 + m21) / (4 * sz), sz], axis=-1)

    choice = jnp.argmax(jnp.stack([trace, m00, m11, m22], axis=-1), axis=-1)
    quat = jnp.stack([qw, qx, qy, qz], axis=-2)
    quat = jnp.take_along_axis(quat, choice[..., None, None], axis=-2)[..., 0, :]
    quat = quat / jnp.linalg.norm(quat, axis=-1, keepdims=True)
    quat = jnp.where(quat[..., :1] < 0.0, -quat, quat)

    vec = quat[..., 1:]
    vec_norm = jnp.sqrt(jnp.sum(vec * vec, axis=-1, keepdims=True) + 1e-12)
    angle = 2.0 * jnp.arctan2(vec_norm, quat[..., :1])
    return vec * (angle / vec_norm)


def calculate_relative_site_quatities(data, rel_site_ids, rel_body_ids, body_rootid, backend,
                                      main_site_id=0):
    """
    Args:
        main_site_id: index INTO `rel_site_ids` of the reference site. The other
            sites' positions/rotations/velocities are computed relative to it.
            Defaults to 0 (first site in the list).
    """

    if backend == np:
        R = np_R
    else:
        R = jnp_R

    # get site positions and rotations
    site_xpos_traj = data.site_xpos
    site_xmat_traj = data.site_xmat
    site_xpos_traj = site_xpos_traj[rel_site_ids]
    site_xmat_traj = site_xmat_traj[rel_site_ids]

    # get relevant properties and calculate site velocities
    site_root_body_id = body_rootid[rel_body_ids]
    site_xvel = calc_site_velocities(rel_site_ids, data, rel_body_ids, site_root_body_id, backend)
    main_site_xvel = site_xvel[main_site_id]
    site_xvel = backend.delete(site_xvel, main_site_id, axis=0)

    # calculate the rotation matrix from main site to the other sites
    main_site_xmat_traj = site_xmat_traj[main_site_id].reshape(3, 3)
    site_xmat_traj = backend.delete(site_xmat_traj, main_site_id, axis=0).reshape(-1, 3, 3)
    rel_rot_mat = calculate_relative_rotation_matrices(main_site_xmat_traj, site_xmat_traj, backend)

    # calculate relative quantities
    main_site_xpos_traj = site_xpos_traj[main_site_id]
    site_xpos_traj = backend.delete(site_xpos_traj, main_site_id, axis=0)
    site_rpos = calc_rel_positions(site_xpos_traj, main_site_xpos_traj, backend)
    site_rangles = R.from_matrix(rel_rot_mat).as_rotvec() if backend == np else rotvec_from_matrix_jax(rel_rot_mat)
    site_rvel = calculate_relative_velocity_in_local_frame(main_site_xvel, site_xvel, main_site_xmat_traj,
                                                           rel_rot_mat, backend)

    return site_rpos, site_rangles, site_rvel


def safe_norm(x, axis=-1):
    """
    Euclidean norm over ``axis`` whose gradient is zero, not NaN, at the zero vector.

    Args:
        x (jnp.ndarray): Input array.
        axis (int): Axis to reduce.

    Returns:
        jnp.ndarray: The norms.
    """
    sq = jnp.sum(x * x, axis=axis)
    positive = sq > 0.0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, sq, 1.0)), 0.0)


def quaternion_angular_distance(q1, q2, backend):
    """
    Calculate the angular distance between two rotations represented by quaternions.

    Args:
        q1 (array): First quaternion. Shape (4,) or (batch_size, 4). (quaternions is expected to be scalar last)
        q2 (array): Second quaternion. Shape (4,) or (batch_size, 4). (quaternions is expected to be scalar last)

    Returns:
        array: Angular distance between the two rotations. Shape (batch_size,)
    """

    if backend == np:
        # Compute the relative rotation and extract its angle in radians
        return (np_R.from_quat(q1).inv() * np_R.from_quat(q2)).magnitude()

    # Same angle as jax.scipy's magnitude(), whose plain square root of the vector part's squared norm has a
    # NaN gradient when the two rotations are equal.
    q1 = q1 / jnp.linalg.norm(q1, axis=-1, keepdims=True)
    q2 = q2 / jnp.linalg.norm(q2, axis=-1, keepdims=True)
    v1, w1, v2, w2 = q1[..., :3], q1[..., 3:], q2[..., :3], q2[..., 3:]
    vec = w1 * v2 - w2 * v1 - jnp.cross(v1, v2)  # vector part of q1^-1 q2
    w = jnp.sum(q1 * q2, axis=-1)                 # its scalar part
    return 2.0 * jnp.arctan2(safe_norm(vec), jnp.abs(w))


def quat2angle(quat, backend):
    """
    Converts a quaternion to an angle. (quaternions is expected to be scalar last)
    """
    if backend == np:
        R = np_R
    else:
        R = jnp_R
    return R.from_quat(quat).as_rotvec()


def quat_scalarfirst2scalarlast(quat):
    """
    Converts a quaternion from scalar-first to scalar-last representation.
    """
    return quat[..., [1, 2, 3, 0]]


def quat_scalarlast2scalarfirst(quat):
    """
    Converts a quaternion from scalar-last to scalar-first representation.
    """
    return quat[..., [3, 0, 1, 2]]


def atleast_3d(tensor, backend):
    """
    Ensures the tensor has at least 3 dimensions by adding axes at the front if necessary.

    Args:
        tensor: Input tensor (numpy or JAX array).
        backend: The numerical backend, either numpy or jax.numpy.

    Returns:
        A tensor with at least 3 dimensions.
    """
    while tensor.ndim < 3:
        tensor = backend.expand_dims(tensor, axis=0)
    
    return tensor


def _column_runs(ind, offsets):
    """
    Splits static column indices into runs: stretches of consecutive columns of one array, or of negative indices.

    Args:
        ind (np.ndarray): Indices into the concatenation of some arrays.
        offsets (np.ndarray): The first column of each of the arrays in their concatenation, and its total width.

    Returns:
        List of (array, first column in the array, length) per run; the array is -1 for negative indices.

    """
    if len(ind) == 0:
        return []
    prev, cur = ind[:-1], ind[1:]
    joined = np.where(prev < 0, cur < 0, (cur == prev + 1) & ~np.isin(cur, offsets))
    starts = np.concatenate([[0], np.flatnonzero(~joined) + 1])
    lengths = np.diff(np.concatenate([starts, [len(ind)]]))
    arrays = np.where(ind[starts] < 0, -1, np.searchsorted(offsets, ind[starts], side="right") - 1)
    return [(int(a), int(i - offsets[a]), int(n)) for a, i, n in zip(arrays, ind[starts], lengths)]


def take_columns(x, ind, max_runs=64):
    """
    Returns the columns ``ind`` of ``x``, for static indices. ``x`` can be a list of arrays, standing for their
    concatenation along the last axis, and a negative index gives a zero.

    Unique indices that form a few runs of consecutive columns are sliced and concatenated, forward and backward.
    Otherwise the columns are gathered, and so is the gradient: on batched arrays a gather is slow, and the scatter
    that JAX would use for its gradient is very slow.

    Args:
        x (jnp.ndarray or list of jnp.ndarray): Array, or arrays, to select from.
        ind (np.ndarray): Indices into the last axis of x.
        max_runs (int): Largest number of runs to slice instead of gathering.

    Returns:
        The selected columns of x.

    """
    parts = tuple(x) if isinstance(x, (list, tuple)) else (x,)
    ind = np.asarray(ind)
    offsets = np.concatenate([[0], np.cumsum([p.shape[-1] for p in parts])]).astype(int)
    width, bounds = offsets[-1], list(zip(offsets[:-1], offsets[1:]))
    picked = ind[ind >= 0]

    def gather(sources, index):
        zero = jnp.zeros(sources[0].shape[:-1] + (1,), sources[0].dtype)
        padded = jnp.concatenate(tuple(sources) + (zero,), axis=-1)
        return padded[..., np.where(index < 0, padded.shape[-1] - 1, index)]

    if len(np.unique(picked)) < len(picked):
        return gather(parts, ind)

    def blocks(sources, runs):
        out = [jnp.zeros(sources[0].shape[:-1] + (n,), sources[0].dtype) if a < 0 else sources[a][..., i:i + n]
               for a, i, n in runs]
        return out[0] if len(out) == 1 else jnp.concatenate(out, axis=-1)

    # the inverse map, for the gradient: the position in the result of each column of the parts, or -1
    inverse = np.full(width, -1)
    inverse[picked] = np.flatnonzero(ind >= 0)
    runs = _column_runs(ind, offsets)
    inverse_runs = [_column_runs(inverse[a:b], np.array([0, len(ind)])) for a, b in bounds]
    sliced = len(ind) > 0 and max(len(runs), *(len(r) for r in inverse_runs)) <= max_runs

    @jax.custom_vjp
    def take(parts):
        return blocks(parts, runs) if sliced else gather(parts, ind)

    def backward(_, g):
        if sliced:
            return (tuple(blocks((g,), r) for r in inverse_runs),)
        return (tuple(gather((g,), inverse[a:b]) for a, b in bounds),)

    take.defvjp(lambda parts: (take(parts), None), backward)
    return take(parts)


# ---------------------------------------------------------------------------
# Rotation helpers in MuJoCo's quaternion convention (w, x, y, z)
#
# Unlike the SciPy-based functions above, which expect scalar-last quaternions, everything below takes
# scalar-first quaternions, batched along the leading axes. The quaternion functions are written for JAX
# and are differentiated through (the learned dynamics of SuperMimic): norms carry a small epsilon so that
# their gradients stay finite at zero rotation. Do not replace them by exact norms.
# ---------------------------------------------------------------------------

def eps_norm(v):
    """
    Norm along the last axis, kept as an axis of size 1, with a small epsilon under the square root so that the
    gradient is finite at zero. See :func:`safe_norm` for the exact norm with a zero gradient at zero.
    """
    # sqrt has an infinite derivative at 0; the epsilon keeps gradients finite at zero rotation.
    return jnp.sqrt(jnp.sum(v * v, axis=-1, keepdims=True) + 1e-12)


def quat_mul(a, b):
    """Hamilton product a * b."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return jnp.stack([aw * bw - ax * bx - ay * by - az * bz,
                      aw * bx + ax * bw + ay * bz - az * by,
                      aw * by - ax * bz + ay * bw + az * bx,
                      aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def quat_conj(q):
    """Conjugate, the inverse of a unit quaternion."""
    return q * jnp.array([1.0, -1.0, -1.0, -1.0])


def quat_rotate(q, v):
    """Rotates the vector v by q."""
    qv, w = q[..., 1:], q[..., :1]
    t = 2.0 * jnp.cross(qv, v)
    return v + w * t + jnp.cross(qv, t)


def quat_rotate_inv(q, v):
    """Rotates the vector v by the inverse of q."""
    return quat_rotate(quat_conj(q), v)


def quat_normalize(q):
    """Unit quaternion of q."""
    return q / eps_norm(q)


def heading_quat(q):
    """Rotation about world z by the yaw of the body x-axis."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    yaw = jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    zero = jnp.zeros_like(yaw)
    return jnp.stack([jnp.cos(0.5 * yaw), zero, zero, jnp.sin(0.5 * yaw)], axis=-1)


def quat_from_rotvec(v):
    """Quaternion of the rotation vector v."""
    th = eps_norm(v)
    return jnp.concatenate([jnp.cos(0.5 * th), v * (jnp.sin(0.5 * th) / th)], axis=-1)


def rotvec_from_quat(q):
    """Rotation vector of q, with an angle in [0, pi]."""
    q = jnp.where(q[..., :1] < 0.0, -q, q)
    s = eps_norm(q[..., 1:])
    angle = 2.0 * jnp.arctan2(s, q[..., :1])
    return q[..., 1:] * (angle / s)


def rot6d_from_quat(q):
    """6D rotation of q: the first column of the rotation matrix, then the second (c0, c1)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    c0 = jnp.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], axis=-1)
    c1 = jnp.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], axis=-1)
    return jnp.concatenate([c0, c1], axis=-1)


def quat_to_mat(q, backend=jnp):
    """Rotation matrix of the scalar-first quaternion q, shape (..., 4) -> (..., 3, 3)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    rows = [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
    return backend.stack(rows, axis=-1).reshape(q.shape[:-1] + (3, 3))


def heading_mat(R, backend=jnp):
    """Rotation about the world z axis by the heading of the rotation matrix R (3, 3): the yaw of its x axis."""
    n = backend.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2 + 1e-12)
    c, s = R[0, 0] / n, R[1, 0] / n
    zero, one = backend.zeros_like(c), backend.ones_like(c)
    return backend.stack([backend.stack([c, -s, zero]), backend.stack([s, c, zero]), backend.stack([zero, zero, one])])


def rot6d_from_mat(R, backend=jnp):
    """
    6D rotation of the rotation matrix R (3, 3), row by row: (r00, r01, r10, r11, r20, r21).

    Note the layout differs from :func:`rot6d_from_quat`, which returns the two columns one after the other.
    Both are inputs of trained networks, so neither can be changed without invalidating checkpoints.
    """
    return backend.ravel(R[:, :2])


def rotation_angle(mat_a, mat_b, backend=jnp):
    """Angle between rotation matrices of shape (..., 3, 3), with a finite gradient at equal rotations."""
    rel = backend.einsum("...ji,...jk->...ik", mat_a, mat_b)
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    axis = backend.stack([rel[..., 2, 1] - rel[..., 1, 2], rel[..., 0, 2] - rel[..., 2, 0],
                          rel[..., 1, 0] - rel[..., 0, 1]], axis=-1)
    sq = backend.sum(axis * axis, axis=-1)
    sin2 = backend.where(sq > 0.0, backend.sqrt(backend.where(sq > 0.0, sq, 1.0)), 0.0)
    return backend.arctan2(sin2, trace - 1.0)
