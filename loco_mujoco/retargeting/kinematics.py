import copy
from typing import Callable, Sequence, Tuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx


def _quat_to_mat_np(quat: np.ndarray) -> np.ndarray:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=np.float64))
    return mat.reshape(3, 3)


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def quat_to_mat(q: jax.Array) -> jax.Array:
    """Convert a scalar-first quaternion to a rotation matrix."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jnp.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_mul(a: jax.Array, b: jax.Array) -> jax.Array:
    """Multiply two scalar-first quaternions."""
    w1, x1, y1, z1 = a[0], a[1], a[2], a[3]
    w2, x2, y2, z2 = b[0], b[1], b[2], b[3]
    return jnp.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def check_floating_base(model: mujoco.MjModel) -> None:
    """Check that a model has one free root joint and otherwise only hinge and slide joints.

    Raises:
        ValueError: If the model has a different joint layout.

    """
    free = mujoco.mjtJoint.mjJNT_FREE
    if model.njnt == 0 or model.jnt_type[0] != free or model.jnt_qposadr[0] != 0:
        raise ValueError("Expected a free root joint at the start of qpos.")
    if (model.jnt_type[1:] == free).any() or (model.jnt_type == mujoco.mjtJoint.mjJNT_BALL).any():
        raise ValueError("Only hinge and slide joints are supported below the free root joint.")


def compile_forward_kinematics(
    model: mujoco.MjModel,
    site_ids: Sequence[int],
    point_bodies: Sequence[int] = (),
    point_positions: Sequence[np.ndarray] = (),
) -> Callable[[jax.Array], Tuple[jax.Array, jax.Array, jax.Array]]:
    """Compile a MuJoCo kinematic tree into a pure JAX forward kinematics function.

    The tree is unrolled at trace time and restricted to the ancestors of the requested sites and
    points. It compiles faster than compile_mjx_forward_kinematics, but runs slower on GPU.

    Args:
        model: The MuJoCo model.
        site_ids: Ids of the sites to compute poses for.
        point_bodies: Body ids of additional points to compute positions for.
        point_positions: Position of each additional point in its body frame.

    Returns:
        A function mapping qpos to site positions (S, 3), site rotation matrices (S, 3, 3), and
        point positions (P, 3).

    """
    check_floating_base(model)

    needed = set()
    for body in [int(model.site_bodyid[s]) for s in site_ids] + [int(b) for b in point_bodies]:
        while body != 0:
            needed.add(body)
            body = int(model.body_parentid[body])

    plan = []
    for body in sorted(needed):
        joints = []
        for j in range(model.body_jntadr[body], model.body_jntadr[body] + model.body_jntnum[body]):
            adr = int(model.jnt_qposadr[j])
            axis = np.array(model.jnt_axis[j])
            joints.append((int(model.jnt_type[j]), adr, np.array(model.jnt_pos[j]), axis,
                           _skew(axis), float(model.qpos0[adr])))
        plan.append((body, int(model.body_parentid[body]), np.array(model.body_pos[body]),
                     _quat_to_mat_np(model.body_quat[body]), joints))

    site_bodies = [int(model.site_bodyid[s]) for s in site_ids]
    site_pos = [np.array(model.site_pos[s]) for s in site_ids]
    site_rot = [_quat_to_mat_np(model.site_quat[s]) for s in site_ids]
    points = [(int(b), np.array(p)) for b, p in zip(point_bodies, point_positions)]
    eye = np.eye(3)

    def forward_kinematics(qpos: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        xpos = {0: jnp.zeros(3, qpos.dtype)}
        xmat = {0: jnp.eye(3, dtype=qpos.dtype)}
        for body, parent, body_pos, body_rot, joints in plan:
            pos = xpos[parent] + xmat[parent] @ body_pos
            rot = xmat[parent] @ body_rot
            for joint_type, adr, joint_pos, axis, axis_skew, qpos0 in joints:
                if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                    pos = qpos[adr:adr + 3]
                    rot = quat_to_mat(qpos[adr + 3:adr + 7])
                elif joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                    angle = qpos[adr] - qpos0
                    anchor = pos + rot @ joint_pos
                    rot = rot @ (eye + jnp.sin(angle) * axis_skew
                                 + (1 - jnp.cos(angle)) * (axis_skew @ axis_skew))
                    pos = anchor - rot @ joint_pos
                else:
                    pos = pos + rot @ (axis * (qpos[adr] - qpos0))
            xpos[body], xmat[body] = pos, rot

        out_pos = jnp.stack([xpos[b] + xmat[b] @ p for b, p in zip(site_bodies, site_pos)])
        out_rot = jnp.stack([xmat[b] @ r for b, r in zip(site_bodies, site_rot)])
        if points:
            out_points = jnp.stack([xpos[b] + xmat[b] @ p for b, p in points])
        else:
            out_points = jnp.zeros((0, 3), qpos.dtype)
        return out_pos, out_rot, out_points

    return forward_kinematics


def compile_mjx_forward_kinematics(
    model: mujoco.MjModel,
    site_ids: Sequence[int],
    point_bodies: Sequence[int] = (),
    point_positions: Sequence[np.ndarray] = (),
) -> Callable[[jax.Array], Tuple[jax.Array, jax.Array, jax.Array]]:
    """Create a forward kinematics function based on mjx.kinematics.

    Same interface as compile_forward_kinematics. Collisions are disabled on a copy of the model,
    since kinematics do not need them and MJX does not support all geom pairs.

    Args:
        model: The MuJoCo model.
        site_ids: Ids of the sites to compute poses for.
        point_bodies: Body ids of additional points to compute positions for.
        point_positions: Position of each additional point in its body frame.

    Returns:
        A function mapping qpos to site positions (S, 3), site rotation matrices (S, 3, 3), and
        point positions (P, 3).

    """
    check_floating_base(model)

    kinematic_model = copy.copy(model)
    kinematic_model.geom_contype[:] = 0
    kinematic_model.geom_conaffinity[:] = 0
    mjx_model = mjx.put_model(kinematic_model)
    mjx_data = mjx.make_data(mjx_model)

    sites = jnp.asarray(site_ids, dtype=jnp.int32)
    bodies = jnp.asarray(point_bodies, dtype=jnp.int32)
    local = jnp.asarray(np.array(point_positions).reshape(-1, 3))

    def forward_kinematics(qpos: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        data = mjx.kinematics(mjx_model, mjx_data.replace(qpos=qpos))
        points = data.xpos[bodies] + jnp.einsum("nij,nj->ni", data.xmat[bodies], local)
        return data.site_xpos[sites], data.site_xmat[sites], points

    return forward_kinematics


def retract(qpos: jax.Array, delta: jax.Array) -> jax.Array:
    """Apply a tangent space update to a configuration with a free root joint.

    Args:
        qpos: Configuration of size nq.
        delta: Update of size nv, with the root's angular part in the body frame.

    Returns:
        The updated configuration.

    """
    dq = jnp.concatenate([jnp.ones(1, qpos.dtype), 0.5 * delta[3:6]])
    quat = quat_mul(qpos[3:7], dq / jnp.linalg.norm(dq))
    return jnp.concatenate([qpos[:3] + delta[:3], quat / jnp.linalg.norm(quat),
                            qpos[7:] + delta[6:]])
