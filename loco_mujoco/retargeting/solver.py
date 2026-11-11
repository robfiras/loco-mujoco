from typing import Callable, NamedTuple, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from loco_mujoco.retargeting.kinematics import retract


class SolverConfig(NamedTuple):
    """Settings of the batched Gauss-Newton solver.

    Attributes:
        iterations: Solver steps per frame, warm started from the previous frame.
        init_iterations: Solver steps on the first frame of a clip.
        damping: Levenberg-Marquardt damping, also acts as temporal smoothing. Too little lets
            near redundant joints, such as a many jointed spine, jump between frames.
        floor_weight: Weight of the one-sided floor penetration residual.
        floor_margin: Height above the floor below which the floor residual becomes active.
        contact_weight: Weight pulling the lowest point of a body in contact onto the floor.
        contact_softness: Height differences over which points count as equally low. Without it,
            the solve flickers between the points of a body resting on the floor.
        log_rotation: Use the rotation vector as rotation error, which keeps pulling at large
            errors, instead of its sine.
        limit_passes: Solves per step for the active set of joints stopped by their limits.

    """
    iterations: int = 4
    init_iterations: int = 100
    damping: float = 1.0
    floor_weight: float = 50.0
    floor_margin: float = 0.0
    contact_weight: float = 30.0
    contact_softness: float = 0.01
    log_rotation: bool = True
    limit_passes: int = 3


def rotation_error(target: jax.Array, rot: jax.Array, log: bool = True) -> jax.Array:
    """Rotation error between stacks of rotation matrices, shape (S, 3).

    Args:
        target: Target rotation matrices, shape (S, 3, 3).
        rot: Rotation matrices, shape (S, 3, 3).
        log: Return the rotation vector instead of its sine.

    Returns:
        The rotation errors.

    """
    rel = jnp.einsum("nji,njk->nik", target, rot)
    sine = 0.5 * jnp.stack([rel[:, 2, 1] - rel[:, 1, 2],
                            rel[:, 0, 2] - rel[:, 2, 0],
                            rel[:, 1, 0] - rel[:, 0, 1]], axis=-1)
    if not log:
        return sine
    squared = jnp.sum(sine ** 2, axis=-1, keepdims=True)
    small = squared < 1e-8
    norm = jnp.sqrt(jnp.where(small, 1.0, squared))
    cosine = 0.5 * (jnp.trace(rel, axis1=-2, axis2=-1)[:, None] - 1.0)
    # the gain diverges towards 180 degrees, bound it to keep the Jacobian usable in float32
    divisor = jnp.where(cosine < 0.0, jnp.maximum(norm, 0.05), norm)
    return sine * jnp.where(small, 1.0, jnp.arctan2(norm, cosine) / divisor)


def soft_min(values: jax.Array, softness: float) -> jax.Array:
    """Mean of values weighted towards the lowest, never below the minimum."""
    return values @ jax.nn.softmax(-values / softness) if softness > 0.0 else values.min()


def make_solver(
    forward_kinematics: Callable,
    nv: int,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    pos_weights: np.ndarray,
    rot_weights: np.ndarray,
    config: SolverConfig = SolverConfig(),
    dof_damping: Optional[np.ndarray] = None,
    limit_dofs: Optional[np.ndarray] = None,
    contact_groups: Sequence[Sequence[int]] = (),
) -> Callable[..., jax.Array]:
    """Create a function retargeting a batch of clips.

    Args:
        forward_kinematics: Function from compile_forward_kinematics.
        nv: Number of degrees of freedom.
        joint_lower: Lower limits of all joints below the root, shape (nv - 6,).
        joint_upper: Upper limits of all joints below the root, shape (nv - 6,).
        pos_weights: Position weight of each site, shape (S,).
        rot_weights: Rotation weight of each site, shape (S,).
        config: Solver settings.
        dof_damping: Additional damping per degree of freedom, shape (nv,). Keeps joints near
            singularities from jumping to another solution.
        limit_dofs: Mask of the degrees of freedom whose limits are handled inside the solve,
            shape (nv,), all by default. All others are only clipped after each step.
        contact_groups: Indices of the contact points of each body that can touch the floor.

    Returns:
        A jitted function mapping initial configurations (B, nq), target site positions
        (B, T, S, 3), target site rotations (B, T, S, 3, 3) and, with contact groups, the contact
        state of each group (B, T, G) to configurations (B, T, nq).

    """
    pos_w = jnp.asarray(pos_weights)[:, None]
    rot_w = jnp.asarray(rot_weights)[:, None]
    lower, upper = jnp.asarray(joint_lower), jnp.asarray(joint_upper)
    damping = config.damping + (jnp.zeros(nv) if dof_damping is None else jnp.asarray(dof_damping))
    limited = jnp.ones(nv - 6, bool) if limit_dofs is None else jnp.asarray(limit_dofs, bool)[6:]

    def residual(qpos, target_pos, target_rot, contact):
        pos, rot, points = forward_kinematics(qpos)
        floor = config.floor_weight * jax.nn.relu(config.floor_margin - points[:, 2])
        lowest = jnp.stack([soft_min(points[jnp.asarray(group), 2], config.contact_softness)
                            for group in contact_groups]) if contact_groups else jnp.zeros(0, qpos.dtype)
        return jnp.concatenate([(pos_w * (pos - target_pos)).ravel(),
                                (rot_w * rotation_error(target_rot, rot, config.log_rotation)).ravel(),
                                floor, config.contact_weight * contact * lowest])

    def step(qpos, target):
        def local_residual(delta):
            return residual(retract(qpos, delta), *target)

        zero = jnp.zeros(nv, qpos.dtype)
        r = local_residual(zero)
        jac = jax.jacfwd(local_residual)(zero)
        hessian = jac.T @ jac + jnp.diag(damping.astype(qpos.dtype))
        gradient = jac.T @ r
        room_low, room_high = lower - qpos[7:], upper - qpos[7:]

        # a joint stopped by its limit must be taken out of the solve, otherwise all other joints
        # keep compensating for a motion that the limit never lets happen
        def solve_free(pinned, pinned_delta):
            free = 1.0 - pinned
            reduced = hessian * free[:, None] * free[None, :] + jnp.diag(pinned)
            rhs = -free * (gradient + hessian @ pinned_delta)
            return jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(reduced), rhs) + pinned_delta

        pinned = jnp.zeros(nv, qpos.dtype)
        pinned_delta = jnp.zeros(nv, qpos.dtype)
        delta = solve_free(pinned, pinned_delta)
        for _ in range(config.limit_passes - 1):
            clipped = jnp.clip(delta[6:], room_low, room_high)
            over = (clipped != delta[6:]) & limited
            pinned = pinned.at[6:].max(over.astype(qpos.dtype))
            pinned_delta = pinned_delta.at[6:].set(jnp.where(over, clipped, pinned_delta[6:]))
            delta = solve_free(pinned, pinned_delta)
        delta = jnp.where(jnp.isfinite(delta).all(), delta, jnp.zeros_like(delta))
        qpos = retract(qpos, delta)
        return qpos.at[7:].set(jnp.clip(qpos[7:], lower, upper))

    def frame(qpos, target):
        for _ in range(config.iterations):
            qpos = step(qpos, target)
        return qpos, qpos

    def clip(qpos, target_pos, target_rot, contact):
        target = (target_pos, target_rot, contact)
        first = jax.tree.map(lambda x: x[0], target)
        qpos = jax.lax.fori_loop(0, config.init_iterations, lambda _, q: step(q, first), qpos)
        return jax.lax.scan(frame, qpos, target)[1]

    solve = jax.jit(jax.vmap(clip))

    def solve_clips(qpos, target_pos, target_rot, contact=None):
        if contact is None:
            contact = jnp.zeros(target_pos.shape[:2] + (len(contact_groups),), target_pos.dtype)
        return solve(qpos, target_pos, target_rot, contact.astype(target_pos.dtype))

    return solve_clips


def pad_clips(arrays: Tuple[np.ndarray, ...], length: int) -> np.ndarray:
    """Stack clips of different lengths, repeating each clip's last frame up to the given length."""
    return np.stack([np.concatenate([a, np.repeat(a[-1:], length - len(a), axis=0)])
                     for a in arrays])
