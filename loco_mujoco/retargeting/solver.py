from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from loco_mujoco.retargeting.kinematics import retract


class SolverConfig(NamedTuple):
    """Settings of the batched Gauss-Newton solver.

    Attributes:
        iterations: Solver steps per frame, warm started from the previous frame.
        init_iterations: Solver steps on the first frame of a clip.
        damping: Levenberg-Marquardt damping, also acts as temporal smoothing.
        floor_weight: Weight of the one-sided floor penetration residual.
        floor_margin: Height above the floor below which the floor residual becomes active.

    """
    iterations: int = 4
    init_iterations: int = 100
    damping: float = 1e-2
    floor_weight: float = 50.0
    floor_margin: float = 0.0


def rotation_error(target: jax.Array, rot: jax.Array) -> jax.Array:
    """Smooth rotation error between stacks of rotation matrices, shape (S, 3)."""
    rel = jnp.einsum("nji,njk->nik", target, rot)
    return 0.5 * jnp.stack([rel[:, 2, 1] - rel[:, 1, 2],
                            rel[:, 0, 2] - rel[:, 2, 0],
                            rel[:, 1, 0] - rel[:, 0, 1]], axis=-1)


def make_solver(
    forward_kinematics: Callable,
    nv: int,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    pos_weights: np.ndarray,
    rot_weights: np.ndarray,
    config: SolverConfig = SolverConfig(),
) -> Callable[[jax.Array, jax.Array, jax.Array], jax.Array]:
    """Create a function retargeting a batch of clips.

    Args:
        forward_kinematics: Function from compile_forward_kinematics.
        nv: Number of degrees of freedom.
        joint_lower: Lower limits of all joints below the root, shape (nv - 6,).
        joint_upper: Upper limits of all joints below the root, shape (nv - 6,).
        pos_weights: Position weight of each site, shape (S,).
        rot_weights: Rotation weight of each site, shape (S,).
        config: Solver settings.

    Returns:
        A jitted function mapping initial configurations (B, nq), target site positions
        (B, T, S, 3) and target site rotations (B, T, S, 3, 3) to configurations (B, T, nq).

    """
    pos_w = jnp.asarray(pos_weights)[:, None]
    rot_w = jnp.asarray(rot_weights)[:, None]
    lower, upper = jnp.asarray(joint_lower), jnp.asarray(joint_upper)

    def residual(qpos, target_pos, target_rot):
        pos, rot, points = forward_kinematics(qpos)
        floor = config.floor_weight * jax.nn.relu(config.floor_margin - points[:, 2])
        return jnp.concatenate([(pos_w * (pos - target_pos)).ravel(),
                                (rot_w * rotation_error(target_rot, rot)).ravel(),
                                floor])

    def step(qpos, target):
        target_pos, target_rot = target

        def local_residual(delta):
            return residual(retract(qpos, delta), target_pos, target_rot)

        zero = jnp.zeros(nv, qpos.dtype)
        r = local_residual(zero)
        jac = jax.jacfwd(local_residual)(zero)
        hessian = jac.T @ jac + config.damping * jnp.eye(nv, dtype=qpos.dtype)
        delta = -jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(hessian), jac.T @ r)
        qpos = retract(qpos, delta)
        return qpos.at[7:].set(jnp.clip(qpos[7:], lower, upper))

    def frame(qpos, target):
        for _ in range(config.iterations):
            qpos = step(qpos, target)
        return qpos, qpos

    def clip(qpos, target_pos, target_rot):
        first = (target_pos[0], target_rot[0])
        qpos = jax.lax.fori_loop(0, config.init_iterations, lambda _, q: step(q, first), qpos)
        return jax.lax.scan(frame, qpos, (target_pos, target_rot))[1]

    return jax.jit(jax.vmap(clip))


def pad_clips(arrays: Tuple[np.ndarray, ...], length: int) -> np.ndarray:
    """Stack clips of different lengths, repeating each clip's last frame up to the given length."""
    return np.stack([np.concatenate([a, np.repeat(a[-1:], length - len(a), axis=0)])
                     for a in arrays])
