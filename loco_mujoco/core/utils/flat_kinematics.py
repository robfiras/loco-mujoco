"""Lean forward kinematics for a floating-base tree of hinge joints, for use inside long differentiated rollouts."""

import jax.numpy as jnp
import mujoco
import numpy as np

from loco_mujoco.core.utils.math import quat_mul as _quat_mul, quat_to_mat as _quat_to_mat


def _cross(a, b):
    """Cross product along the last axis, written out in components."""
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    return jnp.stack([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx], axis=-1)


def _rotate(mat, v):
    """Rotate the vectors ``v`` (..., 3) with the rotation matrices ``mat`` (..., 3, 3)."""
    return jnp.sum(mat * v[..., None, :], axis=-1)


class FlatKinematics:
    """
    Forward kinematics and velocities of one floating tree of hinge joints, body by body. Gives the same ``xpos``,
    ``xquat``, ``xmat``, ``site_xpos``, ``site_xmat`` and ``cvel`` as ``mjx.kinematics``, ``mjx.com_pos`` and
    ``mjx.com_vel``, with far fewer operations, which matters inside long differentiated rollouts.

    Only these fields are updated. ``subtree_com`` holds the center of mass of the whole tree for every body, which is
    the entry ``cvel`` refers to; all other kinematic fields of the data stay as they are.

    Every body is a few products and sums of small arrays, stacked once at the end: batched over many states, neither
    the forward nor the backward pass has a gather or a scatter.

    Args:
        model (mujoco.MjModel): Model with a free joint on the first body, and at most one hinge on every other body.

    """

    def __init__(self, model: mujoco.MjModel):
        hinge = mujoco.mjtJoint.mjJNT_HINGE
        assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE and model.jnt_bodyid[0] == 1, \
            "FlatKinematics needs a free joint on the first body."
        assert np.all(model.jnt_type[1:] == hinge) and np.all(model.body_jntnum[2:] <= 1) \
            and np.all(model.body_parentid[2:] >= 1), "FlatKinematics needs one tree with at most one hinge per body."
        self._parent, self._pos, self._quat = model.body_parentid, model.body_pos, model.body_quat
        self._jnt, self._axis, self._anchor = model.body_jntadr, model.jnt_axis, model.jnt_pos
        self._qpos, self._dof, self._qpos0 = model.jnt_qposadr, model.jnt_dofadr, model.qpos0
        # a hinge angle a turns the body frame to cos(a/2) * quat + sin(a/2) * turn
        self._turn = np.zeros((model.nbody, 4))
        self._anchor_in_parent = np.zeros((model.nbody, 3))
        for b in range(2, model.nbody):
            if model.body_jntadr[b] >= 0:
                j = model.body_jntadr[b]
                mujoco.mju_mulQuat(self._turn[b], model.body_quat[b], np.concatenate([[0.0], model.jnt_axis[j]]))
                mujoco.mju_rotVecQuat(self._anchor_in_parent[b], model.jnt_pos[j], model.body_quat[b])
        self._ipos, self._mass = model.body_ipos[1:], model.body_mass[1:]
        self._site_body, self._site_pos = model.site_bodyid, model.site_pos
        self._site_mat = np.zeros((model.nsite, 9))
        for s in range(model.nsite):
            mujoco.mju_quat2Mat(self._site_mat[s], model.site_quat[s])
        self._site_mat = self._site_mat.reshape(-1, 3, 3)

    def __call__(self, data):
        """
        Returns data with the fields above brought up to its qpos and qvel. Works on a single state.

        Args:
            data (mjx.Data): The simulation data of a single (unbatched) state.

        Returns:
            mjx.Data: Copy of the data with updated ``xpos``, ``xquat``, ``xmat``, ``cvel``, ``subtree_com``,
            ``site_xpos`` and ``site_xmat``.

        """
        qpos, qvel = data.qpos, data.qvel
        root = qpos[3:7] / jnp.linalg.norm(qpos[3:7])
        pos, quat = [jnp.zeros(3), qpos[:3]], [jnp.array([1.0, 0.0, 0.0, 0.0]), root]
        mat = [jnp.eye(3), _quat_to_mat(root)]
        anchor = [None, None]  # of each body's hinge, in the world
        bodies = range(2, len(self._parent))
        for b in bodies:
            par, j = self._parent[b], self._jnt[b]
            local = self._quat[b]
            if j >= 0:
                half = 0.5 * (qpos[self._qpos[j]] - self._qpos0[self._qpos[j]])
                local = jnp.cos(half) * local + jnp.sin(half) * self._turn[b]
            quat.append(_quat_mul(quat[par], local))
            mat.append(_quat_to_mat(quat[b]))
            p = pos[par] + _rotate(mat[par], self._pos[b])
            anchor.append(p)
            if j >= 0 and np.any(self._anchor[j] != 0):  # the hinge is off the body frame: the body turns around it
                anchor[b] = p + _rotate(mat[par], self._anchor_in_parent[b])
                p = anchor[b] - _rotate(mat[b], self._anchor[j])
            pos.append(p)

        xpos, xquat, xmat = jnp.stack(pos), jnp.stack(quat), jnp.stack(mat)
        xipos = xpos[1:] + _rotate(xmat[1:], self._ipos)
        com = jnp.sum(self._mass[:, None] * xipos, axis=0) / np.sum(self._mass)

        # velocities in cvel's convention: angular, then linear at the center of mass
        w = _rotate(mat[1], qvel[3:6])
        vel = [jnp.zeros(6), jnp.concatenate([w, qvel[:3] + _cross(w, com - pos[1])])]
        for b in bodies:
            par, j = self._parent[b], self._jnt[b]
            vel.append(vel[par])
            if j >= 0:
                axis = _rotate(mat[b], self._axis[j])
                vel[b] = vel[par] + qvel[self._dof[j]] * jnp.concatenate([axis, _cross(axis, com - anchor[b])])

        data = data.replace(xpos=xpos, xquat=xquat, xmat=xmat, cvel=jnp.stack(vel),
                            subtree_com=jnp.broadcast_to(com, data.subtree_com.shape))
        if len(self._site_body) == 0:
            return data
        site_pos, site_mat = jnp.stack([pos[b] for b in self._site_body]), jnp.stack([mat[b] for b in self._site_body])
        return data.replace(site_xpos=site_pos + _rotate(site_mat, self._site_pos),
                            site_xmat=jnp.sum(site_mat[:, :, :, None] * self._site_mat[:, None, :, :], axis=-2))
