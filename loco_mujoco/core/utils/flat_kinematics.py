import jax.numpy as jnp
import mujoco
import numpy as np


def _quat_mul(a, b):
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return jnp.concatenate([aw * bw - jnp.sum(av * bv, axis=-1, keepdims=True),
                            aw * bv + bw * av + jnp.cross(av, bv)], axis=-1)


def _quat_rotate(q, v):
    w, u = q[..., :1], q[..., 1:]
    t = 2.0 * jnp.cross(u, v)
    return v + w * t + jnp.cross(u, t)


def _quat_to_mat(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    rows = [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
    return jnp.stack(rows, axis=-1).reshape(q.shape[:-1] + (3, 3))


class FlatKinematics:
    """
    Forward kinematics and velocities of one floating tree of hinge joints, level by level of the tree. Gives the
    same ``xpos``, ``xquat``, ``xmat``, ``site_xpos``, ``site_xmat`` and ``cvel`` as ``mjx.kinematics``, ``mjx.com_pos``
    and ``mjx.com_vel``, with far fewer operations, which matters inside long differentiated rollouts.

    Only these fields are updated. ``subtree_com`` holds the center of mass of the whole tree for every body, which is
    the entry ``cvel`` refers to; all other kinematic fields of the data stay as they are.

    Args:
        model (mujoco.MjModel): Model with a free joint on the first body, and at most one hinge on every other body.

    """

    def __init__(self, model: mujoco.MjModel):
        hinge = mujoco.mjtJoint.mjJNT_HINGE
        assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE and model.jnt_bodyid[0] == 1, \
            "FlatKinematics needs a free joint on the first body."
        assert np.all(model.jnt_type[1:] == hinge) and np.all(model.body_jntnum[2:] <= 1) \
            and np.all(model.body_parentid[2:] >= 1), "FlatKinematics needs one tree with at most one hinge per body."
        self._levels = []  # per level: bodies, positions of their parents in the level above, and static data
        depth = np.zeros(model.nbody, dtype=int)
        depth[1] = 1
        for b in range(2, model.nbody):
            depth[b] = depth[model.body_parentid[b]] + 1
        slot = {1: 0}
        for d in range(2, depth.max() + 1):
            bodies = np.flatnonzero(depth == d)
            jnt = np.array([model.body_jntadr[b] for b in bodies])
            has = jnt >= 0
            jnt = np.where(has, jnt, 0)
            self._levels.append(dict(
                bodies=bodies, parents=[slot[model.body_parentid[b]] for b in bodies], has_joint=has[:, None],
                pos=model.body_pos[bodies], quat=model.body_quat[bodies], axis=model.jnt_axis[jnt],
                anchor=model.jnt_pos[jnt], qpos=model.jnt_qposadr[jnt], dof=model.jnt_dofadr[jnt],
                qpos0=model.qpos0[model.jnt_qposadr[jnt]]))
            slot.update({b: i for i, b in enumerate(bodies)})
        self._order = np.argsort(np.concatenate([[0, 1]] + [lv["bodies"] for lv in self._levels]))
        self._ipos, self._iquat, self._mass = model.body_ipos[1:], model.body_iquat[1:], model.body_mass[1:]
        self._site_body, self._site_pos, self._site_quat = model.site_bodyid, model.site_pos, model.site_quat

    def __call__(self, data):
        """Returns data with the fields above brought up to its qpos and qvel. Works on a single state."""
        qpos, qvel = data.qpos, data.qvel
        p = qpos[None, :3]
        q = qpos[None, 3:7] / jnp.linalg.norm(qpos[3:7])
        positions, quats, hinges = [p], [q], []
        for lv in self._levels:
            take = lambda x: jnp.stack([x[i] for i in lv["parents"]])  # noqa: E731, static slices keep gradients cheap
            pq = take(q)
            quat = _quat_mul(pq, lv["quat"])
            pos = take(p) + _quat_rotate(pq, lv["pos"])
            anchor = pos + _quat_rotate(quat, lv["anchor"])
            axis = _quat_rotate(quat, lv["axis"])
            half = 0.5 * (qpos[lv["qpos"]] - lv["qpos0"])[:, None]
            turned = _quat_mul(quat, jnp.concatenate([jnp.cos(half), jnp.sin(half) * lv["axis"]], axis=-1))
            quat = jnp.where(lv["has_joint"], turned, quat)
            pos = jnp.where(lv["has_joint"], anchor - _quat_rotate(quat, lv["anchor"]), pos)
            p, q = pos, quat
            positions.append(p)
            quats.append(q)
            hinges.append((axis, anchor))

        all_pos, all_quat = jnp.concatenate(positions), jnp.concatenate(quats)
        world = jnp.zeros((1, 3)), jnp.array([[1.0, 0.0, 0.0, 0.0]])
        xpos = jnp.concatenate([world[0], all_pos])[self._order]
        xquat = jnp.concatenate([world[1], all_quat])[self._order]
        xmat = _quat_to_mat(xquat)
        xipos = xpos[1:] + _quat_rotate(xquat[1:], self._ipos)
        com = jnp.sum(self._mass[:, None] * xipos, axis=0) / np.sum(self._mass)

        # velocities in cvel's convention: angular, then linear at the center of mass
        w = _quat_rotate(quats[0][0], qvel[3:6])
        vel = jnp.concatenate([w, qvel[:3] + jnp.cross(w, com - positions[0][0])])[None]
        velocities = [vel]
        for lv, (axis, anchor) in zip(self._levels, hinges):
            speed = jnp.where(lv["has_joint"], qvel[lv["dof"]][:, None], 0.0)
            dof = jnp.concatenate([axis, jnp.cross(axis, com - anchor)], axis=-1)
            vel = jnp.stack([vel[i] for i in lv["parents"]]) + speed * dof
            velocities.append(vel)
        cvel = jnp.concatenate([jnp.zeros((1, 6)), jnp.concatenate(velocities)])[self._order]

        site_quat = _quat_mul(xquat[self._site_body], self._site_quat)
        return data.replace(xpos=xpos, xquat=xquat, xmat=xmat, cvel=cvel,
                            subtree_com=jnp.broadcast_to(com, data.subtree_com.shape),
                            site_xpos=xpos[self._site_body] + _quat_rotate(xquat[self._site_body], self._site_pos),
                            site_xmat=_quat_to_mat(site_quat))
