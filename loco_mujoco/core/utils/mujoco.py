"""Helpers to look up joint and geom indices in MuJoCo models and to query collisions."""

import mujoco
import numpy as np
import jax
import jax.numpy as jnp


def mj_jnt_name2id(name, model):
    """
    Get the joint ID (in the Mujoco datastructure) from the joint name.
    """
    for i in range(model.njnt):
        j = model.joint(i)
        if j.name == name:
            return i
    raise ValueError(f"Joint name {name} not found in model!")


def mj_jntname2qposid(j_name, model):
    """
    Get qpos index of a joint in mujoco data structure.

    Args:
        j_name (str): joint name.
        model (mjModel): mujoco model.

    Returns:
        list of qpos index in MjData corresponding to that joint.
    """
    j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
    if j_id == -1:
        raise ValueError(f"Joint name {j_name} not found in model.")

    return mj_jntid2qposid(j_id, model)


def mj_jntname2qvelid(j_name, model):
    """
    Get qvel index of a joint in mujoco data structure.

    Args:
        j_name (str): joint name.
        model (mjModel): mujoco model.

    Returns:
        list of qvel index in MjData corresponding to that joint.

    """
    j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
    if j_id == -1:
        raise ValueError(f"Joint name {j_name} not found in model.")

    return mj_jntid2qvelid(j_id, model)


def mj_jntid2qposid(j_id, model):
    """
    Get qpos index of a joint in mujoco data structure.

    Args:
        j_id (int): joint id.
        model (mjModel): mujoco model.

    Returns:
        list of qpos index in MjData corresponding to that joint.
    """
    start_qpos_id = model.jnt_qposadr[j_id]
    jnt_type = model.jnt_type[j_id]

    if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
        qpos_id = [i for i in range(start_qpos_id, start_qpos_id+7)]
    else:
        qpos_id = [start_qpos_id]

    return qpos_id


def mj_jntid2qvelid(j_id, model):
    """
    Get qvel index of a joint in mujoco data structure.

    Args:
        j_id (int): joint id.
        model (mjModel): mujoco model.

    Returns:
        list of qvel index in MjData corresponding to that joint.

    """
    start_qvel_id = model.jnt_dofadr[j_id]
    jnt_type = model.jnt_type[j_id]

    if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
        qvel_id = [i for i in range(start_qvel_id, start_qvel_id+6)]
    else:
        qvel_id = [start_qvel_id]

    return qvel_id


def mj_spec_find_geom_id(spec, geom_name):
    """
    Find geom id in mujoco specification.

    Args:
        spec (MjSpec): Mujoco specification.
        geom_name (str): geom name.

    Returns:
        int: geom id in Mujoco specification.
    """
    for i, g in enumerate(spec.geoms):
        if g.name == geom_name:
            return i
    raise ValueError(f"Geom {geom_name} not found in spec.")


def mj_get_collision_dist_and_normal(geom_id1, geom_id2, data, backend):
    """
    Get the distance and normal of the collision between two geoms.

    Args:
        geom_id1 (int): geom id in Mujoco model.
        geom_id2 (int): geom id in Mujoco model.
        data: Mujoco data structure.
        backend: np or jnp.

    Returns:
        tuple: distance and normal vector.
    """
    if backend == jnp:
        mask = (jnp.array([geom_id1, geom_id2]) == data.contact.geom).all(axis=1)
        mask |= (jnp.array([geom_id2, geom_id1]) == data.contact.geom).all(axis=1)
        idx = jnp.where(mask, data.contact.dist, 1e4).argmin()
        dist = data.contact.dist[idx] * mask[idx]
        normal = (dist < 0) * data.contact.frame[idx, 0, :3]
    else:
        found = False
        dist = 0.0
        normal = np.zeros(3)
        for i in range(data.ncon):
            con = data.contact[i]
            if ((con.geom[0] == geom_id1 and con.geom[1] == geom_id2) or
                (con.geom[0] == geom_id2 and con.geom[1] == geom_id1)):
                dist = con.dist
                normal = con.frame[:3] if dist < 0 else np.zeros(3)
                found = True
                break

    return dist, normal


def mj_check_collisions(geom_id1, geom_id2, data, backend):
    """
    Check if two geoms collide.

    Args:
        geom_id1 (int): geom id in Mujoco model.
        geom_id2 (int): geom id in Mujoco model.
        data: Mujoco data structure.
        backend: np or jnp.

    Returns:
        bool: True if geoms collide, False otherwise.
    """

    def _is_in_contact(con_id, res):
        con = data.contact[con_id]
        is_in_contact = np.logical_or(np.logical_and(con.geom1 == geom_id1, con.geom2 == geom_id2),
                                      np.logical_and(con.geom1 == geom_id2, con.geom2 == geom_id1))
        return np.logical_or(res, is_in_contact)

    if backend == jnp:
        return mj_get_collision_dist_and_normal(geom_id1, geom_id2, data, backend)[0] < 0
    else:
        return np.any([_is_in_contact(i, False) for i in range(data.ncon)])
