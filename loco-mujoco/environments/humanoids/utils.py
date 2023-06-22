import numpy as np
from mushroom_rl.utils.angles import euler_to_quat, quat_to_euler


def convert_traj_euler_to_quat(euler_traj, offset=0):
    """
    Convert humanoid trajectory from euler to quaternion.

    Args:
        euler_traj (np.ndarray): trajectory with euler angles;
        offset (int, 0): number of observation to skip.

    Returns:
        The converted trajectory.

    """
    euler_traj = euler_traj.copy()
    is_vect = len(euler_traj.shape) < 2

    if is_vect:
        quat_traj = np.zeros((euler_traj.shape[0] + 1, ))
        quat_traj[:3-offset] = euler_traj[0:3-offset]
        quat_traj[3-offset:7-offset] = euler_to_quat(euler_traj[
                                                     3-offset:6-offset])
        quat_traj[7-offset:] = euler_traj[6-offset:]
    else:
        quat_traj = np.zeros((euler_traj.shape[0] + 1, euler_traj.shape[1]))
        quat_traj[:3-offset, :] = euler_traj[0:3-offset, :]
        quat_traj[3-offset:7-offset, :] = euler_to_quat(euler_traj[
                                                        3-offset:6-offset, :])
        quat_traj[7-offset:, :] = euler_traj[6-offset:, :]

    return quat_traj


def convert_traj_quat_to_euler(quat_traj, offset=0):
    """
    Convert humanoid trajectory from quaternion to euler.

    Args:
        quat_traj (np.ndarray): trajectory with quaternions;
        offset (int, 0): number of observation to skip.

    Returns:
        The converted trajectory.

    """
    quat_traj = quat_traj.copy()
    is_vect = len(quat_traj.shape) < 2

    if is_vect:
        euler_traj = np.zeros((quat_traj.shape[0] - 1, ))
        euler_traj[:3-offset] = quat_traj[0:3-offset]
        euler_traj[3-offset:6-offset] = quat_to_euler(quat_traj[
                                                      3-offset:7-offset])
        euler_traj[6-offset:] = quat_traj[7-offset:]
    else:
        euler_traj = np.zeros((quat_traj.shape[0] - 1, quat_traj.shape[1]))
        euler_traj[:3-offset, :] = quat_traj[0:3-offset, :]
        euler_traj[3-offset:6-offset, :] = quat_to_euler(quat_traj[
                                                         3-offset:7-offset, :])
        euler_traj[6-offset:, :] = quat_traj[7-offset:, :]

    return euler_traj
