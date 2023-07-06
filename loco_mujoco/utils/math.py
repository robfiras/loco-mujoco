import numpy as np


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

    mat = np.dot(mat.reshape((3, 3)), np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])).reshape((9,))
    angle = np.arctan2(mat[3], mat[0])

    return angle


def angle2mat_xy(angle):
    """
    Converts a rotation angle in the x-y-plane to a rotation matrix.

    Args:
        angle (float): Angle to be converted.

    Returns:
        np.array of shape (3, 3)

    """

    mat = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

    # rotate with the default arrow rotation (else the arrow is vertical)
    arrow = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0]).reshape((3, 3))
    mat = np.dot(mat, arrow)

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
