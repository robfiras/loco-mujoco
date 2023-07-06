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
