import numpy as np


class RewardInterface:
    """
    Interface to specify a reward function.

    """

    def __call__(self, state, action, next_state, absorbing):
        """
        Compute the reward.

        Args:
            state (np.ndarray): last state;
            action (np.ndarray): applied action;
            next_state (np.ndarray): current state.

        Returns:
            The reward for the current transition.

        """
        raise NotImplementedError

    def reset_state(self):
        """
        Reset the state of the object.

        """
        pass


class NoReward(RewardInterface):
    """
    A reward function that returns always 0.

    """

    def __call__(self, state, action, next_state, absorbing):
        return 0


class CustomReward(RewardInterface):

    def __init__(self, reward_callback=None):
        self._reward_callback = reward_callback

    def __call__(self, state, action, next_state, absorbing):
        if self._reward_callback is not None:
            return self._reward_callback(state, action, next_state)
        else:
            return 0


class TargetVelocityReward(RewardInterface):

    def __init__(self, target_velocity, x_vel_idx):
        self._target_vel = target_velocity
        self._x_vel_idx = x_vel_idx

    def __call__(self, state, action, next_state, absorbing):
        x_vel = state[self._x_vel_idx]
        return np.exp(- np.square(x_vel - self._target_vel))


class MultiTargetVelocityReward(RewardInterface):

    def __init__(self, target_velocity, x_vel_idx, env_id_len, scalings):
        self._target_vel = target_velocity
        self._env_id_len = env_id_len
        self._scalings = scalings
        self._x_vel_idx = x_vel_idx

    def __call__(self, state, action, next_state, absorbing):
        x_vel = state[self._x_vel_idx]
        env_id = state[-self._env_id_len:]

        # convert binary array to index
        ind = np.packbits(env_id.astype(int), bitorder='big') >> (8 - env_id.shape[0])
        ind = ind[0]
        scaling = self._scalings[ind]

        # calculate target vel
        target_vel = self._target_vel * scaling

        return np.exp(- np.square(x_vel - target_vel))


class PosReward(RewardInterface):

    def __init__(self, pos_idx):
        self._pos_idx = pos_idx

    def __call__(self, state, action, next_state, absorbing):
        pos = state[self._pos_idx]
        return pos
