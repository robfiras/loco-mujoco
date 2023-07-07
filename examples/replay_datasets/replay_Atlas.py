import numpy as np

from loco_mujoco.environments import Atlas


def experiment(seed=0):

    np.random.seed(seed)

    # define env and data frequencies
    env_freq = 1000  # hz, added here as a reminder
    traj_data_freq = 500    # hz, added here as a reminder
    desired_contr_freq = 100     # hz
    n_substeps = env_freq//desired_contr_freq

    # set a reward for logging
    reward_callback = lambda state, action, next_state: np.exp(- np.square(state[14] - 1.25))  # x-velocity as reward

    # prepare trajectory params
    traj_params = dict(traj_path="../datasets/humanoids/02-constspeed_ATLAS.npz",
                       traj_dt=(1 / traj_data_freq),
                       control_dt=(1 / desired_contr_freq))

    # MDP
    gamma = 0.99
    horizon = 1000
    mdp = Atlas(gamma=gamma, horizon=horizon, traj_params=traj_params)

    mdp.play_trajectory_demo()


if __name__ == '__main__':
    experiment()
