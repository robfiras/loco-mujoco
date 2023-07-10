import numpy as np

from loco_mujoco.environments import UnitreeA1


def experiment(seed=0):

    np.random.seed(seed)

    # define env and data frequencies
    env_freq = 1000  # hz, added here as a reminder
    traj_data_freq = 500    # hz, added here as a reminder
    desired_contr_freq = 100     # hz

    # prepare trajectory params
    traj_params = dict(traj_path="../datasets/quadrupeds/states_2023_02_23_19_48_33.npz",
                       traj_dt=(1 / traj_data_freq),
                       control_dt=(1 / desired_contr_freq),
                       clip_trajectory_to_joint_ranges=True)

    # MDP
    gamma = 0.99
    horizon = 1000
    mdp = UnitreeA1(gamma=gamma, horizon=horizon, traj_params=traj_params)

    mdp.play_trajectory_demo()


if __name__ == '__main__':
    experiment()
