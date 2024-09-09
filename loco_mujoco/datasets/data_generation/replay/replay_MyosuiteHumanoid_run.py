import numpy as np
from loco_mujoco.environments import MyoSuiteHumanoid


def experiment():
    np.random.seed(1)

    # define env and data frequencies
    env_freq = 1000  # hz, added here as a reminder
    traj_data_freq = 500    # hz, frequency of the mocap dataset
    desired_contr_freq = 100     # hz, this will also be the dataset frequency after downsampling
    n_substeps = env_freq//desired_contr_freq

    # prepare trajectory params
    traj_params = dict(traj_path="../generated_data/myosuite_humanoid_running.npz",
                       traj_dt=(1/traj_data_freq),
                       control_dt=(1/desired_contr_freq),
                       clip_trajectory_to_joint_ranges=False)

    # MDP
    gamma = 0.99
    horizon = 1000
    mdp = MyoSuiteHumanoid(gamma=gamma, horizon=horizon, n_substeps=n_substeps, traj_params=traj_params)

    mdp.play_trajectory(record=False)


if __name__ == '__main__':
    experiment()
