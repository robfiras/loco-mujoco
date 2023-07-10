import numpy as np

from loco_mujoco.environments import ReducedHumanoidTorque


def experiment(seed=0):

    np.random.seed(seed)

    # define env and data frequencies
    env_freq = 1000  # hz, added here as a reminder
    traj_data_freq = 500    # hz, added here as a reminder
    desired_contr_freq = 100     # hz
    n_substeps = env_freq//desired_contr_freq

    # prepare trajectory params
    traj_params = dict(traj_path="../datasets/humanoids/02-constspeed_reduced_humanoid.npz",
                       traj_dt=(1/traj_data_freq),
                       control_dt=(1/desired_contr_freq),
                       clip_trajectory_to_joint_ranges=True)

    # MDP
    gamma = 0.99
    horizon = 1000
    mdp = ReducedHumanoidTorque(gamma=gamma, horizon=horizon, n_substeps=n_substeps,
                                traj_params=traj_params, use_box_feet=False)

    mdp.play_trajectory_demo()


if __name__ == '__main__':
    experiment()
