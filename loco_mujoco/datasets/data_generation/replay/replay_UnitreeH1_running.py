import numpy as np

from loco_mujoco.environments import UnitreeH1


def experiment():
    np.random.seed(1)

    # define env and data frequencies
    env_freq = 1000  # hz, added here as a reminder
    traj_data_freq = 500    # hz, added here as a reminder
    desired_contr_freq = 100     # hz, this will also be the dataset frequency after downsampling
    n_substeps = env_freq//desired_contr_freq

    # prepare trajectory params
    traj_params = dict(traj_path="../generated_data/05-run_UnitreeH1.npz",
                       traj_dt=(1 / traj_data_freq),
                       control_dt=(1 / desired_contr_freq))

    # MDP
    gamma = 0.99
    horizon = 1000
    mdp = UnitreeH1(gamma=gamma, horizon=horizon, n_substeps=n_substeps, traj_params=traj_params,
                    disable_arms=False, disable_back_joint=False)

    mdp.play_trajectory()


if __name__ == '__main__':
    experiment()
