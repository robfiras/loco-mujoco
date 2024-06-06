import numpy as np
from loco_mujoco.environments import Stompy
from loco_mujoco.utils import video2gif


def experiment():
    np.random.seed(1)

    # define env and data frequencies
    env_freq = 1000  # hz, added here as a reminder
    traj_data_freq = 500    # hz, frequency of the mocap dataset
    desired_contr_freq = 100     # hz, this will also be the dataset frequency after downsampling
    n_substeps = env_freq//desired_contr_freq

    # prepare trajectory params
    traj_params = dict(traj_path="generated_data/walk_stompy.npz",
                       traj_dt=(1/traj_data_freq),
                       control_dt=(1/desired_contr_freq),
                       clip_trajectory_to_joint_ranges=False)

    # MDP
    gamma = 0.99
    horizon = 1000
    mdp = Stompy(gamma=gamma, horizon=horizon, n_substeps=n_substeps, traj_params=traj_params,
                disable_arms=False, disable_back_joint=False)

    mdp.play_trajectory(
        record=True, n_episodes=1, n_steps_per_episode=500, 
        recorder_params=dict(
            path="test_video", 
            tag="stompy",
            video_name="stompy_walk")
        )

    video2gif("test_video/stompy/stompy_walk.mp4", duration=4.0, fps=15, scale=720)


if __name__ == '__main__':
    experiment()
