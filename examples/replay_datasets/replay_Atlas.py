import numpy as np

from loco_mujoco import LocoEnv


def experiment(seed=0):

    np.random.seed(seed)

    mdp = LocoEnv.make("Atlas.walk", random_env_reset=False, disable_back_joint=True, default_camera_mode="top_static",
                       camera_params=dict(top_static=dict(distance=5.0, elevation=-90.0, azimuth=90.0, lookat=[5.0, 0.0, 0.0])))

    mdp.play_trajectory(n_steps_per_episode=250)


if __name__ == '__main__':
    experiment()
