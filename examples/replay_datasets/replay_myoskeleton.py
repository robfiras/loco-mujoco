import numpy as np

from loco_mujoco import LocoEnv


def experiment(seed=0):

    np.random.seed(seed)

    mdp = LocoEnv.make("MyoSkeleton.walk")

    mdp.play_trajectory(n_steps_per_episode=1000, record=True)


if __name__ == '__main__':
    experiment()
