import numpy as np

from loco_mujoco import LocoEnv


def experiment(seed=0):

    np.random.seed(seed)

    mdp = LocoEnv.make("ReducedHumanoidTorque")

    mdp.play_trajectory_demo(250)


if __name__ == '__main__':
    experiment()
