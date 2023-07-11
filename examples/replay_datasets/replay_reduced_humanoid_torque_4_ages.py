import numpy as np

from loco_mujoco import LocoEnv


def experiment(seed=0):

    np.random.seed(seed)

    mdp = LocoEnv.make("ReducedHumanoidTorque4Ages.walk.1", disable_arms=False)

    mdp.play_trajectory()


if __name__ == '__main__':
    experiment()
