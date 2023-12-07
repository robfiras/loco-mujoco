import numpy as np

from loco_mujoco import LocoEnv


def experiment(seed=0):

    np.random.seed(seed)

    mdp = LocoEnv.make("HumanoidMuscle.run.real")
    dataset = mdp.create_dataset()

    mdp.play_trajectory(n_steps_per_episode=500)


if __name__ == '__main__':
    experiment()
