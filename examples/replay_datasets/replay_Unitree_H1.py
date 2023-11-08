import numpy as np

from loco_mujoco import LocoEnv


def experiment(seed=0):

    np.random.seed(seed)

    mdp = LocoEnv.make("UnitreeH1.run")

    mdp.play_trajectory(n_episodes=3, n_steps_per_episode=500)


if __name__ == '__main__':
    experiment()
