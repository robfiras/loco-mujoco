import numpy as np

from loco_mujoco import LocoEnv


def experiment(seed=0):

    np.random.seed(seed)

    mdp = LocoEnv.make("Kuavo.run.real")

    mdp.play_trajectory_from_velocity(n_episodes=10, n_steps_per_episode=500)


if __name__ == '__main__':
    experiment()
