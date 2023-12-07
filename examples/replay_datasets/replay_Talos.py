import numpy as np

from loco_mujoco import LocoEnv


def experiment(seed=0):

    np.random.seed(seed)

    mdp = LocoEnv.make("Talos.walk.perfect", disable_arms=True)
    dataset = mdp.create_dataset()

    mdp.play_trajectory(n_episodes=30, n_steps_per_episode=500)


if __name__ == '__main__':
    experiment()
