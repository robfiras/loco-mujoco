import numpy as np
from loco_mujoco import LocoEnv

np.random.seed(0)
mdp = LocoEnv.make("Talos.walk.perfect")

mdp.play_trajectory_from_velocity(n_episodes=30, n_steps_per_episode=500)
