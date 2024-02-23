import numpy as np
from loco_mujoco import LocoEnv

np.random.seed(0)
mdp = LocoEnv.make("HumanoidMuscle.run.real")

mdp.play_trajectory(n_steps_per_episode=500)
