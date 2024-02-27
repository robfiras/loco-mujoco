import numpy as np
from loco_mujoco import LocoEnv
import gymnasium as gym

# create the environment and task
env = gym.make("LocoMujoco", env_name="UnitreeH1.run.real", render_mode="human")

# get the dataset for the chosen environment and task
expert_data = env.create_dataset()

action_dim = env.action_space.shape[0]

env.reset()
env.render()
terminated = False
i = 0

while True:
    if i == 1000 or terminated:
        env.reset()
        i = 0
    action = np.random.randn(action_dim)
    nstate, reward, terminated, truncated, info = env.step(action)

    env.render()
    i += 1
