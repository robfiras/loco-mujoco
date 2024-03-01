import numpy as np
from loco_mujoco import LocoEnv
import gymnasium as gym


# define what ever reward function you want
def my_reward_function(state, action, next_state):
    return -np.mean(action)     # here we just return the negative mean of the action


# create the environment and task together with the reward function
env = gym.make("LocoMujoco", env_name="UnitreeH1.run.real", reward_type="custom",
               reward_params=dict(reward_callback=my_reward_function))

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

    # HERE is your favorite RL algorithm

    env.render()
    i += 1
