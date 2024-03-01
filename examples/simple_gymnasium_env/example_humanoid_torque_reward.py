import numpy as np
import loco_mujoco
import gymnasium as gym
import numpy as np


def my_reward_function(state, action, next_state):
    return -np.mean(action)


env = gym.make("LocoMujoco", env_name="HumanoidTorque.run", reward_type="custom",
               reward_params=dict(reward_callback=my_reward_function), render_mode="human")

action_dim = env.action_space.shape[0]

env.reset()
env.render()
absorbing = False
i = 0

while True:
    if i == 1000 or absorbing:
        env.reset()
        i = 0
    action = np.random.randn(action_dim)
    nstate, _, absorbing, _,  _ = env.step(action)

    env.render()
    i += 1
