import numpy as np
import loco_mujoco
import gymnasium as gym


env = gym.make("LocoMujoco", env_name="HumanoidTorque.run", render_mode="human")
action_dim = env.action_space.shape[0]

env.reset()
img = env.render()
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
