import numpy as np
from loco_mujoco import LocoEnv

# create the environment and task
env = LocoEnv.make("UnitreeH1.run")

# get the dataset for the chosen environment and task
expert_data = env.create_dataset()

action_dim = env.info.action_space.shape[0]

env.reset()
env.render()
absorbing = False
i = 0

while True:
    if i == 1000 or absorbing:
        env.reset()
        i = 0
    action = np.random.randn(action_dim) * 3
    nstate, reward, absorbing, info = env.step(action)

    env.render()
    i += 1
