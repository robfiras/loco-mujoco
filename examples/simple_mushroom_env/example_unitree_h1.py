import numpy as np
from loco_mujoco import LocoEnv


env = LocoEnv.make("UnitreeH1.run")
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
    nstate, _, absorbing, _ = env.step(action)

    env.render()
    i += 1
