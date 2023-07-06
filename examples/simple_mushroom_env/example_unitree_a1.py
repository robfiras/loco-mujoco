import numpy as np
from loco_mujoco.environments import UnitreeA1


env = UnitreeA1(random_start=False, use_foot_forces=False)

action_dim = env.info.action_space.shape[0]

env.reset()
env.render()
absorbing = False
i = 0

while True:
    if i == 1000 or absorbing:
        env.reset()
        i = 0
    action = np.random.randn(action_dim)*0.2
    nstate, _, absorbing, _ = env.step(action)

    env.render()
    i += 1
