import numpy as np
from loco_mujoco import LocoEnv
import time

env = LocoEnv.make("HumanoidTorque", domain_randomization_config="./domain_randomization_humanoid.yaml")

action_dim = env.info.action_space.shape[0]

np.random.seed(0)
env.reset()
env.render()
absorbing = False
i = 0

start_time = time.time()
for j in range(100):
    if i == 1000 or absorbing:
        env.reset()
        i = 0
        print(j)
    action = np.random.randn(action_dim)
    nstate, _, absorbing, _ = env.step(action)

    env.render()
    i += 1
print("Time needed: ", time.time() - start_time)