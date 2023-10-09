import numpy as np
from loco_mujoco import LocoEnv


env = LocoEnv.make("HumanoidTorque4Ages", domain_randomization_config="./domain_randomization_humanoid.yaml")

action_dim = env.info.action_space.shape[0]

env.reset()
env.render()
absorbing = False
i = 0

while True:
    if i == 1000 or absorbing:
        env.reset()
        i = 0
    action = np.random.randn(action_dim)
    nstate, _, absorbing, _ = env.step(action)

    env.render()
    i += 1
