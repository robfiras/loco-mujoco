import numpy as np
from loco_mujoco import LocoEnv


env = LocoEnv.make("UnitreeA1.simple")
dataset = env.create_dataset()
states = dataset["states"]
list = states[env._get_idx("q_trunk_list")]
tilt = states[env._get_idx("q_trunk_tilt")]


print("min list: ", np.min(list))
print("min tilt: ", np.min(tilt))
print("max list: ", np.max(list))
print("max tilt: ", np.max(tilt))

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
