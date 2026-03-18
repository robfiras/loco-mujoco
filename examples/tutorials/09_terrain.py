import numpy as np
from loco_mujoco import ImitationFactory


env, traj = ImitationFactory.make("FourierGR1T2", default_dataset_conf=dict(task="stepinplace1"),
                            terrain_type="RoughTerrain", terrain_params=dict(random_min_height=-0.05,
                                                                             random_max_height=0.05,))

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
    nstate, reward, absorbing, done, info = env.step(action)

    env.render()
    i += 1
