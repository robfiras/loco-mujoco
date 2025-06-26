import numpy as np
from loco_mujoco import ImitationFactory




# create the environment and task
env = ImitationFactory.make("MyoSkeleton", default_dataset_conf=dict(task="walk"))

# get the dataset for the chosen environment and task -- can be used for GAIL-like algorithms
#expert_data = env.create_dataset()

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
