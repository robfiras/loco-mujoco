import numpy as np
from loco_mujoco import ImitationFactory

# switch from torque control to position control (envs can use torque *or* position control as default!)
# note: the gains can also be an array of length action_dim!
env, traj = ImitationFactory.make("FourierGR1T2", default_dataset_conf=dict(task="stepinplace1"),
                            control_type="PDControl", control_params=dict(p_gain=100, d_gain=1))

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
