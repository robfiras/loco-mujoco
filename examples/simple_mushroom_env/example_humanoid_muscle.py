import numpy as np
from loco_mujoco import LocoEnv

def reward_callback(state, action, next_state):
    # action penalty
    act_pen = -np.linalg.norm(action)
    # reward for high acceleration
    vel_state = state[17:]
    vel_next_state = next_state[17:]
    acc = vel_next_state - vel_state
    acc_rew = np.linalg.norm(acc)
    return 1.0*acc_rew + 0.1*act_pen

reward_type = "custom"
reward_params = dict(reward_callback=reward_callback)

env = LocoEnv.make("HumanoidMuscle", reward_type=reward_type, reward_params=reward_params)

action_dim = env.info.action_space.shape[0]

obs = env.reset()
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
