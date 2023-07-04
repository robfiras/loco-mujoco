import numpy as np
from loco_mujoco.environments import ReducedHumanoidTorque4Ages


if __name__ == '__main__':

    env = ReducedHumanoidTorque4Ages(use_box_feet=True, random_start=False,
                                     disable_arms=True, reward_type="multi_target_velocity",
                                     reward_params=dict(target_velocity=1.25))

    action_dim = env.info.action_space.shape[0]

    env_mask = env.get_mask(obs_to_hide=())

    env.reset()
    env.render()
    absorbing = False
    i = 0
    while True:
        if i == 200 or absorbing:
            env.reset()
            i = 0
        action = np.random.randn(action_dim)
        nstate, _, absorbing, _ = env.step(action)

        env.render()
        i += 1