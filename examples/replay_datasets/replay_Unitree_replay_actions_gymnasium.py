import gymnasium as gym
import numpy as np
from loco_mujoco import LocoEnv


def experiment():

    env = gym.make(
        "LocoMujoco",
        env_name="UnitreeA1.simple.perfect",
        render_mode="human",
        random_start=False,
        init_step_no=0,
    )

    expert_dataset = env.create_dataset()
    expert_actions = expert_dataset["actions"]

    env.reset()
    env.render()

    i = 0
    while i < 1000:
        action = expert_actions[i, :]
        nstate, reward, terminated, truncated, info = env.step(action)

        env.render()
        i += 1


if __name__ == "__main__":
    experiment()
