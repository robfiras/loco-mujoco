import numpy as np

from loco_mujoco import LocoEnv


def experiment(seed=0):

    np.random.seed(seed)

    envs = ["Atlas.walk", "Atlas.carry",
            "HumanoidTorque.walk", "HumanoidTorque.run",
            "HumanoidTorque4Ages.walk.1", "HumanoidTorque4Ages.walk.2",
            "HumanoidTorque4Ages.walk.2", "HumanoidTorque4Ages.walk.4", "HumanoidTorque4Ages.walk.all",
            "HumanoidTorque4Ages.run.1", "HumanoidTorque4Ages.run.2",
            "HumanoidTorque4Ages.run.2", "HumanoidTorque4Ages.run.4", "HumanoidTorque4Ages.run.all",
            "UnitreeA1.simple", "UnitreeA1.hard"]

    replay_params = dict(n_episodes=5, n_steps_per_episode=500, record=True)

    for env in envs:

        mdp = LocoEnv.make(env, disable_arms=False) if "Humanoid" in env or "Atlas.walk" in env \
            else LocoEnv.make(env)

        mdp.play_trajectory(recorder_params=dict(tag=env, path="./record_all"), **replay_params)


if __name__ == '__main__':
    experiment()
