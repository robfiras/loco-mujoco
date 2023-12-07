import numpy as np

from loco_mujoco import LocoEnv
from loco_mujoco.utils import video2gif


def experiment(seed=0):

    np.random.seed(seed)

    envs = LocoEnv.get_all_task_names()

    for env in envs:
        replay_params = dict(n_episodes=15, n_steps_per_episode=250, record=True) if "Unitree.hard" in env or \
                                                                                     ".all" in env or ".carry" in env \
            else dict(n_episodes=3, n_steps_per_episode=500, record=True)

        if "H1" in env:

            if "Humanoid" in env or "Atlas.walk" in env:
                env_params = dict(disable_arms=False)
            else:
                env_params = dict()

            if "4Ages" in env or "carry" in env:
                env_params["random_env_reset"] = False

            if "Humanoid" in env:
                env_params["use_box_feet"] = False

            if "carry" in env:
                disable_arms = True
            else:
                disable_arms = False

            mdp = LocoEnv.make(env, headless=True, disable_arms=disable_arms, **env_params)

            save_path_video = "./record_all"
            mdp.play_trajectory(recorder_params=dict(tag=env, path=save_path_video), **replay_params)

            path_video = save_path_video + "/" + env + "/recording.mp4"

            duration = 10.0 if ".all" in env or ".carry" in env or "Unitree.hard" in env else 4.0
            video2gif(path_video, duration=duration)


if __name__ == '__main__':
    experiment()
