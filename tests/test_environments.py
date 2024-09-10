from pathlib import Path
import numpy as np

import loco_mujoco
from loco_mujoco import LocoEnv
import gymnasium as gym


N_EPISODES = 500
N_STEPS = 1000
N_EPISODES_REP = 5
N_STEPS_REP = 500


def run_environment(env, n_episodes, n_steps):
    obs_dim = env.info.observation_space.shape[0]
    action_dim = env.info.action_space.shape[0]
    dataset = []

    obs = env.reset()
    dataset.append(obs)
    absorbing = False

    for i in range(n_episodes):
        for j in range(n_steps):
            if absorbing:
                env.reset()
                break

            assert obs_dim == obs.shape[0]

            action = np.random.randn(action_dim) * 0.1
            obs, _, absorbing, _ = env.step(action)
            dataset.append(obs)

            j += 1

    return np.array(dataset)


def run_environment_gymnasium(env, n_episodes, n_steps):
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    dataset = []

    obs, _ = env.reset()
    dataset.append(obs)
    absorbing = False

    for i in range(n_episodes):
        for j in range(n_steps):
            if absorbing:
                env.reset()
                break

            assert obs_dim == obs.shape[0]

            action = np.random.randn(action_dim) * 0.1
            obs, _, absorbing, _, _ = env.step(action)
            dataset.append(obs)

            j += 1

    return np.array(dataset)


def test_all_environments():

    path = Path("./test_datasets")
    task_names = loco_mujoco.get_all_task_names()

    for task_name in task_names:

        # todo: include perfect datasets and MyoSkeleton into tests (the latter requires updating Mujoco >= 3.0)
        if "perfect" not in task_name and "MyoSkeleton" not in task_name:
            np.random.seed(0)

            print(f"Testing {task_name}...")
            # --- native environment ---
            task_env = LocoEnv.make(task_name, debug=True)
            dataset = run_environment(task_env, N_EPISODES, N_STEPS)

            np.random.seed(0)
            # --- run gymnasium environment ---
            task_env = gym.make("LocoMujoco", env_name=task_name, debug=True)
            dataset_gym = run_environment_gymnasium(task_env, N_EPISODES, N_STEPS)

            file_name = task_name + ".npy"
            dataset_path = Path(loco_mujoco.__file__).resolve().parent.parent / "tests" / path / file_name

            test_dataset = np.load(dataset_path)

            assert np.allclose(dataset, test_dataset)
            assert np.allclose(dataset_gym, test_dataset)


def test_replays():

    task_names = loco_mujoco.get_all_task_names()

    for task_name in task_names:

        # todo: include perfect datasets and MyoSkeleton into tests (the latter requires updating Mujoco >= 3.0)
        if "perfect" not in task_name and "MyoSkeleton" not in task_name:
            np.random.seed(0)

            print(f"Testing Replay {task_name}...")

            # --- native environment ---
            task_env = LocoEnv.make(task_name, debug=True)

            task_env.play_trajectory(n_episodes=N_EPISODES_REP, n_steps_per_episode=N_STEPS_REP, render=False)

            task_env = LocoEnv.make(task_name, debug=True)
            task_env.play_trajectory(n_episodes=N_EPISODES_REP, n_steps_per_episode=N_STEPS_REP, render=False)

            # --- gymnasium environment ---
            task_env = gym.make("LocoMujoco", env_name=task_name, debug=True)

            task_env.play_trajectory(n_episodes=N_EPISODES_REP, n_steps_per_episode=N_STEPS_REP,  render=False)

            task_env = gym.make("LocoMujoco", env_name=task_name, debug=True)
            task_env.play_trajectory(n_episodes=N_EPISODES_REP, n_steps_per_episode=N_STEPS_REP, render=False)

