"""
Factory for plain reinforcement learning tasks.
"""

from typing import Tuple

from .base import TaskFactory
from loco_mujoco.environments.base import LocoEnv


class RLFactory(TaskFactory):
    """
    A factory class to create reinforcement learning (RL) environments.

    Use :meth:`make` to create an RL environment based on the specified environment name. No trajectory is loaded;
    by default, the task is to follow a randomly sampled root velocity.

    """

    @staticmethod
    def make(env_name: str,
             init_state_type: str = "DefaultInitialStateHandler",
             terminal_state_type: str = "HeightBasedTerminalStateHandler",
             goal_type: str = "GoalRandomRootVelocity",
             reward_type: str = "TargetVelocityGoalReward",
             **kwargs) -> Tuple[LocoEnv, None]:
        """
        Create and return an RL environment based on the specified environment name.

        Args:
            env_name (str): The name of the registered environment to create.
            init_state_type (str, optional): The initial state handler to use.
                Defaults to "DefaultInitialStateHandler".
            terminal_state_type (str, optional): The terminal state handler to use.
                Defaults to "HeightBasedTerminalStateHandler".
            goal_type (str, optional): The goal to use. Defaults to "GoalRandomRootVelocity".
            reward_type (str, optional): The reward function to use. Defaults to "TargetVelocityGoalReward".
            **kwargs: Additional keyword arguments to pass to the environment constructor.

        Returns:
            Tuple[LocoEnv, None]: The requested RL environment and None, as RL tasks have no trajectory.

        Raises:
            KeyError: If the specified environment is not registered.

        """

        if env_name not in LocoEnv.registered_envs:
            raise KeyError(f"Environment '{env_name}' is not a registered LocoMuJoCo environment.")

        # Get environment class
        env_cls = LocoEnv.registered_envs[env_name]

        # Create and return the environment
        env = env_cls(init_state_type=init_state_type,
                      terminal_state_type=terminal_state_type,
                      goal_type=goal_type,
                      reward_type=reward_type,
                      **kwargs)
        return env, None
