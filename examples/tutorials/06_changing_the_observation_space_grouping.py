import numpy as np
import jax

from loco_mujoco.core import ObservationType
from loco_mujoco import ImitationFactory


observation_spec = [
    ObservationType.FreeJointPosNoXY(obs_name="free_joint", xml_name="root", group="prioritized"),
    ObservationType.FreeJointVel(obs_name="free_joint_vel", xml_name="root", group="prioritized"),
    ObservationType.JointPos(obs_name="joint_pos", xml_name="left_hip_pitch", group=["prioritized", "policy"]),
    ObservationType.JointVel(obs_name="joint_vel1", xml_name="right_hip_pitch", group=["prioritized", "policy"]),
    ObservationType.JointVel(obs_name="joint_vel2", xml_name="left_knee", group=["prioritized", "policy"]),
    ObservationType.BodyPos(obs_name="head_pos", xml_name="head", group=["prioritized", "policy"]),
    ObservationType.LastAction(obs_name="last_action", group=["prioritized", "policy"]),
    # define many more in the order you want ...
]

env, traj = ImitationFactory.make("ToddlerBot", observation_spec=observation_spec,
                            default_dataset_conf=dict(task="walk"))

# checkout the detailed observation space (opens webbrowser)
env.create_observation_summary()

# get observation masks
policy_mask = env.obs_container.get_obs_ind_by_group("policy")
prioritized_mask = env.obs_container.get_obs_ind_by_group("prioritized")

# get observations
obs = env.reset()
prioritized_obs = obs[prioritized_mask]
policy_obs = obs[policy_mask]



