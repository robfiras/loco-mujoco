import os
import jax
import numpy as np
import time

from loco_mujoco import ImitationFactory


# can increase the speed by ~30% on some GPUs
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True ')


randomization_params = {
    "prosthesis_side": "left_side",

    "randomize_prosthesis_dof_damping": True,
    "randomization_dof_names": ['ankle_angle', 'subtalar_angle', 'mtp_angle'],
    "prosthesis_dof_damping_range": [0, 10],

    "randomize_prosthesis_joint_stiffness": False, #True,
    "randomization_joint_names": ['ankle_angle', 'subtalar_angle', 'mtp_angle'],
    "prosthesis_joint_stiffness_range":[], # [0, 100],

    "randomize_prosthesis_body_position": True,
    "randomization_body_position_names": ['calcn'],
    "prosthesis_body_position_range": {
        # 'x': [0.4, 0.5],
        # 'y': [0.0, 0.1],
        # 'z': [0.5, 0.9]
    },
    "randomize_prosthesis_body_orientation": True, #True, #False, #True,
    "randomization_body_orientation_names": ['calcn'], #['toe'], #['calcn'], #['foot_box'], #['calcn'],
    "prosthesis_body_orientation_range": {
        # 'x': [0.1, 0.2],
        'y': [0.8, 0.9],
        # 'z': [0.1, 0.2]
        #'x': [-0.01, 0.01],
        # 'y': [0.8, 0.9]#, #[0.15, 0.3], #(9-17°) #[0.8, 0.9],
        #'z': [-0.01, 0.01]
    }
}


# create env
env = ImitationFactory.make(
    "MjxSkeletonMuscleProsthesis",
    prosthesis_side="left_side",
    prosthesis_type="transtibial",
    delete_joints = False,
    joint_stiffness = 50,
    joint_damping = 1,
    default_dataset_conf=dict(task="walk"),
    domain_randomization_type="ProsthesisRandomizer",
    domain_randomization_params=randomization_params
)

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

