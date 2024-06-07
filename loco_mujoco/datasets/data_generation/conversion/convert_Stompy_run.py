"""Script to convert raw mocap data to a format that can be used by the Stompy environment."""
import os
import numpy as np
from loco_mujoco.utils.dataset import adapt_mocap


if __name__ == "__main__":
    joint_conf = dict(
        pelvis_tx=(0.67, 0.0),
        pelvis_tz=(0.67, 0.0),
        pelvis_ty=(0.67, -0.73),
        pelvis_tilt=(0.7, -0.05),
        pelvis_list=(0.7, 0.0),
        pelvis_rotation=(0.7, 0.0),
        knee_angle_l=(1.0, 0.0),
        knee_angle_r=(-1.0, 0.0),
        lumbar_rotation=(0.7, 0.0),

        hip_flexion_r=(1.0, 0.0),
        hip_flexion_l=(-1.0, 0.0),
        hip_rotation_r=(-1.0, 0.0),
        hip_rotation_l=(1.0, 0.0),
        hip_adduction_r=(-1.0, 0.0),
        hip_adduction_l=(1.0, 0.0),
        ankle_angle_r=(-1.0, -0.1),
        ankle_angle_l=(1.0, 0.1),

        subtalar_angle_r=(-1.0, 0.0),
        subtalar_angle_l=(1.0, 0.0),
        arm_flex_r=(1.0, 0.0),
        arm_add_r=(-1.0, 0.0),
        arm_rot_r=(-1.0, 0.0),

        pro_sup_r=(1.0, 0.0),
        wrist_flex_r=(-1.0, 0.0),
        wrist_dev_r=(-1.0, 0.0),
        arm_flex_l=(-1.0, 0.0),
        arm_add_l=(1.0, 0.0),
        arm_rot_l=(-1.0, 0.0),
        elbow_flex_r=(1.0, 0.0),
        elbow_flex_l=(-1.0, 0.0),
    )

    path_mat = "../00_raw_mocap_data/raw_running_mocap_data.mat"
    dir_target_path = "../generated_data"
    if not os.path.exists(dir_target_path):
        os.makedirs(dir_target_path)
    target_path = os.path.join(dir_target_path, "run_stompy.npz")

    rename_map = dict(
        knee_angle_l="joint_legs_1_left_leg_1_knee_revolute",
        knee_angle_r="joint_legs_1_right_leg_1_knee_revolute",
        elbow_flex_l="joint_left_arm_2_x6_2_dof_x6",
        elbow_flex_r="joint_right_arm_1_x6_2_dof_x6",
        hip_adduction_l="joint_legs_1_x8_2_dof_x8", # roll
        hip_adduction_r="joint_legs_1_x8_1_dof_x8", # roll
        hip_flexion_l="joint_legs_1_left_leg_1_x10_1_dof_x10", # pitch
        hip_flexion_r="joint_legs_1_right_leg_1_x10_2_dof_x10", # pitch
        #hip_rotation_l="joint_legs_1_x8_2_dof_x8", # yaw
        hip_rotation_l="joint_legs_1_left_leg_1_x8_1_dof_x8", # yaw
        #hip_rotation_r="joint_legs_1_x8_1_dof_x8", # yaw
        hip_rotation_r="joint_legs_1_right_leg_1_x8_1_dof_x8", # yaw
        lumbar_rotation="joint_torso_1_x8_1_dof_x8",
        ankle_angle_l="joint_legs_1_left_leg_1_ankle_revolute",
        ankle_angle_r="joint_legs_1_right_leg_1_ankle_revolute",
        arm_flex_r="joint_right_arm_1_x8_1_dof_x8",
        arm_flex_l="joint_left_arm_2_x8_1_dof_x8",
    )

    unavailable_keys = [
        "joint_head_1_x4_1_dof_x4",

        "joint_right_arm_1_x8_2_dof_x8",
        "joint_right_arm_1_x6_1_dof_x6",
        "joint_right_arm_1_x4_1_dof_x4",
        "joint_right_arm_1_hand_1_x4_1_dof_x4",
        "joint_right_arm_1_hand_1_slider_1",
        "joint_right_arm_1_hand_1_slider_2",
        "joint_right_arm_1_hand_1_x4_2_dof_x4",

        "joint_left_arm_2_x8_2_dof_x8",
        "joint_left_arm_2_x6_1_dof_x6",
        "joint_left_arm_2_x4_1_dof_x4",
        "joint_left_arm_2_hand_1_x4_1_dof_x4",
        "joint_left_arm_2_hand_1_slider_1",
        "joint_left_arm_2_hand_1_slider_2",
        "joint_left_arm_2_hand_1_x4_2_dof_x4",

        "joint_legs_1_left_leg_1_x10_2_dof_x10",
        "joint_legs_1_right_leg_1_x10_1_dof_x10",
        "joint_legs_1_right_leg_1_x6_1_dof_x6",
        "joint_legs_1_left_leg_1_x6_1_dof_x6",
        "joint_legs_1_right_leg_1_x4_1_dof_x4",
        "joint_legs_1_left_leg_1_x4_1_dof_x4",

    ]

    dataset = adapt_mocap(path_mat, joint_conf=joint_conf, unavailable_keys=unavailable_keys, rename_map=rename_map,
                          discard_first=28500, discard_last=12500)

    np.savez(file=target_path, **dataset)