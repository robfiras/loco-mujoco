import os
import numpy as np
from loco_mujoco.utils.dataset import adapt_mocap

if __name__ == "__main__":

    # the first entry of each tuple is a multiplier and the second is an offset.
    joint_conf = dict(
                      pelvis_tx=(1.0, 0.0),
                      pelvis_tz=(1.0, 0.0),
                      pelvis_ty=(1.125, -1.11),
                      pelvis_tilt=(1.0, 0.0),
                      pelvis_list=(1.0, 0.0),
                      pelvis_rotation=(1.0, 0.0),

                      hip_flexion_r=(1.0, 0.0),
                      hip_adduction_r=(-1.0, 0.0),
                      hip_rotation_r=(1.0, 0.0),
                      knee_angle_r=(1.0, 0.0),
                      ankle_angle_r=(1.0, 0.09),
                      subtalar_angle_r=(-1.0, 0.0),

                      hip_flexion_l=(1.0, 0.0),
                      hip_adduction_l=(1.0, 0.0),
                      hip_rotation_l=(-1.0, 0.0),
                      knee_angle_l=(-1.0, 0.0),
                      ankle_angle_l=(1.0, 0.06),
                      subtalar_angle_l=(1.0, 0.0),

                      arm_flex_r=(1.0, 0.0),
                      arm_add_r=(-1.0, 0.0),
                      arm_rot_r=(-1.0, 0.0),
                      elbow_flex_r=(1.0, 0.0),
                      pro_sup_r=(1.0, 0.0),
                      wrist_flex_r=(-1.0, 0.0),
                      wrist_dev_r=(-1.0, 0.0),

                      arm_flex_l=(-1.0, 0.0),
                      arm_add_l=(1.0, 0.0),
                      arm_rot_l=(-1.0, 0.0),
                      elbow_flex_l=(-1.0, 0.0),
                      pro_sup_l=(1.0, 0.0),
                      wrist_flex_l=(1.0, 0.0),
                      wrist_dev_l=(1.0, 0.0)
    )

    path_mat = "../00_raw_mocap_data/raw_running_mocap_data.mat"
    dir_target_path = "../generated_data"
    if not os.path.exists(dir_target_path):
        os.makedirs(dir_target_path)
    target_path = os.path.join(dir_target_path, "05-run_Kuavo.npz")

    # do seme renaming of the joint names
    rename_map = dict(
                        # right leg
                        hip_flexion_r="r_leg_pitch",
                        hip_adduction_r="r_leg_roll",
                        hip_rotation_r="r_leg_yaw",
                        knee_angle_r="r_knee",
                        ankle_angle_r="r_foot_pitch",
                        subtalar_angle_r="r_foot_roll",
                        # left leg
                        hip_flexion_l="l_leg_pitch",
                        hip_adduction_l="l_leg_roll",
                        hip_rotation_l="l_leg_yaw",
                        knee_angle_l="l_knee",
                        ankle_angle_l="l_foot_pitch",
                        subtalar_angle_l="l_foot_roll",
                        # right arm
                        arm_flex_r="r_arm_pitch",
                        arm_add_r="r_arm_roll",
                        arm_rot_r="r_arm_yaw",
                        elbow_flex_r="r_forearm_pitch",
                        pro_sup_r="r_forearm_yaw",
                        wrist_flex_r="r_hand_pitch",
                        wrist_dev_r="r_hand_roll",
                        # left arm
                        arm_flex_l="l_arm_pitch",
                        arm_add_l="l_arm_roll",
                        arm_rot_l="l_arm_yaw",
                        elbow_flex_l="l_forearm_pitch",
                        pro_sup_l="l_forearm_yaw",
                        wrist_flex_l="l_hand_pitch",
                        wrist_dev_l="l_hand_roll"
                      )

    unavailable_keys = []
    dataset = adapt_mocap(path_mat, joint_conf=joint_conf, unavailable_keys=unavailable_keys, rename_map=rename_map,
                          discard_first=25000, discard_last=1000)

    np.savez(file=target_path, **dataset)
