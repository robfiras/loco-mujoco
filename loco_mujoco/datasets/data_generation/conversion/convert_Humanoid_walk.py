import os
import numpy as np
from loco_mujoco.utils.dataset import adapt_mocap

if __name__ == "__main__":

    # the first entry of each tuple is a multiplier and the second is an offset.
    joint_conf = dict(pelvis_tx=(1.0, 0.0),
                      pelvis_tz=(1.0, 0.0),
                      pelvis_ty=(1.125, -1.11),
                      pelvis_tilt=(1.0, 0.0),
                      pelvis_list=(1.0, 0.0),
                      pelvis_rotation=(1.0, 0.0),
                      hip_flexion_r=(1.0, 0.0),
                      hip_adduction_r=(1.0, 0.0),
                      hip_rotation_r=(1.0, 0.0),
                      knee_angle_r=(1.0, 0.0),
                      ankle_angle_r=(1.0, 0.09),
                      subtalar_angle_r=(1.0, 0.0),
                      mtp_angle_r=(1.0, 0.0),
                      hip_flexion_l=(1.0, 0.0),
                      hip_adduction_l=(1.0, 0.0),
                      hip_rotation_l=(1.0, 0.0),
                      knee_angle_l=(1.0, 0.0),
                      ankle_angle_l=(1.0, 0.06),
                      subtalar_angle_l=(1.0, 0.0),
                      mtp_angle_l=(1.0, 0.0),
                      lumbar_extension=(1.0, 0.0),
                      lumbar_bending=(1.0, 0.0),
                      lumbar_rotation=(1.0, 0.0),
                      arm_flex_r=(1.0, 0.0),
                      arm_add_r=(1.0, 0.0),
                      arm_rot_r=(1.0, 0.0),
                      elbow_flex_r=(1.0, 0.0),
                      pro_sup_r=(1.0, 0.0),
                      wrist_flex_r=(1.0, 0.0),
                      wrist_dev_r=(1.0, 0.0),
                      arm_flex_l=(1.0, 0.0),
                      arm_add_l=(1.0, 0.0),
                      arm_rot_l=(1.0, 0.0),
                      elbow_flex_l=(1.0, 0.0),
                      pro_sup_l=(1.0, 0.0),
                      wrist_flex_l=(1.0, 0.0),
                      wrist_dev_l=(1.0, 0.0))

    unavailable_keys = []

    path_mat = "../00_raw_mocap_data/raw_walking_motion_capture.mat"
    dir_target_path = "../generated_data"
    if not os.path.exists(dir_target_path):
        os.makedirs(dir_target_path)
    target_path = os.path.join(dir_target_path, "02-constspeed_humanoid.npz")

    dataset = adapt_mocap(path_mat, joint_conf=joint_conf, unavailable_keys=unavailable_keys,
                          discard_first=5000, discard_last=1000)

    np.savez(file=target_path, **dataset)
