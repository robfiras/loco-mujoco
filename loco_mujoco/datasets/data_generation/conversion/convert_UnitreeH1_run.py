import os
import numpy as np
from loco_mujoco.utils.dataset import adapt_mocap


if __name__ == "__main__":

    joint_conf = dict(pelvis_tx=(0.95, 0.0),
                      pelvis_tz=(-1.0, 0.0),
                      pelvis_ty=(0.8, -0.77),
                      pelvis_tilt=(0.5, -0.13),
                      pelvis_list=(0.5, 0.0),
                      pelvis_rotation=(1., 0.0),
                      lumbar_extension=(1., 0.25),
                      lumbar_bending=(1., 0.0),
                      lumbar_rotation=(1., 0.0),
                      arm_rot_r=(1., 0.2),
                      arm_add_r=(1., 0.25),
                      arm_flex_r=(-1.0, 0.0),
                      elbow_flex_r=(-1., np.pi/2 + 0.25),
                      pro_sup_r=(1., 0.0),
                      arm_rot_l=(-1., -0.2),
                      arm_add_l=(-1., -0.25),
                      arm_flex_l=(-1.0, 0.0),
                      elbow_flex_l=(-1., np.pi/2 + 0.25),
                      pro_sup_l=(1., 0.0),
                      hip_adduction_l=(-0.7, 0.02),
                      hip_flexion_l=(-1.0, -0.1),
                      hip_rotation_l=(-0.7, 0.0),
                      knee_angle_l=(-1.0, 0.0),
                      ankle_angle_l=(-1.0, -0.06),
                      hip_adduction_r=(0.7, -0.02),
                      hip_flexion_r=(-1.0, -0.1),
                      hip_rotation_r=(0.7, 0.),
                      knee_angle_r=(-1.0, 0.0),
                      ankle_angle_r=(-1.0, -0.06))

    path_mat = "../00_raw_mocap_data/raw_running_mocap_data.mat"
    dir_target_path = "../generated_data"
    if not os.path.exists(dir_target_path):
        os.makedirs(dir_target_path)
    target_path = os.path.join(dir_target_path, "05-run_UnitreeH1.npz")

    # do some renaming of the joint names
    rename_map = dict(lumbar_extension="back_bky",
                      lumbar_bending="back_bkx",
                      lumbar_rotation="back_bkz",
                      arm_flex_r="r_arm_shy",
                      arm_rot_r="r_arm_shz",
                      arm_add_r="r_arm_shx",
                      elbow_flex_r="right_elbow",
                      pro_sup_r="r_arm_wry",
                      arm_flex_l="l_arm_shy",
                      arm_rot_l="l_arm_shz",
                      arm_add_l="l_arm_shx",
                      elbow_flex_l="left_elbow",
                      pro_sup_l="l_arm_wry",
                      )

    unavailable_keys = []
    dataset = adapt_mocap(path_mat, joint_conf=joint_conf, unavailable_keys=unavailable_keys, rename_map=rename_map,
                          discard_first=28500, discard_last=12500)

    np.savez(file=target_path, **dataset)
