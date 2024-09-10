import os
import numpy as np
from loco_mujoco.environments.humanoids import MyoSkeleton
from loco_mujoco.utils.dataset import adapt_mocap

from scipy.spatial.transform import Rotation as R


def reorder_shoulder_orientation(keys, old, new):
    # adapt arm data rotation  --> data shoulder rotation order is 'zxy' new one is 'yxy'
    arr = np.vstack([dataset[k] for k in keys]).T
    arr = R.from_euler(old, arr).as_euler(new).T
    for i, k in enumerate(keys):
        dataset[k] = arr[i]


# get the xml_handle from the environment
xml_handle = MyoSkeleton().xml_handle

all_joints = []
for prefix in ["q_", "dq_"]:
    for j in xml_handle.find_all("joint"):
        all_joints.append(j.name)

# define the joints for which we have data
joint_conf = dict(pelvis_tx=(1.0, 0.0),
                  pelvis_tz=(1.0, 0.0),
                  pelvis_ty=(1.0, -1.01),
                  pelvis_tilt=(1.0, -0.22),
                  pelvis_list=(1.0, 0.0),
                  pelvis_rotation=(1.0, 0.0),
                  hip_flexion_r=(1.0, 0.2),
                  hip_adduction_r=(1.0, 0.0),
                  hip_rotation_r=(1.0, 0.0),
                  knee_angle_r=(-1.0, 0.0),
                  ankle_angle_r=(1.0, 0.15),
                  hip_flexion_l=(1.0, 0.2),
                  hip_adduction_l=(1.0, 0.0),
                  hip_rotation_l=(1.0, 0.0),
                  knee_angle_l=(-1.0, 0.0),
                  ankle_angle_l=(1.0, 0.1),
                  lumbar_extension=(1.0, 0.25),
                  lumbar_bending=(1.0, 0.0),
                  lumbar_rotation=(1.0, 0.0),
                  arm_flex_r=(1.0, 0.0),
                  arm_add_r=(-1.0, 0.0),
                  arm_rot_r=(1.0, 0.0),
                  elbow_flex_r=(1.0, 0.0),
                  pro_sup_r=(1.0, -np.pi/2),
                  arm_flex_l=(1.0, 0.0),
                  arm_add_l=(-1.0, 0.0),
                  arm_rot_l=(1.0, 0.0),
                  elbow_flex_l=(1.0, 0.0),
                  pro_sup_l=(1.0, -np.pi/2)
                  )

unavailable_keys = dict()
for j in all_joints:
    if j not in joint_conf.keys():
        jh = xml_handle.find("joint", j)
        unavailable_keys[j] = jh.ref if jh.ref is not None else 0.0


rename_map = dict(lumbar_extension="L5_S1_Flex_Ext",
                  lumbar_bending="L5_S1_Lat_Bending",
                  lumbar_rotation="L5_S1_axial_rotation",
                  arm_flex_r='elv_angle_r',
                  arm_add_r='shoulder_elv_r',
                  arm_rot_r='shoulder1_r2_r',
                  arm_flex_l='elv_angle_l',
                  arm_add_l='shoulder_elv_l',
                  arm_rot_l='shoulder1_r2_l',
                  pro_sup_r='pro_sup'
                  )

path_mat = "../00_raw_mocap_data/raw_walking_motion_capture.mat"
dir_target_path = "../generated_data"
if not os.path.exists(dir_target_path):
    os.makedirs(dir_target_path)
target_path = os.path.join(dir_target_path, "myosuite_humanoid_walking.npz")

dataset = adapt_mocap(path_mat, joint_conf=joint_conf, unavailable_keys=unavailable_keys, rename_map=rename_map,
                      discard_first=5000, discard_last=1000)

old_order = 'zxy'
new_order = 'yxy'
keys_right = ['q_elv_angle_r', 'q_shoulder_elv_r', 'q_shoulder1_r2_r']
keys_left = ['q_elv_angle_l', 'q_shoulder_elv_l', 'q_shoulder1_r2_l']
reorder_shoulder_orientation(keys_right, old_order, new_order)
reorder_shoulder_orientation(keys_left, old_order, new_order)

# recalculating the velocity for the arm data
dt = 1.0 / 500.0
for k in keys_right+keys_left:
    data = dataset[k]
    vel_data = np.zeros_like(data)
    vel_data[:-1] = np.diff(data) / dt
    dataset["d"+k] = vel_data

# remove the last data point
for k, i in dataset.items():
    dataset[k] = i[:-1]

np.savez(file=target_path, **dataset)


