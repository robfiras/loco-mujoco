# Dataset Generation
This directory contains the code to generate the datasets from raw motion capture data. The raw motion capture dataset
contains joint angles and velocities of a human subject performing a specific motion. The Humanoid model from LocoMujoco (Torque or Muscle) 
uses the kinematics from the human subject, which is why it does not need adaption. To optimize the dataset for each humanoid robot, we apply linear transformations to the joint angles and
velocities. The transformations are tuned manually. Here we explain the pipeline to generate these datasets.
This pipeline can be used to create datasets for new humanoid robots.

### Download the Raw Motion Capture Data
Because the raw motion capture data is not used within LocoMuJoCo, it is not automatically downloaded with the other datasets.
Therefore, you need to download the raw motion capture data manually:

```python
from loco_mujoco.utils.dataset import download_raw_mocap_datasets

download_raw_mocap_datasets()
```
This funciton will download and extract the raw motion capture data  to `00_raw_mocal_data`.

### Generate and Optimize the Datasets for a specific Humanoid
For each humanoid, a joint configuration is needed. This joint configuration is a dictionary containing a tuple
of multipliers and offsets for each joint. The multipliers and offsets are used to scale and shift the joint angles and velocities.
We include two examples for the Humanoid (Torque or Muscle) and the UnitreeH1 in the `conversion` directory. 
Here is the file to convert the raw motion capture dataset for running to the kinematics of the UnitreeH1:

```python
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
                          discard_first=25000, discard_last=1000)

    np.savez(file=target_path, **dataset)
```

As can be seen,  the function adapt_mocap takes the joint configuration containing linear transformations, a list of dictionary
for unavailable keys, and a rename map. The unavailable keys are the keys of joints that are not present in the raw motion capture data but are available
in the model. The rename map is a dictionary that maps the keys of the raw motion capture data to the keys of the model.
`discard_first` and `discard_last` are used to remove the first and last obs from the dataset.
This function converts the dataset and saves it to the `generated_data` directory. 



### Replaying the Dataset
We find the linear transformations by manually tuning the joint configuration. Therefore, it is important to visually inspect the 
generated dataset. In the `replay` directory, we provide two examples. Here is the one for the UnitreeH1:

```python
import numpy as np

from loco_mujoco.environments import UnitreeH1


def experiment():
    np.random.seed(1)

    # define env and data frequencies
    env_freq = 1000  # hz, added here as a reminder
    traj_data_freq = 500    # hz, added here as a reminder
    desired_contr_freq = 100     # hz, this will also be the dataset frequency after downsampling
    n_substeps = env_freq//desired_contr_freq

    # prepare trajectory params
    traj_params = dict(traj_path="../generated_data/05-run_UnitreeH1.npz",
                       traj_dt=(1 / traj_data_freq),
                       control_dt=(1 / desired_contr_freq))

    # MDP
    gamma = 0.99
    horizon = 1000
    mdp = UnitreeH1(gamma=gamma, horizon=horizon, n_substeps=n_substeps, traj_params=traj_params,
                    disable_arms=False, disable_back_joint=False)

    mdp.play_trajectory()


if __name__ == '__main__':
    experiment()
```

Mostly likely, the generated dataset will not be good on the first trial. You have to iteratively adjust the joint configuration
until the replayed motion looks good. This might take some time to get a feeling for it. The most important part is to 
find the joint that have the opposite sign in the model and the raw motion capture data. As an example, the hip flexion joint in the UnitreeH1 has 
opposite sign to the raw motion capture data, which is why the multiplier is set to -1.0.
