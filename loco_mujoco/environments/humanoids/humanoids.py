from loco_mujoco.environments.humanoids.base_humanoid import BaseHumanoid
from loco_mujoco.environments.humanoids.base_humanoid_4_ages import BaseHumanoid4Ages
from loco_mujoco.environments import ValidTaskConf
from loco_mujoco.utils import check_validity_task_mode_dataset


class HumanoidTorque(BaseHumanoid):

    """
    Description
    ------------

    Mujoco environment of a humanoid model with one torque actuator per joint.

    Tasks
    -----------------
    * **Walking**: The robot has to walk forward with a fixed speed of 1.25 m/s.
    * **Running**: Run forward with a fixed speed of 2.5 m/s.


    Dataset Types
    -----------------
    The available dataset types for this environment can be found at: :ref:`env-label`.


    Observation Space
    -----------------

    The observation space has the following properties *by default* (i.e., only obs with Disabled == False):

    | For walking and running task: :code:`(min=-inf, max=inf, dim=36, dtype=float32)`

    Some observations are **disabled by default**, but can be turned on. The detailed observation space is:

    ===== ============================================================================= ========= ======== ======== === ========================
    Index Description                                                                   Min       Max      Disabled Dim Units
    ===== ============================================================================= ========= ======== ======== === ========================
    0     Position of Joint pelvis_ty                                                   -inf      inf      False    1   Position [m]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    1     Position of Joint pelvis_tilt                                                 -inf      inf      False    1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    2     Position of Joint pelvis_list                                                 -inf      inf      False    1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    3     Position of Joint pelvis_rotation                                             -inf      inf      False    1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    4     Position of Joint hip_flexion_r                                               -0.787    0.787    False    1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    5     Position of Joint hip_adduction_r                                             -0.524    0.524    False    1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    6     Position of Joint hip_rotation_r                                              -2.0944   2.0944   False    1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    7     Position of Joint knee_angle_r                                                -2.0944   0.174533 False    1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    8     Position of Joint ankle_angle_r                                               -1.5708   1.5708   False    1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    9     Position of Joint hip_flexion_l                                               -0.787    0.787    False    1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    10    Position of Joint hip_adduction_l                                             -0.524    0.524    False    1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    11    Position of Joint hip_rotation_l                                              -2.0944   2.0944   False    1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    12    Position of Joint knee_angle_l                                                -2.0944   0.174533 False    1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    13    Position of Joint ankle_angle_l                                               -1.0472   1.0472   False    1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    14    Position of Joint lumbar_extension                                            -1.5708   0.377    False    1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    15    Position of Joint lumbar_bending                                              -0.754    0.754    False    1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    16    Position of Joint lumbar_rotation                                             -0.754    0.754    False    1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    17    Position of Joint arm_flex_r                                                  -1.5708   1.5708   True     1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    18    Position of Joint arm_add_r                                                   -2.0944   1.5708   True     1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    19    Position of Joint arm_rot_r                                                   -1.5708   1.5708   True     1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    20    Position of Joint elbow_flex_r                                                -0.3      2.618    True     1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    21    Position of Joint pro_sup_r                                                   -0.6      1.5708   True     1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    22    Position of Joint wrist_flex_r                                                -1.22173  1.22173  True     1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    23    Position of Joint wrist_dev_r                                                 -0.436332 0.610865 True     1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    24    Position of Joint arm_flex_l                                                  -1.5708   1.5708   True     1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    25    Position of Joint arm_add_l                                                   -2.0944   1.5708   True     1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    26    Position of Joint arm_rot_l                                                   -1.5708   1.5708   True     1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    27    Position of Joint elbow_flex_l                                                -0.3      2.618    True     1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    28    Position of Joint pro_sup_l                                                   -0.6      1.5708   True     1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    29    Position of Joint wrist_flex_l                                                -1.22173  1.22173  True     1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    30    Position of Joint wrist_dev_l                                                 -0.436332 0.610865 True     1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    31    Velocity of Joint pelvis_tx                                                   -inf      inf      False    1   Velocity [m/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    32    Velocity of Joint pelvis_tz                                                   -inf      inf      False    1   Velocity [m/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    33    Velocity of Joint pelvis_ty                                                   -inf      inf      False    1   Velocity [m/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    34    Velocity of Joint pelvis_tilt                                                 -inf      inf      False    1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    35    Velocity of Joint pelvis_list                                                 -inf      inf      False    1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    36    Velocity of Joint pelvis_rotation                                             -inf      inf      False    1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    37    Velocity of Joint hip_flexion_r                                               -inf      inf      False    1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    38    Velocity of Joint hip_adduction_r                                             -inf      inf      False    1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    39    Velocity of Joint hip_rotation_r                                              -inf      inf      False    1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    40    Velocity of Joint knee_angle_r                                                -inf      inf      False    1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    41    Velocity of Joint ankle_angle_r                                               -inf      inf      False    1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    42    Velocity of Joint hip_flexion_l                                               -inf      inf      False    1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    43    Velocity of Joint hip_adduction_l                                             -inf      inf      False    1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    44    Velocity of Joint hip_rotation_l                                              -inf      inf      False    1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    45    Velocity of Joint knee_angle_l                                                -inf      inf      False    1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    46    Velocity of Joint ankle_angle_l                                               -inf      inf      False    1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    47    Velocity of Joint lumbar_extension                                            -inf      inf      False    1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    48    Velocity of Joint lumbar_bending                                              -inf      inf      False    1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    49    Velocity of Joint lumbar_rotation                                             -inf      inf      False    1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    50    Velocity of Joint arm_flex_r                                                  -inf      inf      True     1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    51    Velocity of Joint arm_add_r                                                   -inf      inf      True     1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    52    Velocity of Joint arm_rot_r                                                   -inf      inf      True     1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    53    Velocity of Joint elbow_flex_r                                                -inf      inf      True     1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    54    Velocity of Joint pro_sup_r                                                   -inf      inf      True     1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    55    Velocity of Joint wrist_flex_r                                                -inf      inf      True     1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    56    Velocity of Joint wrist_dev_r                                                 -inf      inf      True     1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    57    Velocity of Joint arm_flex_l                                                  -inf      inf      True     1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    58    Velocity of Joint arm_add_l                                                   -inf      inf      True     1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    59    Velocity of Joint arm_rot_l                                                   -inf      inf      True     1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    60    Velocity of Joint elbow_flex_l                                                -inf      inf      True     1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    61    Velocity of Joint pro_sup_l                                                   -inf      inf      True     1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    62    Velocity of Joint wrist_flex_l                                                -inf      inf      True     1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    63    Velocity of Joint wrist_dev_l                                                 -inf      inf      True     1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    64    3D linear Forces between Back Right Foot and Floor                            0.0       inf      True     3   Force [N]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    67    3D linear Forces between Front Right Foot and Floor (If box feet is disabled) 0.0       inf      True     3   Force [N]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    70    3D linear Forces between Back Left Foot and Floor                             0.0       inf      True     3   Force [N]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    73    3D linear Forces between Front Left Foot and Floor (If box feet is disabled)  0.0       inf      True     3   Force [N]
    ===== ============================================================================= ========= ======== ======== === ========================

    Action Space
    ------------

    | The action space has the following properties *by default* (i.e., only actions with Disabled == False):
    | :code:`(min=-1, max=1, dim=13, dtype=float32)`

    The action range in LocoMuJoCo is always standardized, i.e. in [-1.0, 1.0].
    The XML of the environment specifies for each actuator a *gearing* ratio, which is used to scale the
    the action to the actual control range of the actuator.

    Some actions are **disabled by default**, but can be turned on. The detailed action space is:

    ===== ==================== =========== =========== ========
    Index Name in XML          Control Min Control Max Disabled
    ===== ==================== =========== =========== ========
    0     mot_lumbar_ext       -1.0        1.0         False
    ----- -------------------- ----------- ----------- --------
    1     mot_lumbar_bend      -1.0        1.0         False
    ----- -------------------- ----------- ----------- --------
    2     mot_lumbar_rot       -1.0        1.0         False
    ----- -------------------- ----------- ----------- --------
    3     mot_shoulder_flex_r  -1.0        1.0         True
    ----- -------------------- ----------- ----------- --------
    4     mot_shoulder_add_r   -1.0        1.0         True
    ----- -------------------- ----------- ----------- --------
    5     mot_shoulder_rot_r   -1.0        1.0         True
    ----- -------------------- ----------- ----------- --------
    6     mot_elbow_flex_r     -1.0        1.0         True
    ----- -------------------- ----------- ----------- --------
    7     mot_pro_sup_r        -1.0        1.0         True
    ----- -------------------- ----------- ----------- --------
    8     mot_wrist_flex_r     -1.0        1.0         True
    ----- -------------------- ----------- ----------- --------
    9     mot_wrist_dev_r      -1.0        1.0         True
    ----- -------------------- ----------- ----------- --------
    10    mot_shoulder_flex_l  -1.0        1.0         True
    ----- -------------------- ----------- ----------- --------
    11    mot_shoulder_add_l   -1.0        1.0         True
    ----- -------------------- ----------- ----------- --------
    12    mot_shoulder_rot_l   -1.0        1.0         True
    ----- -------------------- ----------- ----------- --------
    13    mot_elbow_flex_l     -1.0        1.0         True
    ----- -------------------- ----------- ----------- --------
    14    mot_pro_sup_l        -1.0        1.0         True
    ----- -------------------- ----------- ----------- --------
    15    mot_wrist_flex_l     -1.0        1.0         True
    ----- -------------------- ----------- ----------- --------
    16    mot_wrist_dev_l      -1.0        1.0         True
    ----- -------------------- ----------- ----------- --------
    17    mot_hip_flexion_r    -1.0        1.0         False
    ----- -------------------- ----------- ----------- --------
    18    mot_hip_adduction_r  -1.0        1.0         False
    ----- -------------------- ----------- ----------- --------
    19    mot_hip_rotation_r   -1.0        1.0         False
    ----- -------------------- ----------- ----------- --------
    20    mot_knee_angle_r     -1.0        1.0         False
    ----- -------------------- ----------- ----------- --------
    21    mot_ankle_angle_r    -1.0        1.0         False
    ----- -------------------- ----------- ----------- --------
    22    mot_subtalar_angle_r -1.0        1.0         True
    ----- -------------------- ----------- ----------- --------
    23    mot_mtp_angle_r      -1.0        1.0         True
    ----- -------------------- ----------- ----------- --------
    24    mot_hip_flexion_l    -1.0        1.0         False
    ----- -------------------- ----------- ----------- --------
    25    mot_hip_adduction_l  -1.0        1.0         False
    ----- -------------------- ----------- ----------- --------
    26    mot_hip_rotation_l   -1.0        1.0         False
    ----- -------------------- ----------- ----------- --------
    27    mot_knee_angle_l     -1.0        1.0         False
    ----- -------------------- ----------- ----------- --------
    28    mot_ankle_angle_l    -1.0        1.0         False
    ----- -------------------- ----------- ----------- --------
    29    mot_subtalar_angle_l -1.0        1.0         True
    ----- -------------------- ----------- ----------- --------
    30    mot_mtp_angle_l      -1.0        1.0         True
    ===== ==================== =========== =========== ========

    Rewards
    --------

    The default reward function is based on the distance between the current center of mass velocity and the
    desired velocity in the x-axis. The desired velocity is given by the dataset to imitate.

    **Class**: :class:`loco_mujoco.utils.reward.TargetVelocityReward`

    Initial States
    ---------------

    The initial state is sampled by default from the dataset to imitate.

    Terminal States
    ----------------

    The terminal state is reached when the humanoid falls, or rather starts falling. The condition to check if the robot
    is falling is based on the orientation of the robot, the height of the center of mass, and the orientation of the
    back joint. More details can be found in the  :code:`_has_fallen` method of the environment.

    Methods
    ------------

    """

    valid_task_confs = ValidTaskConf(tasks=["walk", "run"],
                                     data_types=["real", "perfect"])

    def __init__(self, **kwargs):
        """
        Constructor.

        """

        if "use_muscles" in kwargs.keys():
            assert kwargs["use_muscles"] is False, "Activating muscles in this environment not allowed. "
            del kwargs["use_muscles"]

        super(HumanoidTorque, self).__init__(use_muscles=False, **kwargs)

    @staticmethod
    def generate(task="walk", dataset_type="real", **kwargs):

        check_validity_task_mode_dataset(HumanoidTorque.__name__, task, None, dataset_type,
                                         *HumanoidTorque.valid_task_confs.get_all())

        if dataset_type == "real":
            if task == "walk":
                path = "datasets/humanoids/real/02-constspeed_reduced_humanoid.npz"
            elif task == "run":
                path = "datasets/humanoids/real/05-run_reduced_humanoid.npz"
        elif dataset_type == "perfect":
            if "use_foot_forces" in kwargs.keys():
                assert kwargs["use_foot_forces"] is False
            if "disable_arms" in kwargs.keys():
                assert kwargs["disable_arms"] is True
            if "use_box_feet" in kwargs.keys():
                assert kwargs["use_box_feet"] is True

            if task == "walk":
                path = "datasets/humanoids/perfect/humanoid_torque_walk/perfect_expert_dataset_det.npz"
            elif task == "run":
                path = "datasets/humanoids/perfect/humanoid_torque_run/perfect_expert_dataset_det.npz"

        return BaseHumanoid.generate(HumanoidTorque, path, task, dataset_type, **kwargs)


class HumanoidMuscle(BaseHumanoid):

    """
    Description
    ------------

    Mujoco environment of a humanoid model with muscle actuation.

    .. note:: This Humanoid consists of 92 muscles on the lower limb. The upper body is torque actuated.

    Tasks
    -----------------
    * **Walking**: The robot has to walk forward with a fixed speed of 1.25 m/s.
    * **Running**: Run forward with a fixed speed of 2.5 m/s.


    Dataset Types
    -----------------
    The available dataset types for this environment can be found at: :ref:`env-label`.


    Observation Space
    -----------------

    The observation space has the following properties *by default* (i.e., only obs with Disabled == False):

    | For walking and running task: :code:`(min=-inf, max=inf, dim=36, dtype=float32)`

    Some observations are **disabled by default**, but can be turned on. The detailed observation space is:

    ===== ============================================================================= ========= ======== ======== === ========================
    Index Description                                                                   Min       Max      Disabled Dim Units
    ===== ============================================================================= ========= ======== ======== === ========================
    0     Position of Joint pelvis_ty                                                   -100.0    200.0    False    1   Position [m]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    1     Position of Joint pelvis_tilt                                                 -1.5708   1.5708   False    1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    2     Position of Joint pelvis_list                                                 -1.5708   1.5708   False    1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    3     Position of Joint pelvis_rotation                                             -1.5708   1.5708   False    1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    4     Position of Joint hip_flexion_r                                               -0.787    0.787    False    1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    5     Position of Joint hip_adduction_r                                             -0.524    0.524    False    1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    6     Position of Joint hip_rotation_r                                              -2.0944   2.0944   False    1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    7     Position of Joint knee_angle_r                                                -2.0944   0.174533 False    1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    8     Position of Joint ankle_angle_r                                               -1.5708   1.5708   False    1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    9     Position of Joint hip_flexion_l                                               -0.787    0.787    False    1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    10    Position of Joint hip_adduction_l                                             -0.524    0.524    False    1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    11    Position of Joint hip_rotation_l                                              -2.0944   2.0944   False    1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    12    Position of Joint knee_angle_l                                                -2.0944   0.174533 False    1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    13    Position of Joint ankle_angle_l                                               -1.0472   1.0472   False    1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    14    Position of Joint lumbar_extension                                            -1.5708   0.377    False    1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    15    Position of Joint lumbar_bending                                              -0.754    0.754    False    1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    16    Position of Joint lumbar_rotation                                             -0.754    0.754    False    1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    17    Position of Joint arm_flex_r                                                  -1.5708   1.5708   True     1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    18    Position of Joint arm_add_r                                                   -2.0944   1.5708   True     1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    19    Position of Joint arm_rot_r                                                   -1.5708   1.5708   True     1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    20    Position of Joint elbow_flex_r                                                0.0       2.618    True     1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    21    Position of Joint pro_sup_r                                                   0.0       1.5708   True     1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    22    Position of Joint wrist_flex_r                                                -1.22173  1.22173  True     1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    23    Position of Joint wrist_dev_r                                                 -0.436332 0.610865 True     1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    24    Position of Joint arm_flex_l                                                  -1.5708   1.5708   True     1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    25    Position of Joint arm_add_l                                                   -2.0944   1.5708   True     1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    26    Position of Joint arm_rot_l                                                   -1.5708   1.5708   True     1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    27    Position of Joint elbow_flex_l                                                0.0       2.618    True     1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    28    Position of Joint pro_sup_l                                                   0.0       1.5708   True     1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    29    Position of Joint wrist_flex_l                                                -1.22173  1.22173  True     1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    30    Position of Joint wrist_dev_l                                                 -0.436332 0.610865 True     1   Angle [rad]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    31    Velocity of Joint pelvis_tx                                                   -inf      inf      False    1   Velocity [m/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    32    Velocity of Joint pelvis_tz                                                   -inf      inf      False    1   Velocity [m/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    33    Velocity of Joint pelvis_ty                                                   -inf      inf      False    1   Velocity [m/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    34    Velocity of Joint pelvis_tilt                                                 -inf      inf      False    1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    35    Velocity of Joint pelvis_list                                                 -inf      inf      False    1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    36    Velocity of Joint pelvis_rotation                                             -inf      inf      False    1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    37    Velocity of Joint hip_flexion_r                                               -inf      inf      False    1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    38    Velocity of Joint hip_adduction_r                                             -inf      inf      False    1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    39    Velocity of Joint hip_rotation_r                                              -inf      inf      False    1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    40    Velocity of Joint knee_angle_r                                                -inf      inf      False    1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    41    Velocity of Joint ankle_angle_r                                               -inf      inf      False    1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    42    Velocity of Joint hip_flexion_l                                               -inf      inf      False    1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    43    Velocity of Joint hip_adduction_l                                             -inf      inf      False    1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    44    Velocity of Joint hip_rotation_l                                              -inf      inf      False    1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    45    Velocity of Joint knee_angle_l                                                -inf      inf      False    1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    46    Velocity of Joint ankle_angle_l                                               -inf      inf      False    1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    47    Velocity of Joint lumbar_extension                                            -inf      inf      False    1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    48    Velocity of Joint lumbar_bending                                              -inf      inf      False    1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    49    Velocity of Joint lumbar_rotation                                             -inf      inf      False    1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    50    Velocity of Joint arm_flex_r                                                  -inf      inf      True     1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    51    Velocity of Joint arm_add_r                                                   -inf      inf      True     1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    52    Velocity of Joint arm_rot_r                                                   -inf      inf      True     1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    53    Velocity of Joint elbow_flex_r                                                -inf      inf      True     1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    54    Velocity of Joint pro_sup_r                                                   -inf      inf      True     1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    55    Velocity of Joint wrist_flex_r                                                -inf      inf      True     1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    56    Velocity of Joint wrist_dev_r                                                 -inf      inf      True     1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    57    Velocity of Joint arm_flex_l                                                  -inf      inf      True     1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    58    Velocity of Joint arm_add_l                                                   -inf      inf      True     1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    59    Velocity of Joint arm_rot_l                                                   -inf      inf      True     1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    60    Velocity of Joint elbow_flex_l                                                -inf      inf      True     1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    61    Velocity of Joint pro_sup_l                                                   -inf      inf      True     1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    62    Velocity of Joint wrist_flex_l                                                -inf      inf      True     1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    63    Velocity of Joint wrist_dev_l                                                 -inf      inf      True     1   Angular Velocity [rad/s]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    64    3D linear Forces between Back Right Foot and Floor                            0.0       inf      True     3   Force [N]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    67    3D linear Forces between Front Right Foot and Floor (If box feet is disabled) 0.0       inf      True     3   Force [N]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    70    3D linear Forces between Back Left Foot and Floor                             0.0       inf      True     3   Force [N]
    ----- ----------------------------------------------------------------------------- --------- -------- -------- --- ------------------------
    73    3D linear Forces between Front Left Foot and Floor (If box feet is disabled)  0.0       inf      True     3   Force [N]
    ===== ============================================================================= ========= ======== ======== === ========================

    Action Space
    ------------

    .. note:: This Humanoid consists of 92 muscles on the lower limb. The upper body is torque actuated. The upper
     body torque actuators have "mot_" for motor in their name.

    | The action space has the following properties *by default* (i.e., only actions with Disabled == False):
    | :code:`(min=-1, max=1, dim=92, dtype=float32)`

    The action range in LocoMuJoCo is always standardized, i.e. in [-1.0, 1.0].
    In Mujoco, muscles are actuated in a range of [0, 1], which is whý the action space is scaled and shifted
    by LocoMuJoCo from [-1.0, 1.0] to [0, 1]. The strength of each muscle is defined in the XML.
    For the motors, the XML of the environment specifies for each actuator a *gearing* ratio, which is used to scale the
    the action to the actual control range of the actuator.

    Some actions are **disabled by default**, but can be turned on. The detailed action space is:

    ===== =================== =========== =========== ========
    Index Name in XML         Control Min Control Max Disabled
    ===== =================== =========== =========== ========
    0     mot_shoulder_flex_r -1.0        1.0         True
    ----- ------------------- ----------- ----------- --------
    1     mot_shoulder_add_r  -1.0        1.0         True
    ----- ------------------- ----------- ----------- --------
    2     mot_shoulder_rot_r  -1.0        1.0         True
    ----- ------------------- ----------- ----------- --------
    3     mot_elbow_flex_r    -1.0        1.0         True
    ----- ------------------- ----------- ----------- --------
    4     mot_pro_sup_r       -1.0        1.0         True
    ----- ------------------- ----------- ----------- --------
    5     mot_wrist_flex_r    -1.0        1.0         True
    ----- ------------------- ----------- ----------- --------
    6     mot_wrist_dev_r     -1.0        1.0         True
    ----- ------------------- ----------- ----------- --------
    7     mot_shoulder_flex_l -1.0        1.0         True
    ----- ------------------- ----------- ----------- --------
    8     mot_shoulder_add_l  -1.0        1.0         True
    ----- ------------------- ----------- ----------- --------
    9     mot_shoulder_rot_l  -1.0        1.0         True
    ----- ------------------- ----------- ----------- --------
    10    mot_elbow_flex_l    -1.0        1.0         True
    ----- ------------------- ----------- ----------- --------
    11    mot_pro_sup_l       -1.0        1.0         True
    ----- ------------------- ----------- ----------- --------
    12    mot_wrist_flex_l    -1.0        1.0         True
    ----- ------------------- ----------- ----------- --------
    13    mot_wrist_dev_l     -1.0        1.0         True
    ----- ------------------- ----------- ----------- --------
    14    glut_med1_r         -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    15    glut_med2_r         -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    16    glut_med3_r         -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    17    glut_min1_r         -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    18    glut_min2_r         -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    19    glut_min3_r         -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    20    semimem_r           -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    21    semiten_r           -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    22    bifemlh_r           -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    23    bifemsh_r           -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    24    sar_r               -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    25    add_long_r          -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    26    add_brev_r          -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    27    add_mag1_r          -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    28    add_mag2_r          -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    29    add_mag3_r          -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    30    tfl_r               -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    31    pect_r              -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    32    grac_r              -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    33    glut_max1_r         -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    34    glut_max2_r         -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    35    glut_max3_r         -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    36    iliacus_r           -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    37    psoas_r             -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    38    quad_fem_r          -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    39    gem_r               -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    40    peri_r              -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    41    rect_fem_r          -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    42    vas_med_r           -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    43    vas_int_r           -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    44    vas_lat_r           -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    45    med_gas_r           -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    46    lat_gas_r           -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    47    soleus_r            -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    48    tib_post_r          -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    49    flex_dig_r          -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    50    flex_hal_r          -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    51    tib_ant_r           -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    52    per_brev_r          -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    53    per_long_r          -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    54    per_tert_r          -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    55    ext_dig_r           -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    56    ext_hal_r           -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    57    glut_med1_l         -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    58    glut_med2_l         -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    59    glut_med3_l         -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    60    glut_min1_l         -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    61    glut_min2_l         -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    62    glut_min3_l         -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    63    semimem_l           -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    64    semiten_l           -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    65    bifemlh_l           -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    66    bifemsh_l           -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    67    sar_l               -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    68    add_long_l          -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    69    add_brev_l          -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    70    add_mag1_l          -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    71    add_mag2_l          -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    72    add_mag3_l          -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    73    tfl_l               -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    74    pect_l              -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    75    grac_l              -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    76    glut_max1_l         -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    77    glut_max2_l         -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    78    glut_max3_l         -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    79    iliacus_l           -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    80    psoas_l             -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    81    quad_fem_l          -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    82    gem_l               -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    83    peri_l              -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    84    rect_fem_l          -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    85    vas_med_l           -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    86    vas_int_l           -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    87    vas_lat_l           -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    88    med_gas_l           -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    89    lat_gas_l           -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    90    soleus_l            -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    91    tib_post_l          -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    92    flex_dig_l          -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    93    flex_hal_l          -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    94    tib_ant_l           -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    95    per_brev_l          -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    96    per_long_l          -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    97    per_tert_l          -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    98    ext_dig_l           -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    99    ext_hal_l           -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    100   ercspn_r            -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    101   ercspn_l            -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    102   intobl_r            -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    103   intobl_l            -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    104   extobl_r            -1.0        1.0         False
    ----- ------------------- ----------- ----------- --------
    105   extobl_l            -1.0        1.0         False
    ===== =================== =========== =========== ========

    Rewards
    --------

    The default reward function is based on the distance between the current center of mass velocity and the
    desired velocity in the x-axis. The desired velocity is given by the dataset to imitate.

    **Class**: :class:`loco_mujoco.utils.reward.TargetVelocityReward`

    Initial States
    ---------------

    The initial state is sampled by default from the dataset to imitate.

    Terminal States
    ----------------

    The terminal state is reached when the humanoid falls, or rather starts falling. The condition to check if the robot
    is falling is based on the orientation of the robot, the height of the center of mass, and the orientation of the
    back joint. More details can be found in the  :code:`_has_fallen` method of the environment.

    Methods
    ------------

    """

    valid_task_confs = ValidTaskConf(tasks=["walk", "run"],
                                     data_types=["real", "perfect"],
                                     non_combinable=[("run", None, "perfect")])

    def __init__(self, **kwargs):
        """
        Constructor.

        """

        if "use_muscles" in kwargs.keys():
            assert kwargs["use_muscles"] is True, "Activating torque actuators in this environment not allowed. "
            del kwargs["use_muscles"]

        super(HumanoidMuscle, self).__init__(use_muscles=True, **kwargs)

    @staticmethod
    def generate(task="walk", dataset_type="real", **kwargs):

        check_validity_task_mode_dataset(HumanoidMuscle.__name__, task, None, dataset_type,
                                         *HumanoidMuscle.valid_task_confs.get_all())

        if dataset_type == "real":
            if task == "walk":
                path = "datasets/humanoids/real/02-constspeed_reduced_humanoid.npz"
            elif task == "run":
                path = "datasets/humanoids/real/05-run_reduced_humanoid.npz"
        elif dataset_type == "perfect":
            if "use_foot_forces" in kwargs.keys():
                assert kwargs["use_foot_forces"] is False
            if "disable_arms" in kwargs.keys():
                assert kwargs["disable_arms"] is True
            if "use_box_feet" in kwargs.keys():
                assert kwargs["use_box_feet"] is True

            if task == "walk":
                path = "datasets/humanoids/perfect/humanoid_muscle_walk/perfect_expert_dataset_det.npz"

        return BaseHumanoid.generate(HumanoidMuscle, path, task, dataset_type, **kwargs)


class HumanoidTorque4Ages(BaseHumanoid4Ages):
    """
    Mujoco environment of 4 simplified humanoid models with one torque actuator per joint.
    At the beginning of each episode, one of the four humanoid models are randomly
    sampled and used to simulate a trajectory. The different humanoids should
    resemble an adult, a teenager (∼12 years), a child (∼5 years), and a
    toddler (∼1-2 years). This environment can be partially observable by
    using state masks to hide the humanoid type indicator from the policy.

    .. note::
        This environment is almost identical to the HumanoidTorque environment. The only differences are
        a small addition to the observation space, a scaled reward function, and a slightly different initial
        state distribution.

    Observation Space
    -----------------

    **Additionally** to the HumanoidTorque model, the HumanoidTorque4Ages model has an identifier to give information
    about the current scaling. The identifier is a binary number (bits). The length of the binary number is determined
    by the number of scalings available. By default, it is 4, which is why the length of the identifier is 2
    (Two bits are needed to encode a decimal <=4). By adding the identifier, the environment is not partially
    observable by default. The additional observation space is:

    ===== =============================================== === === ======== === =====
    Index Description                                     Min Max Disabled Dim Units
    ===== =============================================== === === ======== === =====
    77    Binary number identifying the humanoid scaling. 0   1   False    2   None
    ===== =============================================== === === ======== === =====

    Rewards
    --------

    The default reward function is based on the distance between the current center of mass velocity and the
    desired velocity in the x-axis. The desired velocity is given by the dataset to imitate. In comparison to
    the HumanoidTorque model, this reward used a scaled desired target velocity to adapt to smaller humanoids.

    **Class**: :class:`loco_mujoco.utils.reward.MultiTargetVelocityReward`

    Initial States
    ---------------

    At the beginning of each episode, a scaling factor is randomly sampled to choose the type of humanoid.
    Based on that, the initial state is sampled by from the respective humanoid dataset.

    Methods
    ------------

    """

    valid_task_confs = ValidTaskConf(tasks=["walk", "run"],
                                     modes=["all", "1", "2", "3", "4"],
                                     data_types=["real", "perfect"])

    def __init__(self, **kwargs):
        """
        Constructor.

        """

        if "use_muscles" in kwargs.keys():
            assert kwargs["use_muscles"] is False, "Activating muscles in this environment not allowed. "
            del kwargs["use_muscles"]

        super(HumanoidTorque4Ages, self).__init__(use_muscles=False, **kwargs)

    @staticmethod
    def generate(task="walk", mode="all", dataset_type="real", **kwargs):
        """
        Returns a HumanoidTorque4Ages environment corresponding to the specified task.

        Args:
            task (str):
                Main task to solve. Either "walk" or "run".
            mode (str):
                Mode of the environment. Either "all" (sample between all humanoid envs), "1"
                (smallest humanoid), "2" (second smallest humanoid), "3" (teenage humanoid), and "4" (adult humanoid).
            dataset_type (str):
                "real" or "perfect". "real" uses real motion capture data as the
                reference trajectory. This data does not perfectly match the kinematics
                and dynamics of this environment, hence it is more challenging. "perfect" uses
                a perfect dataset.

        Returns:
            An MDP of a set of Torque Humanoid of different sizes.

         """

        check_validity_task_mode_dataset(HumanoidTorque4Ages.__name__, task, mode, dataset_type,
                                         *HumanoidTorque4Ages.valid_task_confs.get_all())

        if task == "walk":
            if dataset_type == "real":
                path = "datasets/humanoids/real/02-constspeed_reduced_humanoid_POMDP"
            elif dataset_type == "perfect":
                path = "datasets/humanoids/perfect/humanoid4ages_torque_walk/" \
                       "HumanoidTorque4Ages_walk_stochastic_dataset"
        elif task == "run":
            if dataset_type == "real":
                path = "datasets/humanoids/real/05-run_reduced_humanoid_POMDP"
            elif dataset_type == "perfect":
                path = "datasets/humanoids/perfect/humanoid4ages_torque_run/" \
                       "HumanoidTorque4Ages_run_stochastic_dataset"

        return BaseHumanoid4Ages.generate(HumanoidTorque4Ages, path, task, mode, dataset_type, **kwargs)


class HumanoidMuscle4Ages(BaseHumanoid4Ages):

    """
    Mujoco environment of 4 simplified humanoid models with muscle actuation.
    At the beginning of each episode, one of the four humanoid models are
    sampled and used to simulate a trajectory. The different humanoids should
    resemble an adult, a teenager (∼12 years), a child (∼5 years), and a
    toddler (∼1-2 years). This environment can be partially observable by
    using state masks to hide the humanoid type indicator from the policy.

    .. note::
        This environment is almost identical to the HumanoidMuscle environment. The only differences are
        a small addition to the observation space, a scaled reward function, and a slightly different initial
        state distribution.

    Observation Space
    -----------------

    **Additionally** to the HumanoidMuscle model, the HumanoidMuscle4Ages model has an identifier to give information
    about the current scaling. The identifier is a binary number (bits). The length of the binary number is determined
    by the number of scalings available. By default, it is 4, which is why the length of the identifier is 2
    (Two bits are needed to encode a decimal <=4). By adding the identifier, the environment is not partially
    observable by default. The additional observation space is:

    ===== =============================================== === === ======== === =====
    Index Description                                     Min Max Disabled Dim Units
    ===== =============================================== === === ======== === =====
    77    Binary number identifying the humanoid scaling. 0   1   False    2   None
    ===== =============================================== === === ======== === =====

    Rewards
    --------

    The default reward function is based on the distance between the current center of mass velocity and the
    desired velocity in the x-axis. The desired velocity is given by the dataset to imitate. In comparison to
    the HumanoidMuscle model, this reward used a scaled desired target velocity to adapt to smaller humanoids.

    **Class**: :class:`loco_mujoco.utils.reward.MultiTargetVelocityReward`

    Initial States
    ---------------

    At the beginning of each episode, a scaling factor is randomly sampled to choose the type of humanoid.
    Based on that, the initial state is sampled by from the respective humanoid dataset.

    Methods
    ------------

    """

    valid_task_confs = ValidTaskConf(tasks=["walk", "run"],
                                     modes=["all", "1", "2", "3", "4"],
                                     data_types=["real"])

    def __init__(self, **kwargs):
        """
                Constructor.

        """

        if "use_muscles" in kwargs.keys():
            assert kwargs["use_muscles"] is True, "Activating torque actuators in this environment not allowed. "
            del kwargs["use_muscles"]

        super(HumanoidMuscle4Ages, self).__init__(use_muscles=True, **kwargs)

    @staticmethod
    def generate(task="walk", mode="all", dataset_type="real", **kwargs):
        """
         Returns a HumanoidMuscle4Ages environment corresponding to the specified task.

        Args:
            task (str):
                Main task to solve. Either "walk" or "run".
            mode (str):
                Mode of the environment. Either "all" (sample between all humanoid envs), "1"
                (smallest humanoid), "2" (second smallest humanoid), "3" (teenage humanoid), and "4" (adult humanoid).
            dataset_type (str):
                "real" or "perfect". "real" uses real motion capture data as the
                reference trajectory. This data does not perfectly match the kinematics
                and dynamics of this environment, hence it is more challenging. "perfect" uses
                a perfect dataset.

        Returns:
            An MDP of a set of Muscle Humanoid of different sizes.
         """

        check_validity_task_mode_dataset(HumanoidMuscle4Ages.__name__, task, mode, dataset_type,
                                         *HumanoidMuscle4Ages.valid_task_confs.get_all())

        if task == "walk":
            if dataset_type == "real":
                path = "datasets/humanoids/real/02-constspeed_reduced_humanoid_POMDP"
        elif task == "run":
            if dataset_type == "real":
                path = "datasets/humanoids/real/05-run_reduced_humanoid_POMDP"

        return BaseHumanoid4Ages.generate(HumanoidMuscle4Ages, path, task, mode, dataset_type, **kwargs)
