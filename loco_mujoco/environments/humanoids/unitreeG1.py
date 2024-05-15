from pathlib import Path
from copy import deepcopy
from dm_control import mjcf

from mushroom_rl.utils.running_stats import *
from mushroom_rl.utils.mujoco import *

from loco_mujoco.environments.humanoids.base_robot_humanoid import BaseRobotHumanoid
from loco_mujoco.utils import check_validity_task_mode_dataset
from loco_mujoco.environments import ValidTaskConf


class UnitreeG1(BaseRobotHumanoid):

    """
    Description
    ------------

    Mujoco simulation of the Unitree G1 robot.

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

    | For walking task: :code:`(min=-inf, max=inf, dim=32, dtype=float32)`
    | For running task: :code:`(min=-inf, max=inf, dim=32, dtype=float32)`

    Some observations are **disabled by default**, but can be turned on. The detailed observation space is:

    ===== ============================================= ===== ==== =========================== === ========================
    Index Description                                   Min   Max  Disabled                    Dim Units
    ===== ============================================= ===== ==== =========================== === ========================
    0     Position of Joint pelvis_ty                   -inf  inf  False                       1   Position [m]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    1     Position of Joint pelvis_tilt                 -inf  inf  False                       1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    2     Position of Joint pelvis_list                 -inf  inf  False                       1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    3     Position of Joint pelvis_rotation             -inf  inf  False                       1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    4     Position of Joint back_bkz                    -2.35 2.35 False                       1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    5     Position of Joint l_arm_shy                   -2.87 2.87 True                        1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    6     Position of Joint l_arm_shx                   -0.34 3.11 True                        1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    7     Position of Joint l_arm_shz                   -1.3  4.45 True                        1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    8     Position of Joint left_elbow                  -1.25 2.61 True                        1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    9     Position of Joint r_arm_shy                   -2.87 2.87 True                        1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    10    Position of Joint r_arm_shx                   -3.11 0.34 True                        1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    11    Position of Joint r_arm_shz                   -4.45 1.3  True                        1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    12    Position of Joint right_elbow                 -1.25 2.61 True                        1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    13    Position of Joint hip_flexion_r               -1.57 1.57 False                       1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    14    Position of Joint hip_adduction_r             -0.43 0.43 False                       1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    15    Position of Joint hip_rotation_r              -0.43 0.43 False                       1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    16    Position of Joint knee_angle_r                -0.26 2.05 False                       1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    17    Position of Joint ankle_angle_r               -0.87 0.52 False                       1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    18    Position of Joint hip_flexion_l               -1.57 1.57 False                       1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    19    Position of Joint hip_adduction_l             -0.43 0.43 False                       1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    20    Position of Joint hip_rotation_l              -0.43 0.43 False                       1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    21    Position of Joint knee_angle_l                -0.26 2.05 False                       1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    22    Position of Joint ankle_angle_l               -0.87 0.52 False                       1   Angle [rad]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    23    Velocity of Joint pelvis_tx                   -inf  inf  False                       1   Velocity [m/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    24    Velocity of Joint pelvis_tz                   -inf  inf  False                       1   Velocity [m/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    25    Velocity of Joint pelvis_ty                   -inf  inf  False                       1   Velocity [m/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    26    Velocity of Joint pelvis_tilt                 -inf  inf  False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    27    Velocity of Joint pelvis_list                 -inf  inf  False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    28    Velocity of Joint pelvis_rotation             -inf  inf  False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    29    Velocity of Joint back_bkz                    -inf  inf  False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    30    Velocity of Joint l_arm_shy                   -inf  inf  True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    31    Velocity of Joint l_arm_shx                   -inf  inf  True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    32    Velocity of Joint l_arm_shz                   -inf  inf  True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    33    Velocity of Joint left_elbow                  -inf  inf  True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    34    Velocity of Joint r_arm_shy                   -inf  inf  True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    35    Velocity of Joint r_arm_shx                   -inf  inf  True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    36    Velocity of Joint r_arm_shz                   -inf  inf  True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    37    Velocity of Joint right_elbow                 -inf  inf  True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    38    Velocity of Joint hip_flexion_r               -inf  inf  False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    39    Velocity of Joint hip_adduction_r             -inf  inf  False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    40    Velocity of Joint hip_rotation_r              -inf  inf  False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    41    Velocity of Joint knee_angle_r                -inf  inf  False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    42    Velocity of Joint ankle_angle_r               -inf  inf  False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    43    Velocity of Joint hip_flexion_l               -inf  inf  False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    44    Velocity of Joint hip_adduction_l             -inf  inf  False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    45    Velocity of Joint hip_rotation_l              -inf  inf  False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    46    Velocity of Joint knee_angle_l                -inf  inf  False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    47    Velocity of Joint ankle_angle_l               -inf  inf  False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    48    Mass of the Weight                            0.0   inf  Only Enabled for Carry Task 1   Mass [kg]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    49    3D linear Forces between Right Foot and Floor 0.0   inf  True                        3   Force [N]
    ----- --------------------------------------------- ----- ---- --------------------------- --- ------------------------
    52    3D linear Forces between Left Foot and Floor  0.0   inf  True                        3   Force [N]
    ===== ============================================= ===== ==== =========================== === ========================

    Action Space
    ------------

    | The action space has the following properties *by default* (i.e., only actions with Disabled == False):
    | :code:`(min=-1, max=1, dim=11, dtype=float32)`

    Some actions are **disabled by default**, but can be turned on. The detailed action space is:

    ===== ======================== =========== =========== ========
    Index Name in XML              Control Min Control Max Disabled
    ===== ======================== =========== =========== ========
    0     back_bkz_actuator        -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    1     l_arm_shy_actuator       -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    2     l_arm_shx_actuator       -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    3     l_arm_shz_actuator       -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    4     left_elbow_actuator      -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    5     r_arm_shy_actuator       -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    6     r_arm_shx_actuator       -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    7     r_arm_shz_actuator       -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    8     right_elbow_actuator     -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    9     hip_flexion_r_actuator   -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    10    hip_adduction_r_actuator -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    11    hip_rotation_r_actuator  -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    12    knee_angle_r_actuator    -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    13    ankle_angle_r_actuator   -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    14    hip_flexion_l_actuator   -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    15    hip_adduction_l_actuator -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    16    hip_rotation_l_actuator  -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    17    knee_angle_l_actuator    -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    18    ankle_angle_l_actuator   -1.0        1.0         False
    ===== ======================== =========== =========== ========


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

    The terminal state is reached when the robot falls, or rather starts falling. The condition to check if the robot
    is falling is based on the orientation of the robot, the height of the center of mass, and the orientation of the
    back joint. More details can be found in the  :code:`_has_fallen` method of the environment.

    Methods
    ------------

    """

    valid_task_confs = ValidTaskConf(tasks=["walk", "run"],
                                     data_types=["real"])

    def __init__(self, disable_arms=True, disable_back_joint=False, **kwargs):
        """
        Constructor.

        """

        xml_path = (Path(__file__).resolve().parent.parent / "data" / "unitree_g1" / "g1.xml").as_posix()
        xml_handle = mjcf.from_path(xml_path)

        self._hold_weight = False   # no weights supported with this envs

        action_spec = self._get_action_specification(xml_handle)

        observation_spec = self._get_observation_specification(xml_handle)

        collision_groups = [("floor", ["floor"]),
                            ("right_foot_1", ["right_foot_1_col"]),
                            ("right_foot_2", ["right_foot_2_col"]),
                            ("right_foot_3", ["right_foot_3_col"]),
                            ("right_foot_4", ["right_foot_4_col"]),
                            ("left_foot_1", ["left_foot_1_col"]),
                            ("left_foot_2", ["left_foot_2_col"]),
                            ("left_foot_3", ["left_foot_3_col"]),
                            ("left_foot_4", ["left_foot_4_col"])]

        self._hidable_obs = ("positions", "velocities", "foot_forces", "weight")

        # --- Modify the xml, the action_spec, and the observation_spec if needed ---
        self._disable_arms = disable_arms
        self._disable_back_joint = disable_back_joint

        if disable_arms or disable_back_joint:
            joints_to_remove, motors_to_remove, equ_constr_to_remove = self._get_xml_modifications()
            obs_to_remove = ["q_" + j for j in joints_to_remove] + ["dq_" + j for j in joints_to_remove]
            observation_spec = [elem for elem in observation_spec if elem[0] not in obs_to_remove]
            action_spec = [ac for ac in action_spec if ac not in motors_to_remove]

            xml_handle = self._delete_from_xml_handle(xml_handle, joints_to_remove,
                                                      motors_to_remove, equ_constr_to_remove)
            if disable_arms:
                xml_handle = self._reorient_arms(xml_handle)

        xml_handles = [xml_handle]

        super().__init__(xml_handles, action_spec, observation_spec, collision_groups, **kwargs)

    def _get_ground_forces(self):
        """
        Returns the ground forces (np.array). By default, 4 ground force sensors are used.
        Environments that use more or less have to override this function.

        """

        grf = np.concatenate([self._get_collision_force("floor", "right_foot_1_col")[:3],
                              self._get_collision_force("floor", "right_foot_2_col")[:3],
                              self._get_collision_force("floor", "right_foot_3_col")[:3],
                              self._get_collision_force("floor", "right_foot_4_col")[:3],
                              self._get_collision_force("floor", "left_foot_1_col")[:3],
                              self._get_collision_force("floor", "left_foot_2_col")[:3],
                              self._get_collision_force("floor", "left_foot_3_col")[:3],
                              self._get_collision_force("floor", "left_foot_4_col")[:3]])

        return grf

    @staticmethod
    def _get_grf_size():
        """
        Returns the size of the ground force vector.

        """

        return 24

    def _get_xml_modifications(self):
        """
        Function that specifies which joints, motors and equality constraints
        should be removed from the Mujoco xml.

        Returns:
            A tuple of lists consisting of names of joints to remove, names of motors to remove,
             and names of equality constraints to remove.

        """

        joints_to_remove = []
        motors_to_remove = []
        equ_constr_to_remove = []

        if self._disable_arms:
            joints_to_remove += ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                                 "right_elbow_pitch_joint", "right_elbow_roll_joint",
                                 "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                                 "left_elbow_pitch_joint", "left_elbow_roll_joint"]
            motors_to_remove += ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                                 "right_elbow_pitch_joint", "right_elbow_roll_joint",
                                 "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                                 "left_elbow_pitch_joint", "left_elbow_roll_joint"]

        if self._disable_back_joint:
            joints_to_remove += ["torso_joint"]
            motors_to_remove += ["torso_joint"]

        return joints_to_remove, motors_to_remove, equ_constr_to_remove

    def _has_fallen(self, obs, return_err_msg=False):
        """
        Checks if a model has fallen.

        Args:
            obs (np.array): Current observation.
            return_err_msg (bool): If True, an error message with violations is returned.

        Returns:
            True, if the model has fallen for the current observation, False otherwise.
            Optionally an error message is returned.

        """

        pelvis_euler = self._get_from_obs(obs, ["q_pelvis_tilt", "q_pelvis_list", "q_pelvis_rotation"])
        pelvis_y_condition = (obs[0] < -0.3) or (obs[0] > 0.1)
        pelvis_tilt_condition = (pelvis_euler[0] < (-np.pi / 4.5)) or (pelvis_euler[0] > (np.pi / 12))
        pelvis_list_condition = (pelvis_euler[1] < -np.pi / 12) or (pelvis_euler[1] > np.pi / 8)
        pelvis_rotation_condition = (pelvis_euler[2] < (-np.pi / 8)) or (pelvis_euler[2] > (np.pi / 8))
        pelvis_condition = (pelvis_y_condition or pelvis_tilt_condition or
                            pelvis_list_condition or pelvis_rotation_condition)

        if return_err_msg:
            error_msg = ""
            if pelvis_y_condition:
                error_msg += "pelvis_y_condition violated.\n"
            elif pelvis_tilt_condition:
                error_msg += "pelvis_tilt_condition violated.\n"
            elif pelvis_list_condition:
                error_msg += "pelvis_list_condition violated.\n"
            elif pelvis_rotation_condition:
                error_msg += "pelvis_rotation_condition violated.\n"

            return pelvis_condition, error_msg
        else:

            return pelvis_condition

    @staticmethod
    def generate(task="walk", dataset_type="real", **kwargs):
        """
        Returns an environment corresponding to the specified task.

        Args:
        task (str): Main task to solve. Either "walk", "run" or "carry". The latter is walking while carrying
        an unknown weight, which makes the task partially observable.
        dataset_type (str): "real" or "perfect". "real" uses real motion capture data as the
        reference trajectory. This data does not perfectly match the kinematics
        and dynamics of this environment, hence it is more challenging. "perfect" uses
        a perfect dataset.

        """
        check_validity_task_mode_dataset(UnitreeG1.__name__, task, None, dataset_type,
                                         *UnitreeG1.valid_task_confs.get_all())
        if dataset_type == "real":
            if task == "run":
                path = "datasets/humanoids/real/05-run_UnitreeG1.npz"
            else:
                path = "datasets/humanoids/real/02-constspeed_UnitreeG1.npz"

        return BaseRobotHumanoid.generate(UnitreeG1, path, task, dataset_type,
                                          clip_trajectory_to_joint_ranges=True, **kwargs)

    @staticmethod
    def _reorient_arms(xml_handle):
        """
        Reorients the elbow to not collide with the hip.

        Args:
            xml_handle: Handle to Mujoco XML.

        Returns:
            Modified Mujoco XML handle.

        """
        # modify the arm orientation
        left_shoulder_pitch_link = xml_handle.find("body", "left_shoulder_pitch_link")
        left_shoulder_pitch_link.quat = [1.0, 0.25, 0.1, 0.0]
        right_elbow_link = xml_handle.find("body", "right_elbow_pitch_link")
        right_elbow_link.quat = [1.0, 0.0, 0.25, 0.0]
        right_shoulder_pitch_link = xml_handle.find("body", "right_shoulder_pitch_link")
        right_shoulder_pitch_link.quat = [1.0, -0.25, 0.1, 0.0]
        left_elbow_link = xml_handle.find("body", "left_elbow_pitch_link")
        left_elbow_link.quat = [1.0, 0.0, 0.25, 0.0]

        return xml_handle

    @staticmethod
    def _get_observation_specification(xml_handle):
        """
        Getter for the observation space specification.

        Returns:
            A list of tuples containing the specification of each observation
            space entry.

        """
        observation_spec = []
        for prefix in ["q_", "dq_"]:
            for j in xml_handle.find_all("joint"):
                obs_type = ObservationType.JOINT_POS if prefix == "q_" else ObservationType.JOINT_VEL
                observation_spec.append((prefix + j.name, j.name, obs_type))
        return observation_spec

    @staticmethod
    def _get_action_specification(xml_handle):
        """
        Getter for the action space specification.

        Returns:
            A list of tuples containing the specification of each action
            space entry.

        """
        action_spec = []
        actuators = xml_handle.find_all("actuator")
        for actuator in actuators:
            action_spec.append(actuator.name)
        return action_spec