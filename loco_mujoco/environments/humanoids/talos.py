from pathlib import Path
from copy import deepcopy
from dm_control import mjcf

from mushroom_rl.utils.running_stats import *
from mushroom_rl.utils.mujoco import *

from loco_mujoco.environments.humanoids.base_robot_humanoid import BaseRobotHumanoid
from loco_mujoco.utils import check_validity_task_mode_dataset
from loco_mujoco.environments import ValidTaskConf


class Talos(BaseRobotHumanoid):

    """
    Description
    ------------

    Mujoco environment of the Talos robot. Optionally, Talos can carry
    a weight. This environment can be partially observable by hiding
    some of the state space entries from the policy using a state mask.
    Hidable entries are "positions", "velocities", "foot_forces",
    or "weight".

    Tasks
    -----------------
    * **Walking**: The robot has to walk forward with a fixed speed of 1.25 m/s.
    * **Carry**: The robot has to walk forward with a fixed speed of 1.25 m/s while carrying a weight.
      The mass is either specified by the user or sampled from a uniformly from [0.1 kg, 1 kg, 5 kg, 10 kg].


    Dataset Types
    -----------------
    The available dataset types for this environment can be found at: :ref:`env-label`.


    Observation Space
    -----------------

    The observation space has the following properties *by default* (i.e., only obs with Disabled == False):

    | For walking task: :code:`(min=-inf, max=inf, dim=34, dtype=float32)`
    | For carry task: :code:`(min=-inf, max=inf, dim=35, dtype=float32)`

    Some observations are **disabled by default**, but can be turned on. The detailed observation space is:

    ===== ============================================= ========== =========== =========================== === ========================
    Index Description                                   Min        Max         Disabled                    Dim Units
    ===== ============================================= ========== =========== =========================== === ========================
    0     Position of Joint pelvis_ty                   -inf       inf         False                       1   Position [m]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    1     Position of Joint pelvis_tilt                 -inf       inf         False                       1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    2     Position of Joint pelvis_list                 -inf       inf         False                       1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    3     Position of Joint pelvis_rotation             -inf       inf         False                       1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    4     Position of Joint back_bkz                    -1.25664   1.25664     False                       1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    5     Position of Joint back_bky                    -0.226893  0.733038    False                       1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    6     Position of Joint l_arm_shz                   -1.5708    0.785398    True                        1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    7     Position of Joint l_arm_shx                   0.00872665 2.87107     True                        1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    8     Position of Joint l_arm_ely                   -2.42601   2.42601     True                        1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    9     Position of Joint l_arm_elx                   -2.23402   0.00349066  True                        1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    10    Position of Joint l_arm_wry                   -2.51327   2.51327     True                        1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    11    Position of Joint l_arm_wrx                   -1.37008   1.37008     True                        1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    12    Position of Joint r_arm_shz                   -0.785398  1.5708      True                        1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    13    Position of Joint r_arm_shx                   -2.87107   -0.00872665 True                        1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    14    Position of Joint r_arm_ely                   -2.42601   2.42601     True                        1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    15    Position of Joint r_arm_elx                   -2.23402   0.00349066  True                        1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    16    Position of Joint r_arm_wry                   -2.51327   2.51327     True                        1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    17    Position of Joint r_arm_wrx                   -1.37008   1.37008     True                        1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    18    Position of Joint hip_flexion_r               -2.095     0.7         False                       1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    19    Position of Joint hip_adduction_r             -0.5236    0.5236      False                       1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    20    Position of Joint hip_rotation_r              -1.5708    0.349066    False                       1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    21    Position of Joint knee_angle_r                0.0        2.618       False                       1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    22    Position of Joint ankle_angle_r               -1.27      0.68        False                       1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    23    Position of Joint hip_flexion_l               -2.095     0.7         False                       1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    24    Position of Joint hip_adduction_l             -0.5236    0.5236      False                       1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    25    Position of Joint hip_rotation_l              -0.349066  1.5708      False                       1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    26    Position of Joint knee_angle_l                0.0        2.618       False                       1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    27    Position of Joint ankle_angle_l               -1.27      0.68        False                       1   Angle [rad]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    28    Velocity of Joint pelvis_tx                   -inf       inf         False                       1   Velocity [m/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    29    Velocity of Joint pelvis_tz                   -inf       inf         False                       1   Velocity [m/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    30    Velocity of Joint pelvis_ty                   -inf       inf         False                       1   Velocity [m/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    31    Velocity of Joint pelvis_tilt                 -inf       inf         False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    32    Velocity of Joint pelvis_list                 -inf       inf         False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    33    Velocity of Joint pelvis_rotation             -inf       inf         False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    34    Velocity of Joint back_bkz                    -inf       inf         False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    35    Velocity of Joint back_bky                    -inf       inf         False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    36    Velocity of Joint l_arm_shz                   -inf       inf         True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    37    Velocity of Joint l_arm_shx                   -inf       inf         True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    38    Velocity of Joint l_arm_ely                   -inf       inf         True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    39    Velocity of Joint l_arm_elx                   -inf       inf         True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    40    Velocity of Joint l_arm_wry                   -inf       inf         True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    41    Velocity of Joint l_arm_wrx                   -inf       inf         True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    42    Velocity of Joint r_arm_shz                   -inf       inf         True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    43    Velocity of Joint r_arm_shx                   -inf       inf         True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    44    Velocity of Joint r_arm_ely                   -inf       inf         True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    45    Velocity of Joint r_arm_elx                   -inf       inf         True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    46    Velocity of Joint r_arm_wry                   -inf       inf         True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    47    Velocity of Joint r_arm_wrx                   -inf       inf         True                        1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    48    Velocity of Joint hip_flexion_r               -inf       inf         False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    49    Velocity of Joint hip_adduction_r             -inf       inf         False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    50    Velocity of Joint hip_rotation_r              -inf       inf         False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    51    Velocity of Joint knee_angle_r                -inf       inf         False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    52    Velocity of Joint ankle_angle_r               -inf       inf         False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    53    Velocity of Joint hip_flexion_l               -inf       inf         False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    54    Velocity of Joint hip_adduction_l             -inf       inf         False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    55    Velocity of Joint hip_rotation_l              -inf       inf         False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    56    Velocity of Joint knee_angle_l                -inf       inf         False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    57    Velocity of Joint ankle_angle_l               -inf       inf         False                       1   Angular Velocity [rad/s]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    58    Mass of the Weight                            0.0        inf         Only Enabled for Carry Task 1   Mass [kg]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    59    3D linear Forces between Right Foot and Floor 0.0        inf         True                        3   Force [N]
    ----- --------------------------------------------- ---------- ----------- --------------------------- --- ------------------------
    62    3D linear Forces between Left Foot and Floor  0.0        inf         True                        3   Force [N]
    ===== ============================================= ========== =========== =========================== === ========================

    Action Space
    ------------

    | The action space has the following properties *by default* (i.e., only actions with Disabled == False):
    | :code:`(min=-1, max=1, dim=12, dtype=float32)`

    The action range in LocoMuJoCo is always standardized, i.e. in [-1.0, 1.0].
    The XML of the environment specifies for each actuator a *gearing* ratio, which is used to scale the
    the action to the actual control range of the actuator.

    Some actions are **disabled by default**, but can be turned on. The detailed action space is:

    ===== ======================== =========== =========== ========
    Index Name in XML              Control Min Control Max Disabled
    ===== ======================== =========== =========== ========
    0     back_bkz_actuator        -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    1     back_bky_actuator        -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    2     l_arm_shz_actuator       -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    3     l_arm_shx_actuator       -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    4     l_arm_ely_actuator       -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    5     l_arm_elx_actuator       -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    6     l_arm_wry_actuator       -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    7     l_arm_wrx_actuator       -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    8     r_arm_shz_actuator       -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    9     r_arm_shx_actuator       -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    10    r_arm_ely_actuator       -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    11    r_arm_elx_actuator       -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    12    r_arm_wry_actuator       -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    13    r_arm_wrx_actuator       -1.0        1.0         True
    ----- ------------------------ ----------- ----------- --------
    14    hip_flexion_r_actuator   -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    15    hip_adduction_r_actuator -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    16    hip_rotation_r_actuator  -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    17    knee_angle_r_actuator    -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    18    ankle_angle_r_actuator   -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    19    hip_flexion_l_actuator   -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    20    hip_adduction_l_actuator -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    21    hip_rotation_l_actuator  -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    22    knee_angle_l_actuator    -1.0        1.0         False
    ----- ------------------------ ----------- ----------- --------
    23    ankle_angle_l_actuator   -1.0        1.0         False
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

    valid_task_confs = ValidTaskConf(tasks=["walk", "carry"],
                                     data_types=["real", "perfect"],
                                     non_combinable=[("carry", None, "perfect")])

    def __init__(self, disable_arms=True, disable_back_joint=False, hold_weight=False,
                 weight_mass=None, **kwargs):
        """
        Constructor.

        """

        if hold_weight:
            assert disable_arms is True, "If you want Talos to carry a weight, please disable the arms. " \
                                         "They will be kept fixed."

        xml_path = (Path(__file__).resolve().parent.parent / "data" / "talos" / "talos.xml").as_posix()

        action_spec = self._get_action_specification()

        observation_spec = self._get_observation_specification()

        collision_groups = [("floor", ["floor"]),
                            ("foot_r", ["right_foot"]),
                            ("foot_l", ["left_foot"])]

        self._hidable_obs = ("positions", "velocities", "foot_forces", "weight")

        # --- Modify the xml, the action_spec, and the observation_spec if needed ---
        self._disable_arms = disable_arms
        self._disable_back_joint = disable_back_joint
        self._hold_weight = hold_weight
        self._weight_mass = weight_mass
        self._valid_weights = [0.1, 1.0, 5.0, 10.0]

        xml_handle = mjcf.from_path(xml_path)
        xml_handles = []

        if disable_arms or hold_weight:

            if disable_arms or disable_back_joint:
                joints_to_remove, motors_to_remove, equ_constr_to_remove = self._get_xml_modifications()
                obs_to_remove = ["q_" + j for j in joints_to_remove] + ["dq_" + j for j in joints_to_remove]
                observation_spec = [elem for elem in observation_spec if elem[0] not in obs_to_remove]
                action_spec = [ac for ac in action_spec if ac not in motors_to_remove]

                xml_handle = self._delete_from_xml_handle(xml_handle, joints_to_remove,
                                                          motors_to_remove, equ_constr_to_remove)

            if hold_weight and weight_mass is not None:
                color_red = np.array([1.0, 0.0, 0.0, 1.0])
                xml_handle = self._add_weight(xml_handle, weight_mass, color_red)
                xml_handles.append(xml_handle)
            elif hold_weight and weight_mass is None:
                for i, w in enumerate(self._valid_weights):
                    color = self._get_box_color(i)
                    current_xml_handle = deepcopy(xml_handle)
                    current_xml_handle = self._add_weight(current_xml_handle, w, color)
                    xml_handles.append(current_xml_handle)
            else:
                xml_handle = self._reorient_arms(xml_handle)
                xml_handles.append(xml_handle)
        else:
            xml_handles.append(xml_handle)

        super().__init__(xml_handles, action_spec, observation_spec, collision_groups, **kwargs)

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
            joints_to_remove += ["l_arm_shz", "l_arm_shx", "l_arm_ely", "l_arm_elx", "l_arm_wry", "l_arm_wrx",
                                 "r_arm_shz", "r_arm_shx", "r_arm_ely", "r_arm_elx", "r_arm_wry", "r_arm_wrx"]
            motors_to_remove += ["l_arm_shz_actuator", "l_arm_shx_actuator", "l_arm_ely_actuator", "l_arm_elx_actuator",
                                 "l_arm_wry_actuator", "l_arm_wrx_actuator", "r_arm_shz_actuator", "r_arm_shx_actuator",
                                 "r_arm_ely_actuator", "r_arm_elx_actuator", "r_arm_wry_actuator", "r_arm_wrx_actuator"]

        if self._disable_back_joint:
            joints_to_remove += ["back_bkz", "back_bky"]
            motors_to_remove += ["back_bkz_actuator", "back_bky_actuator"]

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
        pelvis_rotation_condition = (pelvis_euler[2] < (-np.pi / 10)) or (pelvis_euler[2] > (np.pi / 10))
        pelvis_condition = (pelvis_y_condition or pelvis_tilt_condition or
                            pelvis_list_condition or pelvis_rotation_condition)

        if not self._disable_back_joint:
            back_euler = self._get_from_obs(obs, ["q_back_bky", "q_back_bkz"])

            back_extension_condition = (back_euler[0] < (-np.pi / 4)) or (back_euler[0] > (np.pi / 10))
            back_rotation_condition = (back_euler[1] < -np.pi / 10) or (back_euler[1] > np.pi / 10)
            back_condition = (back_extension_condition or back_rotation_condition)
        else:
            back_condition = back_extension_condition = back_rotation_condition = False

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
            elif back_extension_condition:
                error_msg += "back_extension_condition violated.\n"
            elif back_rotation_condition:
                error_msg += "back_rotation_condition violated.\n"

            return pelvis_condition or back_condition, error_msg
        else:

            return pelvis_condition or back_condition

    def _get_ground_forces(self):
        """
        Returns the ground forces (np.array). By default, 4 ground force sensors are used.
        Environments that use more or less have to override this function.

        """

        grf = np.concatenate([self._get_collision_force("floor", "foot_r")[:3],
                              self._get_collision_force("floor", "foot_l")[:3]])

        return grf

    @staticmethod
    def _get_grf_size():
        """
        Returns the size of the ground force vector.

        """

        return 6

    @staticmethod
    def generate(task="walk", dataset_type="real", **kwargs):
        """
        Returns an environment corresponding to the specified task.

        Args:
            task (str):
                Main task to solve. Either "walk" or "carry". The latter is walking while carrying
                an unknown weight, which makes the task partially observable.
            dataset_type (str):
                "real" or "perfect". "real" uses real motion capture data as the
                reference trajectory. This data does not perfectly match the kinematics
                and dynamics of this environment, hence it is more challenging. "perfect" uses
                a perfect dataset.

        """
        if "disable_arms" in kwargs.keys():
            assert kwargs["disable_arms"] is True,\
                "Activating the arms in the Talos environment is currently not supported."

        check_validity_task_mode_dataset(Talos.__name__, task, None, dataset_type,
                                         *Talos.valid_task_confs.get_all())

        if dataset_type == "real":
            path = "datasets/humanoids/real/02-constspeed_TALOS.npz"
        elif dataset_type == "perfect":
            if "use_foot_forces" in kwargs.keys():
                assert kwargs["use_foot_forces"] is False
            if "disable_arms" in kwargs.keys():
                assert kwargs["disable_arms"] is True
            if "disable_back_joint" in kwargs.keys():
                assert kwargs["disable_back_joint"] is False
            if "hold_weight" in kwargs.keys():
                assert kwargs["hold_weight"] is False

            path = "datasets/humanoids/perfect/talos_walk/perfect_expert_dataset_det.npz"

        return BaseRobotHumanoid.generate(Talos, path, task, dataset_type,
                                          clip_trajectory_to_joint_ranges=True, **kwargs)

    @staticmethod
    def _add_weight(xml_handle, mass, color):
        """
        Adds a weight to the Mujoco XML handle. The weight will
        be hold in front of Talos. Therefore, the arms will be
        reoriented.

        Args:
            xml_handle: Handle to Mujoco XML.

        Returns:
            Modified Mujoco XML handle.

        """

        # find pelvis handle
        pelvis = xml_handle.find("body", "torso_2_link")
        pelvis.add("body", name="weight")
        weight = xml_handle.find("body", "weight")
        weight.add("geom", type="box", size="0.1 0.25 0.1", pos="0.45 0 -0.20", group="0", rgba=color, mass=mass)

        # modify the arm orientation
        arm_right_4_link = xml_handle.find("body", "arm_right_4_link")
        arm_right_4_link.quat = [1.0,  0.0, -0.65, 0.0]
        arm_left_4_link = xml_handle.find("body", "arm_left_4_link")
        arm_left_4_link.quat = [1.0,  0.0, -0.65, 0.0]

        arm_right_6_link = xml_handle.find("body", "arm_right_6_link")
        arm_right_6_link.quat = [1.0,  0.0, -0.0, 1.0]
        arm_left_6_link = xml_handle.find("body", "arm_left_6_link")
        arm_left_6_link.quat = [1.0,  0.0, -0.0, 1.0]

        return xml_handle

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
        arm_right_4_link = xml_handle.find("body", "arm_right_4_link")
        arm_right_4_link.quat = [1.0,  0.0, -0.25, 0.0]
        arm_left_4_link = xml_handle.find("body", "arm_left_4_link")
        arm_left_4_link.quat = [1.0,  0.0, -0.25, 0.0]

        return xml_handle

    @staticmethod
    def _get_observation_specification():
        """
        Getter for the observation space specification.

        Returns:
            A list of tuples containing the specification of each observation
            space entry.

        """

        observation_spec = [# ------------- JOINT POS -------------
                            ("q_pelvis_tx", "pelvis_tx", ObservationType.JOINT_POS),
                            ("q_pelvis_tz", "pelvis_tz", ObservationType.JOINT_POS),
                            ("q_pelvis_ty", "pelvis_ty", ObservationType.JOINT_POS),
                            ("q_pelvis_tilt", "pelvis_tilt", ObservationType.JOINT_POS),
                            ("q_pelvis_list", "pelvis_list", ObservationType.JOINT_POS),
                            ("q_pelvis_rotation", "pelvis_rotation", ObservationType.JOINT_POS),
                            ("q_back_bkz", "back_bkz", ObservationType.JOINT_POS),
                            ("q_back_bky", "back_bky", ObservationType.JOINT_POS),
                            ("q_l_arm_shz", "l_arm_shz", ObservationType.JOINT_POS),
                            ("q_l_arm_shx", "l_arm_shx", ObservationType.JOINT_POS),
                            ("q_l_arm_ely", "l_arm_ely", ObservationType.JOINT_POS),
                            ("q_l_arm_elx", "l_arm_elx", ObservationType.JOINT_POS),
                            ("q_l_arm_wry", "l_arm_wry", ObservationType.JOINT_POS),
                            ("q_l_arm_wrx", "l_arm_wrx", ObservationType.JOINT_POS),
                            ("q_r_arm_shz", "r_arm_shz", ObservationType.JOINT_POS),
                            ("q_r_arm_shx", "r_arm_shx", ObservationType.JOINT_POS),
                            ("q_r_arm_ely", "r_arm_ely", ObservationType.JOINT_POS),
                            ("q_r_arm_elx", "r_arm_elx", ObservationType.JOINT_POS),
                            ("q_r_arm_wry", "r_arm_wry", ObservationType.JOINT_POS),
                            ("q_r_arm_wrx", "r_arm_wrx", ObservationType.JOINT_POS),
                            ("q_hip_flexion_r", "hip_flexion_r", ObservationType.JOINT_POS),
                            ("q_hip_adduction_r", "hip_adduction_r", ObservationType.JOINT_POS),
                            ("q_hip_rotation_r", "hip_rotation_r", ObservationType.JOINT_POS),
                            ("q_knee_angle_r", "knee_angle_r", ObservationType.JOINT_POS),
                            ("q_ankle_angle_r", "ankle_angle_r", ObservationType.JOINT_POS),
                            ("q_hip_flexion_l", "hip_flexion_l", ObservationType.JOINT_POS),
                            ("q_hip_adduction_l", "hip_adduction_l", ObservationType.JOINT_POS),
                            ("q_hip_rotation_l", "hip_rotation_l", ObservationType.JOINT_POS),
                            ("q_knee_angle_l", "knee_angle_l", ObservationType.JOINT_POS),
                            ("q_ankle_angle_l", "ankle_angle_l", ObservationType.JOINT_POS),

                            # ------------- JOINT VEL -------------
                            ("dq_pelvis_tx", "pelvis_tx", ObservationType.JOINT_VEL),
                            ("dq_pelvis_tz", "pelvis_tz", ObservationType.JOINT_VEL),
                            ("dq_pelvis_ty", "pelvis_ty", ObservationType.JOINT_VEL),
                            ("dq_pelvis_tilt", "pelvis_tilt", ObservationType.JOINT_VEL),
                            ("dq_pelvis_list", "pelvis_list", ObservationType.JOINT_VEL),
                            ("dq_pelvis_rotation", "pelvis_rotation", ObservationType.JOINT_VEL),
                            ("dq_back_bkz", "back_bkz", ObservationType.JOINT_VEL),
                            ("dq_back_bky", "back_bky", ObservationType.JOINT_VEL),
                            ("dq_l_arm_shz", "l_arm_shz", ObservationType.JOINT_VEL),
                            ("dq_l_arm_shx", "l_arm_shx", ObservationType.JOINT_VEL),
                            ("dq_l_arm_ely", "l_arm_ely", ObservationType.JOINT_VEL),
                            ("dq_l_arm_elx", "l_arm_elx", ObservationType.JOINT_VEL),
                            ("dq_l_arm_wry", "l_arm_wry", ObservationType.JOINT_VEL),
                            ("dq_l_arm_wrx", "l_arm_wrx", ObservationType.JOINT_VEL),
                            ("dq_r_arm_shz", "r_arm_shz", ObservationType.JOINT_VEL),
                            ("dq_r_arm_shx", "r_arm_shx", ObservationType.JOINT_VEL),
                            ("dq_r_arm_ely", "r_arm_ely", ObservationType.JOINT_VEL),
                            ("dq_r_arm_elx", "r_arm_elx", ObservationType.JOINT_VEL),
                            ("dq_r_arm_wry", "r_arm_wry", ObservationType.JOINT_VEL),
                            ("dq_r_arm_wrx", "r_arm_wrx", ObservationType.JOINT_VEL),
                            ("dq_hip_flexion_r", "hip_flexion_r", ObservationType.JOINT_VEL),
                            ("dq_hip_adduction_r", "hip_adduction_r", ObservationType.JOINT_VEL),
                            ("dq_hip_rotation_r", "hip_rotation_r", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_r", "knee_angle_r", ObservationType.JOINT_VEL),
                            ("dq_ankle_angle_r", "ankle_angle_r", ObservationType.JOINT_VEL),
                            ("dq_hip_flexion_l", "hip_flexion_l", ObservationType.JOINT_VEL),
                            ("dq_hip_adduction_l", "hip_adduction_l", ObservationType.JOINT_VEL),
                            ("dq_hip_rotation_l", "hip_rotation_l", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_l", "knee_angle_l", ObservationType.JOINT_VEL),
                            ("dq_ankle_angle_l", "ankle_angle_l", ObservationType.JOINT_VEL)]

        return observation_spec

    @staticmethod
    def _get_action_specification():
        """
        Getter for the action space specification.

        Returns:
            A list of tuples containing the specification of each action
            space entry.

        """

        action_spec = ["back_bkz_actuator", "back_bky_actuator", "l_arm_shz_actuator",
                       "l_arm_shx_actuator", "l_arm_ely_actuator", "l_arm_elx_actuator", "l_arm_wry_actuator",
                       "l_arm_wrx_actuator", "r_arm_shz_actuator", "r_arm_shx_actuator",
                       "r_arm_ely_actuator", "r_arm_elx_actuator", "r_arm_wry_actuator", "r_arm_wrx_actuator",
                       "hip_flexion_r_actuator", "hip_adduction_r_actuator", "hip_rotation_r_actuator",
                       "knee_angle_r_actuator", "ankle_angle_r_actuator", "hip_flexion_l_actuator",
                       "hip_adduction_l_actuator", "hip_rotation_l_actuator", "knee_angle_l_actuator",
                       "ankle_angle_l_actuator"]

        return action_spec
