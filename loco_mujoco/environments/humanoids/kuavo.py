from pathlib import Path
from copy import deepcopy
from dm_control import mjcf

from mushroom_rl.utils.running_stats import *
from mushroom_rl.utils.mujoco import *

from loco_mujoco.environments.humanoids.base_robot_humanoid import BaseRobotHumanoid
from loco_mujoco.utils import check_validity_task_mode_dataset
from loco_mujoco.environments import ValidTaskConf


class Kuavo(BaseRobotHumanoid):
    """
    Mujoco simulation of the Kuavo robot. Optionally, the Kuavo can carry
    a weight. This environment can be partially observable by hiding
    some of the state space entries from the policy using a state mask.
    Hidable entries are "positions", "velocities", "foot_forces",
    or "weight".

    """

    valid_task_confs = ValidTaskConf(tasks=["walk", "run", "carry"],
                                     data_types=["real"])

    def __init__(self, disable_arms=True, disable_back_joint=True, hold_weight=False,
                 weight_mass=None, **kwargs):
        """
        Constructor.

        """

        if hold_weight:
            assert disable_arms is True, "If you want Unitree H1 to carry a weight, please disable the arms. " \
                                         "They will be kept fixed."

        xml_path = (Path(__file__).resolve().parent.parent / "data" / "kuavo" / "kuavo.xml").as_posix()

        action_spec = self._get_action_specification()

        observation_spec = self._get_observation_specification()

        collision_groups = [("floor", ["floor"]),
                            ("foot_r", ["r_foot"]),
                            ("foot_l", ["l_foot"])]

        self._hidable_obs = ("positions", "velocities", "foot_forces", "weight")

        # --- Modify the xml, the action_spec, and the observation_spec if needed ---
        self._disable_arms = disable_arms
        self._disable_back_joint = disable_back_joint
        self._hold_weight = hold_weight
        self._weight_mass = weight_mass
        self._valid_weights = [0.1, 1.0, 5.0, 10.0]

        if disable_arms or hold_weight:
            xml_handle = mjcf.from_path(xml_path)

            if disable_arms or disable_back_joint:
                joints_to_remove, motors_to_remove, equ_constr_to_remove = self._get_xml_modifications()
                obs_to_remove = ["q_" + j for j in joints_to_remove] + ["dq_" + j for j in joints_to_remove]
                observation_spec = [elem for elem in observation_spec if elem[0] not in obs_to_remove]
                action_spec = [ac for ac in action_spec if ac not in motors_to_remove]

                xml_handle = self._delete_from_xml_handle(xml_handle, joints_to_remove,
                                                          motors_to_remove, equ_constr_to_remove)
            if disable_arms and not hold_weight:
                xml_handle = self._reorient_arms(xml_handle)

            xml_handles = []
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
                xml_handles.append(xml_handle)

        else:
            xml_handles = mjcf.from_path(xml_path)

        super().__init__(xml_handles, action_spec, observation_spec, collision_groups, **kwargs)

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
            joints_to_remove += ["l_shoulder_y", "l_shoulder_z", "l_shoulder_x", "l_elbow", 
                                 "r_shoulder_y", "r_shoulder_z", "r_shoulder_x", "r_elbow"]
            
            motors_to_remove += ["l_shoulder_y", "l_shoulder_z", "l_shoulder_x", "l_elbow", 
                                 "r_shoulder_y", "r_shoulder_z", "r_shoulder_x", "r_elbow"]
            
        if self._disable_back_joint:
            # kuavo has no back joints
            pass

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
        check_validity_task_mode_dataset(Kuavo.__name__, task, None, dataset_type,
                                         *Kuavo.valid_task_confs.get_all())

        if task == "run":
            path = "datasets/humanoids/05-run_Kuavo.npz"
        else:
            path = "datasets/humanoids/02-constspeed_Kuavo.npz"

        return BaseRobotHumanoid.generate(Kuavo, path, task, dataset_type,
                                          clip_trajectory_to_joint_ranges=True, **kwargs)

    @staticmethod
    def _add_weight(xml_handle, mass, color):
        """
        Adds a weight to the Mujoco XML handle. The weight will
        be hold in front of Unitree H1. Therefore, the arms will be
        reoriented.

        Args:
            xml_handle: Handle to Mujoco XML.

        Returns:
            Modified Mujoco XML handle.

        """
        # find pelvis handle
        pelvis = xml_handle.find("body", "torso")
        pelvis.add("body", name="weight")
        weight = xml_handle.find("body", "weight")
        weight.add("geom", type="box", size="0.1 0.18 0.1", pos="0.35 0 0.1", group="0", rgba=color, mass=mass)

        return xml_handle

    @staticmethod
    def _reorient_arms(xml_handle):
        """TODO:
        Reorients the elbow to not collide with the hip.

        Args:
            xml_handle: Handle to Mujoco XML.

        Returns:
            Modified Mujoco XML handle.

        """
        # modify the arm orientation
        # left_shoulder_pitch_link = xml_handle.find("body", "left_shoulder_pitch_link")
        # left_shoulder_pitch_link.quat = [1.0, 0.25, 0.1, 0.0]
        # right_elbow_link = xml_handle.find("body", "right_elbow_link")
        # right_elbow_link.quat = [1.0, 0.0, 0.25, 0.0]
        # right_shoulder_pitch_link = xml_handle.find("body", "right_shoulder_pitch_link")
        # right_shoulder_pitch_link.quat = [1.0, -0.25, 0.1, 0.0]
        # left_elbow_link = xml_handle.find("body", "left_elbow_link")
        # left_elbow_link.quat = [1.0, 0.0, 0.25, 0.0]

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
                            ("q_l_shoulder_y", "l_shoulder_y", ObservationType.JOINT_POS),
                            ("q_l_shoulder_x", "l_shoulder_x", ObservationType.JOINT_POS),
                            ("q_l_shoulder_z", "l_shoulder_z", ObservationType.JOINT_POS),
                            ("q_l_elbow", "l_elbow", ObservationType.JOINT_POS),
                            ("q_r_shoulder_y", "r_shoulder_y", ObservationType.JOINT_POS),
                            ("q_r_shoulder_x", "r_shoulder_x", ObservationType.JOINT_POS),
                            ("q_r_shoulder_z", "r_shoulder_z", ObservationType.JOINT_POS),
                            ("q_r_elbow", "r_elbow", ObservationType.JOINT_POS),
                            ("q_r_hip_z", "r_hip_z", ObservationType.JOINT_POS),
                            ("q_r_hip_x", "r_hip_x", ObservationType.JOINT_POS),
                            ("q_r_hip_y", "r_hip_y", ObservationType.JOINT_POS),
                            ("q_r_knee", "r_knee", ObservationType.JOINT_POS),
                            ("q_r_ankle", "r_ankle", ObservationType.JOINT_POS),
                            ("q_l_hip_z", "l_hip_z", ObservationType.JOINT_POS),
                            ("q_l_hip_x", "l_hip_x", ObservationType.JOINT_POS),
                            ("q_l_hip_y", "l_hip_y", ObservationType.JOINT_POS),
                            ("q_l_knee", "l_knee", ObservationType.JOINT_POS),
                            ("q_l_ankle", "l_ankle", ObservationType.JOINT_POS),

                            # ------------- JOINT VEL -------------  
                            ("dq_pelvis_tx", "pelvis_tx", ObservationType.JOINT_VEL),
                            ("dq_pelvis_tz", "pelvis_tz", ObservationType.JOINT_VEL),
                            ("dq_pelvis_ty", "pelvis_ty", ObservationType.JOINT_VEL),
                            ("dq_pelvis_tilt", "pelvis_tilt", ObservationType.JOINT_VEL),
                            ("dq_pelvis_list", "pelvis_list", ObservationType.JOINT_VEL),
                            ("dq_pelvis_rotation", "pelvis_rotation", ObservationType.JOINT_VEL),
                            ("dq_l_shoulder_y", "l_shoulder_y", ObservationType.JOINT_VEL),
                            ("dq_l_shoulder_x", "l_shoulder_x", ObservationType.JOINT_VEL),
                            ("dq_l_shoulder_z", "l_shoulder_z", ObservationType.JOINT_VEL),
                            ("dq_l_elbow", "l_elbow", ObservationType.JOINT_VEL),
                            ("dq_r_shoulder_y", "r_shoulder_y", ObservationType.JOINT_VEL),
                            ("dq_r_shoulder_x", "r_shoulder_x", ObservationType.JOINT_VEL),
                            ("dq_r_shoulder_z", "r_shoulder_z", ObservationType.JOINT_VEL),
                            ("dq_r_elbow", "r_elbow", ObservationType.JOINT_VEL),
                            ("dq_r_hip_z", "r_hip_z", ObservationType.JOINT_VEL),
                            ("dq_r_hip_x", "r_hip_x", ObservationType.JOINT_VEL),
                            ("dq_r_hip_y", "r_hip_y", ObservationType.JOINT_VEL),
                            ("dq_r_knee", "r_knee", ObservationType.JOINT_VEL),
                            ("dq_r_ankle", "r_ankle", ObservationType.JOINT_VEL),
                            ("dq_l_hip_z", "l_hip_z", ObservationType.JOINT_VEL),
                            ("dq_l_hip_x", "l_hip_x", ObservationType.JOINT_VEL),
                            ("dq_l_hip_y", "l_hip_y", ObservationType.JOINT_VEL),
                            ("dq_l_knee", "l_knee", ObservationType.JOINT_VEL),
                            ("dq_l_ankle", "l_ankle", ObservationType.JOINT_VEL)]
        
                            

        return observation_spec

    @staticmethod
    def _get_action_specification():
        """
        Getter for the action space specification.

        Returns:
            A list of tuples containing the specification of each action
            space entry.

        """
       
        action_spec = ["l_shoulder_y", "l_shoulder_x", "l_shoulder_z", "l_elbow", 
                       "r_shoulder_y", "r_shoulder_x", "r_shoulder_z", "r_elbow", 
                       "r_hip_z", "r_hip_x", "r_hip_y", "r_knee", "r_ankle", 
                       "l_hip_z", "l_hip_x", "l_hip_y", "l_knee", "l_ankle"]

        return action_spec
