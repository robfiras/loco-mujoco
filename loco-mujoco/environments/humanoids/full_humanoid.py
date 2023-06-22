from pathlib import Path

import numpy as np
from dm_control import mjcf

from mushroom_rl.environments.mujoco_envs.humanoids.base_humanoid import BaseHumanoid
from mushroom_rl.utils.running_stats import *
from mushroom_rl.utils.mujoco import *

# optional imports
try:
    mujoco_viewer_available = True
    import mujoco_viewer
except ModuleNotFoundError:
    mujoco_viewer_available = False


class FullHumanoid(BaseHumanoid):
    """
    Mujoco simulation of full humanoid with muscle-actuated lower limb and torque-actuated upper body.

    """
    def __init__(self, use_brick_foots=False, disable_arms=False, tmp_dir_name=None, **kwargs):
        """
        Constructor.

        """
        if use_brick_foots:
            assert tmp_dir_name is not None, "If you want to use brick foots or disable the arms, you have to specify a" \
                                             "directory name for the xml-files to be saved."
        xml_path = (Path(__file__).resolve().parent.parent / "data" / "full_humanoid" / "full_humanoid.xml").as_posix()

        action_spec = [# motors
                       "lumbar_ext", "lumbar_bend", "lumbar_rot", "shoulder_flex_r", "shoulder_add_r", "shoulder_rot_r",
                       "elbow_flex_r", "pro_sup_r", "wrist_flex_r", "wrist_dev_r", "shoulder_flex_l", "shoulder_add_l",
                       "shoulder_rot_l", "elbow_flex_l", "pro_sup_l", "wrist_flex_l", "wrist_dev_l",
                       # muscles
                       "addbrev_r", "addlong_r", "addmagDist_r", "addmagIsch_r", "addmagMid_r", "addmagProx_r",
                       "bflh_r", "bfsh_r", "edl_r", "ehl_r", "fdl_r", "fhl_r", "gaslat_r", "gasmed_r", "glmax1_r",
                       "glmax2_r", "glmax3_r", "glmed1_r", "glmed2_r", "glmed3_r", "glmin1_r", "glmin2_r",
                       "glmin3_r", "grac_r", "iliacus_r", "perbrev_r", "perlong_r", "piri_r", "psoas_r", "recfem_r",
                       "sart_r", "semimem_r", "semiten_r", "soleus_r", "tfl_r", "tibant_r", "tibpost_r", "vasint_r",
                       "vaslat_r", "vasmed_r", "addbrev_l", "addlong_l", "addmagDist_l", "addmagIsch_l", "addmagMid_l",
                       "addmagProx_l", "bflh_l", "bfsh_l", "edl_l", "ehl_l", "fdl_l", "fhl_l", "gaslat_l", "gasmed_l",
                       "glmax1_l", "glmax2_l", "glmax3_l", "glmed1_l", "glmed2_l", "glmed3_l", "glmin1_l", "glmin2_l",
                       "glmin3_l", "grac_l", "iliacus_l", "perbrev_l", "perlong_l", "piri_l", "psoas_l", "recfem_l",
                       "sart_l", "semimem_l", "semiten_l", "soleus_l", "tfl_l", "tibant_l", "tibpost_l", "vasint_l",
                       "vaslat_l", "vasmed_l"]

        observation_spec = [#------------- JOINT POS -------------
                            ("q_pelvis_tx", "pelvis_tx", ObservationType.JOINT_POS),
                            ("q_pelvis_tz", "pelvis_tz", ObservationType.JOINT_POS),
                            ("q_pelvis_ty", "pelvis_ty", ObservationType.JOINT_POS),
                            ("q_pelvis_tilt", "pelvis_tilt", ObservationType.JOINT_POS),
                            ("q_pelvis_list", "pelvis_list", ObservationType.JOINT_POS),
                            ("q_pelvis_rotation", "pelvis_rotation", ObservationType.JOINT_POS),
                            # --- lower limb right ---
                            ("q_hip_flexion_r", "hip_flexion_r", ObservationType.JOINT_POS),
                            ("q_hip_adduction_r", "hip_adduction_r", ObservationType.JOINT_POS),
                            ("q_hip_rotation_r", "hip_rotation_r", ObservationType.JOINT_POS),
                            ("q_knee_angle_r_translation2", "knee_angle_r_translation2", ObservationType.JOINT_POS),
                            ("q_knee_angle_r_translation1", "knee_angle_r_translation1", ObservationType.JOINT_POS),
                            ("q_knee_angle_r", "knee_angle_r", ObservationType.JOINT_POS),
                            ("q_knee_angle_r_rotation2", "knee_angle_r_rotation2", ObservationType.JOINT_POS),
                            ("q_knee_angle_r_rotation3", "knee_angle_r_rotation3", ObservationType.JOINT_POS),
                            ("q_ankle_angle_r", "ankle_angle_r", ObservationType.JOINT_POS),
                            ("q_subtalar_angle_r", "subtalar_angle_r", ObservationType.JOINT_POS),
                            ("q_mtp_angle_r", "mtp_angle_r", ObservationType.JOINT_POS),
                            ("q_knee_angle_r_beta_translation2", "knee_angle_r_beta_translation2", ObservationType.JOINT_POS),
                            ("q_knee_angle_r_beta_translation1", "knee_angle_r_beta_translation1", ObservationType.JOINT_POS),
                            ("q_knee_angle_r_beta_rotation1", "knee_angle_r_beta_rotation1", ObservationType.JOINT_POS),
                            # --- lower limb left ---
                            ("q_hip_flexion_l", "hip_flexion_l", ObservationType.JOINT_POS),
                            ("q_hip_adduction_l", "hip_adduction_l", ObservationType.JOINT_POS),
                            ("q_hip_rotation_l", "hip_rotation_l", ObservationType.JOINT_POS),
                            ("q_knee_angle_l_translation2", "knee_angle_l_translation2", ObservationType.JOINT_POS),
                            ("q_knee_angle_l_translation1", "knee_angle_l_translation1", ObservationType.JOINT_POS),
                            ("q_knee_angle_l", "knee_angle_l", ObservationType.JOINT_POS),
                            ("q_knee_angle_l_rotation2", "knee_angle_l_rotation2", ObservationType.JOINT_POS),
                            ("q_knee_angle_l_rotation3", "knee_angle_l_rotation3", ObservationType.JOINT_POS),
                            ("q_ankle_angle_l", "ankle_angle_l", ObservationType.JOINT_POS),
                            ("q_subtalar_angle_l", "subtalar_angle_l", ObservationType.JOINT_POS),
                            ("q_mtp_angle_l", "mtp_angle_l", ObservationType.JOINT_POS),
                            ("q_knee_angle_l_beta_translation2", "knee_angle_l_beta_translation2", ObservationType.JOINT_POS),
                            ("q_knee_angle_l_beta_translation1", "knee_angle_l_beta_translation1", ObservationType.JOINT_POS),
                            ("q_knee_angle_l_beta_rotation1", "knee_angle_l_beta_rotation1", ObservationType.JOINT_POS),
                            # --- lumbar ---
                            ("q_lumbar_extension", "lumbar_extension", ObservationType.JOINT_POS),
                            ("q_lumbar_bending", "lumbar_bending", ObservationType.JOINT_POS),
                            ("q_lumbar_rotation", "lumbar_rotation", ObservationType.JOINT_POS),
                            # q-- upper body right ---
                            ("q_arm_flex_r", "arm_flex_r", ObservationType.JOINT_POS),
                            ("q_arm_add_r", "arm_add_r", ObservationType.JOINT_POS),
                            ("q_arm_rot_r", "arm_rot_r", ObservationType.JOINT_POS),
                            ("q_elbow_flex_r", "elbow_flex_r", ObservationType.JOINT_POS),
                            ("q_pro_sup_r", "pro_sup_r", ObservationType.JOINT_POS),
                            ("q_wrist_flex_r", "wrist_flex_r", ObservationType.JOINT_POS),
                            ("q_wrist_dev_r", "wrist_dev_r", ObservationType.JOINT_POS),
                            # --- upper body left ---
                            ("q_arm_flex_l", "arm_flex_l", ObservationType.JOINT_POS),
                            ("q_arm_add_l", "arm_add_l", ObservationType.JOINT_POS),
                            ("q_arm_rot_l", "arm_rot_l", ObservationType.JOINT_POS),
                            ("q_elbow_flex_l", "elbow_flex_l", ObservationType.JOINT_POS),
                            ("q_pro_sup_l", "pro_sup_l", ObservationType.JOINT_POS),
                            ("q_wrist_flex_l", "wrist_flex_l", ObservationType.JOINT_POS),
                            ("q_wrist_dev_l", "wrist_dev_l", ObservationType.JOINT_POS),

                            # ------------- JOINT VEL -------------
                            ("dq_pelvis_tx", "pelvis_tx", ObservationType.JOINT_VEL),
                            ("dq_pelvis_tz", "pelvis_tz", ObservationType.JOINT_VEL),
                            ("dq_pelvis_ty", "pelvis_ty", ObservationType.JOINT_VEL),
                            ("dq_pelvis_tilt", "pelvis_tilt", ObservationType.JOINT_VEL),
                            ("dq_pelvis_list", "pelvis_list", ObservationType.JOINT_VEL),
                            ("dq_pelvis_rotation", "pelvis_rotation", ObservationType.JOINT_VEL),
                            # --- lower limb right ---
                            ("dq_hip_flexion_r", "hip_flexion_r", ObservationType.JOINT_VEL),
                            ("dq_hip_adduction_r", "hip_adduction_r", ObservationType.JOINT_VEL),
                            ("dq_hip_rotation_r", "hip_rotation_r", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_r_translation2", "knee_angle_r_translation2", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_r_translation1", "knee_angle_r_translation1", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_r", "knee_angle_r", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_r_rotation2", "knee_angle_r_rotation2", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_r_rotation3", "knee_angle_r_rotation3", ObservationType.JOINT_VEL),
                            ("dq_ankle_angle_r", "ankle_angle_r", ObservationType.JOINT_VEL),
                            ("dq_subtalar_angle_r", "subtalar_angle_r", ObservationType.JOINT_VEL),
                            ("dq_mtp_angle_r", "mtp_angle_r", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_r_beta_translation2", "knee_angle_r_beta_translation2", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_r_beta_translation1", "knee_angle_r_beta_translation1", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_r_beta_rotation1", "knee_angle_r_beta_rotation1", ObservationType.JOINT_VEL),
                            # --- lower limb left ---
                            ("dq_hip_flexion_l", "hip_flexion_l", ObservationType.JOINT_VEL),
                            ("dq_hip_adduction_l", "hip_adduction_l", ObservationType.JOINT_VEL),
                            ("dq_hip_rotation_l", "hip_rotation_l", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_l_translation2", "knee_angle_l_translation2", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_l_translation1", "knee_angle_l_translation1", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_l", "knee_angle_l", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_l_rotation2", "knee_angle_l_rotation2", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_l_rotation3", "knee_angle_l_rotation3", ObservationType.JOINT_VEL),
                            ("dq_ankle_angle_l", "ankle_angle_l", ObservationType.JOINT_VEL),
                            ("dq_subtalar_angle_l", "subtalar_angle_l", ObservationType.JOINT_VEL),
                            ("dq_mtp_angle_l", "mtp_angle_l", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_l_beta_translation2", "knee_angle_l_beta_translation2", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_l_beta_translation1", "knee_angle_l_beta_translation1", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_l_beta_rotation1", "knee_angle_l_beta_rotation1", ObservationType.JOINT_VEL),
                            # --- lumbar ---
                            ("dq_lumbar_extension", "lumbar_extension", ObservationType.JOINT_VEL),
                            ("dq_lumbar_bending", "lumbar_bending", ObservationType.JOINT_VEL),
                            ("dq_lumbar_rotation", "lumbar_rotation", ObservationType.JOINT_VEL),
                            # --- upper body right ---
                            ("dq_arm_flex_r", "arm_flex_r", ObservationType.JOINT_VEL),
                            ("dq_arm_add_r", "arm_add_r", ObservationType.JOINT_VEL),
                            ("dq_arm_rot_r", "arm_rot_r", ObservationType.JOINT_VEL),
                            ("dq_elbow_flex_r", "elbow_flex_r", ObservationType.JOINT_VEL),
                            ("dq_pro_sup_r", "pro_sup_r", ObservationType.JOINT_VEL),
                            ("dq_wrist_flex_r", "wrist_flex_r", ObservationType.JOINT_VEL),
                            ("dq_wrist_dev_r", "wrist_dev_r", ObservationType.JOINT_VEL),
                            # --- upper body left ---
                            ("dq_arm_flex_l", "arm_flex_l", ObservationType.JOINT_VEL),
                            ("dq_arm_add_l", "arm_add_l", ObservationType.JOINT_VEL),
                            ("dq_arm_rot_l", "arm_rot_l", ObservationType.JOINT_VEL),
                            ("dq_elbow_flex_l", "elbow_flex_l", ObservationType.JOINT_VEL),
                            ("dq_pro_sup_l", "pro_sup_l", ObservationType.JOINT_VEL),
                            ("dq_wrist_flex_l", "wrist_flex_l", ObservationType.JOINT_VEL),
                            ("dq_wrist_dev_l", "wrist_dev_l", ObservationType.JOINT_VEL)]

        collision_groups = [("floor", ["floor"]),
                            ("foot_r", ["r_foot"]),
                            ("front_foot_r", ["r_bofoot"]),
                            ("foot_l", ["l_foot"]),
                            ("front_foot_l", ["l_bofoot"])]

        self._use_brick_foots = use_brick_foots
        self._disable_arms = disable_arms
        joints_to_remove = []
        actuators_to_remove = []
        equ_constr_to_remove = []
        if use_brick_foots:
            joints_to_remove +=["subtalar_angle_l", "mtp_angle_l", "subtalar_angle_r", "mtp_angle_r"]
            equ_constr_to_remove += [j + "_constraint" for j in joints_to_remove]
            # ToDo: think about a smarter way to not include foot force twice for bricks
            collision_groups = [("floor", ["floor"]),
                                ("foot_r", ["foot_brick_r"]),
                                ("front_foot_r", ["foot_brick_r"]),
                                ("foot_l", ["foot_brick_l"]),
                                ("front_foot_l", ["foot_brick_l"])]

        if disable_arms:
            joints_to_remove +=["arm_flex_r", "arm_add_r", "arm_rot_r", "elbow_flex_r", "pro_sup_r", "wrist_flex_r",
                                "wrist_dev_r", "arm_flex_l", "arm_add_l", "arm_rot_l", "elbow_flex_l", "pro_sup_l",
                                "wrist_flex_l", "wrist_dev_l"]
            actuators_to_remove += ["shoulder_flex_r", "shoulder_add_r", "shoulder_rot_r", "elbow_flex_r",
                                    "pro_sup_r", "wrist_flex_r", "wrist_dev_r", "shoulder_flex_l",
                                    "shoulder_add_l", "shoulder_rot_l", "elbow_flex_l", "pro_sup_l",
                                    "wrist_flex_l", "wrist_dev_l"]
            equ_constr_to_remove += ["wrist_flex_r_constraint", "wrist_dev_r_constraint",
                                    "wrist_flex_l_constraint", "wrist_dev_l_constraint"]

        if use_brick_foots or disable_arms:
            obs_to_remove = ["q_" + j for j in joints_to_remove] + ["dq_" + j for j in joints_to_remove]
            observation_spec = [elem for elem in observation_spec if elem[0] not in obs_to_remove]
            # ToDo: there are probabily some muscles that act on the foot, but these are not removed when using brick foots.
            action_spec = [ac for ac in action_spec if ac not in actuators_to_remove]
            xml_handle = mjcf.from_path(xml_path)
            xml_handle = self.delete_from_xml_handle(xml_handle, joints_to_remove,
                                                     actuators_to_remove, equ_constr_to_remove)
            if use_brick_foots:
                xml_handle = self.add_brick_foots_to_xml_handle(xml_handle)
            xml_path = self.save_xml_handle(xml_handle, tmp_dir_name)

        super().__init__(xml_path, action_spec, observation_spec, collision_groups, **kwargs)

    def delete_from_xml_handle(self, xml_handle, joints_to_remove, actuators_to_remove, equ_constraints):

        for j in joints_to_remove:
            j_handle = xml_handle.find("joint", j)
            j_handle.remove()
        for m in actuators_to_remove:
            m_handle = xml_handle.find("actuator", m)
            m_handle.remove()
        for e in equ_constraints:
            e_handle = xml_handle.find("equality", e)
            e_handle.remove()

        return xml_handle

    def add_brick_foots_to_xml_handle(self, xml_handle):

        # find foot and attach bricks
        toe_l = xml_handle.find("body", "toes_l")
        toe_l.add("geom", name="foot_brick_l", type="box", size=[0.112, 0.03, 0.05], pos=[-0.09, 0.019, 0.0],
                  rgba=[0.5, 0.5, 0.5, 0.5], euler=[0.0, 0.15, 0.0])
        toe_r = xml_handle.find("body", "toes_r")
        toe_r.add("geom", name="foot_brick_r", type="box", size=[0.112, 0.03, 0.05], pos=[-0.09, 0.019, 0.0],
                  rgba=[0.5, 0.5, 0.5, 0.5], euler=[0.0, -0.15, 0.0])

        # make true foot uncollidable
        foot_r = xml_handle.find("geom", "r_foot")
        bofoot_r = xml_handle.find("geom", "r_bofoot")
        foot_l = xml_handle.find("geom", "l_foot")
        bofoot_l = xml_handle.find("geom", "l_bofoot")
        foot_r.contype = 0
        foot_r.conaffinity = 0
        bofoot_r.contype = 0
        bofoot_r.conaffinity = 0
        foot_l.contype = 0
        foot_l.conaffinity = 0
        bofoot_l.contype = 0
        bofoot_l.conaffinity = 0

        return xml_handle

    def save_xml_handle(self, xml_handle, tmp_dir_name):

        # save new model and return new xml path
        new_model_dir_name = 'new_full_humanoid_with_bricks_model/' +  tmp_dir_name + "/"
        cwd = Path.cwd()
        new_model_dir_path = Path.joinpath(cwd, new_model_dir_name)
        xml_file_name =  "modified_reduced_humanoid.xml"
        mjcf.export_with_assets(xml_handle, new_model_dir_path, xml_file_name)
        new_xml_path = Path.joinpath(new_model_dir_path, xml_file_name)

        return new_xml_path.as_posix()

    @staticmethod
    def has_fallen(state):
        # todo this function has to be adapted for brick feet as well!
        pelvis_euler = state[1:4]
        pelvis_condition = ((state[0] < -0.35) or (state[0] > 0.10)
                            or (pelvis_euler[0] < (-np.pi / 4.5)) or (pelvis_euler[0] > (np.pi / 12))
                            or (pelvis_euler[1] < -np.pi / 12) or (pelvis_euler[1] > np.pi / 8)
                            or (pelvis_euler[2] < (-np.pi / 10)) or (pelvis_euler[2] > (np.pi / 10))
                           )
        lumbar_euler = state[32:35]
        lumbar_condition = ((lumbar_euler[0] < (-np.pi / 6)) or (lumbar_euler[0] > (np.pi / 10))
                            or (lumbar_euler[1] < -np.pi / 10) or (lumbar_euler[1] > np.pi / 10)
                            or (lumbar_euler[2] < (-np.pi / 4.5)) or (lumbar_euler[2] > (np.pi / 4.5))
                            )
        return pelvis_condition or lumbar_condition


if __name__ == '__main__':
    import time

    env = FullHumanoid(n_substeps=10, use_brick_foots=False, disable_arms=True, random_start=False, tmp_dir_name="test")

    action_dim = env.info.action_space.shape[0]

    env.reset()
    env.render()

    absorbing = False

    frequencies = 2*np.pi * np.ones(action_dim) * np.random.uniform(0, 10, action_dim)
    psi = np.zeros_like(frequencies)
    dt = 0.01

    while True:
        psi = psi + dt * frequencies
        #action = np.sin(psi)
        #action = np.random.normal(0.0, 1.0, (action_dim,)) # compare to normal gaussian noise
        action = np.tanh(np.random.normal(0.0, 3.0, (action_dim,))) # compare to normal gaussian noise
        action[:3] = 0.0
        nstate, _, absorbing, _ = env.step(action)

        env.render()
