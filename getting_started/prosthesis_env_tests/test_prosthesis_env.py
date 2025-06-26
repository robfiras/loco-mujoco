import time
import mujoco
from loco_mujoco.environments.humanoids.skeletons import MjxSkeletonMuscle
from loco_mujoco.core import ObservationType

import mujoco.viewer

class MjxSkeletonMuscleProsthesis(MjxSkeletonMuscle):
    """
    Mjx version of SkeletonMuscle with specs for adding a prosthesis.
    """

    mjx_enabled = True

    def __init__(self, timestep: float = 0.002, n_substeps: int = 5, **kwargs):
        """
        Constructor.
        """
        if "prosthesis_side" not in kwargs:
            raise ValueError("Missing required argument: 'prosthesis_side'")
        if "prosthesis_type" not in kwargs:
            raise ValueError("Missing required argument: 'prosthesis_type'")

        side_arg = kwargs.pop("prosthesis_side")
        prosthesis_type = kwargs.pop("prosthesis_type")

        if side_arg == "left_side":
            self.prosthesis_side = "_l"
        elif side_arg == "right_side":
            self.prosthesis_side = "_r"
        else:
            raise ValueError("Invalid prosthesis side. Choose 'left_side' or 'right_side'.")

        # Load model specification and modify it
        spec = mujoco.MjSpec.from_file('/home/nadinebadie/loco-mujoco/loco_mujoco/models/skeleton/skeleton_muscle.xml') #self.get_default_xml_file_path())
        spec = self.replace_leg_level(spec, prosthesis_type)
        model = spec.compile()  # Compile the model from the modified spec
        data = mujoco.MjData(model)

        with mujoco.viewer.launch(model, data) as viewer:
            while viewer.is_running():
                step_start = time.time()
                mujoco.mj_step(model, data) 

                # Pick up changes to the physics state, apply perturbations, update options from GUI
                viewer.sync()

                # Time keeping 
                time_until_new_step = model.opt.timestep - (time.time() - step_start)
                if time_until_new_step > 0:
                    time.sleep(time_until_new_step)


        # data = mujoco.MjData(model)



        # Model option configuration
        model_option_conf = kwargs.pop("model_option_conf", {
            "iterations": 4,
            "ls_iterations": 8,
            "disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP
        })

        super().__init__(timestep=timestep, n_substeps=n_substeps,
                         model_option_conf=model_option_conf, spec=spec, **kwargs)
        

    def replace_leg_level(self, spec, prosthesis_type):
        if prosthesis_type == "None":
            return spec
        elif prosthesis_type == "transtibial":
            return self.transtibial_prosthesis(spec)
        elif prosthesis_type == "transfemoral":
            raise NotImplementedError("Transfemoral prosthesis not implemented yet.")
        else:
            raise ValueError(f"Unknown prosthesis type: {prosthesis_type}")
        

    def remove_sites(self, body):
        muscle_names = []

        for s in body.sites:  
            # print(f"Site: {s.name}")
            if '-P' in s.name:
                muscle_names.append(s.name[:-3])
                # print(f"Muscle name: {muscle_names}")
            # print(f"Removing site: {s.name}")
            s.delete()
        return set(muscle_names)
    

    def remove_tendons(self, spec, muscle_names):
        for t in spec.tendons:
            if any(m in t.name for m in muscle_names):
                t.delete()


    def remove_actuators(self, spec, muscle_names):
        for a in spec.actuators:
            # print(f"Actuator: {a.name}")
            # print(f"Muscle names: {muscle_names}")
            if any(m in a.name for m in muscle_names):
                # print(f"Removing actuator: {a.name}")
                a.delete()


    def remove_joint(self, spec, joint_names):
        for j in spec.joints:
            if j.name in joint_names:
                j.delete()



    def remove_equality(self, spec, joint_names):
        """
        Removes equality constraints associated with the specified joint names.

        Args:
            spec: The model specification object.
            joint_names: A list of joint names whose equality constraints should be removed.
        """
        for e in spec.equalities:  # Use list to avoid iteration issues during deletion
            if any(joint_name in e.name for joint_name in joint_names):
                # print(f"Removing equality constraint: {e.name}")
                e.delete()



    def remove_site_actuator_tendon(self, spec, body):
        muscle_names = self.remove_sites(body)
        self.remove_tendons(spec, muscle_names)
        self.remove_actuators(spec, muscle_names)



    def transtibial_prosthesis(self, spec):
        joint_names = [
            f"ankle_angle{self.prosthesis_side}",
            f"subtalar_angle{self.prosthesis_side}",
            f"mtp_angle{self.prosthesis_side}"
        ]
        self.remove_joint(spec, joint_names)
        self.remove_equality(spec, joint_names)

        calcn = spec.find_body(f"calcn{self.prosthesis_side}")
        # print(f"Calcn Body: {calcn}")
        self.remove_site_actuator_tendon(spec, calcn)

        toe = spec.find_body(f"toes{self.prosthesis_side}")
        # print(f"Toe Body: {toe}")
        self.remove_site_actuator_tendon(spec, toe)

        talus = spec.find_body(f"talus{self.prosthesis_side}")

        # Set color of calcn and toe bodies to blue
        # calcn.rgba = [0.0, 0.0, 1.0, 1.0]
        # toe.rgba = [0.0, 0.0, 1.0, 1.0]

        # get geometries of calcn and toe
        for g in calcn.geoms + toe.geoms + talus.geoms:
            g.rgba = [0.0, 0.0, 1.0, 1.0]

        return spec
    

    def _get_observation_specification(self, spec: mujoco.MjSpec):
        joint_names = []
        for j in spec.joints: 
            joint_names.append(j.name)

        # print(f"Joint names: {joint_names}")
        # print(f"Number of joints: {len(joint_names)}")

        if 'root' in joint_names: 
            joint_names.remove('root')
        # print(f"Joint names without root: {joint_names}")

        observation_spec_joint_pos = []
        observation_spec_joint_vel = []

        for j in joint_names:
            observation_spec_joint_pos.append(ObservationType.JointPos(f"q_{j}", xml_name=j))
            observation_spec_joint_vel.append(ObservationType.JointVel(f"dq_{j}", xml_name=j))

        # print(f"Observation spec joint pos: {observation_spec_joint_pos}")
        # print(f"Observation spec joint vel: {observation_spec_joint_vel}")
        observation_spec = [  # ------------- JOINT POS -------------
                            ObservationType.FreeJointPosNoXY("q_root", xml_name="root"),
                            # --- lower limb right ---
                            # ObservationType.JointPos("q_hip_flexion_r", xml_name="hip_flexion_r"),
                            # ObservationType.JointPos("q_hip_adduction_r", xml_name="hip_adduction_r"),
                            # ObservationType.JointPos("q_hip_rotation_r", xml_name="hip_rotation_r"),
                            # ObservationType.JointPos("q_knee_angle_r", xml_name="knee_angle_r"),
                            # ObservationType.JointPos("q_ankle_angle_r", xml_name="ankle_angle_r"),
                            # ObservationType.JointPos("q_subtalar_angle_r", xml_name="subtalar_angle_r"),
                            # ObservationType.JointPos("q_mtp_angle_r", xml_name="mtp_angle_r"),
                            # # --- lower limb left ---
                            # ObservationType.JointPos("q_hip_flexion_l", xml_name="hip_flexion_l"),
                            # ObservationType.JointPos("q_hip_adduction_l", xml_name="hip_adduction_l"),
                            # ObservationType.JointPos("q_hip_rotation_l", xml_name="hip_rotation_l"),
                            # ObservationType.JointPos("q_knee_angle_l", xml_name="knee_angle_l"),
                            # ObservationType.JointPos("q_ankle_angle_l", xml_name="ankle_angle_l"),
                            # ObservationType.JointPos("q_subtalar_angle_l", xml_name="subtalar_angle_l"),
                            # ObservationType.JointPos("q_mtp_angle_l", xml_name="mtp_angle_l"),
                            # # --- lumbar ---
                            # ObservationType.JointPos("q_lumbar_extension", xml_name="lumbar_extension"),
                            # ObservationType.JointPos("q_lumbar_bending", xml_name="lumbar_bending"),
                            # ObservationType.JointPos("q_lumbar_rotation", xml_name="lumbar_rotation"),
                            # # --- upper body right ---
                            # ObservationType.JointPos("q_arm_flex_r", xml_name="arm_flex_r"),
                            # ObservationType.JointPos("q_arm_add_r", xml_name="arm_add_r"),
                            # ObservationType.JointPos("q_arm_rot_r", xml_name="arm_rot_r"),
                            # ObservationType.JointPos("q_elbow_flex_r", xml_name="elbow_flex_r"),
                            # ObservationType.JointPos("q_pro_sup_r", xml_name="pro_sup_r"),
                            # ObservationType.JointPos("q_wrist_flex_r", xml_name="wrist_flex_r"),
                            # ObservationType.JointPos("q_wrist_dev_r", xml_name="wrist_dev_r"),
                            # # --- upper body left ---
                            # ObservationType.JointPos("q_arm_flex_l", xml_name="arm_flex_l"),
                            # ObservationType.JointPos("q_arm_add_l", xml_name="arm_add_l"),
                            # ObservationType.JointPos("q_arm_rot_l", xml_name="arm_rot_l"),
                            # ObservationType.JointPos("q_elbow_flex_l", xml_name="elbow_flex_l"),
                            # ObservationType.JointPos("q_pro_sup_l", xml_name="pro_sup_l"),
                            # ObservationType.JointPos("q_wrist_flex_l", xml_name="wrist_flex_l"),
                            # ObservationType.JointPos("q_wrist_dev_l", xml_name="wrist_dev_l"),

                            # # ------------- JOINT VEL -------------
                            # ObservationType.FreeJointVel("dq_root", xml_name="root"),
                            # # --- lower limb right ---
                            # ObservationType.JointVel("dq_hip_flexion_r", xml_name="hip_flexion_r"),
                            # ObservationType.JointVel("dq_hip_adduction_r", xml_name="hip_adduction_r"),
                            # ObservationType.JointVel("dq_hip_rotation_r", xml_name="hip_rotation_r"),
                            # ObservationType.JointVel("dq_knee_angle_r", xml_name="knee_angle_r"),
                            # ObservationType.JointVel("dq_ankle_angle_r", xml_name="ankle_angle_r"),
                            # ObservationType.JointVel("dq_subtalar_angle_r", xml_name="subtalar_angle_r"),
                            # ObservationType.JointVel("dq_mtp_angle_r", xml_name="mtp_angle_r"),
                            # # --- lower limb left ---
                            # ObservationType.JointVel("dq_hip_flexion_l", xml_name="hip_flexion_l"),
                            # ObservationType.JointVel("dq_hip_adduction_l", xml_name="hip_adduction_l"),
                            # ObservationType.JointVel("dq_hip_rotation_l", xml_name="hip_rotation_l"),
                            # ObservationType.JointVel("dq_knee_angle_l", xml_name="knee_angle_l"),
                            # ObservationType.JointVel("dq_ankle_angle_l", xml_name="ankle_angle_l"),
                            # ObservationType.JointVel("dq_subtalar_angle_l", xml_name="subtalar_angle_l"),
                            # ObservationType.JointVel("dq_mtp_angle_l", xml_name="mtp_angle_l"),
                            # # --- lumbar ---
                            # ObservationType.JointVel("dq_lumbar_extension", xml_name="lumbar_extension"),
                            # ObservationType.JointVel("dq_lumbar_bending", xml_name="lumbar_bending"),
                            # ObservationType.JointVel("dq_lumbar_rotation", xml_name="lumbar_rotation"),
                            # # --- upper body right ---
                            # ObservationType.JointVel("dq_arm_flex_r", xml_name="arm_flex_r"),
                            # ObservationType.JointVel("dq_arm_add_r", xml_name="arm_add_r"),
                            # ObservationType.JointVel("dq_arm_rot_r", xml_name="arm_rot_r"),
                            # ObservationType.JointVel("dq_elbow_flex_r", xml_name="elbow_flex_r"),
                            # ObservationType.JointVel("dq_pro_sup_r", xml_name="pro_sup_r"),
                            # ObservationType.JointVel("dq_wrist_flex_r", xml_name="wrist_flex_r"),
                            # ObservationType.JointVel("dq_wrist_dev_r", xml_name="wrist_dev_r"),
                            # # --- upper body left ---
                            # ObservationType.JointVel("dq_arm_flex_l", xml_name="arm_flex_l"),
                            # ObservationType.JointVel("dq_arm_add_l", xml_name="arm_add_l"),
                            # ObservationType.JointVel("dq_arm_rot_l", xml_name="arm_rot_l"),
                            # ObservationType.JointVel("dq_elbow_flex_l", xml_name="elbow_flex_l"),
                            # ObservationType.JointVel("dq_pro_sup_l", xml_name="pro_sup_l"),
                            # ObservationType.JointVel("dq_wrist_flex_l", xml_name="wrist_flex_l"),
                            # ObservationType.JointVel("dq_wrist_dev_l", xml_name="wrist_dev_l")
                            ] + observation_spec_joint_pos + observation_spec_joint_vel

        return observation_spec
    


    def _get_action_specification(self, spec: mujoco. MjSpec):
        """
        Getter for the action space specification.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            List[str]: List of action space specification.
        """

        action_spec = []
        for m in spec.actuators:
            action_spec.append(m.name)

        print(f"Action spec: {action_spec}")

        # action_spec = ["mot_lumbar_ext", "mot_lumbar_bend", "mot_lumbar_rot", "mot_shoulder_flex_r",
        #                "mot_shoulder_add_r", "mot_shoulder_rot_r", "mot_elbow_flex_r", "mot_pro_sup_r",
        #                "mot_wrist_flex_r", "mot_wrist_dev_r", "mot_shoulder_flex_l", "mot_shoulder_add_l",
        #                "mot_shoulder_rot_l", "mot_elbow_flex_l", "mot_pro_sup_l", "mot_wrist_flex_l",
        #                "mot_wrist_dev_l", "mot_hip_flexion_r", "mot_hip_adduction_r", "mot_hip_rotation_r",
        #                "mot_knee_angle_r", "mot_ankle_angle_r", "mot_subtalar_angle_r", "mot_mtp_angle_r",
        #                "mot_hip_flexion_l", "mot_hip_adduction_l", "mot_hip_rotation_l", "mot_knee_angle_l",
        #                "mot_ankle_angle_l", "mot_subtalar_angle_l", "mot_mtp_angle_l"]

        return action_spec



    



if __name__ == "__main__":

    prosthesis_side= "right_side"  # or "right_side"
    prosthesis_type = "transtibial"  # or "transfemoral", or "None"

    # if prosthesis_side == "left_side" and prosthesis_type == "transtibial":
    # observation_spec = [
    #     ObservationType.FreeJointPosNoXY(obs_name="free_joint", xml_name="root", group="prioritized"),
    #     ObservationType.FreeJointVel(obs_name="hip_flexion_r", xml_name="root", group="prioritized"),
    # ]
    # env = MjxSkeletonMuscleProsthesis(prosthesis_side=prosthesis_side, prosthesis_type=prosthesis_type)
    env = MjxSkeletonMuscleProsthesis(prosthesis_side=prosthesis_side, prosthesis_type=prosthesis_type)