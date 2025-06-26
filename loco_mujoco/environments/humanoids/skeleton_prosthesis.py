import mujoco 
from loco_mujoco.core import ObservationType
from loco_mujoco.environments.humanoids.skeletons import MjxSkeletonMuscle

class MjxSkeletonMuscleProsthesis(MjxSkeletonMuscle):
    """
    Mjx version of SkeletonMuscle with specs for adding a prosthesis.
    """

    mjx_enabled = True

    def __init__(self, timestep: float = 0.002, n_substeps: int = 5, **kwargs):
        """
        Constructor for MjxSkeletonMuscleProsthesis.
        Args:
            timestep (float): The time step for the simulation.
            n_substeps (int): The number of substeps for the simulation.
            **kwargs: Additional keyword arguments for configuration.
        Raises:
            ValueError: If required arguments are missing.
        """
        if "joint_stiffness" in kwargs:
            self.joint_stiffness = kwargs.pop("joint_stiffness")
        if "joint_damping" in kwargs:
            self.joint_damping = kwargs.pop("joint_damping")
        if "delete_joints" in kwargs:
            self.delete_joints = kwargs.pop("delete_joints")
        if "prosthesis_side" not in kwargs:
            raise ValueError("Missing required argument: 'prosthesis_side'")
        if "prosthesis_type" not in kwargs:
            raise ValueError("Missing required argument: 'prosthesis_type'")
        if "add_sensors" in kwargs:
            self.add_sensors = kwargs.pop("add_sensors")

        
        self.amputated_joint_names = []
        self.amputated_body_names = []

        side_arg = kwargs.pop("prosthesis_side")
        self.prosthesis_type = kwargs.pop("prosthesis_type")

        if side_arg == "left_side":
            self.prosthesis_side = "_l"
        elif side_arg == "right_side":
            self.prosthesis_side = "_r"
        else:
            raise ValueError("Invalid prosthesis side. Choose 'left_side' or 'right_side'.")
        
        print(f"Prosthesis side: {self.prosthesis_side}")
        print(f"Prosthesis type: {self.prosthesis_type}")

        # Load model specification and modify it
        spec = mujoco.MjSpec.from_file(self.get_default_xml_file_path())
        spec = self.replace_leg_level(spec) 
       

        # Model option configuration
        model_option_conf = kwargs.pop("model_option_conf", {
            "iterations": 4,
            "ls_iterations": 8,
            "disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP
        })

        super().__init__(timestep=timestep, n_substeps=n_substeps,
                         model_option_conf=model_option_conf, spec=spec, **kwargs)
        

    def add_force_sensor_prosthesis_side(self, spec, site_name):
        """
        Adds a force sensor to the specified body in the mjcf model specification.
        
        Args:
            spec (mjcf.RootElement): The MJCF root model object.
            body_name (str): The name of the body to attach the force sensor to.
        """
        # body = spec.find('body', body_name)
        # for s in body.sites: 
        #     if 'mimic' in s.name:
        #         site_name = s.name

        # if body is None:
        #     raise ValueError(f"Body '{body_name}' not found in the model specification.")
        
        if self.prosthesis_side == "_l":
            site_name = "left_" + site_name
        elif self.prosthesis_side == "_r":
            site_name = "right_" + site_name

        sensor_site = spec.find_site(site_name)

        # print(f"Adding force sensor to site: {site_name}")

        # print(f"Sensor site: {sensor_site}")
        if sensor_site is None:
            raise ValueError(f"Site '{site_name}' not found in the model specification.")
        
        # Add the force sensor
        force_sensor = spec.add_sensor(
        name=f"{sensor_site.name}_force_sensor",
        type=mujoco.mjtSensor.mjSENS_FORCE,
        objtype=mujoco.mjtObj.mjOBJ_SITE,  # Specify that the sensor is attached to a site object
        objname=sensor_site.name      # Provide the name of the specific site
        )

        return force_sensor
    

    def add_force_sensor(self, spec, site_name):
        """
        Adds a force sensor to the specified body in the mjcf model specification.
        
        Args:
            spec (mjcf.RootElement): The MJCF root model object.
            body_name (str): The name of the body to attach the force sensor to.
        """
        sensor_site = spec.find_site(site_name)
        # print(f"Adding force sensor to site: {site_name}")
        # print(f"Sensor site: {sensor_site}")

        # print(f"Adding force sensor to site: {site_name}")

        # print(f"Sensor site: {sensor_site}")
        if sensor_site is None:
            raise ValueError(f"Site '{site_name}' not found in the model specification.")
        
        # Add the force sensor
        force_sensor = spec.add_sensor(
        name=f"{sensor_site.name}_force_sensor",
        type=mujoco.mjtSensor.mjSENS_FORCE,
        objtype=mujoco.mjtObj.mjOBJ_SITE,  # Specify that the sensor is attached to a site object
        objname=sensor_site.name      # Provide the name of the specific site
        )

        return force_sensor
    

    def add_torque_sensor(self, spec, site_name):
        """
        Adds a torque sensor to the specified body in the mjcf model specification.
        
        Args:
            spec (mjcf.RootElement): The MJCF root model object.
            body_name (str): The name of the body to attach the force sensor to.
        """
        sensor_site = spec.find_site(site_name)
        # print(f"Adding force sensor to site: {site_name}")
        # print(f"Sensor site: {sensor_site}")

        # print(f"Adding force sensor to site: {site_name}")

        # print(f"Sensor site: {sensor_site}")
        if sensor_site is None:
            raise ValueError(f"Site '{site_name}' not found in the model specification.")
        
        # Add the torque sensor
        torque_sensor = spec.add_sensor(
        name=f"{sensor_site.name}_torque_sensor",
        type=mujoco.mjtSensor.mjSENS_TORQUE,
        objtype=mujoco.mjtObj.mjOBJ_SITE,  # Specify that the sensor is attached to a site object
        objname=sensor_site.name      # Provide the name of the specific site
        )

        return torque_sensor



    def increase_joint_stiffness(self, spec):
        """
        Increases the stiffness of specified joints in the model specification.
        Args:
            spec (MjSpec): The model specification object.
        """
        for j in spec.joints:
            if j.name in self.amputated_joint_names:
                # print(f"Increasing stiffness of joint: {j.name}")
                j.stiffness = self.joint_stiffness
                j.damping = self.joint_damping

        
    def replace_leg_level(self, spec):
        """
        Replaces the leg level in the model specification based on the prosthesis type.
        Args:
            spec (MjSpec): The model specification object.
        Returns:
            MjSpec: The modified model specification.
        """
        if self.prosthesis_type == "None":
            return spec
        elif self.prosthesis_type == "transtibial":
            self.amputated_joint_names = [
                f"ankle_angle{self.prosthesis_side}",
                f"subtalar_angle{self.prosthesis_side}",
                f"mtp_angle{self.prosthesis_side}"]
            self.amputated_body_names = [
                f"calcn{self.prosthesis_side}",
                f"toes{self.prosthesis_side}",
                f"talus{self.prosthesis_side}"]
            # if self.delete_joint defined or not 
            if hasattr(self, 'delete_joints') and self.delete_joints:
                spec = self.transtibial_prosthesis(spec)
                return spec
            elif not hasattr(self, 'delete_joints'): # Was initally not defined. To use for older policies 
                spec = self.transtibial_prosthesis(spec)
                return spec
            else:
                spec = self.transtibial_prosthesis_with_joints(spec)
                return spec
        elif self.prosthesis_type == "transfemoral":
            raise NotImplementedError("Transfemoral prosthesis not implemented yet.")
        else:
            raise ValueError(f"Unknown prosthesis type: {self.prosthesis_type}")
        

    def remove_sites(self, body):
        """
        Removes sites from the specified body that are associated with muscles.
        Args:
            body (MjBody): The body from which to remove sites.
        Returns:
            set: A set of muscle names derived from the removed sites.
        """
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
        """
        Removes tendons associated with the specified muscle names.
        Args:
            spec (MjSpec): The model specification object.
            muscle_names (set): A set of muscle names to match against tendon names.
        """
        for t in spec.tendons:
            if any(m in t.name for m in muscle_names):
                t.delete()


    def remove_actuators(self, spec, muscle_names):
        """
        Removes actuators associated with the specified muscle names.
        Args:
            spec (MjSpec): The model specification object.
            muscle_names (set): A set of muscle names to match against actuator names."""
        for a in spec.actuators:
            # print(f"Actuator: {a.name}")
            # print(f"Muscle names: {muscle_names}")
            if any(m in a.name for m in muscle_names):
                # print(f"Removing actuator: {a.name}")
                a.delete()


    def remove_joint(self, spec, amputated_joint_names):
        """
        Removes joints specified in self.amputated_joint_names.

        Args:
            spec (MjSpec): The model specification object.
        """
        for j in spec.joints:
            if j.name in amputated_joint_names:
                j.delete()



    def remove_equality(self, spec, amputated_joint_names):
        """
        Removes equality constraints associated with the specified joint names.

        Args:
            spec: The model specification object.
            amputated_joint_names: A list of joint names whose equality constraints should be removed.
        """
        for e in spec.equalities:  # Use list to avoid iteration issues during deletion
            if any(joint_name in e.name for joint_name in amputated_joint_names):
                # print(f"Removing equality constraint: {e.name}")
                e.delete()

    def remove_site_actuator_tendon(self, spec):
        for b in self.amputated_body_names:
            body = spec.find_body(b)
            muscle_names = self.remove_sites(body)
            self.remove_tendons(spec, muscle_names)
            self.remove_actuators(spec, muscle_names)
            for g in body.geoms:
                g.rgba = [0.0, 0.0, 1.0, 1.0]

    def remove_site_actuator_tendon_old(self, spec, body):
        """
        Removes sites, actuators, and tendons from the specified body.
        Args:
            spec (MjSpec): The model specification object.
            body (MjBody): The body from which to remove sites, actuators, and tendons.
        """
        muscle_names = self.remove_sites(body)
        self.remove_tendons(spec, muscle_names)
        self.remove_actuators(spec, muscle_names)

    
    def transtibial_prosthesis_with_joints(self, spec):
        """Handles transtibial prosthesis while keeping joints but increasing stiffness.
        Args:
            spec (MjSpec): The model specification object.
        Returns:
            MjSpec: The modified model specification with increased joint stiffness.
        """
        self.increase_joint_stiffness(spec)

        self.remove_site_actuator_tendon(spec)

       
        # joint force sensor site name only for evaluation
        if hasattr(self, 'add_sensors') and self.add_sensors:
            joint_force_sensor_site_name = "hip_mimic"
            self.add_force_sensor(spec, f"left_{joint_force_sensor_site_name}")
            self.add_force_sensor(spec, f"right_{joint_force_sensor_site_name}")
            self.add_torque_sensor(spec, f"left_{joint_force_sensor_site_name}")
            self.add_torque_sensor(spec, f"right_{joint_force_sensor_site_name}")
            joint_force_sensor_site_name = "knee_mimic"
            self.add_force_sensor(spec, f"left_{joint_force_sensor_site_name}")
            self.add_force_sensor(spec, f"right_{joint_force_sensor_site_name}")
            self.add_torque_sensor(spec, f"left_{joint_force_sensor_site_name}")
            self.add_torque_sensor(spec, f"right_{joint_force_sensor_site_name}")
            joint_force_sensor_site_name = "foot_mimic"
            self.add_force_sensor(spec, f"left_{joint_force_sensor_site_name}")
            self.add_force_sensor(spec, f"right_{joint_force_sensor_site_name}")
            self.add_torque_sensor(spec, f"left_{joint_force_sensor_site_name}")
            self.add_torque_sensor(spec, f"right_{joint_force_sensor_site_name}")


        return spec


    def transtibial_prosthesis(self, spec):
        """Handles transtibial prosthesis by removing joints, sites, actuators, and tendons.
        Args:
            spec (MjSpec): The model specification object.
        Returns:
            MjSpec: The modified model specification with amputated joints and bodies.
        """
        
        self.remove_joint(spec, self.amputated_joint_names)
        self.remove_equality(spec, self.amputated_joint_names)

        self.remove_site_actuator_tendon(spec)

        # joint force sensor site name only for evaluation
        if hasattr(self, 'add_sensors') and self.add_sensors:
            joint_force_sensor_site_name = "knee_mimic"
            self.add_force_sensor(spec, f"left_{joint_force_sensor_site_name}")
            self.add_force_sensor(spec, f"right_{joint_force_sensor_site_name}")
            self.add_torque_sensor(spec, f"left_{joint_force_sensor_site_name}")
            self.add_torque_sensor(spec, f"right_{joint_force_sensor_site_name}")
            joint_force_sensor_site_name = "foot_mimic"
            self.add_force_sensor(spec, f"left_{joint_force_sensor_site_name}")
            self.add_force_sensor(spec, f"right_{joint_force_sensor_site_name}")
            self.add_torque_sensor(spec, f"left_{joint_force_sensor_site_name}")
            self.add_torque_sensor(spec, f"right_{joint_force_sensor_site_name}")

        # for b in self.amputated_body_names:
        #     body = spec.find_body(b)
        #     self.remove_site_actuator_tendon_old(spec, body)
        #     for g in body.geoms:
        #         g.rgba = [0.0, 0.0, 1.0, 1.0]





        # calcn = spec.find_body(f"calcn{self.prosthesis_side}")
        # # print(f"Calcn Body: {calcn}")
        # self.remove_site_actuator_tendon(spec, calcn)

        # toe = spec.find_body(f"toes{self.prosthesis_side}")
        # # print(f"Toe Body: {toe}")
        # self.remove_site_actuator_tendon(spec, toe)

        # talus = spec.find_body(f"talus{self.prosthesis_side}")

        # # Set color of calcn and toe bodies to blue
        # calcn.rgba = [0.0, 0.0, 1.0, 1.0]
        # toe.rgba = [0.0, 0.0, 1.0, 1.0]

        # # get geometries of calcn and toe
        # for g in calcn.geoms + toe.geoms + talus.geoms:
        #     g.rgba = [0.0, 0.0, 1.0, 1.0]

        return spec
    

    def _get_observation_specification(self, spec: mujoco.MjSpec):
        """
        Getter for the observation space specification.
        Args:
            spec (MjSpec): Specification of the environment.
        Returns:
            List[ObservationType]: List of observation space specification.
        """
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

        # print(f"Action spec: {action_spec}")

        # action_spec = ["mot_lumbar_ext", "mot_lumbar_bend", "mot_lumbar_rot", "mot_shoulder_flex_r",
        #                "mot_shoulder_add_r", "mot_shoulder_rot_r", "mot_elbow_flex_r", "mot_pro_sup_r",
        #                "mot_wrist_flex_r", "mot_wrist_dev_r", "mot_shoulder_flex_l", "mot_shoulder_add_l",
        #                "mot_shoulder_rot_l", "mot_elbow_flex_l", "mot_pro_sup_l", "mot_wrist_flex_l",
        #                "mot_wrist_dev_l", "mot_hip_flexion_r", "mot_hip_adduction_r", "mot_hip_rotation_r",
        #                "mot_knee_angle_r", "mot_ankle_angle_r", "mot_subtalar_angle_r", "mot_mtp_angle_r",
        #                "mot_hip_flexion_l", "mot_hip_adduction_l", "mot_hip_rotation_l", "mot_knee_angle_l",
        #                "mot_ankle_angle_l", "mot_subtalar_angle_l", "mot_mtp_angle_l"]

        return action_spec
