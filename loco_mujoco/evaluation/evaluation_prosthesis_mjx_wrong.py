# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "" 
# os.environ["JAX_PLATFORMS"] = "cpu"
import time 


import jax 
import jax.numpy as jnp
# jax.config.update('jax_platform_name', 'cpu')

# import argparse
import mujoco

import numpy as np

# from omegaconf import OmegaConf, DictConfig


# from loco_mujoco import TaskFactory
# from loco_mujoco.algorithms import PPOJax
# from loco_mujoco.environments.humanoids.skeleton_prosthesis import MjxSkeletonMuscleProsthesis
# from loco_mujoco.core.control_functions.skeleton_muscle import SkeletonMuscleControlFunction


# from omegaconf import OmegaConf

# os.environ["MUJOCO_GL"] = "egl"

# from loco_mujoco.utils import MetricsHandler




class ProsthesisMetricsHandler(): #MetricsHandler): #MetricsHandler):
    """
    Metrics handler for prosthesis evaluation.
    This class extends the MetricsHandler to include specific metrics for prosthesis evaluation.
    """

    def __init__(self, config, env):
        """
        Initialize the ProsthesisMetricsHandler.
        Args:
        config (DictConfig): The configuration dictionary.
        env (MjxSkeletonMuscleProsthesis): The environment instance.
        """
        self.config = config
        self.env = env
        self.model = env.get_model()  # Get the model from the environment


    def get_joint_angles(self, mjx_data):
        """
        Get all joint angles from the mjx model and save in a dictionary with joint name as key and angle as value.

        Args:
            mjx_data: The mjx data object containing joint positions.

        Returns:
            dict: Dictionary mapping joint names to their current angles.
        """
        joint_angles = {}
        for i in range(self.model.njnt):
            joint_name = self.model.joint(i).name
            joint_angle = mjx_data.qpos[i]
            joint_angles[joint_name] = joint_angle
        return joint_angles
    

    def get_joint_vels(self, mjx_data):
        """ Get all joint velocities and save in dictionary with joint name as key and velocity as value.
        """
        joint_velocities = {}
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            joint_velocity = mjx_data.qvel[i]
            joint_velocities[joint_name] = joint_velocity
        return joint_velocities
    
    def get_joint_frces(self, mjx_data):
        """ Get all joint forces and save in dictionary with joint name as key and force as value.
        """
        joint_forces_constraint = {}
        joint_forces_smooth = {}
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            joint_force_constraint = mjx_data.qfrc_constraint[i] # constraint force; joint limits and contacts etc. 
            joint_force_smooth = mjx_data.qfrc_smooth[i] # net unconstrained force; Sum of all forces, e.g., gravity, applied torques etc,
            joint_forces_constraint[joint_name] = joint_force_constraint
            joint_forces_smooth[joint_name] = joint_force_smooth
        return joint_forces_constraint, joint_forces_smooth
    
    def get_joint_trques(self, mjx_data):
        """ 
        Get all joint torques and save in dictionary with joint name as key and torque as value.
        qfrc_actuator in actuator force --> As we are using hinge joints, this is the torque applied by the actuator? (As mjx tutorial?)
        """
        joint_torques = {}
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            joint_torque = mjx_data.qfrc_actuator[i]
            joint_torques[joint_name] = joint_torque
        return joint_torques
    

    def calc_joint_energy_exp(self, joint_torques, joint_vels):
        """Energy expenditure is calculated as joint torques multiplied by joint velocities."""
        joint_energy_expenditure = {}
        for joint_name in joint_torques.keys():
            if joint_name in joint_vels.keys():
                energy_expenditure = joint_torques[joint_name] * joint_vels[joint_name]
                joint_energy_expenditure[joint_name] = energy_expenditure
            else:
                raise ValueError(f"Joint velocity for {joint_name} not found.")
        return joint_energy_expenditure


    def get_muscle_group(self, which_muscles): 
        muscle_groups = {
        "back_muscles": ['ercspn'],
        "torso_muscles": ['intobl', 'extobl'],
        "vasti_muscles": ['vas_int', 'vas_lat', 'vas_med'],
        "rectus_muscles": ['rect_fem'],
        "medial_muscles": ['add_mag1', 'add_mag2', 'add_mag3', 'add_brev', 'add_long', 'grac'],
        "leg_muscles": [
            "glut_med1", "glut_med2", "glut_med3", "glut_min1", "glut_min2", "glut_min3",
            "semimem", "semiten", "bifemlh", "bifemsh", "sar", "add_long", "add_brev",
            "add_mag1", "add_mag2", "add_mag3", "tfl_r", "pect", "grac", "glut_max1",
            "glut_max2", "glut_max3", "iliacus", "psoas", "quad_fem", "gem", "peri",
            "rect_fem", "vas_med_r", "vas_int_r", "vas_lat", "med_gas", "lat_gas", "soleus",
            "tib_post", "flex_dig", "flex_hal", "tib_ant", "per_brev", "per_long",
            "per_tert", "ext_dig", "ext_hal"
        ],
        "all_muscles": [
            "glut_med1", "glut_med2", "glut_med3", "glut_min1", "glut_min2", "glut_min3",
            "semimem", "semiten", "bifemlh", "bifemsh", "sar", "add_long", "add_brev",
            "add_mag1", "add_mag2", "add_mag3", "tfl_r", "pect", "grac", "glut_max1",
            "glut_max2", "glut_max3", "iliacus", "psoas", "quad_fem", "gem", "peri",
            "rect_fem", "vas_med_r", "vas_int_r", "vas_lat", "med_gas", "lat_gas", "soleus",
            "tib_post", "flex_dig", "flex_hal", "tib_ant", "per_brev", "per_long",
            "per_tert", "ext_dig", "ext_hal", "ercspn", "intobl", "extobl"
        ]
        }

        # Determine muscle names based on group and side
        if which_muscles in muscle_groups:
            muscle_name = muscle_groups[which_muscles]
        else:
            raise ValueError(f"Invalid muscle group: {which_muscles}")
        
        return muscle_name



    def get_relevant_ctrl(self, muscle_name, muscle_side, action):
        """
        Extract actions for specific muscle groups and sides.

        Args:
        which_muscles (str): The muscle group to extract actions for. 
                        Options: "back_muscles", "torso_muscles", "leg_muscles", "all_muscles", or specific muscle names.
        muscle_side (str): The side of the body. Options: "left_side", "right_side".
        action (jnp.ndarray): The action array to extract from.

        Returns:
        jnp.ndarray: The extracted actions corresponding to the specified muscle group and side.
        """
    
        # elif any(which_muscles in a.name for a in env.model.actuators):
        #     muscle_name = [a.name for a in env.model.actuators if which_muscles in a.name]
        if muscle_side == 'left_side':
            muscle_name = [name+'_l' for name in muscle_name]
        elif muscle_side == 'right_side':
            muscle_name = [name+'_r' for name in muscle_name]
        # elif which_muscles == "all_muscles":
        #     suffix = '_l' if muscle_side == 'left_side' else '_r'
        #     muscle_name = [name for name in env.info.action_space.names if name.endswith(suffix)]
        else:
            raise ValueError(f"Invalid muscle side: {muscle_side}")
        action_indices = []
        extracted_actions = []

        # Get indices for the specified muscles
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name in muscle_name:
                # print(f"Actuator {i}: {name}")
                action_indices.append(i)
        # action_indices = [i for i, name in enumerate(env.info.action_space.names) if name in muscle_name]

        # Extract the actions corresponding to the specified indices

        # Ensure action_indices is not empty
        if not action_indices:
            raise ValueError(f"No matching muscles found for group '{muscle_name}' and side '{muscle_side}'.")

        # Extract the actions corresponding to the specified indices
        extracted_actions = [action[k] for k in action_indices]
        # extracted_action = action[jnp.array(action_indices)]

        return jnp.array(extracted_actions)
    
    

    def evaluate_action_symmetry(self, all_data, muscle_name, left_indices, right_indices):
        """
        Evaluate the symmetry of the actions in the environment states.
        This function calculates the symmetry of the actions for left and right sides.
        """

        muscle_name_left = [name+'_l' for name in muscle_name]
        muscle_name_right = [name+'_r' for name in muscle_name]

        action_indices_left = []
        action_indices_right = []
        all_actions_left = []
        all_actions_right = []

        for n in range(len(all_data)):
            data = all_data[n]

            # Get indices for the specified muscles
            for i in range(self.model.nu):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                if name in muscle_name_left:
                    print(f"Actuator {i}: {name}")
                    action_indices_left.append(i)
                if name in muscle_name_right:
                    print(f"Actuator {i}: {name}")
                    action_indices_right.append(i)


            for i in range(len(left_indices)):
                for a in action_indices_left:
                    actions_left = data[i].ctrl[a][left_indices[i][0]:left_indices[i][1]]
                    all_actions_left.append(actions_left)
                    print(f"Left Action {a} at step {i}: {actions_left}")

            for i in range(len(right_indices)):
                for a in action_indices_right:
                    actions_right = data[i].ctrl[a][right_indices[i][0]:right_indices[i][1]]
                    all_actions_right.append(actions_right)
                    print(f"Right Action {a} at step {i}: {actions_right}")

            # Calculate the symmetry of the actions
            symmetry_scores = []
            for i in range(len(all_actions_left)):
                left_action = all_actions_left[i]
                right_action = all_actions_right[i]
                if len(left_action) != len(right_action):
                    raise ValueError("Left and right actions must have the same length for symmetry evaluation.")
                # Calculate symmetry score as the absolute difference between left and right actions
                symmetry_score = jnp.abs(left_action - right_action).mean()
                symmetry_scores.append(symmetry_score)
                print(f"Symmetry Score for step {i}: {symmetry_score}")



    def evaluate_angle_symmetry(self, all_data, joint_name, left_indices, right_indices):
        """        Evaluate the symmetry of the joint angles in the environment states.
        This function calculates the symmetry of the joint angles for left and right sides.
        """
        joint_left = [name+'_l' for name in joint_name]
        joint_right = [name+'_r' for name in joint_name]
        joint_indices_left = []
        joint_indices_right = []
        all_joints_left = []
        all_joints_right = []

        for n in range(len(all_data)):
            data = all_data[n]
            # Get indices for the specified joints
            for i in range(self.model.njnt):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if name in joint_left:
                    print(f"Joint {i}: {name}")
                    joint_indices_left.append(i)
                if name in joint_right:
                    print(f"Joint {i}: {name}")
                    joint_indices_right.append(i)
            for i in range(len(left_indices)):
                for j in joint_indices_left:
                    joint_left = data[i].qpos[j][left_indices[i][0]:left_indices[i][1]]
                    all_joints_left.append(joint_left)
                    print(f"Left Joint {j} at step {i}: {joint_left}")
            for i in range(len(right_indices)):
                for j in joint_indices_right:
                    joint_right = data[i].qpos[j][right_indices[i][0]:right_indices[i][1]]
                    all_joints_right.append(joint_right)
                    print(f"Right Joint {j} at step {i}: {joint_right}")
            # Calculate the symmetry of the angles
            symmetry_scores = []
            for i in range(len(all_joints_left)):
                left_joint = all_joints_left[i]
                right_joint = all_joints_right[i]
                if len(left_joint) != len(right_joint):
                    raise ValueError("Left and right joints must have the same length for symmetry evaluation.")
                # Calculate symmetry score as the absolute difference between left and right angles
                symmetry_score = jnp.abs(left_joint - right_joint).mean()
                symmetry_scores.append(symmetry_score)
                print(f"Symmetry Score for step {i}: {symmetry_score}")


    def get_contact_steps(self, mjx_data, foot_side, step):
        """
        Identify the simulation steps where the specified foot (left or right) is in contact with the ground.

        Args:
            mjx_data: The mjx data object containing contact information.
            foot_side (str): 'left' or 'right' to specify which foot to check.
            step (int): The current simulation step index.

        Returns:
            list: A list containing the step indices where the specified foot is in contact with the ground.
        """
        all_step_contact = []
        if foot_side == 'left':
            side_suffix = '_l'
        elif foot_side == 'right':
            side_suffix = '_r'
        else:
            raise ValueError("Invalid foot_side. Expected 'left' or 'right'.")
        for n in range(mjx_data.ncon):
            geom_distance = mjx_data.contact.dist[n]
            if geom_distance <= 0:
                geom_name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, mjx_data.contact.geom2[n])
                if side_suffix in geom_name2:
                    all_step_contact.append(step)
        return all_step_contact

        

        #         if "_l" in geom_name2:
        #             contact_left = step
        #             all_step_contact_left.append(contact_left)
        #         elif "_r" in geom_name2:
        #             contact_right = step
        #             all_step_contact_right.append(contact_right)

        # all_step_contact_left = list(set(all_step_contact_left))  # remove duplicates
        # all_step_contact_right = list(set(all_step_contact_right))

        # return all_step_contact_left, all_step_contact_right


    def get_start_steps(self, all_step_contact):
        """
        Get the start steps for left and right foot based on contact data.
        This function processes the contact data to determine the start of each step.
        """
        # Initialize lists to store the start steps for left and right foot
        all_step_start = [all_step_contact[0]]


        # Iterate through the contact data to find non-consecutive steps
        for m in range(len(all_step_contact) - 1):
            if all_step_contact[m + 1] - all_step_contact[m] > 1:
                all_step_start.append(all_step_contact[m])
        
        print(f"All start steps: {all_step_start}")


        return all_step_start
    

    # def get_start_steps(self, all_step_contact_left, all_step_contact_right):
    #     """
    #     Get the start steps for left and right foot based on contact data.
    #     This function processes the contact data to determine the start of each step.
    #     """
    #     # Initialize lists to store the start steps for left and right foot
    #     all_step_start_left = [all_step_contact_left[0]]
    #     all_step_start_right = [all_step_contact_right[0]]

    #     # Iterate through the contact data to find non-consecutive steps
    #     for m in range(len(all_step_contact_left) - 1):
    #         if all_step_contact_left[m + 1] - all_step_contact_left[m] > 1:
    #             all_step_start_left.append(all_step_contact_left[m])
    #     for m in range(len(all_step_contact_right) - 1):
    #         if all_step_contact_right[m + 1] - all_step_contact_right[m] > 1:
    #             all_step_start_right.append(all_step_contact_right[m])

    #     print(f"All start steps left: {all_step_start_left}")
    #     print(f"All start steps right: {all_step_start_right}")


    #     return all_step_start_left, all_step_start_right



    def get_sensor_data(self, mjx_data):
        """
        Retrieve sensor data from the Mujoco environment.
        This function extracts sensor data from the Mujoco data structure.
        """
        # Retrieve sensor names from the model
        sensor_names = []
        for sensor_id in range(self.model.nsensor):
            sensor_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
            sensor_names.append(sensor_name)

        # print(f"Number of sensors: {self.model.nsensor}")
        # print(f"Sensor names: {sensor_names}")

        # Extract sensor data
        if self.model.nsensor == 1: 
            # tibia sensor is the only sensor available
            sensor_data = mjx_data.sensordata
            # print(f"Human-prosthesis interface sensor data: {sensor_data}") 
            return sensor_data
        else: 
            # get sensor_names and iterate through all sensors
            for sensor_id in range(self.model.nsensor):
                sensor_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
                sensor_data = mjx_data.sensordata[sensor_id]
                print(f"Sensor {sensor_name} data: {sensor_data}")
                # No specific sensor data is saved!!!! FIX!!!!!!!!!!!!!!!!


    
    def get_grf(self, mjx_data, foot_name):
        """
        Get the ground reaction forces (GRF) from the Mujoco data.
        This function extracts the GRF data from the Mujoco data structure.
        """
        # if foot_side == 'left_side':
        #     foot_name = f"{foot_name}_l"
        # elif foot_side == 'right_side':
        #     foot_name = f"{foot_name}_r"

        # Get the body IDs for left and right foot
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, foot_name)


        # Extract the GRF for left and right foot
        grf = mjx_data.cfrc_ext[body_id]


        # self.all_grf_left.append(grf_l)
        # self.all_grf_right.append(grf_r)

        return grf
    

    def filter_start_step_indices(self, all_step_start, min_walk_step_length):
        """
        Filter the start step indices to ensure they are sufficiently spaced apart.
        This function removes start steps that are too close together, based on a minimum step length.
        
        Args:
        all_step_start (list): The list of start step indices.
        min_walk_step_length (int): The minimum length of a walk step to consider it valid.

        Returns:
        list: A filtered list of start step indices.
        """
        # if difference between start_step[i] and start_step[i+1] is smaller than 50 than delete start_step[i]
        filtered_start_steps = []
        for i in range(len(all_step_start) - 1):
            if all_step_start[i + 1] - all_step_start[i] > min_walk_step_length:
                filtered_start_steps.append(all_step_start[i])

        return filtered_start_steps
    


    def get_parameter_per_step(self, parameter_data, start_indices):
        """
        Get the parameter values for each step based on the start indices until the next start index -1 .
        This function extracts the specified parameter values from the saved parameter data.
        
        Args:
        chosen_parameter (str): The parameter to extract. Options: "joint_angles", "joint_velocities", "joint_forces", "joint_torques", "muscle_actions".
        start_indices (list): The start indices of the steps.

        Returns:
        list: A list of parameter values for each step.
        """
        all_parameter_per_step = []
        
        for i in range(len(start_indices) - 1):
            start_index = start_indices[i]
            end_index = start_indices[i + 1] - 1

            parameter_per_step = parameter_data[start_index:end_index]  # Extract the parameter values for the step

            parameter_per_step = jnp.array(parameter_per_step)  # Convert to JAX array for further processing
            all_parameter_per_step.append(parameter_per_step)  # Append the parameter values for the step to the list
        
        return all_parameter_per_step



    

    


    # def extract_steps(self, data, step, all_data):
    #     """
    #     Extract the steps from the data based on the ground reaction forces.
    #     Return the indices of the steps for left and right. Gets the first indix of grf not equal to zero for each step 
    #     """
    #     left_steps = jnp.array([])  # Initialize as an empty array
    #     right_steps = jnp.array([])  # Initialize as an empty array
        

    #     for n in range(len(all_data)):
    #         data = all_data[n]
           
    #         for i in range(data.ncon):
    #             geom1_id = data.contact.geom1[i]
    #             geom2_id = data.contact.geom2[i]

    #             # print('geom1_id:', geom1_id, 'geom2_id:', geom2_id)
                
    #             # Get names if needed
    #             geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
    #             geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)

    #             # print(f"Contact between {geom1_name} and {geom2_name}")

    #             if geom2_name == "foot_box_l":
    #                 # check if force is not zero 
    #                 result = np.zeros(6, dtype=np.float64)  # 3 force + 3 torque
    #                 result = mujoco.mj_contactForce(self.model, data, i, result)
    #                 # Check when 0:3 in results is not zero, save left step index 
    #                 # print(f"Contact Force Result: {result}")
    #                 if np.any(result[0:3] != 0):
    #                     left_step_index = step #data.time[i]
    #                     # print(f"Left step detected at index {left_step_index}")
    #                     left_steps = jnp.array([left_step_index])
    #             elif geom2_name == "foot_box_r":
    #                 # check if force is not zero 
    #                 result = np.zeros(6, dtype=np.float64)  # 3 force + 3 torque
    #                 result = mujoco.mj_contactForce(self.model, data, i, result)
    #                 # print(f"Contact Force Result: {result}")
    #                 # Check when 0:3 in results is not zero, save right step index
    #                 if np.any(result[0:3] != 0):
    #                     right_step_index = step #data.time[i]
    #                     # print(f"Right step detected at index {right_step_index}")
    #                     right_steps = jnp.array([right_step_index]) 

    #     right_indices = []
    #     left_indices = []

    #     # Only get the start of the step, so not consequitive in left_steps and right_steps
    #     if left_steps.size != 0:
    #         for i in range(1, len(left_steps)):
    #             if left_steps[i] == left_steps[i-1] + 1:
    #                 left_steps = jnp.delete(left_steps, i)
    #         for i in range(len(left_steps) - 1):
    #             start_index = left_steps[i]
    #             end_index = left_steps[i + 1] - 1
    #             left_indices.extend([start_index, end_index + 1])
    #         left_indices = jnp.array(left_indices)
    #     if right_steps.size != 0:
    #         for i in range(1, len(right_steps)):
    #             if right_steps[i] == right_steps[i-1] + 1:
    #                 right_steps = jnp.delete(right_steps, i)
    #         for i in range(len(right_steps) - 1):
    #             start_index = right_steps[i]
    #             end_index = right_steps[i + 1] - 1
    #             right_indices.extend([start_index, end_index + 1])
    #         right_indices = jnp.array(right_indices)


    #     # Get start and end of each step 
    #     # left_indices = []
    #     # for i in range(len(left_steps) - 1):
    #     #     start_index = left_steps[i]
    #     #     end_index = left_steps[i + 1] - 1
    #     #     left_indices.extend([start_index, end_index + 1])
    #     # left_indices = jnp.array(left_indices)

    #     # right_indices = []
    #     # for i in range(len(right_steps) - 1):
    #     #     start_index = right_steps[i]
    #     #     end_index = right_steps[i + 1] - 1
    #     #     right_indices.extend([start_index, end_index + 1])
    #     # right_indices = jnp.array(right_indices)
       
    #     return left_indices, right_indices
    

    # def get knee contact force 

    

    
