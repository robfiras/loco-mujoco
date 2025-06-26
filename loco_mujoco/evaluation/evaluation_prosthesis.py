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
import matplotlib.pyplot as plt

from mujoco import mjx

# from omegaconf import OmegaConf, DictConfig


# from loco_mujoco import TaskFactory
# from loco_mujoco.algorithms import PPOJax
# from loco_mujoco.environments.humanoids.skeleton_prosthesis import MjxSkeletonMuscleProsthesis
# from loco_mujoco.core.control_functions.skeleton_muscle import SkeletonMuscleControlFunction


# from omegaconf import OmegaConf

# os.environ["MUJOCO_GL"] = "egl"

from loco_mujoco.utils import MetricsHandler
import numpy as np
import itertools




class ProsthesisMetricsHandler(): #MetricsHandler): #MetricsHandler):
    """
    Metrics handler for prosthesis evaluation.
    This class extends the MetricsHandler to include specific metrics for prosthesis evaluation.
    """

    def __init__(self, env):
        """
        Initialize the ProsthesisMetricsHandler.
        Args:
        config (DictConfig): The configuration dictionary.
        env (MjxSkeletonMuscleProsthesis): The environment instance.
        """
        #self.config = config
        self.env = env
        self.model = env.get_model()  # Get the model from the environment

        # self.all_step_contact_right = []  # List to store start steps for right foot
        # self.all_step_contact_left = []  # List to store start steps for left foot
        # self.all_step_start_right = []  # List to store start steps for right foot
        # self.all_step_start_left = []  # List to store start steps for left foot
        # self.all_grf_left = []  # List to store ground reaction forces for left foot
        # self.all_grf_right = []  # List to store ground reaction forces for right foot
        # super().__init__(config, env)


    # def get_joint_group(self, which_joints):
    #     """
    #     Get the joint angles based on the specified joint group.
        
    #     Args:
    #     which_joints (str): The joint group to extract. Options: "all_joints", "left_side", "right_side".
        
    #     Returns:
    #     list: A list of joint names corresponding to the specified group.
    #     """
    #     joint_groups = {
    #         "all_joints": [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)],
    #         "leg_joints": ["hip_flexion", "hip_adduction", "hip_rotation", "knee_angle", "ankle_angle", "toe_angle", "mtp_angle"],
    #         "hip_joints": ["hip_flexion", "hip_adduction", "hip_rotation"],
    #         "hip_flexion": ["hip_flexion_l"],
    #         "hip_abduction": ["hip_adduction"],
    #         "knee_joint": ["knee_angle"],
    #         "ankle_joint": ["ankle_angle"],
    #         "talus_joint": ["talus_angle"],
    #         "mtp_angle": ["mtp_angle"],
    #     }

    def get_joint_angles(self,mjx_data): 
        """ Get all joint angles and save in dictionary with joint name as key and angle as value.
        """
        joint_angles = {}
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            qpos_address = self.model.jnt_qposadr[i]
            # joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            joint_angle = mjx_data.qpos[...,qpos_address]
            joint_angles[joint_name] = joint_angle
        return joint_angles
    

    def get_joint_vels(self, mjx_data):
        """ Get all joint velocities and save in dictionary with joint name as key and velocity as value.
        """
        joint_velocities = {}
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            dof_address = self.model.jnt_dofadr[i]
            joint_velocity = mjx_data.qvel[...,dof_address]
            joint_velocities[joint_name] = joint_velocity
        return joint_velocities
    
    def get_joint_frces(self, mjx_data):
        """ Get all joint forces and save in dictionary with joint name as key and force as value.
        """
        joint_forces_constraint = {}
        joint_forces_smooth = {}
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            dof_address = self.model.jnt_dofadr[i]
            joint_force_constraint = mjx_data.qfrc_constraint[...,dof_address] # constraint force; joint limits and contacts etc. 
            joint_force_smooth = mjx_data.qfrc_smooth[...,dof_address] # net unconstrained force; Sum of all forces, e.g., gravity, applied torques etc,
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
            dof_address = self.model.jnt_dofadr[i]
            joint_torque = mjx_data.qfrc_actuator[...,dof_address]
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
        extracted_actions = [action[..., k] for k in action_indices]
        # extracted_action = action[jnp.array(action_indices)]

        return jnp.array(extracted_actions)
    
    def calc_mean_grf(self, data):
        f_contact_frame_r = np.zeros(3)
        f_contact_frame_l = np.zeros(3)
        geom1_id = data.contact.geom1
        geom2_id = data.contact.geom2

        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        foot_box_r_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'foot_box_r')
        foot_box_l_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'foot_box_l')

        mj_data = mjx.get_data(self.model, data)
        geom1_id_is_floor = geom1_id == floor_id
        if (geom1_id_is_floor).all():
            for n in range(mj_data[0].ncon):
                # Calculate contact force
                # mj_contactForce takes (model, data, contact_id, force_array)
                # The force_array is a 6-element array: (f_x, f_y, f_z, t_x, t_y, t_z)
                contact_force_raw = np.zeros((6,1), dtype=np.float64)
                mujoco.mj_contactForce(self.model, mj_data[0], n, contact_force_raw)

                frX = mj_data[0].contact.frame[0][0:3] #[0][n][0]
                frY = mj_data[0].contact.frame[0][3:6] #[0][n][1]
                frZ = mj_data[0].contact.frame[0][6:9] #[0][n][2]

                # `contact_force_raw[0:3]` are the force components in the contact frame
                
                if mj_data[0].contact.geom2[n] == foot_box_r_id:
                    # f_contact_frame_r += contact_force_raw[0:3]
                    f_contact_frame_r += frX * contact_force_raw[0] + frY * contact_force_raw[1] + frZ * contact_force_raw[2]
                elif mj_data[0].contact.geom2[n] == foot_box_l_id:
                    # f_contact_frame_l += contact_force_raw[0:3]
                    f_contact_frame_l += frX * contact_force_raw[0] + frY * contact_force_raw[1] + frZ * contact_force_raw[2]

            # force_in_world_frame_l = frX * f_contact_frame_l[0] + frY * f_contact_frame_l[1] + frZ * f_contact_frame_l[2]
            # force_in_world_frame_r = frX * f_contact_frame_r[0] + frY * f_contact_frame_r[1] + frZ * f_contact_frame_r[2]

        return f_contact_frame_l, f_contact_frame_r



    # def evaluate_action_symmetry(self, all_data, muscle_name, left_indices, right_indices):
    #     """
    #     Evaluate the symmetry of the actions in the environment states.
    #     This function calculates the symmetry of the actions for left and right sides.
    #     """

    #     muscle_name_left = [name+'_l' for name in muscle_name]
    #     muscle_name_right = [name+'_r' for name in muscle_name]

    #     action_indices_left = []
    #     action_indices_right = []
    #     all_actions_left = []
    #     all_actions_right = []

    #     for n in range(len(all_data)):
    #         data = all_data[n]

    #         # Get indices for the specified muscles
    #         for i in range(self.model.nu):
    #             name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    #             if name in muscle_name_left:
    #                 print(f"Actuator {i}: {name}")
    #                 action_indices_left.append(i)
    #             if name in muscle_name_right:
    #                 print(f"Actuator {i}: {name}")
    #                 action_indices_right.append(i)


    #         for i in range(len(left_indices)):
    #             for a in action_indices_left:
    #                 actions_left = data[i].ctrl[a][left_indices[i][0]:left_indices[i][1]]
    #                 all_actions_left.append(actions_left)
    #                 print(f"Left Action {a} at step {i}: {actions_left}")

    #         for i in range(len(right_indices)):
    #             for a in action_indices_right:
    #                 actions_right = data[i].ctrl[a][right_indices[i][0]:right_indices[i][1]]
    #                 all_actions_right.append(actions_right)
    #                 print(f"Right Action {a} at step {i}: {actions_right}")

    #         # Calculate the symmetry of the actions
    #         symmetry_scores = []
    #         for i in range(len(all_actions_left)):
    #             left_action = all_actions_left[i]
    #             right_action = all_actions_right[i]
    #             if len(left_action) != len(right_action):
    #                 raise ValueError("Left and right actions must have the same length for symmetry evaluation.")
    #             # Calculate symmetry score as the absolute difference between left and right actions
    #             symmetry_score = jnp.abs(left_action - right_action).mean()
    #             symmetry_scores.append(symmetry_score)
    #             print(f"Symmetry Score for step {i}: {symmetry_score}")


    def get_contact_steps(self, data, step):
        contact_index_r = []  # List to store steps where left foot contacts the ground
        contact_index_l = []  # List to store steps where right foot contacts the ground
       
        geom_distance = data.contact.dist
        geom1_id = data.contact.geom1
        geom2_id = data.contact.geom2

        # check that geom1_index is all 0 and geom_name is floor 
        # get from geom name to index 
        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        foot_box_r_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'foot_box_r')
        foot_box_l_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'foot_box_l')

        geom1_id_is_floor = geom1_id == floor_id
        # if all geom1_id_is lfoor true: get indices wheren dist is smaller= 0 
        if (geom1_id_is_floor).all(): 
            penetration = geom_distance <= 0
            # get indices where geom_distance is smaller than 0
            penetration_indices_geom = jnp.where(penetration[0])[0]
            # check which floot_box_r_id or floot_box_l_id are at penetraction_indices_geom 
            for n in penetration_indices_geom:
                if geom2_id[...,n] == foot_box_r_id: 
                    contact_index_r = step
                elif geom2_id[...,n] == foot_box_l_id: 
                    contact_index_l = step

        return contact_index_l, contact_index_r 

    

    # def get_contact_steps(self, mjx_data, foot_side,  step):
    #     all_step_contact = []  # List to store steps where left foot contacts the ground
    #     all_step_contact = []  # List to store steps where right foot contacts the ground
    #     if 'left' in foot_side:
    #         side_suffix = '_l'
    #     elif 'right' in foot_side:
    #         side_suffix = '_r'
    #     else:
    #         raise ValueError("Invalid foot_side. Expected 'left' or 'right'.")
    #     # Collect all geom2 names and their corresponding distances
    #     for n in range(mjx_data.ncon):
    #         geom2_id = mjx_data.contact.geom2[n]
    #         geom_name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)
    #         geom_distance = mjx_data.contact.dist[n]
    #         # Check if the geom_name2 matches the requested side and the contact is happening
    #         if side_suffix in geom_name2 and (geom_distance <= 0).any():
    #           all_step_contact.append(step)
    #     return contact_index_l, contact_index_r



        

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
        if all_step_contact[0]!= 0:
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
        # Force sensor: creates a 3-axis force sensor. The sensor outputs three numbers, which are the interaction 
        # force between a child and a parent body, expressed in the site frame defining the sensor. The convention 
        # is that the site is attached to the child body, and the force points from the child towards the parent. 
        # The computation here takes into account all forces acting on the system, including contacts as well as 
        # external perturbations. 

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
            #return sensor_data
        else: 
            sensor_data = {}
            # get sensor_names and iterate through all sensors
            for sensor_id in range(self.model.nsensor):
                sensor_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
                start_idx = sensor_id * 3
                end_idx = start_idx + 3
                sensor_data[sensor_name] = mjx_data.sensordata[0][start_idx:end_idx]
                # sensor_data[sensor_name] = mjx_data.sensordata[...,sensor_id]
        
        return sensor_data, sensor_names
                # print(f"Sensor {sensor_name} data: {sensor_data}")
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
        grf = mjx_data.cfrc_ext[0][body_id]


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

    
class PostProcessMetricsHandler(): 

    def get_start_steps(self, all_step_contact, min_walk_step_length):
        """
        Get the start steps for left and right foot based on contact data.
        This function processes the contact data to determine the start of each step.
        Args:
        all_step_contact (list): A list of contact indices where steps occur.
        min_walk_step_length (int): The minimum length of a walk step to consider it valid.
        """
        all_step_start  = []
        if all_step_contact[0]!= 0:
            all_step_start = [all_step_contact[0]]
        for m in range(len(all_step_contact) - 1):
            if all_step_contact[m + 1] - all_step_contact[m] > 1: #min_walk_step_length:
                all_step_start.append(all_step_contact[m+1])
        
        filtered_start_steps = []
        for i in range(len(all_step_start) - 1):
            if all_step_start[i + 1] - all_step_start[i] > min_walk_step_length:
                filtered_start_steps.append(all_step_start[i])
        # if all_step_start[m] - all_start_step[m+1] > min_walk_step_length:
        #     all_step_start.append(all_step_contact[m+1])
        return filtered_start_steps #all_step_start

    @staticmethod
    def _get_step_slices(data, step_indices):
        steps = []
        for i in range(len(step_indices) - 1):
            start = step_indices[i]
            end = step_indices[i + 1]
            steps.append(data[start:end])
        return steps

    def _plot_steps(
        self,
        steps, 
        ax, 
        interp_mode="none", 
        interp_len=None, 
        deg=False, 
        legend_prefix="", 
        side="left", 
        col=None
    ):
        for i, step in enumerate(steps):
            if len(step) < 2:
                continue
            if col is not None:
                y = [float(np.array(a)[0, col] if np.array(a).ndim > 1 else np.array(a)[col]) for a in step]
            else:
                y = [float(a) for a in step]
            if interp_mode == "interp":
                x_old = np.linspace(0, 1, len(y))
                x_new = np.linspace(0, 1, interp_len or min([len(s) for s in steps if len(s) > 1]))
                y = np.interp(x_new, x_old, y)
            if deg:
                y = np.rad2deg(y)
            ax.plot(y, label=f"{legend_prefix}{side} step {i+1}")

    def plot_parameter_per_step(
        self,
        data,
        step_indices_left,
        step_indices_right,
        param_name="parameter",
        interp_mode="none",
        interp_len=None,
        columns=None,
        ylabel=None,
        title=None,
        legend_prefix="",
        deg=False,
    ):
        if isinstance(data, dict):
            nplots = len(data)
            fig, axes = plt.subplots(nplots, 1, figsize=(10, 3 * nplots), sharex=True)
            if nplots == 1:
                axes = [axes]
            for ax, (key, arr) in zip(axes, data.items()):
                self.plot_parameter_per_step(
                    arr,
                    step_indices_left,
                    step_indices_right,
                    param_name=param_name,
                    interp_mode=interp_mode,
                    interp_len=interp_len,
                    columns=columns,
                    ylabel=ylabel or param_name,
                    title=title or key,
                    legend_prefix=key + " ",
                    deg=deg,
                )
                ax.set_title(key)
            plt.tight_layout()
            plt.show()
            return

        if isinstance(data, list) and len(data) > 0 and hasattr(data[0], "shape"):
            arr0 = np.array(data[0])
            ncols = arr0.shape[-1] if arr0.ndim > 1 else 1
            if columns is None:
                columns = list(range(ncols))
            elif isinstance(columns, int):
                columns = [columns]
            nplots = len(columns)
            fig, axes = plt.subplots(nplots, 1, figsize=(10, 3 * nplots), sharex=True)
            if nplots == 1:
                axes = [axes]
            for idx, col in enumerate(columns):
                ax = axes[idx]
                if step_indices_left and len(step_indices_left) > 1:
                    steps = self._get_step_slices(data, step_indices_left)
                    self._plot_steps(
                        steps, ax, interp_mode, interp_len, deg, legend_prefix, "left", col
                    )
                if step_indices_right and len(step_indices_right) > 1:
                    steps = self._get_step_slices(data, step_indices_right)
                    self._plot_steps(
                        steps, ax, interp_mode, interp_len, deg, legend_prefix, "right", col
                    )
                ax.set_ylabel(ylabel or f"{param_name} col {col}")
                ax.legend()
                ax.set_title(title or f"{param_name} col {col}")
            axes[-1].set_xlabel("Interpolated Step (%)" if interp_mode == "interp" else "Step index")
            plt.tight_layout()
            plt.show()
            return

        fig, ax = plt.subplots(figsize=(10, 3))
        if step_indices_left and len(step_indices_left) > 1:
            steps = self._get_step_slices(data, step_indices_left)
            self._plot_steps(
                steps, ax, interp_mode, interp_len, deg, legend_prefix, "left"
            )
        if step_indices_right and len(step_indices_right) > 1:
            steps = self._get_step_slices(data, step_indices_right)
            self._plot_steps(
                steps, ax, interp_mode, interp_len, deg, legend_prefix, "right"
            )
        ax.set_ylabel(ylabel or param_name)
        ax.set_title(title or param_name)
        ax.legend()
        ax.set_xlabel("Interpolated Step (%)" if interp_mode == "interp" else "Step index")
        plt.tight_layout()
        plt.show()

    def plot_parameter_mean_per_step(
        self,
        data,
        step_indices_left,
        step_indices_right,
        param_name="parameter",
        interp_mode="none",
        interp_len=None,
        columns=None,
        ylabel=None,
        title=None,
        legend_prefix="",
        deg=False,
    ):

        def compute_mean_std(steps, interp_mode, interp_len, deg, col=None):
            y_steps = []
            for step in steps:
                if len(step) < 2:
                    continue
                if col is not None:
                    y = [float(np.array(a)[0, col] if np.array(a).ndim > 1 else np.array(a)[col]) for a in step]
                else:
                    y = [float(a) for a in step]
                if interp_mode == "interp":
                    x_old = np.linspace(0, 1, len(y))
                    x_new = np.linspace(0, 1, interp_len or min([len(s) for s in steps if len(s) > 1]))
                    y = np.interp(x_new, x_old, y)
                if deg:
                    y = np.rad2deg(y)
                y_steps.append(y)
            if not y_steps:
                return None, None
            y_steps = np.array(y_steps)
            mean = np.mean(y_steps, axis=0)
            std = np.std(y_steps, axis=0)
            return mean, std

        if isinstance(data, dict):
            nplots = len(data)
            fig, axes = plt.subplots(nplots, 1, figsize=(10, 3 * nplots), sharex=True)
            if nplots == 1:
                axes = [axes]
            for ax, (key, arr) in zip(axes, data.items()):
                self.plot_parameter_mean_per_step(
                    arr,
                    step_indices_left,
                    step_indices_right,
                    param_name=param_name,
                    interp_mode=interp_mode,
                    interp_len=interp_len,
                    columns=columns,
                    ylabel=ylabel or param_name,
                    title=title or key,
                    legend_prefix=key + " ",
                    deg=deg,
                )
                ax.set_title(key)
            plt.tight_layout()
            plt.show()
            return

        if isinstance(data, list) and len(data) > 0 and hasattr(data[0], "shape"):
            arr0 = np.array(data[0])
            ncols = arr0.shape[-1] if arr0.ndim > 1 else 1
            if columns is None:
                columns = list(range(ncols))
            elif isinstance(columns, int):
                columns = [columns]
            nplots = len(columns)
            fig, axes = plt.subplots(nplots, 1, figsize=(10, 3 * nplots), sharex=True)
            if nplots == 1:
                axes = [axes]
            for idx, col in enumerate(columns):
                ax = axes[idx]
                if step_indices_left and len(step_indices_left) > 1:
                    steps = self._get_step_slices(data, step_indices_left)
                    mean, std = compute_mean_std(steps, interp_mode, interp_len, deg, col)
                    if mean is not None:
                        x = np.arange(len(mean))
                        ax.plot(x, mean, label=f"{legend_prefix}left mean")
                        ax.fill_between(x, mean - std, mean + std, alpha=0.2)
                if step_indices_right and len(step_indices_right) > 1:
                    steps = self._get_step_slices(data, step_indices_right)
                    mean, std = compute_mean_std(steps, interp_mode, interp_len, deg, col)
                    if mean is not None:
                        x = np.arange(len(mean))
                        ax.plot(x, mean, label=f"{legend_prefix}right mean")
                        ax.fill_between(x, mean - std, mean + std, alpha=0.2)
                ax.set_ylabel(ylabel or f"{param_name} col {col}")
                ax.legend()
                ax.set_title(title or f"{param_name} col {col}")
            axes[-1].set_xlabel("Interpolated Step (%)" if interp_mode == "interp" else "Step index")
            plt.tight_layout()
            plt.show()
            return

        fig, ax = plt.subplots(figsize=(10, 3))
        if step_indices_left and len(step_indices_left) > 1:
            steps = self._get_step_slices(data, step_indices_left)
            mean, std = compute_mean_std(steps, interp_mode, interp_len, deg)
            if mean is not None:
                x = np.arange(len(mean))
                ax.plot(x, mean, label=f"{legend_prefix}left mean")
                ax.fill_between(x, mean - std, mean + std, alpha=0.2)
        if step_indices_right and len(step_indices_right) > 1:
            steps = self._get_step_slices(data, step_indices_right)
            mean, std = compute_mean_std(steps, interp_mode, interp_len, deg)
            if mean is not None:
                x = np.arange(len(mean))
                ax.plot(x, mean, label=f"{legend_prefix}right mean")
                ax.fill_between(x, mean - std, mean + std, alpha=0.2)
        ax.set_ylabel(ylabel or param_name)
        ax.set_title(title or param_name)
        ax.legend()
        ax.set_xlabel("Interpolated Step (%)" if interp_mode == "interp" else "Step index")
        plt.tight_layout()
        plt.show()

    def plot_joint_parameter_per_step(
        self,
        joint_pairs,
        joint_data,
        step_indices_left,
        step_indices_right,
        parameter_names,
        convert_to_deg=False,
        interp_mode="interp",
        interp_len=None,
    ):
        def get_step_indices(joint):
            if joint.endswith("_l"):
                return step_indices_left
            elif joint.endswith("_r"):
                return step_indices_right
            else:
                return [step_indices_left, step_indices_right]

        if interp_mode == "interp" and interp_len is None:
            step_lengths = []
            for pair in joint_pairs:
                for joint in pair:
                    indices = get_step_indices(joint)
                    if isinstance(indices, list) and indices and isinstance(indices[0], list):
                        for idx in indices:
                            if idx and len(idx) > 1:
                                step_lengths += [idx[i+1] - idx[i] for i in range(len(idx)-1)]
                    elif indices and len(indices) > 1:
                        step_lengths += [indices[i+1] - indices[i] for i in range(len(indices)-1)]
            interp_len = min(step_lengths) if step_lengths else 1

        n_rows = len(joint_pairs)
        n_cols = len(parameter_names)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3 * n_rows), sharex=True)
        if n_rows == 1:
            axes = [axes]
        if n_cols == 1:
            axes = [[ax] for ax in axes] if n_rows > 1 else [[axes]]

        for row_idx, pair in enumerate(joint_pairs):
            for col_idx, parameter_name in enumerate(parameter_names):
                ax = axes[row_idx][col_idx]
                for joint in pair:
                    if joint not in joint_data or parameter_name not in joint_data[joint]:
                        continue
                    indices = get_step_indices(joint)
                    if isinstance(indices, list) and indices and isinstance(indices[0], list):
                        for side_indices, side in zip(indices, ["left", "right"]):
                            self._plot_joint_steps(
                                ax, joint, joint_data[joint][parameter_name], side_indices, interp_mode, interp_len, convert_to_deg, parameter_name, side
                            )
                    else:
                        side = "left" if joint.endswith("_l") else "right" if joint.endswith("_r") else ""
                        self._plot_joint_steps(
                            ax, joint, joint_data[joint][parameter_name], indices, interp_mode, interp_len, convert_to_deg, parameter_name, side
                        )
                ax.set_ylabel(f"{parameter_name.capitalize()} ({'deg' if convert_to_deg and parameter_name == 'angle' else 'rad'})")
                ax.set_title(" / ".join(pair) + f" - {parameter_name}")
                ax.legend()
        for col_idx in range(n_cols):
            axes[-1][col_idx].set_xlabel("Interpolated Step (%)" if interp_mode == "interp" else "Step index")
        plt.tight_layout()
        plt.show()

    def _plot_joint_steps(self, ax, joint, data, step_indices, interp_mode, interp_len, convert_to_deg, parameter_name, side):
        if step_indices is None or len(step_indices) < 2:
            return
        for i in range(len(step_indices) - 1):
            start = step_indices[i]
            end = step_indices[i + 1]
            y = [float(np.array(a).squeeze()) for a in data[start:end]]
            if len(y) < 2:
                continue
            if interp_mode == "interp":
                x_old = np.linspace(0, 1, len(y))
                x_new = np.linspace(0, 1, interp_len)
                y = np.interp(x_new, x_old, y)
            if convert_to_deg and parameter_name in ("angle", "velocity"):
                y = np.rad2deg(y)
            label = f"{joint} {side} step {i+1}" if side else f"{joint} step {i+1}"
            ax.plot(y, label=label)

    def plot_joint_parameter_mean_per_step(
        self,
        joint_pairs,
        joint_data,
        step_indices_left,
        step_indices_right,
        parameter_names,
        convert_to_deg=False,
        interp_mode="interp",
        interp_len=None,
    ):
        def get_step_indices(joint):
            if joint.endswith("_l"):
                return step_indices_left
            elif joint.endswith("_r"):
                return step_indices_right
            else:
                return [step_indices_left, step_indices_right]

        if interp_mode == "interp" and interp_len is None:
            step_lengths = []
            for pair in joint_pairs:
                for joint in pair:
                    indices = get_step_indices(joint)
                    if isinstance(indices, list) and indices and isinstance(indices[0], list):
                        for idx in indices:
                            if idx and len(idx) > 1:
                                step_lengths += [idx[i+1] - idx[i] for i in range(len(idx)-1)]
                    elif indices and len(indices) > 1:
                        step_lengths += [indices[i+1] - indices[i] for i in range(len(indices)-1)]
            interp_len = min(step_lengths) if step_lengths else 1

        n_rows = len(joint_pairs)
        n_cols = len(parameter_names)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3 * n_rows), sharex=True)
        if n_rows == 1:
            axes = [axes]
        if n_cols == 1:
            axes = [[ax] for ax in axes] if n_rows > 1 else [[axes]]

        for row_idx, pair in enumerate(joint_pairs):
            for col_idx, parameter_name in enumerate(parameter_names):
                ax = axes[row_idx][col_idx]
                for joint in pair:
                    if joint not in joint_data or parameter_name not in joint_data[joint]:
                        continue
                    indices = get_step_indices(joint)
                    if isinstance(indices, list) and indices and isinstance(indices[0], list):
                        for side_indices, side in zip(indices, ["left", "right"]):
                            self._plot_joint_mean_std(
                                ax, joint, joint_data[joint][parameter_name], side_indices, interp_mode, interp_len, convert_to_deg, parameter_name, side
                            )
                    else:
                        side = "left" if joint.endswith("_l") else "right" if joint.endswith("_r") else ""
                        self._plot_joint_mean_std(
                            ax, joint, joint_data[joint][parameter_name], indices, interp_mode, interp_len, convert_to_deg, parameter_name, side
                        )
                ax.set_ylabel(f"{parameter_name.capitalize()} ({'deg' if convert_to_deg and parameter_name == 'angle' else 'rad'})")
                ax.set_title(" / ".join(pair) + f" - {parameter_name} (mean ± std)")
                ax.legend()
        for col_idx in range(n_cols):
            axes[-1][col_idx].set_xlabel("Interpolated Step (%)" if interp_mode == "interp" else "Step index")
        plt.tight_layout()
        plt.show()

    def _plot_joint_mean_std(self, ax, joint, data, step_indices, interp_mode, interp_len, convert_to_deg, parameter_name, side):
        if step_indices is None or len(step_indices) < 2:
            return
        y_steps = []
        for i in range(len(step_indices) - 1):
            start = step_indices[i]
            end = step_indices[i + 1]
            y = [float(np.array(a).squeeze()) for a in data[start:end]]
            if len(y) < 2:
                continue
            if interp_mode == "interp":
                x_old = np.linspace(0, 1, len(y))
                x_new = np.linspace(0, 1, interp_len)
                y = np.interp(x_new, x_old, y)
            if convert_to_deg and parameter_name in ("angle", "velocity"):
                y = np.rad2deg(y)
            y_steps.append(y)
        if not y_steps:
            return
        y_steps = np.array(y_steps)
        mean = np.mean(y_steps, axis=0)
        std = np.std(y_steps, axis=0)
        label = f"{joint} {side} mean" if side else f"{joint} mean"
        x = np.arange(len(mean))
        ax.plot(x, mean, label=label)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2)

    def plot_muscle_activations_per_step(
        self,
        muscle_actions,
        muscle_names,
        step_indices_left,
        step_indices_right,
        interp_mode="interp",
        interp_len=None,
    ):
        if interp_mode == "interp" and interp_len is None:
            step_lengths = []
            for name in muscle_names:
                steps = step_indices_left if name.endswith("_l") else step_indices_right if name.endswith("_r") else None
                if steps and len(steps) > 1:
                    step_lengths += [steps[i+1] - steps[i] for i in range(len(steps)-1)]
            interp_len = min(step_lengths) if step_lengths else 1

        n = len(muscle_names)
        fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
        if n == 1:
            axes = [axes]
        for i, name in enumerate(muscle_names):
            ax = axes[i]
            actions = muscle_actions.get(name, None)
            if actions is None:
                ax.set_title(f"{name} (no data)")
                continue
            if name.endswith("_l"):
                steps = step_indices_left
                side = "Left"
            elif name.endswith("_r"):
                steps = step_indices_right
                side = "Right"
            else:
                ax.set_title(f"{name} (unknown side)")
                continue
            if steps and len(steps) > 1:
                for j in range(len(steps) - 1):
                    start = steps[j]
                    end = steps[j + 1]
                    y = [float(np.array(a).squeeze()) for a in actions[start:end]]
                    if len(y) < 2:
                        continue
                    if interp_mode == "interp":
                        x_old = np.linspace(0, 1, len(y))
                        x_new = np.linspace(0, 1, interp_len)
                        y = np.interp(x_new, x_old, y)
                    ax.plot(y, label=f"{side} step {j+1}")
            ax.set_title(f"{name} Activation per Step")
            ax.set_ylabel("Activation Level")
            ax.legend()
        axes[-1].set_xlabel("Interpolated Step (%)" if interp_mode == "interp" else "Step index")
        plt.tight_layout()
        plt.show()


    def plot_muscle_activations_mean_per_step(
        self,
        muscle_actions,
        muscle_names,
        step_indices_left,
        step_indices_right,
        interp_mode="interp",
        interp_len=None,
    ):
        if interp_mode == "interp" and interp_len is None:
            step_lengths = []
            for name in muscle_names:
                steps = step_indices_left if name.endswith("_l") else step_indices_right if name.endswith("_r") else None
                if steps and len(steps) > 1:
                    step_lengths += [steps[i+1] - steps[i] for i in range(len(steps)-1)]
            interp_len = min(step_lengths) if step_lengths else 1

        n = len(muscle_names)
        fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
        if n == 1:
            axes = [axes]
        for i, name in enumerate(muscle_names):
            ax = axes[i]
            actions = muscle_actions.get(name, None)
            if actions is None:
                ax.set_title(f"{name} (no data)")
                continue
            if name.endswith("_l"):
                steps = step_indices_left
                side = "Left"
            elif name.endswith("_r"):
                steps = step_indices_right
                side = "Right"
            else:
                ax.set_title(f"{name} (unknown side)")
                continue
            y_steps = []
            if steps and len(steps) > 1:
                for j in range(len(steps) - 1):
                    start = steps[j]
                    end = steps[j + 1]
                    y = [float(np.array(a).squeeze()) for a in actions[start:end]]
                    if len(y) < 2:
                        continue
                    if interp_mode == "interp":
                        x_old = np.linspace(0, 1, len(y))
                        x_new = np.linspace(0, 1, interp_len)
                        y = np.interp(x_new, x_old, y)
                    y_steps.append(y)
            if y_steps:
                y_steps = np.array(y_steps)
                mean = np.mean(y_steps, axis=0)
                std = np.std(y_steps, axis=0)
                x = np.arange(len(mean))
                ax.plot(x, mean, label=f"{side} mean")
                ax.fill_between(x, mean - std, mean + std, alpha=0.2)
            ax.set_title(f"{name} Activation Mean per Step")
            ax.set_ylabel("Activation Level")
            ax.legend()
        axes[-1].set_xlabel("Interpolated Step (%)" if interp_mode == "interp" else "Step index")
        plt.tight_layout()
        plt.show()






    def plot_muscle_activation_symmetry(
        self,
        muscle_names_list,
        all_actuator_names,
        all_actions,
        all_step_start_left,
        all_step_start_right,
        interp_len=100,
    ):
        def find_actuator_index(actuator_names, name):
            try:
                return actuator_names.index(name)
            except ValueError:
                print(f"{name} not found in all_actuator_names.")
                return None

        def collect_activations(actions, idx):
            return [actions[i][0][idx] for i in range(len(actions))]

        def compute_interpolated_steps(acts, step_starts, interp_len):
            steps = []
            if step_starts and len(step_starts) > 1:
                for j in range(len(step_starts) - 1):
                    start, end = step_starts[j], step_starts[j + 1]
                    y = [float(a) for a in acts[start:end]]
                    if len(y) < 2:
                        continue
                    x_old = np.linspace(0, 1, len(y))
                    x_new = np.linspace(0, 1, interp_len)
                    y_interp = np.interp(x_new, x_old, y)
                    steps.append(y_interp)
            return steps

        for muscle_name in muscle_names_list:
            left_name = muscle_name + "_l"
            right_name = muscle_name + "_r"
            idx_left = find_actuator_index(all_actuator_names, left_name)
            idx_right = find_actuator_index(all_actuator_names, right_name)
            if idx_left is None or idx_right is None:
                continue

            left_acts = collect_activations(all_actions, idx_left)
            right_acts = collect_activations(all_actions, idx_right)

            left_steps = compute_interpolated_steps(left_acts, all_step_start_left, interp_len)
            right_steps = compute_interpolated_steps(right_acts, all_step_start_right, interp_len)

            if left_steps and right_steps:
                mean_left = np.mean(left_steps, axis=0)
                std_left = np.std(left_steps, axis=0)
                mean_right = np.mean(right_steps, axis=0)
                std_right = np.std(right_steps, axis=0)
                diff = mean_left - mean_right
                x = np.linspace(0, 100, interp_len)
                plt.figure(figsize=(10, 5))
                plt.plot(x, mean_left, label=f"{muscle_name}_l mean", color='blue')
                plt.fill_between(x, mean_left - std_left, mean_left + std_left, color='blue', alpha=0.2)
                plt.plot(x, mean_right, label=f"{muscle_name}_r mean", color='red')
                plt.fill_between(x, mean_right - std_right, mean_right + std_right, color='red', alpha=0.2)
                plt.plot(x, diff, label="Difference (left - right)", color='black', linestyle='--')
                plt.axhline(0, color='gray', linestyle=':', linewidth=1)
                plt.title(f"Activation symmetry: {muscle_name}_l vs {muscle_name}_r")
                plt.xlabel("Interpolated Step (%)")
                plt.ylabel("Activation / Difference")
                plt.legend()
                plt.tight_layout()
                plt.show()
            else:
                print(f"Not enough steps for {muscle_name} to compute difference.")





    def plot_grf_component_symmetry(
        self,
        grf_components,
        all_grf_l,
        all_grf_r,
        all_step_start_left,
        all_step_start_right,
        interp_len=100,
    ):
        """
        Plot mean and std of GRF components for left and right steps, and their difference.
        """
        for i, comp in enumerate(grf_components):
            # Compute mean per time step in a walk step for left
            left_steps = []
            if all_grf_l and all_step_start_left and len(all_step_start_left) > 1:
                for j in range(len(all_step_start_left) - 1):
                    start, end = all_step_start_left[j], all_step_start_left[j + 1]
                    y = [float(np.array(a)[i]) for a in all_grf_l[start:end]]
                    if len(y) < 2:
                        continue
                    x_old = np.linspace(0, 1, len(y))
                    x_new = np.linspace(0, 1, interp_len)
                    y_interp = np.interp(x_new, x_old, y)
                    left_steps.append(y_interp)
            # Compute mean per time step in a walk step for right
            right_steps = []
            if all_grf_r and all_step_start_right and len(all_step_start_right) > 1:
                for j in range(len(all_step_start_right) - 1):
                    start, end = all_step_start_right[j], all_step_start_right[j + 1]
                    y = [float(np.array(a)[i]) for a in all_grf_r[start:end]]
                    if len(y) < 2:
                        continue
                    x_old = np.linspace(0, 1, len(y))
                    x_new = np.linspace(0, 1, interp_len)
                    y_interp = np.interp(x_new, x_old, y)
                    right_steps.append(y_interp)
            # Calculate mean and std
            if left_steps and right_steps:
                mean_left = np.mean(left_steps, axis=0)
                std_left = np.std(left_steps, axis=0)
                mean_right = np.mean(right_steps, axis=0)
                std_right = np.std(right_steps, axis=0)
                diff = mean_left - mean_right
                x = np.linspace(0, 100, interp_len)
                plt.figure(figsize=(10, 5))
                # Plot mean and std for left
                plt.plot(x, mean_left, label=f"Left mean", color='blue')
                plt.fill_between(x, mean_left - std_left, mean_left + std_left, color='blue', alpha=0.2)
                # Plot mean and std for right
                plt.plot(x, mean_right, label=f"Right mean", color='red')
                plt.fill_between(x, mean_right - std_right, mean_right + std_right, color='red', alpha=0.2)
                # Plot difference
                plt.plot(x, diff, label="Difference (left - right)", color='black', linestyle='--')
                plt.axhline(0, color='gray', linestyle=':', linewidth=1)
                plt.title(f"GRF {comp}: mean, std, and difference per time step")
                plt.xlabel("Interpolated Step (%)")
                plt.ylabel(f"{comp} (N or Nm)")
                plt.legend()
                plt.tight_layout()
                plt.show()
            else:
                print(f"Not enough steps for GRF {comp} to compute difference.")



    def plot_sensor_force_symmetry(
        self,
        sensor_force_names,
        all_sensor_force_data,
        all_step_start_left,
        all_step_start_right,
        interp_len=100,
    ):
        sensor_pairs = []
        for name in sensor_force_names:
            if name.startswith("left_"):
                right_name = name.replace("left_", "right_")
                if right_name in sensor_force_names:
                    sensor_pairs.append((name, right_name))

        for left_sensor, right_sensor in sensor_pairs:
            left_data = all_sensor_force_data[left_sensor]
            right_data = all_sensor_force_data[right_sensor]

            # Collect interpolated steps for left
            left_steps = []
            if all_step_start_left and len(all_step_start_left) > 1:
                for j in range(len(all_step_start_left) - 1):
                    start, end = all_step_start_left[j], all_step_start_left[j + 1]
                    y = [float(np.array(a)[1]) for a in left_data[start:end]]
                    if len(y) < 2:
                        continue
                    x_old = np.linspace(0, 1, len(y))
                    x_new = np.linspace(0, 1, interp_len)
                    y_interp = np.interp(x_new, x_old, y)
                    left_steps.append(y_interp)
            # Collect interpolated steps for right
            right_steps = []
            if all_step_start_right and len(all_step_start_right) > 1:
                for j in range(len(all_step_start_right) - 1):
                    start, end = all_step_start_right[j], all_step_start_right[j + 1]
                    y = [float(np.array(a)[1]) for a in right_data[start:end]]
                    if len(y) < 2:
                        continue
                    x_old = np.linspace(0, 1, len(y))
                    x_new = np.linspace(0, 1, interp_len)
                    y_interp = np.interp(x_new, x_old, y)
                    right_steps.append(y_interp)
            # Calculate mean and std across steps
            if left_steps and right_steps:
                mean_left = np.mean(left_steps, axis=0)
                std_left = np.std(left_steps, axis=0)
                mean_right = np.mean(right_steps, axis=0)
                std_right = np.std(right_steps, axis=0)
                diff = mean_left - mean_right
                x = np.linspace(0, 100, interp_len)

                plt.figure(figsize=(10, 5))
                plt.plot(x, mean_left, label=f"{left_sensor} mean", color='blue')
                plt.fill_between(x, mean_left - std_left, mean_left + std_left, color='blue', alpha=0.2)
                plt.plot(x, mean_right, label=f"{right_sensor} mean", color='red')
                plt.fill_between(x, mean_right - std_right, mean_right + std_right, color='red', alpha=0.2)
                plt.plot(x, diff, label="Difference (left - right)", color='black', linestyle='--')
                plt.axhline(0, color='gray', linestyle=':', linewidth=1)
                plt.title(f"Force Sensor (y) per time step: {left_sensor} vs {right_sensor}")
                plt.xlabel("Interpolated Step (%)")
                plt.ylabel("Force Value")
                plt.legend()
                plt.tight_layout()
                plt.show()
            else:
                print(f"Not enough steps for {left_sensor} or {right_sensor} to compute difference.")


            

    def plot_joint_angle_symmetry(
        self,
        joint_names_list,
        joint_data,
        step_indices_left,
        step_indices_right,
        parameter_name="angle",
        convert_to_deg=False,
        interp_mode="interp",
        interp_len=None,
    ):
        """
        Plot mean and difference of joint parameter (e.g., angle or velocity) for symmetry assessment.
        joint_names_list: list of joint base names, e.g. ["knee_angle", "ankle_angle", ...]
        joint_data: dict of {joint_name: {parameter_name: list/array}}
        step_indices_left, step_indices_right: step start indices for left/right
        parameter_name: "angle", "velocity", etc.
        """
        def get_step_indices(joint):
            if joint.endswith("_l"):
                return step_indices_left
            elif joint.endswith("_r"):
                return step_indices_right
            else:
                return None

        # Convert joint_names_list to left/right pairs
        joint_pairs = [(name + "_l", name + "_r") for name in joint_names_list]

        # Determine interpolation length if needed
        if interp_mode == "interp" and interp_len is None:
            step_lengths = []
            for left_joint, right_joint in joint_pairs:
                for joint in (left_joint, right_joint):
                    indices = get_step_indices(joint)
                    if indices and len(indices) > 1:
                        step_lengths += [indices[i+1] - indices[i] for i in range(len(indices)-1)]
            interp_len = min(step_lengths) if step_lengths else 1

        for left_joint, right_joint in joint_pairs:
            left_indices = get_step_indices(left_joint)
            right_indices = get_step_indices(right_joint)
            left_data = joint_data.get(left_joint, {}).get(parameter_name, None)
            right_data = joint_data.get(right_joint, {}).get(parameter_name, None)
            if left_data is None or right_data is None:
                print(f"Missing data for {left_joint} or {right_joint}")
                continue

            def collect_steps(data, indices):
                steps = []
                if indices and len(indices) > 1:
                    for j in range(len(indices) - 1):
                        start, end = indices[j], indices[j + 1]
                        y = [float(np.array(a).squeeze()) for a in data[start:end]]
                        if len(y) < 2:
                            continue
                        if interp_mode == "interp":
                            x_old = np.linspace(0, 1, len(y))
                            x_new = np.linspace(0, 1, interp_len)
                            y = np.interp(x_new, x_old, y)
                        if convert_to_deg and parameter_name in ("angle", "velocity"):
                            y = np.rad2deg(y)
                        steps.append(y)
                return steps

            left_steps = collect_steps(left_data, left_indices)
            right_steps = collect_steps(right_data, right_indices)

            if left_steps and right_steps:
                mean_left = np.mean(left_steps, axis=0)
                std_left = np.std(left_steps, axis=0)
                mean_right = np.mean(right_steps, axis=0)
                std_right = np.std(right_steps, axis=0)
                diff = mean_left - mean_right
                x = np.linspace(0, 100, interp_len)
                plt.figure(figsize=(10, 5))
                plt.plot(x, mean_left, label=f"{left_joint} mean", color='blue')
                plt.fill_between(x, mean_left - std_left, mean_left + std_left, color='blue', alpha=0.2)
                plt.plot(x, mean_right, label=f"{right_joint} mean", color='red')
                plt.fill_between(x, mean_right - std_right, mean_right + std_right, color='red', alpha=0.2)
                plt.plot(x, diff, label="Difference (left - right)", color='black', linestyle='--')
                plt.axhline(0, color='gray', linestyle=':', linewidth=1)
                plt.title(f"Joint symmetry: {left_joint} vs {right_joint} ({parameter_name})")
                plt.xlabel("Interpolated Step (%)")
                plt.ylabel(f"{parameter_name} ({'deg' if convert_to_deg and parameter_name in ('angle', 'velocity') else 'rad'})")
                plt.legend()
                plt.tight_layout()
                plt.show()
            else:
                print(f"Not enough steps for {left_joint} or {right_joint} to compute symmetry.")



    # @staticmethod
    # def plot_muscle_activation_symmetry_all_runs(
    #     muscle_names_list,
    #     all_loaded_data,
    #     run_step_data,
    #     interp_len=100,
    #     ):
    #     """
    #     Plot muscle activation symmetry (mean left - mean right per time step) for each muscle in muscle_names_list, for all runs.
    #     """
    #     for muscle_name in muscle_names_list:
    #         plt.figure(figsize=(12, 6))
    #         for run_key, run_dict in all_loaded_data.items():
    #             all_actuator_names = run_dict["all_actuator_names"]
    #             all_actions = run_dict["all_actions"]
    #             step_data = run_step_data[run_key]
    #             all_step_start_left = step_data["step_start_left"]
    #             all_step_start_right = step_data["step_start_right"]

    #             def find_actuator_index(actuator_names, name):
    #                 try:
    #                     return actuator_names.index(name)
    #                 except ValueError:
    #                     print(f"{name} not found in all_actuator_names.")
    #                     return None

    #             idx_left = find_actuator_index(all_actuator_names, muscle_name + "_l")
    #             idx_right = find_actuator_index(all_actuator_names, muscle_name + "_r")
    #             if idx_left is None or idx_right is None:
    #                 continue

    #             # Collect left and right activations per step
    #             left_steps = []
    #             right_steps = []
    #             if all_step_start_left and len(all_step_start_left) > 1:
    #                 for j in range(len(all_step_start_left) - 1):
    #                     start, end = all_step_start_left[j], all_step_start_left[j + 1]
    #                     y = [float(np.array(a)[0][idx_left]) for a in all_actions[start:end]]
    #                     if len(y) < 2:
    #                         continue
    #                     x_old = np.linspace(0, 1, len(y))
    #                     x_new = np.linspace(0, 1, interp_len)
    #                     y_interp = np.interp(x_new, x_old, y)
    #                     left_steps.append(y_interp)
    #             if all_step_start_right and len(all_step_start_right) > 1:
    #                 for j in range(len(all_step_start_right) - 1):
    #                     start, end = all_step_start_right[j], all_step_start_right[j + 1]
    #                     y = [float(np.array(a)[0][idx_right]) for a in all_actions[start:end]]
    #                     if len(y) < 2:
    #                         continue
    #                     x_old = np.linspace(0, 1, len(y))
    #                     x_new = np.linspace(0, 1, interp_len)
    #                     y_interp = np.interp(x_new, x_old, y)
    #                     right_steps.append(y_interp)
    #             # Plot if enough steps
    #             if left_steps and right_steps:
    #                 mean_left = np.mean(left_steps, axis=0)
    #                 std_left = np.std(left_steps, axis=0)
    #                 mean_right = np.mean(right_steps, axis=0)
    #                 std_right = np.std(right_steps, axis=0)
    #                 diff = mean_left - mean_right
    #                 x = np.linspace(0, 100, interp_len)
    #                 plt.plot(x, mean_left, label=f"{run_key} {muscle_name}_l mean")
    #                 plt.fill_between(x, mean_left - std_left, mean_left + std_left, alpha=0.15)
    #                 plt.plot(x, mean_right, label=f"{run_key} {muscle_name}_r mean", linestyle='--')
    #                 plt.fill_between(x, mean_right - std_right, mean_right + std_right, alpha=0.15)
    #                 plt.plot(x, diff, label=f"{run_key} L-R", linestyle=':')
    #         plt.axhline(0, color='gray', linestyle=':', linewidth=1)
    #         plt.title(f"Muscle activation symmetry: {muscle_name} (all runs)")
    #         plt.xlabel("Interpolated Step (%)")
    #         plt.ylabel("Activation")
    #         plt.legend()
    #         plt.tight_layout()
    #         plt.show()


    @staticmethod
    def plot_muscle_activation_symmetry_all_runs(
        muscle_names_list,
        all_loaded_data,
        run_step_data,
        interp_len=100,
    ):
        """
        Plot mean left of all runs in one plot, mean right of all runs in another, and the difference in a third.
        Each plot includes all runs, shown side by side.
        """
        for muscle_name in muscle_names_list:
            mean_left_all = []
            std_left_all = []
            mean_right_all = []
            std_right_all = []
            diff_all = []
            run_keys = list(all_loaded_data.keys())
            for run_key in run_keys:
                run_dict = all_loaded_data[run_key]
                all_actuator_names = run_dict["all_actuator_names"]
                all_actions = run_dict["all_actions"]
                step_data = run_step_data[run_key]
                all_step_start_left = step_data["step_start_left"]
                all_step_start_right = step_data["step_start_right"]

                def find_actuator_index(actuator_names, name):
                    try:
                        return actuator_names.index(name)
                    except ValueError:
                        print(f"{name} not found in all_actuator_names.")
                        return None

                idx_left = find_actuator_index(all_actuator_names, muscle_name + "_l")
                idx_right = find_actuator_index(all_actuator_names, muscle_name + "_r")
                if idx_left is None or idx_right is None:
                    continue

                # Collect left and right activations per step
                left_steps = []
                right_steps = []
                if all_step_start_left and len(all_step_start_left) > 1:
                    for j in range(len(all_step_start_left) - 1):
                        start, end = all_step_start_left[j], all_step_start_left[j + 1]
                        y = [float(np.array(a)[0][idx_left]) for a in all_actions[start:end]]
                        if len(y) < 2:
                            continue
                        x_old = np.linspace(0, 1, len(y))
                        x_new = np.linspace(0, 1, interp_len)
                        y_interp = np.interp(x_new, x_old, y)
                        left_steps.append(y_interp)
                if all_step_start_right and len(all_step_start_right) > 1:
                    for j in range(len(all_step_start_right) - 1):
                        start, end = all_step_start_right[j], all_step_start_right[j + 1]
                        y = [float(np.array(a)[0][idx_right]) for a in all_actions[start:end]]
                        if len(y) < 2:
                            continue
                        x_old = np.linspace(0, 1, len(y))
                        x_new = np.linspace(0, 1, interp_len)
                        y_interp = np.interp(x_new, x_old, y)
                        right_steps.append(y_interp)
                # Save means/stds if enough steps
                if left_steps and right_steps:
                    mean_left = np.mean(left_steps, axis=0)
                    std_left = np.std(left_steps, axis=0)
                    mean_right = np.mean(right_steps, axis=0)
                    std_right = np.std(right_steps, axis=0)
                    diff = mean_left - mean_right
                    mean_left_all.append((run_key, mean_left, std_left))
                    mean_right_all.append((run_key, mean_right, std_right))
                    diff_all.append((run_key, diff))

            x = np.linspace(0, 100, interp_len)
            fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
            # Plot mean left
            for run_key, mean_left, std_left in mean_left_all:
                axes[0].plot(x, mean_left, label=f"{run_key} {muscle_name}_l mean")
                axes[0].fill_between(x, mean_left - std_left, mean_left + std_left, alpha=0.15)
            axes[0].set_title(f"{muscle_name}_l mean (all runs)")
            axes[0].set_xlabel("Interpolated Step (%)")
            axes[0].set_ylabel("Activation")
            axes[0].legend()
            # Plot mean right
            for run_key, mean_right, std_right in mean_right_all:
                axes[1].plot(x, mean_right, label=f"{run_key} {muscle_name}_r mean")
                axes[1].fill_between(x, mean_right - std_right, mean_right + std_right, alpha=0.15)
            axes[1].set_title(f"{muscle_name}_r mean (all runs)")
            axes[1].set_xlabel("Interpolated Step (%)")
            axes[1].set_ylabel("Activation")
            axes[1].legend()
            # Plot difference
            for run_key, diff in diff_all:
                axes[2].plot(x, diff, label=f"{run_key} L-R")
            axes[2].axhline(0, color='gray', linestyle=':', linewidth=1)
            axes[2].set_title(f"{muscle_name} L-R difference (all runs)")
            axes[2].set_xlabel("Interpolated Step (%)")
            axes[2].set_ylabel("Activation Difference")
            axes[2].legend()
            plt.tight_layout()
            plt.show()
            plt.close()

    # @staticmethod
    # def plot_joint_angle_symmetry_all_runs(
    #         joint_names_list,
    #         all_loaded_data,
    #         run_step_data,
    #         parameter_name="angle",
    #         convert_to_deg=False,
    #         interp_mode="interp",
    #         interp_len=100,
    # ):
    #     """
    #     Plot mean and difference of joint parameter (e.g., angle or velocity) for symmetry assessment for all runs.
    #     joint_names_list: list of joint base names, e.g. ["knee_angle", "ankle_angle", ...]
    #     all_loaded_data: dict of {run_key: loaded_data}
    #     run_step_data: dict of {run_key: {"step_start_left": [...], "step_start_right": [...]} }
    #     parameter_name: "angle", "velocity", etc.
    #     """

    #     joint_pairs = [(name + "_l", name + "_r") for name in joint_names_list]
    #     color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #     color_iter = itertools.cycle(color_cycle)

    #     run_colors = {}  # Dictionary to store assigned colors for each run
    #     for left_joint, right_joint in joint_pairs:
    #         plt.figure(figsize=(12, 6))
    #         has_data = False  # Track if any run has data for this parameter
    #         for run_key, run_dict in all_loaded_data.items():
    #             joint_data = {}
    #             # Build joint_data dict for this run
    #             evaluation_joint_names = run_dict.get("evaluation_joint_names")
    #             joint_parameters = ["angle", "velocity", "forces_constraint", "forces_smooth", "torques", "energy_exp"]
    #             for joint_name in evaluation_joint_names:
    #                 joint_data[joint_name] = {}
    #                 for key in joint_parameters:
    #                     param = run_dict.get(f"{joint_name}_{key}")
    #                     if param is not None:
    #                         joint_data[joint_name][key] = param
    #                         joint_data[joint_name][key] = param

    #             step_indices_left = run_step_data[run_key]["step_start_left"]
    #             step_indices_right = run_step_data[run_key]["step_start_right"]
    #             left_data = joint_data.get(left_joint, {}).get(parameter_name, None)
    #             right_data = joint_data.get(right_joint, {}).get(parameter_name, None)
    #             if left_data is None or right_data is None:
    #                 print(f"[{run_key}] Missing data for {left_joint} or {right_joint}")
    #                 continue

    #             def collect_steps(data, indices):
    #                 steps = []
    #                 if indices and len(indices) > 1:
    #                     for j in range(len(indices) - 1):
    #                         start, end = indices[j], indices[j + 1]
    #                         y = [float(np.array(a).squeeze()) for a in data[start:end]]
    #                         if len(y) < 2:
    #                             continue
    #                         if interp_mode == "interp":
    #                             x_old = np.linspace(0, 1, len(y))
    #                             x_new = np.linspace(0, 1, interp_len)
    #                             y = np.interp(x_new, x_old, y)
    #                         if convert_to_deg and parameter_name in ("angle", "velocity"):
    #                             y = np.rad2deg(y)
    #                         steps.append(y)
    #                 return steps

    #             left_steps = collect_steps(left_data, step_indices_left)
    #             right_steps = collect_steps(right_data, step_indices_right)

    #             if left_steps and right_steps:
    #                 mean_left = np.mean(left_steps, axis=0)
    #                 std_left = np.std(left_steps, axis=0)
    #                 mean_right = np.mean(right_steps, axis=0)
    #                 std_right = np.std(right_steps, axis=0)
    #                 diff = mean_left - mean_right
    #                 x = np.linspace(0, 100, interp_len)
    #                 # Assign a color for this run
    #                 color = run_colors.get(run_key)
    #                 if color is None:
    #                     color = next(color_iter)
    #                     run_colors[run_key] = color
    #                 plt.plot(x, mean_left, label=f"{run_key} {left_joint} mean", color=color)
    #                 plt.fill_between(x, mean_left - std_left, mean_left + std_left, alpha=0.15, color=color)
    #                 plt.plot(x, mean_right, label=f"{run_key} {right_joint} mean", linestyle='--', color=color)
    #                 plt.fill_between(x, mean_right - std_right, mean_right + std_right, alpha=0.15, color=color)
    #                 plt.plot(x, diff, label=f"{run_key} Diff (L-R)", linestyle=':', color=color)
    #                 has_data = True
    #             else:
    #                 print(f"[{run_key}] Not enough steps for {left_joint} or {right_joint} to compute symmetry.")

    #         if has_data:
    #             plt.axhline(0, color='gray', linestyle=':', linewidth=1)
    #             plt.title(f"Joint symmetry (all runs): {left_joint} vs {right_joint} ({parameter_name})")
    #             plt.xlabel("Interpolated Step (%)")
    #             plt.ylabel(f"{parameter_name} ({'deg' if convert_to_deg and parameter_name in ('angle', 'velocity') else 'rad'})")
    #             plt.legend()
    #             plt.tight_layout()
    #             plt.show()
    #         else:
    #             plt.close()


    @staticmethod
    def plot_joint_angle_symmetry_all_runs(
        joint_names_list,
        all_loaded_data,
        run_step_data,
        parameter_name="angle",
        convert_to_deg=False,
        interp_mode="interp",
        interp_len=100,
        plot_baseline = False, 
        baseline_data=None,  # can be added for angle and velocity from loco-mujoco data 
    ):
        """
        Plot mean of right of all runs, mean of left, and the difference, side by side in one figure.
        Only plot runs where both left and right data are available.
        """
        joint_pairs = [(name + "_l", name + "_r") for name in joint_names_list]
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_iter = itertools.cycle(color_cycle)

        for left_joint, right_joint in joint_pairs:
            run_keys = list(all_loaded_data.keys())
            mean_left_all = []
            std_left_all = []
            mean_right_all = []
            std_right_all = []
            diff_all = []
            for run_key in run_keys:
                run_dict = all_loaded_data[run_key]
                joint_data = {}
                evaluation_joint_names = run_dict.get("evaluation_joint_names")
                joint_parameters = ["angle", "velocity", "forces_constraint", "forces_smooth", "torques", "energy_exp"]
                for joint_name in evaluation_joint_names:
                    joint_data[joint_name] = {}
                    for key in joint_parameters:
                        param = run_dict.get(f"{joint_name}_{key}")
                        if param is not None:
                            joint_data[joint_name][key] = param

                step_indices_left = run_step_data[run_key]["step_start_left"]
                step_indices_right = run_step_data[run_key]["step_start_right"]
                left_data = joint_data.get(left_joint, {}).get(parameter_name, None)
                right_data = joint_data.get(right_joint, {}).get(parameter_name, None)
                # Only include runs where both left and right data are available
                if left_data is None or right_data is None:
                    continue

                def collect_steps(data, indices):
                    steps = []
                    if indices and len(indices) > 1:
                        for j in range(len(indices) - 1):
                            start, end = indices[j], indices[j + 1]
                            y = [float(np.array(a).squeeze()) for a in data[start:end]]
                            if len(y) < 2:
                                continue
                            if interp_mode == "interp":
                                x_old = np.linspace(0, 1, len(y))
                                x_new = np.linspace(0, 1, interp_len)
                                y = np.interp(x_new, x_old, y)
                            if convert_to_deg and parameter_name in ("angle", "velocity"):
                                y = np.rad2deg(y)
                            steps.append(y)
                    return steps

                left_steps = collect_steps(left_data, step_indices_left)
                right_steps = collect_steps(right_data, step_indices_right)

                if left_steps and right_steps:
                    mean_left = np.mean(left_steps, axis=0)
                    std_left = np.std(left_steps, axis=0)
                    mean_right = np.mean(right_steps, axis=0)
                    std_right = np.std(right_steps, axis=0)
                    diff = mean_left - mean_right
                    mean_left_all.append((run_key, mean_left, std_left))
                    mean_right_all.append((run_key, mean_right, std_right))
                    diff_all.append((run_key, diff))

            x = np.linspace(0, 100, interp_len)
            # Plot all three plots side by side
            if mean_left_all or mean_right_all or diff_all:
                fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
                # Plot mean of right of all runs
                if mean_right_all:
                    for run_key, mean_right, std_right in mean_right_all:
                        axes[0].plot(x, mean_right, label=f"{run_key} {right_joint} mean")
                        axes[0].fill_between(x, mean_right - std_right, mean_right + std_right, alpha=0.15)
                    if plot_baseline and parameter_name in ["angle", "velocity"] and baseline_data is not None:
                        axes[0].plot(x, baseline_data[right_joint][parameter_name], label=f"Baseline {right_joint} mean", color='black')
                    axes[0].set_title(f"{right_joint} mean (all runs)")
                    axes[0].set_xlabel("Interpolated Step (%)")
                    axes[0].set_ylabel(f"{parameter_name} ({'deg' if convert_to_deg and parameter_name in ('angle', 'velocity') else 'rad'})")
                    axes[0].legend()
                # Plot mean of left of all runs
                if mean_left_all:
                    for run_key, mean_left, std_left in mean_left_all:
                        axes[1].plot(x, mean_left, label=f"{run_key} {left_joint} mean")
                        axes[1].fill_between(x, mean_left - std_left, mean_left + std_left, alpha=0.15)
                    if plot_baseline and parameter_name in ["angle", "velocity"] and baseline_data is not None:
                        axes[1].plot(x, baseline_data[right_joint][parameter_name], label=f"Baseline {left_joint} mean", color='black')
                    axes[1].set_title(f"{left_joint} mean (all runs)")
                    axes[1].set_xlabel("Interpolated Step (%)")
                    axes[1].set_ylabel(f"{parameter_name} ({'deg' if convert_to_deg and parameter_name in ('angle', 'velocity') else 'rad'})")
                    axes[1].legend()
                # Plot difference (left - right) of all runs
                if diff_all:
                    for run_key, diff in diff_all:
                        axes[2].plot(x, diff, label=f"{run_key} Diff (L-R)")
                    axes[2].axhline(0, color='gray', linestyle=':', linewidth=1)
                    axes[2].set_title(f"{left_joint} - {right_joint} difference (all runs)")
                    axes[2].set_xlabel("Interpolated Step (%)")
                    axes[2].set_ylabel(f"{parameter_name} difference ({'deg' if convert_to_deg and parameter_name in ('angle', 'velocity') else 'rad'})")
                    axes[2].legend()
                plt.tight_layout()
                plt.show()
                plt.close()


    #@staticmethod 
    # def plot_sensor_force_symmetry_all_runs(all_loaded_data, run_step_data, interp_len=100, sensor_force_names=None):
    #     """
    #     Plot mean, std, and left-right difference for each left/right sensor force pair across all runs.
    #     """
    #     # Use sensor_force_names from the first run if not provided
    #     if sensor_force_names is None:
    #         first_run = next(iter(all_loaded_data))
    #         sensor_force_names = all_loaded_data[first_run]["sensor_force_names"]

    #     # Find sensor pairs (left/right)
    #     sensor_pairs = []
    #     for name in sensor_force_names:
    #         if name.startswith("left_"):
    #             right_name = name.replace("left_", "right_")
    #             if right_name in sensor_force_names:
    #                 sensor_pairs.append((name, right_name))

    #     for left_sensor, right_sensor in sensor_pairs:
    #         plt.figure(figsize=(12, 6))
    #         for run_key in all_loaded_data:
    #             run_dict = all_loaded_data[run_key]
    #             step_data = run_step_data[run_key]
    #             left_data = run_dict["all_sensor_force"][left_sensor]
    #             right_data = run_dict["all_sensor_force"][right_sensor]
    #             left_steps = []
    #             right_steps = []
    #             all_step_start_left = step_data["step_start_left"]
    #             all_step_start_right = step_data["step_start_right"]

    #             # Interpolate left steps
    #             if all_step_start_left and len(all_step_start_left) > 1:
    #                 for j in range(len(all_step_start_left) - 1):
    #                     start, end = all_step_start_left[j], all_step_start_left[j + 1]
    #                     y = [float(np.array(a)[1]) for a in left_data[start:end]]
    #                     if len(y) < 2:
    #                         continue
    #                     x_old = np.linspace(0, 1, len(y))
    #                     x_new = np.linspace(0, 1, interp_len)
    #                     y_interp = np.interp(x_new, x_old, y)
    #                     left_steps.append(y_interp)
    #             # Interpolate right steps
    #             if all_step_start_right and len(all_step_start_right) > 1:
    #                 for j in range(len(all_step_start_right) - 1):
    #                     start, end = all_step_start_right[j], all_step_start_right[j + 1]
    #                     y = [float(np.array(a)[1]) for a in right_data[start:end]]
    #                     if len(y) < 2:
    #                         continue
    #                     x_old = np.linspace(0, 1, len(y))
    #                     x_new = np.linspace(0, 1, interp_len)
    #                     y_interp = np.interp(x_new, x_old, y)
    #                     right_steps.append(y_interp)
    #             # Plot if enough steps
    #             if left_steps and right_steps:
    #                 mean_left = np.mean(left_steps, axis=0)
    #                 std_left = np.std(left_steps, axis=0)
    #                 mean_right = np.mean(right_steps, axis=0)
    #                 std_right = np.std(right_steps, axis=0)
    #                 diff = mean_left - mean_right
    #                 x = np.linspace(0, 100, interp_len)
    #                 plt.plot(x, mean_left, label=f"{run_key} {left_sensor} mean")
    #                 plt.fill_between(x, mean_left - std_left, mean_left + std_left, alpha=0.15)
    #                 plt.plot(x, mean_right, label=f"{run_key} {right_sensor} mean", linestyle='--')
    #                 plt.fill_between(x, mean_right - std_right, mean_right + std_right, alpha=0.15)
    #                 plt.plot(x, diff, label=f"{run_key} Left-Right", linestyle=':')
    #         plt.axhline(0, color='gray', linestyle=':', linewidth=1)
    #         plt.title(f"Force Sensor (y) per time step: {left_sensor} vs {right_sensor} (all runs)")
    #         plt.xlabel("Interpolated Step (%)")
    #         plt.ylabel("Force Value")
    #         plt.legend()
    #         plt.tight_layout()
    #         plt.show()


    @staticmethod
    def plot_sensor_force_symmetry_all_runs(
        all_loaded_data, run_step_data, direction, interp_len=100, sensor_force_names=None
    ):
        """
        Plot mean of right of all runs, mean of left, and the difference, side by side in one figure.
        Each plot includes all runs.
        direction: "x", "y", or "z" to specify the force direction. [0, 1, 2] for x, y, z respectively.
        """
        if direction == "x":
            force_direction = 0
        elif direction == "y":
            force_direction = 1
        elif direction == "z":
            force_direction = 2
        # Use sensor_force_names from the first run if not provided
        if sensor_force_names is None:
            first_run = next(iter(all_loaded_data))
            sensor_force_names = all_loaded_data[first_run]["sensor_force_names"]

        # Find sensor pairs (left/right)
        sensor_pairs = []
        for name in sensor_force_names:
            if name.startswith("left_"):
                right_name = name.replace("left_", "right_")
                if right_name in sensor_force_names:
                    sensor_pairs.append((name, right_name))

        for left_sensor, right_sensor in sensor_pairs:
            # Prepare data for all runs
            run_keys = list(all_loaded_data.keys())
            mean_left_all = []
            std_left_all = []
            mean_right_all = []
            std_right_all = []
            diff_all = []
            for run_key in run_keys:
                run_dict = all_loaded_data[run_key]
                step_data = run_step_data[run_key]
                left_data = run_dict["all_sensor_force"][left_sensor]
                right_data = run_dict["all_sensor_force"][right_sensor]
                all_step_start_left = step_data["step_start_left"]
                all_step_start_right = step_data["step_start_right"]

                # Interpolate left steps
                left_steps = []
                if all_step_start_left and len(all_step_start_left) > 1:
                    for j in range(len(all_step_start_left) - 1):
                        start, end = all_step_start_left[j], all_step_start_left[j + 1]
                        y = [float(np.array(a)[force_direction]) for a in left_data[start:end]]
                        if len(y) < 2:
                            continue
                        x_old = np.linspace(0, 1, len(y))
                        x_new = np.linspace(0, 1, interp_len)
                        y_interp = np.interp(x_new, x_old, y)
                        left_steps.append(y_interp)
                # Interpolate right steps
                right_steps = []
                if all_step_start_right and len(all_step_start_right) > 1:
                    for j in range(len(all_step_start_right) - 1):
                        start, end = all_step_start_right[j], all_step_start_right[j + 1]
                        # Plot y axis in site coordinate system [1]
                        y = [float(np.array(a)[force_direction]) for a in right_data[start:end]]
                        if len(y) < 2:
                            continue
                        x_old = np.linspace(0, 1, len(y))
                        x_new = np.linspace(0, 1, interp_len)
                        y_interp = np.interp(x_new, x_old, y)
                        right_steps.append(y_interp)
                # Compute means and stds
                if left_steps and right_steps:
                    mean_left = np.mean(left_steps, axis=0)
                    std_left = np.std(left_steps, axis=0)
                    mean_right = np.mean(right_steps, axis=0)
                    std_right = np.std(right_steps, axis=0)
                    diff = mean_left - mean_right
                    mean_left_all.append((run_key, mean_left, std_left))
                    mean_right_all.append((run_key, mean_right, std_right))
                    diff_all.append((run_key, diff))
                else:
                    mean_left_all.append((run_key, None, None))
                    mean_right_all.append((run_key, None, None))
                    diff_all.append((run_key, None))

            x = np.linspace(0, 100, interp_len)
            # Plot all three plots side by side
            fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
            # Plot mean of right of all runs
            for run_key, mean_right, std_right in mean_right_all:
                if mean_right is not None:
                    axes[0].plot(x, mean_right, label=f"{run_key} {right_sensor} mean")
                    axes[0].fill_between(x, mean_right - std_right, mean_right + std_right, alpha=0.15)
            axes[0].set_title(f"{right_sensor} mean (all runs)")
            axes[0].set_xlabel("Interpolated Step (%)")
            axes[0].set_ylabel("Force Value")
            axes[0].legend()

            # Plot mean of left of all runs
            for run_key, mean_left, std_left in mean_left_all:
                if mean_left is not None:
                    axes[1].plot(x, mean_left, label=f"{run_key} {left_sensor} mean")
                    axes[1].fill_between(x, mean_left - std_left, mean_left + std_left, alpha=0.15)
            axes[1].set_title(f"{left_sensor} mean (all runs)")
            axes[1].set_xlabel("Interpolated Step (%)")
            axes[1].set_ylabel("Force Value")
            axes[1].legend()

            # Plot difference (left - right) of all runs
            for run_key, diff in diff_all:
                if diff is not None:
                    axes[2].plot(x, diff, label=f"{run_key} Left-Right")
            axes[2].axhline(0, color='gray', linestyle=':', linewidth=1)
            axes[2].set_title(f"{left_sensor} - {right_sensor} difference (all runs)")
            axes[2].set_xlabel("Interpolated Step (%)")
            axes[2].set_ylabel("Force Value Difference")
            axes[2].legend()

            plt.tight_layout()
            plt.show()
            plt.close()



    @staticmethod
    def plot_grf_component_symmetry_all_runs(
        grf_components,
        all_loaded_data,
        run_step_data,
        interp_len=100,
    ):
        """
        Plot mean of right of all runs, mean of left, and the difference, side by side in one figure.
        Each plot includes all runs.
        """
        for comp_idx, comp in enumerate(grf_components):
            run_keys = list(all_loaded_data.keys())
            mean_left_all = []
            std_left_all = []
            mean_right_all = []
            std_right_all = []
            diff_all = []
            for run_key in run_keys:
                run_dict = all_loaded_data[run_key]
                step_data = run_step_data[run_key]
                all_grf_l = np.array(run_dict["all_grf_l"])
                all_grf_r = np.array(run_dict["all_grf_r"])
                all_step_start_left = step_data["step_start_left"]
                all_step_start_right = step_data["step_start_right"]

                # Left steps
                left_steps = []
                if all_grf_l is not None and all_step_start_left and len(all_step_start_left) > 1:
                    for j in range(len(all_step_start_left) - 1):
                        start, end = all_step_start_left[j], all_step_start_left[j + 1]
                        y = [float(np.array(a)[comp_idx]) for a in all_grf_l[start:end]]
                        if len(y) < 2:
                            continue
                        x_old = np.linspace(0, 1, len(y))
                        x_new = np.linspace(0, 1, interp_len)
                        y_interp = np.interp(x_new, x_old, y)
                        left_steps.append(y_interp)
                # Right steps
                right_steps = []
                if all_grf_r is not None and all_step_start_right and len(all_step_start_right) > 1:
                    for j in range(len(all_step_start_right) - 1):
                        start, end = all_step_start_right[j], all_step_start_right[j + 1]
                        y = [float(np.array(a)[comp_idx]) for a in all_grf_r[start:end]]
                        if len(y) < 2:
                            continue
                        x_old = np.linspace(0, 1, len(y))
                        x_new = np.linspace(0, 1, interp_len)
                        y_interp = np.interp(x_new, x_old, y)
                        right_steps.append(y_interp)
                # Save means/stds if enough steps
                if left_steps and right_steps:
                    mean_left = np.mean(left_steps, axis=0)
                    std_left = np.std(left_steps, axis=0)
                    mean_right = np.mean(right_steps, axis=0)
                    std_right = np.std(right_steps, axis=0)
                    diff = mean_left - mean_right
                    mean_left_all.append((run_key, mean_left, std_left))
                    mean_right_all.append((run_key, mean_right, std_right))
                    diff_all.append((run_key, diff))

            x = np.linspace(0, 100, interp_len)
            fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
            # Plot mean of right of all runs
            for run_key, mean_right, std_right in mean_right_all:
                axes[0].plot(x, mean_right, label=f"{run_key} Right mean", linestyle='--')
                axes[0].fill_between(x, mean_right - std_right, mean_right + std_right, alpha=0.15)
            axes[0].set_title(f"{comp} Right mean (all runs)")
            axes[0].set_xlabel("Interpolated Step (%)")
            axes[0].set_ylabel(f"{comp} (N or Nm)")
            axes[0].legend()
            # Plot mean of left of all runs
            for run_key, mean_left, std_left in mean_left_all:
                axes[1].plot(x, mean_left, label=f"{run_key} Left mean", linestyle='-')
                axes[1].fill_between(x, mean_left - std_left, mean_left + std_left, alpha=0.15)
            axes[1].set_title(f"{comp} Left mean (all runs)")
            axes[1].set_xlabel("Interpolated Step (%)")
            axes[1].set_ylabel(f"{comp} (N or Nm)")
            axes[1].legend()
            # Plot difference (left - right) of all runs
            for run_key, diff in diff_all:
                axes[2].plot(x, diff, label=f"{run_key} Left-Right", linestyle=':')
            axes[2].axhline(0, color='gray', linestyle=':', linewidth=1)
            axes[2].set_title(f"{comp} Left-Right difference (all runs)")
            axes[2].set_xlabel("Interpolated Step (%)")
            axes[2].set_ylabel(f"{comp} difference (N or Nm)")
            axes[2].legend()
            plt.tight_layout()
            plt.show()
            plt.close()