# Helper function to run evaluation loop and collect metrics
def run_evaluation_loop(env_state, n_steps, train_state, rng):
    obs = env_state.observation
    step_total = 0

    # Local copies for metrics
    local_all_foot_ground_contact_left = []
    local_all_foot_ground_contact_right = []
    local_all_grf_l = []
    local_all_grf_r = []
    local_all_actions = []
    local_all_actuator_names = []
    local_all_sensor_force = {}
    local_joint_data = {k: {kk: [] for kk in v.keys()} for k, v in joint_data.items()}
    local_run_activations = {f"run_{mg}_activation_left": [] for mg in evaluation_muscle_groups}
    local_run_activations.update({f"run_{mg}_activation_right": [] for mg in evaluation_muscle_groups})

    for i in range(n_steps):
        rng, _rng = jax.random.split(rng)
        action, train_state = sample_actions(train_state, obs, _rng)
        action = jnp.atleast_2d(action)
        env_state = jit_step(env_state, action)
        obs = env_state.observation

        # Metrics collection (same as in your commented-out loop)
        # Foot ground contact indices 
        contact_left, contact_right = prosthesis_metrics_handler.get_contact_steps(env_state.data, i)
        local_all_foot_ground_contact_left.append(contact_left)
        local_all_foot_ground_contact_right.append(contact_right)

        # GRF 
        grf_l = prosthesis_metrics_handler.get_grf(env_state.data, f"{foot_name}_l")
        grf_r = prosthesis_metrics_handler.get_grf(env_state.data, f"{foot_name}_r")
        local_all_grf_l.append(grf_l)
        local_all_grf_r.append(grf_r)

        # Joint data
        joint_angles = prosthesis_metrics_handler.get_joint_angles(env_state.data)
        joint_velocities = prosthesis_metrics_handler.get_joint_vels(env_state.data)
        joint_forces_constraint, joint_forces_smooth = prosthesis_metrics_handler.get_joint_frces(env_state.data)
        joint_torques = prosthesis_metrics_handler.get_joint_trques(env_state.data)
        joint_energy_exp = prosthesis_metrics_handler.calc_joint_energy_exp(joint_torques, joint_velocities)
        for joint_name, angle in joint_angles.items():
            local_joint_data[joint_name]["angle"].append(angle)
        for joint_name, velocity in joint_velocities.items():
            local_joint_data[joint_name]["velocity"].append(velocity)
        for joint_name, force in joint_forces_constraint.items():
            local_joint_data[joint_name]["forces_constraint"].append(force)
        for joint_name, force in joint_forces_smooth.items():
            local_joint_data[joint_name]["forces_smooth"].append(force)
        for joint_name, torque in joint_torques.items():
            local_joint_data[joint_name]["torques"].append(torque)
        for joint_name, energy_exp in joint_energy_exp.items():
            local_joint_data[joint_name]["energy_exp"].append(energy_exp)

        # Sensor data
        sensor_force, sensor_force_names = prosthesis_metrics_handler.get_sensor_data(env_state.data)
        for sensor_name, force in sensor_force.items():
            if sensor_name not in local_all_sensor_force:
                local_all_sensor_force[sensor_name] = []
            local_all_sensor_force[sensor_name].append(force)

        # Muscle activations
        for i_act in range(model.nu):
            if model.actuator_dyntype[i_act] == mujoco.mjtDyn.mjDYN_MUSCLE:
                action = action.at[..., i_act].set(muscle_skeleton_control_activation.adapted_sigmoid(action[..., i_act]))
        local_all_actions.append(action)
        local_all_actuator_names = []
        for a in range(model.nu):
            actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            local_all_actuator_names.append(actuator_name)
        for muscle_group, muscle_names in evaluation_muscle_names.items():
            locals()[f"{muscle_group}_activation_left"] = prosthesis_metrics_handler.get_relevant_ctrl(muscle_names, left_side, action)
            locals()[f"{muscle_group}_activation_right"] = prosthesis_metrics_handler.get_relevant_ctrl(muscle_names, right_side, action)
            local_run_activations[f"run_{muscle_group}_activation_left"].append(locals()[f"{muscle_group}_activation_left"])
            local_run_activations[f"run_{muscle_group}_activation_right"].append(locals()[f"{muscle_group}_activation_right"])
        step_total += n_envs

    # Return all collected data
    return {
        "all_foot_ground_contact_left": local_all_foot_ground_contact_left,
        "all_foot_ground_contact_right": local_all_foot_ground_contact_right,
        "all_grf_l": local_all_grf_l,
        "all_grf_r": local_all_grf_r,
        "all_actions": local_all_actions,
        "all_actuator_names": local_all_actuator_names,
        "all_sensor_force": local_all_sensor_force,
        "joint_data": local_joint_data,
        "run_activations": local_run_activations,
        "sensor_force_names": sensor_force_names,
        "step_total": step_total,
    }

# Iterate over params in randomization_params_names 
for param_name in randomization_params_names:
    if param_name in randomization_params:
        if randomization_params[f"randomize_{param_name}"]:
            # set other "randomize_param_name" to False
            for other_param in randomization_params_names:
                if other_param != param_name:
                    env._domain_randomizer[f"randomize_{other_param}"] = False

            if "stiffness" in param_name or "damping" in param_name:
                min_val, max_val = randomization_params[f"{param_name}_range"]
                for i in range(min_val, max_val, randomization_increments[param_name]):
                    env._domain_randomizer[f"{param_name}_range"] = [min_val + i, min_val + i]
                    # Reset to new domain randomization state
                    env_state = jit_reset(env_keys)
                    # Run evaluation loop for this setting
                    eval_data = run_evaluation_loop(env_state, n_steps, train_state, rng)
                    # Save or process eval_data as needed

            elif "position" in param_name or "orientation" in param_name:
                range_dict = randomization_params[f"{param_name}_range"]
                directions = list(range_dict.keys())
                for axis in directions:
                    min_val = np.array(range_dict[axis][0])
                    max_val = np.array(range_dict[axis][1])
                    for n in range(min_val, max_val, randomization_increments[param_name]):
                        env._domain_randomizer[f"{param_name}_range"][axis] = [min_val + n, max_val + n]
                        for other_axis in directions:
                            if other_axis != axis:
                                env._domain_randomizer[f"randomize_{param_name}"][other_axis] = [0, 0]
                        # Reset to new domain randomization state
                        env_state = jit_reset(env_keys)
                        # Run evaluation loop for this setting
                        eval_data = run_evaluation_loop(env_state, n_steps, train_state, rng)
                        # Save or process eval_data as needed

