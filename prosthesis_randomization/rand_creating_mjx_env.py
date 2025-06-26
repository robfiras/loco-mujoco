import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
jax.config.update('jax_platform_name', 'cpu')
import time

from loco_mujoco import ImitationFactory


# can increase the speed by ~30% on some GPUs
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True ')


randomization_params = {
    "prosthesis_side": "left_side",

    "randomize_prosthesis_dof_damping": True,
    "randomization_dof_names": ['ankle_angle', 'subtalar_angle', 'mtp_angle'],
    "prosthesis_dof_damping_range": [0, 10],

    "randomize_prosthesis_joint_stiffness": False, #True,
    "randomization_joint_names": ['ankle_angle', 'subtalar_angle', 'mtp_angle'],
    "prosthesis_joint_stiffness_range":[], # [0, 100],

    "randomize_prosthesis_body_position": True,
    "randomization_body_position_names": ['calcn'],
    "prosthesis_body_position_range": {
        # 'x': [0.4, 0.5],
        # 'y': [0.0, 0.1],
        # 'z': [0.5, 0.9]
    },
    "randomize_prosthesis_body_orientation": True, #True, #False, #True,
    "randomization_body_orientation_names": ['calcn'], #['toe'], #['calcn'], #['foot_box'], #['calcn'],
    "prosthesis_body_orientation_range": {
        # 'x': [0.1, 0.2],
        'y': [0.8, 0.9],
        # 'z': [0.1, 0.2]
        #'x': [-0.01, 0.01],
        # 'y': [0.8, 0.9]#, #[0.15, 0.3], #(9-17°) #[0.8, 0.9],
        #'z': [-0.01, 0.01]
    }
}




# randomization_params = {
#     "prosthesis_side": "left_side",
#     "randomize_prosthesis_dof_damping": True,
#     "randomization_joint_names": ['ankle_angle', 'subtalar_angle', 'mtp_angle'],
#     # "randomization_joints": ['ankle_angle', 'subtalar_angle', 'mtp_angle'],
#     "prosthesis_dof_damping_range": [0, 10],
#     "randomize_prosthesis_joint_stiffness": True,
#     "randomization_dof_names": ['ankle_angle', 'subtalar_angle', 'mtp_angle'],
#     "prosthesis_joint_stiffness_range": [0, 100],
#     "randomization_body_position_names": ['calcn'], #['toe'], #['calcn'], #['foot_box'], #['calcn'],
#     # "randomization_bodies": ['calcn'], #['toe'], #['calcn'], #['foot_box'], #['calcn'],
#     "randomize_prosthesis_body_position": False, #True,
#     "prosthesis_body_position_range": {
#         # 'x': [0.4, 0.5],
#         # 'y': [0.0, 0.1],
#         # 'z': [0.5, 0.9]
#         # 'x': [-0.001, 0.001],
#         # 'y': [-0.001, 0.001],
#         # 'z': [0.9, 1]
#     },
#     "randomize_prosthesis_body_orientation": True, #False, #True,
#     "randomization_body_orientation_names": ['calcn'], #['toe'], #['calcn'], #['foot_box'], #['calcn'],
#     "prosthesis_body_orientation_range": {
#         # 'x': [0.1, 0.2],
#         # 'y': [0.8, 0.9],
#         # 'z': [0.1, 0.2]
#         #'x': [-0.01, 0.01],
#         'y': [0.8, 0.9]#, #[0.15, 0.3], #(9-17°) #[0.8, 0.9],
#         #'z': [-0.01, 0.01]
#     }
# }

# randomization_params = {
#     # gravity
#     "randomize_gravity": True,
#     "gravity_range": [9.51, 10.11],

#     # geom properties
#     "randomize_geom_friction_tangential": True,
#     "geom_friction_tangential_range": [0.8, 1.2],
#     "randomize_geom_friction_torsional": True,
#     "geom_friction_torsional_range": [0.003, 0.007],
#     "randomize_geom_friction_rolling": True,
#     "geom_friction_rolling_range": [0.00008, 0.00012],
#     "randomize_geom_damping": True,
#     "geom_damping_range": [72, 88],
#     "randomize_geom_stiffness": True,
#     "geom_stiffness_range": [900, 1100],

#     # joint properties
#     "randomize_joint_damping": True,
#     "joint_damping_range": [0.3, 1.5],
#     "randomize_joint_stiffness": True,
#     "joint_stiffness_range": [0.9, 1.1],
#     "randomize_joint_friction_loss": True,
#     "joint_friction_loss_range": [0.0, 0.2],
#     "randomize_joint_armature": True,
#     "joint_armature_range": [0.08, 0.12],

#     # base mass
#     "randomize_base_mass": True,
#     "base_mass_to_add_range": [-2.0, 2.0],

#     # COM
#     "randomize_com_displacement": True,
#     "com_displacement_range": [-0.15, 0.15],

#     # link mass
#     "randomize_link_mass": True,
#     "link_mass_multiplier_range": {
#         "root_body": [0.5, 1.9],
#         "other_bodies": [0.8, 1.2],
#     },

#     # PD Gains (if PDControl is used)
#     "add_p_gains_noise": True,
#     "add_d_gains_noise": True,
#     "p_gains_noise_scale": 0.1,
#     "d_gains_noise_scale": 0.1,

#     # Observation Noise
#     "add_joint_pos_noise": True,
#     "joint_pos_noise_scale": 0.003,
#     "add_joint_vel_noise": True,
#     "joint_vel_noise_scale": 0.08,
#     "add_gravity_noise": True,
#     "gravity_noise_scale": 0.015,
#     "add_free_joint_lin_vel_noise": True,
#     "lin_vel_noise_scale": 0.1,
#     "add_free_joint_ang_vel_noise": True,
#     "ang_vel_noise_scale": 0.02,
# }


# create env
env = ImitationFactory.make(
    "MjxSkeletonMuscleProsthesis",
    prosthesis_side="left_side",
    prosthesis_type="transtibial",
    delete_joints = False,
    joint_stiffness = 50,
    joint_damping = 1,
    default_dataset_conf=dict(task="walk"),
    domain_randomization_type="ProsthesisRandomizer", #"DefaultRandomizer", #"ProsthesisRandomizer",
    domain_randomization_params=randomization_params
)

# create keys
key = jax.random.key(0)
n_envs = 1
keys = jax.random.split(key, n_envs + 1)
key, env_keys = keys[0], keys[1:]

# jit and vmap all functions needed
rng_reset = jax.jit(jax.vmap(env.mjx_reset))
rng_step = jax.jit(jax.vmap(env.mjx_step_test))
# rng_step = jax.jit(jax.vmap(env.mjx_step))
rng_sample_uni_action = jax.jit(jax.vmap(env.sample_action_space))

# reset env
state = rng_reset(env_keys)

step = 0
previous_time = time.time()
LOGGING_FREQUENCY = 100000
i = 0
while i < 100000:

    # step
    keys = jax.random.split(key, n_envs + 1)
    key, action_keys = keys[0], keys[1:]
    action = rng_sample_uni_action(action_keys)
    state, sys = rng_step(state, action)

    print(f"sys.jnt_stiffness: {sys.jnt_stiffness}")
    print(f"sys.dof_damping: {sys.dof_damping}")
    print(f"sys.body_pos: {sys.body_pos}")
    print(f"sys.body_quat: {sys.body_quat}")

    # parallel render
    env.mjx_render_domain_randomization(state)

    step += n_envs

    # log speed (disable rendering for accurate speed measurement)
    if step % LOGGING_FREQUENCY == 0:
        current_time = time.time()
        print(f"{int(LOGGING_FREQUENCY / (current_time - previous_time))} steps per second.")
        previous_time = current_time

    i+=1

