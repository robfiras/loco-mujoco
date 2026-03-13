import jax
import jax.numpy as jnp
import numpy as np
import mujoco

from loco_mujoco import LocoEnv
from loco_mujoco.core.trajectory import Trajectory, TrajectoryInfo, TrajectoryModel, TrajectoryData


"""
This is a minimal example explaining the trajectory interface.
Here a standing still trajectory is created and replayed in the Unitree H1 environment

The example shows how to:
    - create a trajectory 
    - save the trajectory
    - load the trajectory
    - replay the trajectory
    
"""

N_steps = 1000

# create the environment
env = LocoEnv.make("UnitreeH1", init_state_type="DefaultInitialStateHandler")

# reset the env
key = jax.random.PRNGKey(0)
env.reset(key)

# get the model and data of the environment
model = env.get_model()
data = env.get_data()

# get the initial qpos and qvel of the environment
qpos = data.qpos
qvel = data.qvel

# stack qpos and qvel to a trajectory
qpos = np.tile(qpos, (N_steps, 1))
qvel = np.tile(qvel, (N_steps, 1))

# create a trajectory info -- this stores basic information about the trajectory
njnt = model.njnt
jnt_type = model.jnt_type
jnt_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(njnt)]
traj_info = TrajectoryInfo(jnt_names, model=TrajectoryModel(njnt, jnp.array(jnt_type)), frequency=1/env.dt)

# create a trajectory data -- this stores the actual trajectory data
traj_data = TrajectoryData(jnp.array(qpos), jnp.array(qvel), split_points=jnp.array([0, N_steps]))

# combine them to a trajectory
traj = Trajectory(traj_info, traj_data)

# example: save the trajectory
traj.save("trajectory.npz")

# example: load the trajectory
traj = Trajectory.load("trajectory.npz")

# add the trajectory to the environment
env.load_trajectory(traj)

# replay the trajectory
env.play_trajectory(n_steps_per_episode=N_steps)



