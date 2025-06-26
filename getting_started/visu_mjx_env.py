import time 
import jax 
import mujoco
import mujoco.viewer
from loco_mujoco import ImitationFactory
from loco_mujoco.environments.humanoids.skeletons import MjxSkeletonMuscle


def main(): 
    # env = ImitationFactory.make("MjxSkeletonMuscle", default_dataset_conf=dict(task="walk"))
    env = MjxSkeletonMuscle()
    spec = mujoco.MjSpec.from_file(env.get_default_xml_file_path())
    # spec = env.
    # xml_path = '/home/nadinebadie/loco-mujoco/loco_mujoco/models/skeleton/skeleton_muscle.xml'
    model = spec.compile() #mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # # create keys
    # key = jax.random.key(0)
    # n_envs = 100
    # keys = jax.random.split(key, n_envs + 1)
    # key, env_keys = keys[0], keys[1:]

    
    jit_reset = jax.jit(env.mjx_reset)
    jit_step = jax.jit(env.mjx_step) #model, data)
    state = jit_reset(jax.random.PRNGKey(0)) #env_keys) #jax.random.PRNGKey(0))



    with mujoco.viewer.launch(model, data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_WORLD_FRM] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        while viewer.is_running():
            step_start = time.time()

            state = state.replace(data=state.data.replace(mocap_pos=data.mocap_pos, xfrc_applied=data.xfrc_applied))
            state = jit_step(state, data.ctrl)

            data.qpos = state.data.qpos
            mujoco.mj_froward(model, data)

            data.sensordata[0] = state.reward
            data.sensordata[1:] = state.info['target_angle']    


            # Pick up changes to the physics state, apply perturbations, update options from GUI
            viewer.sync()

            # Time keeping 
            time_until_new_step = model.opt.timestep - (time.time() - step_start)
            if time_until_new_step > 0:
                time.sleep(time_until_new_step)
    

    pass

if __name__ == "__main__":
    main()