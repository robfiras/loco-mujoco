import time 
import jax 
import mujoco
import mujoco.viewer
from loco_mujoco import ImitationFactory
from loco_mujoco.environments.humanoids.skeletons import MjxSkeletonMuscle
from mujoco import mjx

def main(): 
    # env = ImitationFactory.make("MjxSkeletonMuscle", default_dataset_conf=dict(task="walk"))
    #env = MjxSkeletonMuscle()
    spec = mujoco.MjSpec.from_file("/home/nadinebadie/loco-mujoco/loco_mujoco/models/skeleton/skeleton_muscle.xml") #env.get_default_xml_file_path())
    # body_name = "tibia_l"
    # body = spec.find_body(body_name)
    

    # sensor_site_name = "right_knee_mimic"
    # # get this site from the model
    # sensor_site = spec.find_site(sensor_site_name)
    # if sensor_site is None:
    #     raise ValueError(f"Site '{sensor_site_name}' not found in the model.")

    # force_sensor = spec.add_sensor(
    # name="my_force_sensor",
    # type=mujoco.mjtSensor.mjSENS_FORCE,
    # objtype=mujoco.mjtObj.mjOBJ_SITE,  # Specify that the sensor is attached to a site object
    # objname=sensor_site.name      # Provide the name of the specific site
    # )

    model = spec.compile() #mujoco.MjModel.from_xml_path(xml_path)


    data = mujoco.MjData(model)

    # # print(f"number of sensors: {model.nsensor}")

    # # iterate over sensors and print their names
    # for sensor_id in range(model.nsensor):
    #     sensor = model.sensor(sensor_id)
    #     print(f"Sensor name: {sensor.name}, type: {sensor.type}, objtype: {sensor.objtype}")

    # # get sensor data
    # sensor_data = mujoco.mj_get_sensor(model, data, "my_force_sensor")
    # print(f"Sensor data for '{force_sensor.name}': {sensor_data}")

    # # Get the ID of the site by its name
    # site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, sensor_site_name)
    # if site_id == -1:
    #     raise ValueError(f"Site '{sensor_site_name}' not found in the model.")
    # print(f"Site ID for '{sensor_site_name}': {site_id}")

    # # get body id for toes_l
    # body_name = "toes_l"
    # body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)


    # create keys
    key = jax.random.key(0)
    n_envs = 100
    keys = jax.random.split(key, n_envs + 1)
    key, env_keys = keys[0], keys[1:]


    with mujoco.viewer.launch(model, data) as viewer:
        # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_WORLD_FRM] = True
        # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
        # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

        # viewer.sync()
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data) 

            # Pick up changes to the physics state, apply perturbations, update options from GUI
            viewer.sync()

            # Time keeping 
            time_until_new_step = model.opt.timestep - (time.time() - step_start)
            if time_until_new_step > 0:
                time.sleep(time_until_new_step)

    pass


if __name__ == "__main__":
    main()