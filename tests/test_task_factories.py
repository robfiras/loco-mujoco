import gc

from loco_mujoco import RLFactory, ImitationFactory
from loco_mujoco.task_factories import DefaultDatasetConf, LAFAN1DatasetConf, CustomDatasetConf
import gymnasium as gym

from test_conf import *


# Set Jax-backend to CPU
jax.config.update('jax_platform_name', 'cpu')
print(f"Jax backend device: {jax.default_backend()} \n")


def get_custom_traj(env_name):
    """ Generates a custom trajectory for testing. (all zeros) """
    N_steps = 100

    # create the environment
    env_cls = LocoEnv.registered_envs[env_name]
    env = env_cls()

    # reset the env
    key = jax.random.PRNGKey(0)
    env.reset(key)

    # get the model and data of the environment
    model = env.model
    data = env.data

    # get the initial qpos and qvel of the environment
    qpos = data.qpos
    qvel = data.qvel

    # stack qpos and qvel to a trajectory
    qpos = np.tile(qpos, (N_steps, 1))
    qvel = np.tile(qvel, (N_steps, 1))

    # create a trajectory info -- this stores basic information about the trajectory
    njnt = model.njnt
    jnt_type = model.jnt_type.copy()
    jnt_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(njnt)]
    traj_info = TrajectoryInfo(jnt_names, model=TrajectoryModel(njnt, jnp.array(jnt_type)), frequency=1 / env.dt)

    # create a trajectory data -- this stores the actual trajectory data
    traj_data = TrajectoryData(jnp.array(qpos), jnp.array(qvel), split_points=jnp.array([0, N_steps]))

    # combine them to a trajectory
    traj = Trajectory(traj_info, traj_data)

    return traj


@pytest.mark.parametrize("env_name", get_numpy_env_names())
def test_RLFactory_numpy(env_name):
    env, traj = RLFactory.make(env_name)
    assert isinstance(env, LocoEnv.registered_envs[env_name])


@pytest.mark.parametrize("env_name", get_jax_env_names())
def test_RLFactory_jax(env_name):
    gc.collect()
    env, traj = RLFactory.make(env_name)
    assert isinstance(env, LocoEnv.registered_envs[env_name])


@pytest.mark.parametrize("env_name", get_numpy_env_names())
def test_Gymasium(env_name):
    env = gym.make("LocoMujoco", env_name=env_name)
    assert isinstance(env.unwrapped, LocoEnv.registered_envs[env_name])


@pytest.mark.parametrize("env_name", get_numpy_env_names_default_dataset_conf())
def test_ImitationFactoryDefaultDatasetConf(env_name):
    task = "balance"
    env, traj = ImitationFactory.make(env_name, default_dataset_conf=DefaultDatasetConf(task))
    assert isinstance(env, LocoEnv.registered_envs[env_name])


@pytest.mark.parametrize("env_name", get_numpy_env_names_lafan1_dataset_conf())
def test_ImitationFactorLafan1(env_name):
    dataset_name = "dance1_subject1"
    env, traj = ImitationFactory.make(env_name, lafan1_dataset_conf=LAFAN1DatasetConf(dataset_name))
    assert isinstance(env, LocoEnv.registered_envs[env_name])


@pytest.mark.parametrize("env_name", get_numpy_env_names())
def test_ImitationFactoryCustomDataset(env_name):
    traj = get_custom_traj(env_name)
    env, traj = ImitationFactory.make(env_name, custom_dataset_conf=CustomDatasetConf(traj))
    assert isinstance(env, LocoEnv.registered_envs[env_name])
