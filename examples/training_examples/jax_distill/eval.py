"""Play the distilled student policy."""
import os
import argparse

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms.experimental import VanillaDaggerJax

from omegaconf import OmegaConf

os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True '

parser = argparse.ArgumentParser(description='Evaluate a distilled VanillaDagger student.')
parser.add_argument('--path', type=str, required=True,
                    help='Path to the agent pkl file (VanillaDaggerJax_saved.pkl)')
parser.add_argument('--use_mujoco', action='store_true',
                    help='Use MuJoCo for evaluation instead of Mjx')
parser.add_argument('--use_teacher', action='store_true',
                    help='Play the frozen teacher instead of the student (sanity check)')
parser.add_argument('--deterministic', action='store_true',
                    help='Use the mean action instead of sampling')
parser.add_argument('--n_steps', type=int, default=10000)
args = parser.parse_args()

agent_conf, agent_state = VanillaDaggerJax.load_agent(args.path)
config = agent_conf.config

factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
OmegaConf.set_struct(config, False)
config.experiment.env_params["headless"] = False
env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

VanillaDaggerJax.play_policy(
    env, agent_conf, agent_state,
    deterministic=args.deterministic,
    n_steps=args.n_steps, n_envs=1, record=True,
    use_mujoco=args.use_mujoco,
    use_teacher=args.use_teacher,
    traj=traj,
)
