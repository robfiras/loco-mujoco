"""Evaluate the distilled student.

Two modes (select via flags):

  * Metric mode (default): jit + parallel eval over many envs / steps using
    VanillaDaggerJax.build_eval_fn. Reports mean episode return/length.
    Fast, no rendering.

  * Play mode (`--play`): step-by-step rollout with VanillaDaggerJax.
    play_policy, renders / records a video. One env at a time.

Use `--use_teacher` to run the teacher instead of the student
(sanity check).
"""
import os
import argparse

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms.experimental import VanillaDaggerJax

os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True '


parser = argparse.ArgumentParser(description='Evaluate a distilled VanillaDagger student.')
parser.add_argument('--path', type=str, required=True,
                    help='Path to the saved agent (VanillaDaggerJax_saved.pkl)')
parser.add_argument('--play', action='store_true',
                    help='Switch to step-by-step play_policy (renders / records)')
parser.add_argument('--use_mujoco', action='store_true',
                    help='Use MuJoCo for play-mode eval instead of Mjx (only with --play)')
parser.add_argument('--use_teacher', action='store_true',
                    help='Play the frozen teacher instead of the student (sanity check; --play only)')
parser.add_argument('--deterministic', action='store_true',
                    help='Use the mean action in --play mode (metric mode is already deterministic)')
parser.add_argument('--n_steps', type=int, default=500,
                    help='Number of eval steps (per parallel env in metric mode)')
parser.add_argument('--n_envs', type=int, default=None,
                    help='Parallel envs for metric mode (defaults to config.eval_num_envs / num_envs)')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()


agent_conf, agent_state = VanillaDaggerJax.load_agent(args.path)
config = agent_conf.config

factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
OmegaConf.set_struct(config, False)
if args.play:
    config.experiment.env_params["headless"] = False
env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)


if args.play:
    VanillaDaggerJax.play_policy(
        env, agent_conf, agent_state,
        deterministic=args.deterministic,
        n_steps=args.n_steps, n_envs=1, record=True,
        use_mujoco=args.use_mujoco,
        use_teacher=args.use_teacher,
        traj=traj,
    )
else:
    # Inject eval sizing from CLI if provided.
    if args.n_envs is not None:
        config.experiment.eval_num_envs = int(args.n_envs)
    if args.n_steps is not None:
        config.experiment.eval_num_steps = int(args.n_steps)

    eval_fn = VanillaDaggerJax.build_eval_fn(env, agent_conf)
    eval_fn = jax.jit(eval_fn)

    rng = jax.random.PRNGKey(args.seed)
    out = eval_fn(rng, agent_state, traj)
    s = out["eval_summary"]
    n_envs = int(getattr(config.experiment, "eval_num_envs",
                         config.experiment.num_envs))
    n_steps = int(getattr(config.experiment, "eval_num_steps", 500))
    print(f"\n[eval] {n_envs} envs × {n_steps} steps, student deterministic")
    print(f"  mean episode return: {float(s.mean_episode_return):.3f}")
    print(f"  mean episode length: {float(s.mean_episode_length):.1f}")
    print(f"  max timestep       : {int(s.max_timestep)}\n")
