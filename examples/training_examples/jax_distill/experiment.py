"""
Single-teacher DAgger distillation.

Loads a pretrained PPO teacher from a checkpoint, distills it into a student
via Vanilla DAgger. Rollouts use a sticky per-env mixture of student/teacher;
the label is always the teacher's action at the observed state.
"""
import os
import sys
import jax
import jax.numpy as jnp
import wandb
from dataclasses import fields
from omegaconf import DictConfig, OmegaConf, open_dict

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.algorithms.experimental import VanillaDaggerJax

import hydra
from hydra.core.hydra_config import HydraConfig
import traceback


def load_teacher(teacher_ckpt: str):
    """Load teacher params + run_stats from a saved PPOJax checkpoint.

    Any JaxRLAlgorithmBase agent that exposes `train_state.params` /
    `train_state.run_stats` works the same way — swap `PPOJax.load_agent`
    for the relevant class.
    """
    teacher_conf, teacher_state = PPOJax.load_agent(teacher_ckpt)
    return (teacher_state.train_state.params, teacher_state.train_state.run_stats,
            teacher_conf)


@hydra.main(version_base=None, config_path="./", config_name="conf")
def experiment(config: DictConfig):
    try:
        os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True '
        result_dir = HydraConfig.get().runtime.output_dir

        # wandb
        wandb.login()
        config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        run = wandb.init(project=config.wandb.project, config=config_dict)

        # env
        factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
        env, traj = factory.make(**config.experiment.env_params,
                                 **config.experiment.task_factory.params)

        # Load teacher. The DAgger `teacher` sub-config must match the teacher's
        # architecture (hidden_layers / activation / etc.) so init can build a
        # matching network.
        teacher_params, teacher_run_stats, teacher_conf = load_teacher(config.teacher_ckpt)
        print(f"[distill] loaded teacher from {config.teacher_ckpt}")

        # Chunked training so we can log per-chunk metrics and save intermediates.
        total_timesteps = int(config.experiment.total_timesteps)
        timesteps_per_chunk = int(config.experiment.timesteps_per_chunk)
        with open_dict(config.experiment):
            config.experiment.total_timesteps = timesteps_per_chunk

        agent_conf = VanillaDaggerJax.init_agent_conf(env, config)
        updates_per_chunk = int(agent_conf.config.experiment.num_updates)
        n_chunks = max(1, total_timesteps // (timesteps_per_chunk))

        rng = jax.random.PRNGKey(0)
        agent_state = VanillaDaggerJax.init_agent_state(
            env, agent_conf, rng,
            teacher_params=teacher_params,
            teacher_run_stats=teacher_run_stats,
        )

        train_fn = VanillaDaggerJax.build_train_fn(env, agent_conf)
        train_fn = jax.jit(train_fn)

        global_step_offset = 0
        for chunk in range(n_chunks):
            print(f"[distill] chunk {chunk + 1}/{n_chunks}")
            out = train_fn(rng, agent_state, traj)
            agent_state = out["agent_state"]

            if not config.experiment.debug:
                metrics = out["training_metrics"]
                for i in range(len(metrics.mean_episode_return)):
                    step = global_step_offset + int(metrics.max_timestep[i])
                    payload = {f"Training/{f.name}": getattr(metrics, f.name)[i]
                               for f in fields(metrics) if f.name != "max_timestep"}
                    run.log(payload, step=step)
                global_step_offset += int(metrics.max_timestep[-1])

        # Save the student.
        save_path = VanillaDaggerJax.save_agent(result_dir, agent_conf, agent_state)
        run.config.update({"agent_save_path": save_path})

        # Record a video of the student.
        VanillaDaggerJax.play_policy(
            env, agent_conf, agent_state,
            deterministic=True, n_steps=200, n_envs=20, record=True, traj=traj,
        )
        run.log({"Student Video": wandb.Video(env.video_file_path)})

        wandb.finish()

    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    experiment()
