"""
Multi-teacher DAgger distillation with per-chunk trajectory + teacher swapping.

At every training chunk: pick a task uniformly at random, swap in its
pretrained teacher and its trajectory, and keep training. The replay buffer
persists across swaps — the entire point of this setup — so data collected
under a previous teacher/traj remains useful.
"""
import os
import sys
import random
import jax
import jax.numpy as jnp
import wandb
from dataclasses import fields
from omegaconf import DictConfig, OmegaConf, open_dict

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.algorithms.experimental import VanillaDaggerJax
from loco_mujoco.task_factories import ImitationFactory
from loco_mujoco.task_factories.dataset_confs import DefaultDatasetConf

import hydra
from hydra.core.hydra_config import HydraConfig
import traceback


def load_teacher_params(teacher_ckpt: str):
    """Load (teacher_params, teacher_run_stats) from a saved PPO checkpoint."""
    _conf, teacher_state = PPOJax.load_agent(teacher_ckpt)
    return teacher_state.train_state.params, teacher_state.train_state.run_stats


@hydra.main(version_base=None, config_path="./", config_name="conf_traj_swap")
def experiment(config: DictConfig):
    try:
        os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True '
        result_dir = HydraConfig.get().runtime.output_dir

        wandb.login()
        config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        run = wandb.init(project=config.wandb.project, config=config_dict)

        # Env bootstrap — uses the first task as its initial trajectory.
        factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
        env, traj = factory.make(**config.experiment.env_params,
                                 **config.experiment.task_factory.params)

        # Preload every (traj, teacher) pair so chunk boundaries are fast:
        # no disk I/O or re-init inside the training loop.
        dataset_type = config.traj_swap.dataset_type
        tasks = []
        preloaded = {}
        for task_entry in config.traj_swap.tasks:
            task_name = task_entry.task
            ckpt = task_entry.teacher_ckpt
            dconf = DefaultDatasetConf(task=task_name, dataset_type=dataset_type)
            preloaded[task_name] = {
                "traj": ImitationFactory.get_default_traj(env, dconf),
                "teacher_params": None,
                "teacher_run_stats": None,
            }
            params, run_stats = load_teacher_params(ckpt)
            preloaded[task_name]["teacher_params"] = params
            preloaded[task_name]["teacher_run_stats"] = run_stats
            tasks.append(task_name)
            print(f"[swap] preloaded task={task_name} teacher={ckpt}")

        # Chunked training
        total_timesteps = int(config.experiment.total_timesteps)
        timesteps_per_chunk = int(config.experiment.timesteps_per_chunk)
        with open_dict(config.experiment):
            config.experiment.total_timesteps = timesteps_per_chunk

        agent_conf = VanillaDaggerJax.init_agent_conf(env, config)
        n_chunks = max(1, total_timesteps // timesteps_per_chunk)

        # Bootstrap state with the FIRST task's teacher so it matches the conf
        # shape. (Subsequent chunks just swap teacher_params/run_stats.)
        first_task = tasks[0]
        rng = jax.random.PRNGKey(0)
        agent_state = VanillaDaggerJax.init_agent_state(
            env, agent_conf, rng,
            teacher_params=preloaded[first_task]["teacher_params"],
            teacher_run_stats=preloaded[first_task]["teacher_run_stats"],
        )

        train_fn = VanillaDaggerJax.build_train_fn(env, agent_conf)
        train_fn = jax.jit(train_fn)

        rng_py = random.Random(int(config.get("swap_seed", 0)))
        global_step_offset = 0
        for chunk in range(n_chunks):
            task = rng_py.choice(tasks)
            entry = preloaded[task]

            # Swap traj into the env (so env's trajectory-conditioned obs match).
            traj = env.process_trajectory(entry["traj"])

            # Swap teacher params + null env_state/last_obs so the env resets
            # cleanly for the new traj. Replay buffer + rollout_state stay.
            agent_state = agent_state.replace(
                teacher_params=entry["teacher_params"],
                teacher_run_stats=entry["teacher_run_stats"],
                env_state=None,
                last_obs=None,
            )

            print(f"[swap] chunk={chunk + 1}/{n_chunks} task={task}")

            out = train_fn(rng, agent_state, traj)
            agent_state = out["agent_state"]

            if not config.experiment.debug:
                metrics = out["training_metrics"]
                for i in range(len(metrics.mean_episode_return)):
                    step = global_step_offset + int(metrics.max_timestep[i])
                    payload = {f"Training/{f.name}": getattr(metrics, f.name)[i]
                               for f in fields(metrics) if f.name != "max_timestep"}
                    payload["Training/chunk_task"] = task
                    payload["Training/chunk_task_index"] = tasks.index(task)
                    payload[f"Training/mean_episode_return__{task}"] = metrics.mean_episode_return[i]
                    payload[f"Training/mean_episode_length__{task}"] = metrics.mean_episode_length[i]
                    run.log(payload, step=step)
                global_step_offset += int(metrics.max_timestep[-1])

        # Save final student (with the last-used teacher baked in; swap later
        # via `.replace(teacher_params=...)` if you reload for more training).
        save_path = VanillaDaggerJax.save_agent(result_dir, agent_conf, agent_state)
        run.config.update({"agent_save_path": save_path})

        # Record video on the last swapped-in task.
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
