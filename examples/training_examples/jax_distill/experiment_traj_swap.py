"""
Multi-teacher DAgger distillation with per-chunk (expert, traj) swapping.

Config is a flat list of (expert_ckpt, traj) pairs under `pairs`. An expert
may appear multiple times (once per trajectory it's qualified for) — that's
the natural way to express "this expert handles N trajs" and also gives you
free weighting (list an entry more often to sample it more often).

At each training chunk:
  1. Pick one pair at random.
  2. Swap its trajectory into the env via `env.process_trajectory`.
  3. Swap its teacher params / run_stats onto the agent state.
  4. Null env_state + last_obs so the env resets cleanly for the new traj.

Replay buffer + rollout mixture state persist across swaps — the whole
point of the VanillaDagger layout. Gradient updates sample from the union
of every (expert, traj) pair ever visited.

Expert checkpoints are cached by path: if the same ckpt appears in multiple
pairs it's loaded from disk only once.
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


def _fresh_copy(tree):
    """Allocate a fresh device copy of every leaf in `tree`.

    We pass teacher params to the jit'd train_fn which donates the whole
    agent_state (to avoid OOM from double-allocation of the replay buffer).
    Donation invalidates every array on the input — including the teacher
    params. If those arrays are also referenced from an outside cache,
    subsequent lookups would hit "Array has been deleted". Copying on each
    lookup keeps the cache's arrays alive.
    """
    return jax.tree.map(jnp.copy, tree)


@hydra.main(version_base=None, config_path="./", config_name="conf_traj_swap")
def experiment(config: DictConfig):
    try:
        os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True '
        result_dir = HydraConfig.get().runtime.output_dir

        wandb.login()
        config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        run = wandb.init(project=config.wandb.project, config=config_dict)

        # Env bootstrap — uses the first pair's task as initial trajectory.
        factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
        env, traj = factory.make(**config.experiment.env_params,
                                 **config.experiment.task_factory.params)

        # Preload everything once. Experts are cached by ckpt path so an
        # expert that appears in K pairs is only deserialised once.
        dataset_type = config.traj_swap.dataset_type
        expert_cache = {}
        traj_cache = {}
        pairs = []
        for entry in config.traj_swap.pairs:
            ckpt = entry.expert_ckpt
            task = entry.traj

            if ckpt not in expert_cache:
                expert_cache[ckpt] = load_teacher_params(ckpt)
                print(f"[swap] loaded expert {ckpt}")

            if task not in traj_cache:
                dconf = DefaultDatasetConf(task=task, dataset_type=dataset_type)
                traj_cache[task] = ImitationFactory.get_default_traj(env, dconf)
                print(f"[swap] loaded traj {task}")

            pairs.append((ckpt, task))

        print(f"[swap] {len(pairs)} (expert, traj) pairs, "
              f"{len(expert_cache)} unique experts, {len(traj_cache)} unique trajs")

        # Chunked training
        total_timesteps = int(config.experiment.total_timesteps)
        timesteps_per_chunk = int(config.experiment.timesteps_per_chunk)
        with open_dict(config.experiment):
            config.experiment.total_timesteps = timesteps_per_chunk

        agent_conf = VanillaDaggerJax.init_agent_conf(env, config)
        n_chunks = max(1, total_timesteps // timesteps_per_chunk)

        # Bootstrap with the first pair's teacher so shapes are concrete.
        rng = jax.random.PRNGKey(0)
        first_ckpt, _ = pairs[0]
        cached_params, cached_run_stats = expert_cache[first_ckpt]
        agent_state = VanillaDaggerJax.init_agent_state(
            env, agent_conf, rng,
            teacher_params=_fresh_copy(cached_params),
            teacher_run_stats=_fresh_copy(cached_run_stats),
        )

        train_fn = VanillaDaggerJax.build_train_fn(env, agent_conf)
        # donate_argnums=(1,) donates `agent_state` to the output so XLA can
        # alias the replay buffer + env_state in place — without this, the
        # input agent_state and its output copy both live in GPU memory
        # during the call, roughly doubling peak usage.
        train_fn = jax.jit(train_fn, donate_argnums=(1,))

        rng_py = random.Random(int(config.get("swap_seed", 0)))
        global_step_offset = 0
        for chunk in range(n_chunks):
            ckpt, task = rng_py.choice(pairs)
            cached_params, cached_run_stats = expert_cache[ckpt]

            # Swap traj into the env (env's trajectory-conditioned obs must match).
            traj = env.process_trajectory(traj_cache[task])

            # Swap teacher + null env carry so the env resets cleanly for the
            # new traj. Replay buffer + rollout_state stay.
            # Fresh-copy teacher params: train_fn donates the agent_state, which
            # would otherwise invalidate the arrays the cache still references.
            agent_state = agent_state.replace(
                teacher_params=_fresh_copy(cached_params),
                teacher_run_stats=_fresh_copy(cached_run_stats),
                env_state=None,
                last_obs=None,
            )

            print(f"[swap] chunk={chunk + 1}/{n_chunks} expert={os.path.basename(ckpt)} traj={task}")

            out = train_fn(rng, agent_state, traj)
            new_agent_state = out["agent_state"]

            if not config.experiment.debug:
                metrics = out["training_metrics"]
                for i in range(len(metrics.mean_episode_return)):
                    step = global_step_offset + int(metrics.max_timestep[i])
                    payload = {f"Training/{f.name}": getattr(metrics, f.name)[i]
                               for f in fields(metrics) if f.name != "max_timestep"}
                    payload["Training/chunk_expert"] = os.path.basename(ckpt)
                    payload["Training/chunk_traj"] = task
                    payload[f"Training/mean_episode_return__{task}"] = metrics.mean_episode_return[i]
                    payload[f"Training/mean_episode_length__{task}"] = metrics.mean_episode_length[i]
                    run.log(payload, step=step)
                global_step_offset += int(metrics.max_timestep[-1])

            # Release the previous chunk's pytree (and `out`) before the next
            # train_fn call so we don't hold two full agent_states in GPU
            # memory simultaneously.
            del agent_state, out
            agent_state = new_agent_state

        # Save final student (with the last-used teacher baked in; swap later
        # via `.replace(teacher_params=...)` if you reload for more training).
        save_path = VanillaDaggerJax.save_agent(result_dir, agent_conf, agent_state)
        run.config.update({"agent_save_path": save_path})

        # Record video on the last swapped-in traj.
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
