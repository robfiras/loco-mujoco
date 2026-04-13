import os
import sys
import jax
import jax.numpy as jnp
import wandb
from dataclasses import fields
from omegaconf import DictConfig, OmegaConf, open_dict
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms.experimental import SACJax

import hydra
from hydra.core.hydra_config import HydraConfig
import traceback


@hydra.main(version_base=None, config_path="./", config_name="conf_resume")
def experiment(config: DictConfig):
    try:

        os.environ['XLA_FLAGS'] = (
            '--xla_gpu_triton_gemm_any=True ')

        result_dir = HydraConfig.get().runtime.output_dir

        # setup wandb
        wandb.login()
        config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        run = wandb.init(project=config.wandb.project, config=config_dict)

        # get task factory and create env
        factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
        env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

        # chunked training setup
        total_timesteps = int(config.experiment.total_timesteps)
        timesteps_per_chunk = int(config.experiment.timesteps_per_chunk)
        num_envs = int(config.experiment.num_envs)

        # override total_timesteps to per-chunk so init_agent_conf computes num_updates per chunk
        with open_dict(config.experiment):
            config.experiment.total_timesteps = timesteps_per_chunk

        agent_conf = SACJax.init_agent_conf(env, config)

        # compute n_chunks from actual num_updates
        updates_per_chunk = int(agent_conf.config.experiment.num_updates)
        total_updates = total_timesteps // num_envs
        n_chunks = total_updates // updates_per_chunk

        # initialize agent state and build training function
        rng = jax.random.PRNGKey(0)
        agent_state = SACJax.init_agent_state(env, agent_conf, rng)

        train_fn = SACJax.build_train_fn(env, agent_conf)
        train_fn = jax.jit(train_fn)

        for chunk in range(n_chunks):
            out = train_fn(rng, agent_state, traj)

            agent_state = out["agent_state"]

            if not config.experiment.debug:
                training_metrics = out["training_metrics"]

                for i in range(len(training_metrics.mean_episode_return)):
                    step = int(training_metrics.max_timestep[i])
                    metrics_to_log = {f"Training/{field.name}": getattr(training_metrics, field.name)[i]
                                      for field in fields(training_metrics) if field.name != "max_timestep"}
                    run.log(metrics_to_log, step=step)

        # save final agent state
        save_path = SACJax.save_agent(result_dir, agent_conf, agent_state)
        run.config.update({"agent_save_path": save_path})

        # record video with trained agent
        SACJax.play_policy(env, agent_conf, agent_state, deterministic=True, n_steps=200, n_envs=20, record=True,
                           traj=traj)
        video_file = env.video_file_path
        run.log({"Agent Video": wandb.Video(video_file)})

        wandb.finish()

    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    experiment()
