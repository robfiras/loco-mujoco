import os
import sys
import jax
import jax.numpy as jnp
import wandb
from dataclasses import fields
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms.experimental import SACJax

import hydra
from omegaconf import DictConfig, OmegaConf
import traceback


@hydra.main(version_base=None, config_path="./", config_name="conf")
def experiment(config: DictConfig):
    try:

        os.environ['XLA_FLAGS'] = (
            '--xla_gpu_triton_gemm_any=True ')

        # Accessing the current sweep number
        result_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

        # setup wandb
        wandb.login()
        config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        run = wandb.init(project=config.wandb.project, config=config_dict)

        # get task factory
        factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

        # create env
        env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

        # get initial agent configuration
        agent_conf = SACJax.init_agent_conf(env, config)

        # initialize agent state
        rng = jax.random.PRNGKey(0)
        agent_state = SACJax.init_agent_state(env, agent_conf, rng)

        # build training function
        train_fn = SACJax.build_train_fn(env, agent_conf)
        train_fn = jax.jit(train_fn)

        out = train_fn(rng, agent_state, traj)

        # save agent state
        agent_state = out["agent_state"]
        save_path = SACJax.save_agent(result_dir, agent_conf, agent_state)
        run.config.update({"agent_save_path": save_path})

        import time
        t_start = time.time()
        # get the metrics and log them
        if not config.experiment.debug:
            training_metrics = out["training_metrics"]

            for i in range(len(training_metrics.mean_episode_return)):
                metrics_to_log = {f"Training/{field.name}": getattr(training_metrics, field.name)[i]
                                  for field in fields(training_metrics) if field.name != "max_timestep"}
                run.log(metrics_to_log, step=int(training_metrics.max_timestep[i]))

        print(f"Time taken to log metrics: {time.time() - t_start}s")

        # run the environment with the trained agent to record video
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
