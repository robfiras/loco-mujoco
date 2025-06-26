import os
import sys
import jax
import wandb
import jax.numpy as jnp
import traceback

# Hydra:  key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line 
import hydra 
# from hydra.core.hydra_config import HydraConfig

from dataclasses import fields
from loco_mujoco.utils.metrics import QuantityContainer


from omegaconf import DictConfig, OmegaConf #OmegaConf is a YAML based hierarchical configuration system, with support for merging configurations from multiple sources

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.utils import MetricsHandler
from loco_mujoco import ImitationFactory



# Set MUJOCO_GL to egl
os.environ["MUJOCO_GL"] = "egl"  # Use EGL for rendering, which is more compatible with headless environments


@hydra.main(version_base=None, config_path="./", config_name="conf")
def experiment(config: DictConfig):
    try: 
        # can increase the speed by ~30% on some GPUs
        # os.environ['XLA_FLAGS'] = (
        #     '--xla_gpu_triton_gemm_any=True ')
        
        # Accessing the current sweep number
        result_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir


        # Extract date and time from the result directory path for wandb run name
        result_dir_parts = result_dir.split("/")
        result_dir_date, result_dir_time = result_dir_parts[-2], result_dir_parts[-1]
        formatted_result_dir = f"{result_dir_date}_{result_dir_time}_"


        # setup wandb 
        wandb.login()
        config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        run = wandb.init(project=config.wandb.project, name= formatted_result_dir + config.wandb.run_name, config=config_dict)


        # get task factory
        factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

        # create env
        env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

        # get initial agent configuration
        agent_conf = PPOJax.init_agent_conf(env, config) #PPOJax.init_agent_conf(env, config)

        # setup metric handler (optional)
        mh = MetricsHandler(config, env) if config.experiment.validation.active else None

        # build training function
        train_fn = PPOJax.build_train_fn(env, agent_conf, mh=mh) #PPOJax.build_train_fn(env, agent_conf, mh=mh)


        # jit and vmap training function
        train_fn = jax.jit(jax.vmap(train_fn)) if config.experiment.n_seeds > 1 else jax.jit(train_fn)

        # get rng keys and run training
        # skip_seed = 1
        # rngs = [jax.random.PRNGKey(i) for i in range(skip_seed,config.experiment.n_seeds+1)]  # create rngs from seed
        rngs = [jax.random.PRNGKey(i) for i in range(config.experiment.n_seeds+1)]  # create rngs from seed
        rng, _rng = rngs[0], jnp.squeeze(jnp.vstack(rngs[1:]))
        out = train_fn(_rng)


        # save agent state
        agent_state = out["agent_state"]
        save_path = PPOJax.save_agent(result_dir, agent_conf, agent_state) #PPOJax.save_agent(result_dir, agent_conf, agent_state)
        run.config.update({"agent_save_path": save_path})

        import time
        t_start = time.time()
        # get the metrics and log them
        if not config.experiment.debug:
            training_metrics = out["training_metrics"]
            validation_metrics = out["validation_metrics"]

            # calculate mean across seeds
            training_metrics = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), training_metrics)
            validation_metrics = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), validation_metrics)

            for i in range(len(training_metrics.mean_episode_return)):
                run.log({"Mean Episode Return": training_metrics.mean_episode_return[i],
                         "Mean Episode Length": training_metrics.mean_episode_length[i]},
                        step=int(training_metrics.max_timestep[i]))

                if (i+1) % config.experiment.validation_interval == 0 and config.experiment.validation.active:
                    run.log({"Validation Info/Mean Episode Return": validation_metrics.mean_episode_return[i],
                             "Validation Info/Mean Episode Length": validation_metrics.mean_episode_length[i]},
                            step=int(training_metrics.max_timestep[i]))

                    # log all measures
                    metrics_to_log = {}
                    for field in fields(validation_metrics):
                        attr = getattr(validation_metrics, field.name)
                        if isinstance(attr, QuantityContainer):
                            measure_name = field.name
                            for field_attr in fields(attr):
                                attr_name = field_attr.name
                                attr_value = getattr(attr, attr_name)
                                if attr_value.size > 0:
                                    metrics_to_log[f"Validation Measures/{measure_name}/{attr_name}"] = attr_value[i]

                    run.log(metrics_to_log, step=int(training_metrics.max_timestep[i]))

                    # metric for used for wandb sweep (optional)
                    site_rpos = validation_metrics.euclidean_distance.site_rpos[i]
                    site_rrotvec = validation_metrics.euclidean_distance.site_rpos[i]
                    site_rvel = validation_metrics.euclidean_distance.site_rpos[i]
                    run.log({"Metric for Sweep": site_rpos + site_rrotvec + site_rvel},
                            step=int(training_metrics.max_timestep[i]))

        print(f"Time taken to log metrics: {time.time() - t_start}s")

        # run the environment with the trained agent to record video
        PPOJax.play_policy(env, agent_conf, agent_state, deterministic=True, n_steps=200, n_envs=20, record=True,
                           train_state_seed=0)
        # PPOJax.play_policy(env, agent_conf, agent_state, deterministic=True, n_steps=200, n_envs=20, record=True,
        #                    train_state_seed=0)
        video_file = env.video_file_path
        run.log({"Agent Video": wandb.Video(video_file)})

        wandb.finish()

    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    experiment()
