"""S2PG2 obs-ablation experiment (Unitree H1 walking, joint-pos OR joint-vel removed).

Direct port of ias_cluster `JMLR_obstest_ablation/experiment_s2pg.py` with
S2PGPPOJax → S2PG2Jax. Tests whether the decoupled action / z-policy +
V-fitting cost-return reward of S2PG2 helps in the POMDP regime where the
obs is missing one of (joint position, joint velocity).

Pick a config at the CLI:
    python experiment.py --config-name=conf_no_jointvel
    python experiment.py --config-name=conf_no_jointpos
"""
import os
import sys
import jax
import jax.numpy as jnp
import wandb
from dataclasses import fields

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms.experimental import S2PG2Jax
from loco_mujoco.utils.metrics import QuantityContainer
from loco_mujoco.utils import MetricsHandler

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import traceback


@hydra.main(version_base=None, config_path="./", config_name="conf_no_jointvel")
def experiment(config: DictConfig):
    try:
        os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True '

        result_dir = HydraConfig.get().runtime.output_dir

        wandb.login()
        config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        run = wandb.init(project=config.wandb.project, config=config_dict)

        factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
        env, traj = factory.make(**config.experiment.env_params,
                                 **config.experiment.task_factory.params)

        agent_conf = S2PG2Jax.init_agent_conf(env, config)

        rngs = [jax.random.PRNGKey(i) for i in range(config.experiment.n_seeds + 1)]
        rng, _rng = rngs[0], jnp.squeeze(jnp.vstack(rngs[1:]))
        agent_state = jax.vmap(lambda r: S2PG2Jax.init_agent_state(env, agent_conf, r))(_rng) \
            if config.experiment.n_seeds > 1 else S2PG2Jax.init_agent_state(env, agent_conf, _rng)

        mh = MetricsHandler(config, env) if config.experiment.validation.active else None

        train_fn = S2PG2Jax.build_train_fn(env, agent_conf, mh=mh)
        train_fn = jax.jit(jax.vmap(train_fn, in_axes=(0, 0, None))) \
            if config.experiment.n_seeds > 1 else jax.jit(train_fn)

        out = train_fn(_rng, agent_state, traj)

        agent_state = out["agent_state"]
        save_path = S2PG2Jax.save_agent(result_dir, agent_conf, agent_state)
        run.config.update({"agent_save_path": save_path})

        if not config.experiment.debug:
            training_metrics = out["training_metrics"]
            validation_metrics = out["validation_metrics"]

            training_metrics = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), training_metrics)
            validation_metrics = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), validation_metrics)

            for i in range(len(training_metrics.mean_episode_return)):
                metrics_to_log = {f"Training/{field.name}": getattr(training_metrics, field.name)[i]
                                  for field in fields(training_metrics) if field.name != "max_timestep"}
                run.log(metrics_to_log, step=int(training_metrics.max_timestep[i]))

                if (i + 1) % config.experiment.validation_interval == 0 and config.experiment.validation.active:
                    run.log({"Validation Info/Mean Episode Return": validation_metrics.mean_episode_return[i],
                             "Validation Info/Mean Episode Length": validation_metrics.mean_episode_length[i]},
                            step=int(training_metrics.max_timestep[i]))

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

                    site_rpos = validation_metrics.euclidean_distance.site_rpos[i]
                    site_rrotvec = validation_metrics.euclidean_distance.site_rpos[i]
                    site_rvel = validation_metrics.euclidean_distance.site_rpos[i]
                    run.log({"Metric for Sweep": site_rpos + site_rrotvec + site_rvel},
                            step=int(training_metrics.max_timestep[i]))

        S2PG2Jax.play_policy(env, agent_conf, agent_state, deterministic=True,
                             n_steps=200, n_envs=20, record=True,
                             train_state_seed=0, traj=traj)
        video_file = env.video_file_path
        try:
            if video_file is not None:
                run.log({"Agent Video": wandb.Video(video_file)})
            else:
                print("[warn] env.video_file_path is None — skipping video upload.")
        except Exception as e:
            print(f"[warn] wandb video upload failed: {e!r} — continuing.")

        wandb.finish()

    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    experiment()
