import pytest
from jax import make_jaxpr
from omegaconf import open_dict

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax, GAILJax, AMPJax
from loco_mujoco.algorithms.experimental import S2PGPPOJax, BPTTPPOJax, HistoryPPOJax, SACJax, TD3Jax, VanillaDaggerJax
from loco_mujoco.utils import MetricsHandler

from test_conf import *


# Set Jax-backend to CPU
jax.config.update('jax_platform_name', 'cpu')
print(f"Jax backend device: {jax.default_backend()} \n")


def _params_allclose(a, b):
    return all(
        jnp.allclose(x, y)
        for x, y in zip(
            jax.tree_util.tree_leaves(a),
            jax.tree_util.tree_leaves(b),
        )
    )


def test_PPO_Jax_build_train_fn(ppo_rl_config):

    config = ppo_rl_config

    # get task factory
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

    # create env
    env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

    # get initial agent configuration
    agent_conf = PPOJax.init_agent_conf(env, config)

    # initialize agent state
    rngs = [jax.random.PRNGKey(i) for i in range(config.experiment.n_seeds+1)]
    rng, _rng = rngs[0], jnp.squeeze(jnp.vstack(rngs[1:]))
    agent_state = jax.vmap(lambda r: PPOJax.init_agent_state(env, agent_conf, r))(_rng) \
        if config.experiment.n_seeds > 1 else PPOJax.init_agent_state(env, agent_conf, _rng)

    # build training function
    train_fn = PPOJax.build_train_fn(env, agent_conf)

    # jit and vmap training function
    train_fn = jax.jit(jax.vmap(train_fn, in_axes=(0, 0, None))) if config.experiment.n_seeds > 1 else jax.jit(train_fn)

    # Use make_jaxpr to check if the function compiles correctly
    try:
        jaxpr = make_jaxpr(train_fn)(_rng, agent_state, traj)

        assert jaxpr is not None
    except Exception as e:
        pytest.fail(f"JAX function compilation failed: {e}")


def test_PPO_save_and_load_agent(ppo_rl_config, tmp_path):
    """Train for a few steps, save agent, load it, and verify params match."""
    config = OmegaConf.create(OmegaConf.to_container(ppo_rl_config, resolve=True))
    with open_dict(config.experiment):
        config.experiment.total_timesteps = 64
        config.experiment.num_envs = 4
        config.experiment.num_steps = 8
        config.experiment.num_minibatches = 32
        config.experiment.validation.num = 1

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)
    agent_conf = PPOJax.init_agent_conf(env, config)
    rng = jax.random.PRNGKey(0)
    agent_state = PPOJax.init_agent_state(env, agent_conf, rng)
    train_fn = PPOJax.build_train_fn(env, agent_conf)
    train_fn = jax.jit(train_fn)

    result = train_fn(rng, agent_state, traj)
    agent_state = result["agent_state"]

    save_path = PPOJax.save_agent(tmp_path, agent_conf, agent_state)
    assert save_path.exists()

    loaded_conf, loaded_state = PPOJax.load_agent(save_path)
    assert loaded_conf is not None
    assert loaded_state is not None

    assert _params_allclose(agent_state.train_state.params, loaded_state.train_state.params)
    assert _params_allclose(agent_state.train_state.run_stats, loaded_state.train_state.run_stats)


def test_PPO_eval_fn(ppo_rl_config):
    """Standalone eval_fn should jit, run, and produce finite metrics without
    mutating the agent's training state (mirrors the VanillaDagger eval test)."""
    config = ppo_rl_config

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

    agent_conf = PPOJax.init_agent_conf(env, config)
    rng = jax.random.PRNGKey(0)
    agent_state = PPOJax.init_agent_state(env, agent_conf, rng)

    eval_fn = jax.jit(PPOJax.build_eval_fn(env, agent_conf))

    try:
        jaxpr = make_jaxpr(eval_fn)(rng, agent_state, traj)
        assert jaxpr is not None
    except Exception as e:
        pytest.fail(f"JAX eval function compilation failed: {e}")

    out = eval_fn(rng, agent_state, traj)
    assert "eval_summary" in out
    assert "validation_metrics" in out
    assert jnp.isfinite(out["eval_summary"].mean_episode_return)
    assert jnp.isfinite(out["eval_summary"].mean_episode_length)


def test_PPO_adaptive_kl_lr(ppo_rl_config):
    """With anneal_lr off and desired_kl set, PPO uses an inject_hyperparams
    optimizer and the RSL-style adaptive-KL learning-rate branch. Exercise it
    by running a short training chunk."""
    config = OmegaConf.create(OmegaConf.to_container(ppo_rl_config, resolve=True))
    with open_dict(config.experiment):
        config.experiment.total_timesteps = 64
        config.experiment.num_envs = 4
        config.experiment.num_steps = 8
        config.experiment.num_minibatches = 32
        config.experiment.validation.num = 1
        config.experiment.anneal_lr = False
        config.experiment.desired_kl = 0.01

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)
    agent_conf = PPOJax.init_agent_conf(env, config)
    rng = jax.random.PRNGKey(0)
    agent_state = PPOJax.init_agent_state(env, agent_conf, rng)
    train_fn = jax.jit(PPOJax.build_train_fn(env, agent_conf))

    result = train_fn(rng, agent_state, traj)
    # training completed and produced a finite adapted learning rate
    lr = result["training_metrics"].learning_rate
    assert jnp.all(jnp.isfinite(lr))


@pytest.mark.parametrize("algorithm", ("GAIL", "AMP"))
def test_Imitation_save_and_load_agent(algorithm, imitation_config, tmp_path):
    """Train imitation agent for a few steps, save, load, and verify params match."""
    alg_cls = GAILJax if algorithm == "GAIL" else AMPJax
    config = OmegaConf.create(OmegaConf.to_container(imitation_config, resolve=True))
    with open_dict(config.experiment):
        config.experiment.total_timesteps = 64
        config.experiment.num_envs = 4
        config.experiment.num_steps = 8
        config.experiment.num_minibatches = 32
        config.experiment.n_seeds = 1
        config.experiment.validation.num = 1
        config.experiment.validation.active = False

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)
    expert_dataset = env.create_dataset()
    agent_conf = alg_cls.init_agent_conf(env, config)
    agent_conf = agent_conf.add_expert_dataset(expert_dataset)

    rng = jax.random.PRNGKey(0)
    agent_state = alg_cls.init_agent_state(env, agent_conf, rng)
    train_fn = alg_cls.build_train_fn(env, agent_conf, mh=None)
    train_fn = jax.jit(train_fn)

    result = train_fn(rng, agent_state, traj)
    agent_state = result["agent_state"]

    save_path = alg_cls.save_agent(tmp_path, agent_conf, agent_state)
    assert save_path.exists()

    loaded_conf, loaded_state = alg_cls.load_agent(save_path)
    assert loaded_conf is not None
    assert loaded_state is not None
    # Loaded conf has expert_dataset=None (not saved)
    assert loaded_conf.expert_dataset is None

    assert _params_allclose(agent_state.train_state.params, loaded_state.train_state.params)
    assert _params_allclose(agent_state.train_state.run_stats, loaded_state.train_state.run_stats)
    assert _params_allclose(agent_state.disc_train_state.params, loaded_state.disc_train_state.params)
    assert _params_allclose(agent_state.disc_train_state.run_stats, loaded_state.disc_train_state.run_stats)


@pytest.mark.parametrize("algorithm", ("GAIL", "AMP"))
def test_Imitation_Jax_build_train_fn(algorithm, imitation_config):

    alg_cls = GAILJax if algorithm == "GAIL" else AMPJax

    config = imitation_config

    # get task factory
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

    # create env
    env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

    # create an expert dataset
    expert_dataset = env.create_dataset()

    # get initial agent configuration
    agent_conf = alg_cls.init_agent_conf(env, config)
    agent_conf = agent_conf.add_expert_dataset(expert_dataset)

    # setup metric handler (optional)
    mh = MetricsHandler(config, env) if config.experiment.validation.active else None

    # initialize agent state
    rngs = [jax.random.PRNGKey(i) for i in range(config.experiment.n_seeds+1)]
    rng, _rng = rngs[0], jnp.squeeze(jnp.vstack(rngs[1:]))
    agent_state = jax.vmap(lambda r: alg_cls.init_agent_state(env, agent_conf, r))(_rng) \
        if config.experiment.n_seeds > 1 else alg_cls.init_agent_state(env, agent_conf, _rng)

    # build training function
    train_fn = alg_cls.build_train_fn(env, agent_conf, mh=mh)

    # jit and vmap training function
    train_fn = jax.jit(jax.vmap(train_fn, in_axes=(0, 0, None))) if config.experiment.n_seeds > 1 else jax.jit(train_fn)

    # Use make_jaxpr to check if the function compiles correctly
    try:
        jaxpr = make_jaxpr(train_fn)(_rng, agent_state, traj)

        assert jaxpr is not None
    except Exception as e:
        pytest.fail(f"JAX function compilation failed: {e}")


def test_S2PG_PPO_build_train_fn(s2pg_ppo_config):

    config = s2pg_ppo_config

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

    agent_conf = S2PGPPOJax.init_agent_conf(env, config)

    rngs = [jax.random.PRNGKey(i) for i in range(config.experiment.n_seeds + 1)]
    rng, _rng = rngs[0], jnp.squeeze(jnp.vstack(rngs[1:]))
    agent_state = jax.vmap(lambda r: S2PGPPOJax.init_agent_state(env, agent_conf, r))(_rng) \
        if config.experiment.n_seeds > 1 else S2PGPPOJax.init_agent_state(env, agent_conf, _rng)

    train_fn = S2PGPPOJax.build_train_fn(env, agent_conf)
    train_fn = jax.jit(jax.vmap(train_fn, in_axes=(0, 0, None))) \
        if config.experiment.n_seeds > 1 else jax.jit(train_fn)

    try:
        jaxpr = make_jaxpr(train_fn)(_rng, agent_state, traj)
        assert jaxpr is not None
    except Exception as e:
        pytest.fail(f"JAX function compilation failed: {e}")


def test_S2PG_PPO_save_and_load_agent(s2pg_ppo_config, tmp_path):
    """Train S2PG for a few steps, save agent, load it, and verify params match."""
    config = OmegaConf.create(OmegaConf.to_container(s2pg_ppo_config, resolve=True))
    with open_dict(config.experiment):
        config.experiment.total_timesteps = 64
        config.experiment.num_envs = 4
        config.experiment.num_steps = 8
        config.experiment.num_minibatches = 32
        config.experiment.validation.num = 1

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)
    agent_conf = S2PGPPOJax.init_agent_conf(env, config)
    rng = jax.random.PRNGKey(0)
    agent_state = S2PGPPOJax.init_agent_state(env, agent_conf, rng)
    train_fn = jax.jit(S2PGPPOJax.build_train_fn(env, agent_conf))

    result = train_fn(rng, agent_state, traj)
    agent_state = result["agent_state"]

    save_path = S2PGPPOJax.save_agent(tmp_path, agent_conf, agent_state)
    assert save_path.exists()

    loaded_conf, loaded_state = S2PGPPOJax.load_agent(save_path)
    assert loaded_conf is not None
    assert loaded_state is not None

    assert _params_allclose(agent_state.train_state.params, loaded_state.train_state.params)
    assert _params_allclose(agent_state.train_state.run_stats, loaded_state.train_state.run_stats)


def test_BPTT_PPO_build_train_fn(bptt_ppo_config):

    config = bptt_ppo_config

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

    agent_conf = BPTTPPOJax.init_agent_conf(env, config)

    rngs = [jax.random.PRNGKey(i) for i in range(config.experiment.n_seeds + 1)]
    rng, _rng = rngs[0], jnp.squeeze(jnp.vstack(rngs[1:]))
    agent_state = jax.vmap(lambda r: BPTTPPOJax.init_agent_state(env, agent_conf, r))(_rng) \
        if config.experiment.n_seeds > 1 else BPTTPPOJax.init_agent_state(env, agent_conf, _rng)

    train_fn = BPTTPPOJax.build_train_fn(env, agent_conf)
    train_fn = jax.jit(jax.vmap(train_fn, in_axes=(0, 0, None))) \
        if config.experiment.n_seeds > 1 else jax.jit(train_fn)

    try:
        jaxpr = make_jaxpr(train_fn)(_rng, agent_state, traj)
        assert jaxpr is not None
    except Exception as e:
        pytest.fail(f"JAX function compilation failed: {e}")


def test_BPTT_PPO_save_and_load_agent(bptt_ppo_config, tmp_path):
    """Train BPTT-PPO for a few steps, save agent, load it, and verify params match."""
    config = OmegaConf.create(OmegaConf.to_container(bptt_ppo_config, resolve=True))
    with open_dict(config.experiment):
        config.experiment.total_timesteps = 64
        config.experiment.num_envs = 4
        config.experiment.num_steps = 8
        config.experiment.num_minibatches = 4
        config.experiment.validation.num = 1

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)
    agent_conf = BPTTPPOJax.init_agent_conf(env, config)
    rng = jax.random.PRNGKey(0)
    agent_state = BPTTPPOJax.init_agent_state(env, agent_conf, rng)
    train_fn = jax.jit(BPTTPPOJax.build_train_fn(env, agent_conf))

    result = train_fn(rng, agent_state, traj)
    agent_state = result["agent_state"]

    save_path = BPTTPPOJax.save_agent(tmp_path, agent_conf, agent_state)
    assert save_path.exists()

    loaded_conf, loaded_state = BPTTPPOJax.load_agent(save_path)
    assert loaded_conf is not None
    assert loaded_state is not None

    assert _params_allclose(agent_state.train_state.params, loaded_state.train_state.params)
    assert _params_allclose(agent_state.train_state.run_stats, loaded_state.train_state.run_stats)


@pytest.mark.parametrize("encoder_type", ("mlp", "transformer"))
def test_History_PPO_build_train_fn(encoder_type, history_ppo_config):

    config = OmegaConf.create(OmegaConf.to_container(history_ppo_config, resolve=True))
    with open_dict(config.experiment):
        config.experiment.encoder_type = encoder_type

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

    agent_conf = HistoryPPOJax.init_agent_conf(env, config)

    rngs = [jax.random.PRNGKey(i) for i in range(config.experiment.n_seeds + 1)]
    rng, _rng = rngs[0], jnp.squeeze(jnp.vstack(rngs[1:]))
    agent_state = jax.vmap(lambda r: HistoryPPOJax.init_agent_state(env, agent_conf, r))(_rng) \
        if config.experiment.n_seeds > 1 else HistoryPPOJax.init_agent_state(env, agent_conf, _rng)

    train_fn = HistoryPPOJax.build_train_fn(env, agent_conf)
    train_fn = jax.jit(jax.vmap(train_fn, in_axes=(0, 0, None))) \
        if config.experiment.n_seeds > 1 else jax.jit(train_fn)

    try:
        jaxpr = make_jaxpr(train_fn)(_rng, agent_state, traj)
        assert jaxpr is not None
    except Exception as e:
        pytest.fail(f"JAX function compilation failed: {e}")


def test_History_PPO_save_and_load_agent(history_ppo_config, tmp_path):
    """Train History-PPO for a few steps, save agent, load it, and verify params match."""
    config = OmegaConf.create(OmegaConf.to_container(history_ppo_config, resolve=True))
    with open_dict(config.experiment):
        config.experiment.total_timesteps = 64
        config.experiment.num_envs = 4
        config.experiment.num_steps = 8
        config.experiment.num_minibatches = 32
        config.experiment.validation.num = 1

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)
    agent_conf = HistoryPPOJax.init_agent_conf(env, config)
    rng = jax.random.PRNGKey(0)
    agent_state = HistoryPPOJax.init_agent_state(env, agent_conf, rng)
    train_fn = jax.jit(HistoryPPOJax.build_train_fn(env, agent_conf))

    result = train_fn(rng, agent_state, traj)
    agent_state = result["agent_state"]

    save_path = HistoryPPOJax.save_agent(tmp_path, agent_conf, agent_state)
    assert save_path.exists()

    loaded_conf, loaded_state = HistoryPPOJax.load_agent(save_path)
    assert loaded_conf is not None
    assert loaded_state is not None

    assert _params_allclose(agent_state.train_state.params, loaded_state.train_state.params)
    assert _params_allclose(agent_state.train_state.run_stats, loaded_state.train_state.run_stats)


def test_S2PG_PPO_adaptive_kl_lr(s2pg_ppo_config):
    """With anneal_lr off and desired_kl set, S2PG-PPO uses an inject_hyperparams
    optimizer and exercises the adaptive-KL learning-rate branch."""
    config = OmegaConf.create(OmegaConf.to_container(s2pg_ppo_config, resolve=True))
    with open_dict(config.experiment):
        config.experiment.total_timesteps = 64
        config.experiment.num_envs = 4
        config.experiment.num_steps = 8
        config.experiment.num_minibatches = 32
        config.experiment.validation.num = 1
        config.experiment.anneal_lr = False
        config.experiment.desired_kl = 0.01

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)
    agent_conf = S2PGPPOJax.init_agent_conf(env, config)
    rng = jax.random.PRNGKey(0)
    agent_state = S2PGPPOJax.init_agent_state(env, agent_conf, rng)
    train_fn = jax.jit(S2PGPPOJax.build_train_fn(env, agent_conf))

    result = train_fn(rng, agent_state, traj)
    assert jnp.all(jnp.isfinite(result["training_metrics"].learning_rate))


def test_BPTT_PPO_adaptive_kl_lr(bptt_ppo_config):
    """With anneal_lr off and desired_kl set, BPTT-PPO uses an inject_hyperparams
    optimizer and exercises the adaptive-KL learning-rate branch."""
    config = OmegaConf.create(OmegaConf.to_container(bptt_ppo_config, resolve=True))
    with open_dict(config.experiment):
        config.experiment.total_timesteps = 64
        config.experiment.num_envs = 4
        config.experiment.num_steps = 8
        config.experiment.num_minibatches = 4
        config.experiment.validation.num = 1
        config.experiment.anneal_lr = False
        config.experiment.desired_kl = 0.01

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)
    agent_conf = BPTTPPOJax.init_agent_conf(env, config)
    rng = jax.random.PRNGKey(0)
    agent_state = BPTTPPOJax.init_agent_state(env, agent_conf, rng)
    train_fn = jax.jit(BPTTPPOJax.build_train_fn(env, agent_conf))

    result = train_fn(rng, agent_state, traj)
    assert jnp.all(jnp.isfinite(result["training_metrics"].learning_rate))


def test_History_PPO_adaptive_kl_lr(history_ppo_config):
    """With anneal_lr off and desired_kl set, History-PPO uses an inject_hyperparams
    optimizer and exercises the adaptive-KL learning-rate branch."""
    config = OmegaConf.create(OmegaConf.to_container(history_ppo_config, resolve=True))
    with open_dict(config.experiment):
        config.experiment.total_timesteps = 64
        config.experiment.num_envs = 4
        config.experiment.num_steps = 8
        config.experiment.num_minibatches = 32
        config.experiment.validation.num = 1
        config.experiment.anneal_lr = False
        config.experiment.desired_kl = 0.01

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)
    agent_conf = HistoryPPOJax.init_agent_conf(env, config)
    rng = jax.random.PRNGKey(0)
    agent_state = HistoryPPOJax.init_agent_state(env, agent_conf, rng)
    train_fn = jax.jit(HistoryPPOJax.build_train_fn(env, agent_conf))

    result = train_fn(rng, agent_state, traj)
    assert jnp.all(jnp.isfinite(result["training_metrics"].learning_rate))


def test_SAC_build_train_fn(sac_config):

    config = sac_config

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

    # Use make_jaxpr to check if the function compiles correctly
    try:
        jaxpr = make_jaxpr(train_fn)(rng, agent_state, traj)
        assert jaxpr is not None
    except Exception as e:
        pytest.fail(f"JAX function compilation failed: {e}")


def test_SAC_save_and_load_agent(sac_config, tmp_path):
    """Train SAC for a few steps, save agent, load it, and verify params match."""
    config = OmegaConf.create(OmegaConf.to_container(sac_config, resolve=True))

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)
    agent_conf = SACJax.init_agent_conf(env, config)
    rng = jax.random.PRNGKey(0)
    agent_state = SACJax.init_agent_state(env, agent_conf, rng)
    train_fn = jax.jit(SACJax.build_train_fn(env, agent_conf))

    result = train_fn(rng, agent_state, traj)
    agent_state = result["agent_state"]

    save_path = SACJax.save_agent(tmp_path, agent_conf, agent_state)
    assert save_path.exists()

    loaded_conf, loaded_state = SACJax.load_agent(save_path)
    assert loaded_conf is not None
    assert loaded_state is not None

    assert _params_allclose(agent_state.actor_state.params, loaded_state.actor_state.params)
    assert _params_allclose(agent_state.actor_state.run_stats, loaded_state.actor_state.run_stats)
    assert _params_allclose(agent_state.critic_state.params, loaded_state.critic_state.params)
    assert _params_allclose(agent_state.critic_state.run_stats, loaded_state.critic_state.run_stats)


def test_TD3_build_train_fn(td3_config):
    config = td3_config

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

    agent_conf = TD3Jax.init_agent_conf(env, config)
    rng = jax.random.PRNGKey(0)
    agent_state = TD3Jax.init_agent_state(env, agent_conf, rng)

    train_fn = TD3Jax.build_train_fn(env, agent_conf)
    train_fn = jax.jit(train_fn)

    try:
        jaxpr = make_jaxpr(train_fn)(rng, agent_state, traj)
        assert jaxpr is not None
    except Exception as e:
        pytest.fail(f"JAX function compilation failed: {e}")


def test_TD3_save_and_load_agent(td3_config, tmp_path):
    """Train TD3 for a few steps, save agent, load it, and verify params match."""
    config = OmegaConf.create(OmegaConf.to_container(td3_config, resolve=True))

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)
    agent_conf = TD3Jax.init_agent_conf(env, config)
    rng = jax.random.PRNGKey(0)
    agent_state = TD3Jax.init_agent_state(env, agent_conf, rng)
    train_fn = jax.jit(TD3Jax.build_train_fn(env, agent_conf))

    result = train_fn(rng, agent_state, traj)
    agent_state = result["agent_state"]

    save_path = TD3Jax.save_agent(tmp_path, agent_conf, agent_state)
    assert save_path.exists()

    loaded_conf, loaded_state = TD3Jax.load_agent(save_path)
    assert loaded_conf is not None
    assert loaded_state is not None

    assert _params_allclose(agent_state.actor_state.params, loaded_state.actor_state.params)
    assert _params_allclose(agent_state.actor_state.run_stats, loaded_state.actor_state.run_stats)
    assert _params_allclose(agent_state.critic_state.params, loaded_state.critic_state.params)
    assert _params_allclose(agent_state.critic_state.run_stats, loaded_state.critic_state.run_stats)


def test_TD3_pessimism_penalty(td3_config):
    """With `pessimism_penalty` set, TD3 aggregates the twin Q-targets with
    Motivo-style ensemble pessimism (mean - k*|Q1-Q2|) instead of min(Q1,Q2).
    Run a short training chunk to exercise that branch."""
    config = OmegaConf.create(OmegaConf.to_container(td3_config, resolve=True))
    with open_dict(config.experiment):
        config.experiment.pessimism_penalty = 1.0

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)
    agent_conf = TD3Jax.init_agent_conf(env, config)
    rng = jax.random.PRNGKey(0)
    agent_state = TD3Jax.init_agent_state(env, agent_conf, rng)
    train_fn = jax.jit(TD3Jax.build_train_fn(env, agent_conf))

    result = train_fn(rng, agent_state, traj)
    # training ran and produced finite actor/critic params
    leaves = jax.tree_util.tree_leaves(result["agent_state"].critic_state.params)
    assert all(jnp.all(jnp.isfinite(x)) for x in leaves)


def test_TD3_categorical_critic(td3_config):
    """With `critic_loss='categorical'` (+ num_atoms>1) TD3 uses a C51-style
    distributional critic: the target twin is picked per-sample by scalar Q, the
    next-state distribution is projected onto the shifted support, and the loss is
    a cross-entropy over both twins' sown log-probs. Run a short chunk to drive
    the categorical target build (607-626) and categorical critic loss (648-675)."""
    config = OmegaConf.create(OmegaConf.to_container(td3_config, resolve=True))
    with open_dict(config.experiment):
        config.experiment.critic_loss = "categorical"
        config.experiment.num_atoms = 51
        config.experiment.min_v = -10.0
        config.experiment.max_v = 10.0

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)
    agent_conf = TD3Jax.init_agent_conf(env, config)
    rng = jax.random.PRNGKey(0)
    agent_state = TD3Jax.init_agent_state(env, agent_conf, rng)
    train_fn = jax.jit(TD3Jax.build_train_fn(env, agent_conf))

    result = train_fn(rng, agent_state, traj)
    leaves = jax.tree_util.tree_leaves(result["agent_state"].critic_state.params)
    assert all(jnp.all(jnp.isfinite(x)) for x in leaves)


def test_TD3_crossq_batch_norm(td3_config):
    """With `use_batch_norm=True` TD3 uses the XQC/CrossQ joint forward: the online
    critic sees concat([obs, nobs]) / concat([act, next_act]) in one pass so
    BatchNorm normalises both batches together, then splits. Drives the BN critic
    net build + the CrossQ branch of the MSE critic loss (676-688)."""
    config = OmegaConf.create(OmegaConf.to_container(td3_config, resolve=True))
    with open_dict(config.experiment):
        config.experiment.use_batch_norm = True

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)
    agent_conf = TD3Jax.init_agent_conf(env, config)
    rng = jax.random.PRNGKey(0)
    agent_state = TD3Jax.init_agent_state(env, agent_conf, rng)
    train_fn = jax.jit(TD3Jax.build_train_fn(env, agent_conf))

    result = train_fn(rng, agent_state, traj)
    # BN keeps a batch_stats collection in the critic bundle, and params stay finite
    leaves = jax.tree_util.tree_leaves(result["agent_state"].critic_state.params)
    assert all(jnp.all(jnp.isfinite(x)) for x in leaves)
    assert "batch_stats" in result["agent_state"].critic_state.run_stats


def test_VanillaDagger_build_train_fn(vanilla_dagger_config):
    config = vanilla_dagger_config

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

    agent_conf = VanillaDaggerJax.init_agent_conf(env, config)
    rng = jax.random.PRNGKey(0)
    agent_state = VanillaDaggerJax.init_agent_state(env, agent_conf, rng)

    train_fn = VanillaDaggerJax.build_train_fn(env, agent_conf)
    train_fn = jax.jit(train_fn)

    try:
        jaxpr = make_jaxpr(train_fn)(rng, agent_state, traj)
        assert jaxpr is not None
    except Exception as e:
        pytest.fail(f"JAX function compilation failed: {e}")


def test_VanillaDagger_save_and_load_agent(vanilla_dagger_config, tmp_path):
    """Train DAgger briefly, save, reload, and verify student+teacher params match.
    The replay buffer is intentionally *not* serialized."""
    config = OmegaConf.create(OmegaConf.to_container(vanilla_dagger_config, resolve=True))

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)
    agent_conf = VanillaDaggerJax.init_agent_conf(env, config)
    rng = jax.random.PRNGKey(0)
    agent_state = VanillaDaggerJax.init_agent_state(env, agent_conf, rng)
    train_fn = jax.jit(VanillaDaggerJax.build_train_fn(env, agent_conf))

    result = train_fn(rng, agent_state, traj)
    agent_state = result["agent_state"]

    save_path = VanillaDaggerJax.save_agent(tmp_path, agent_conf, agent_state)
    assert save_path.exists()

    loaded_conf, loaded_state = VanillaDaggerJax.load_agent(save_path)
    assert loaded_conf is not None
    assert loaded_state is not None

    assert _params_allclose(agent_state.student_train_state.params,
                            loaded_state.student_train_state.params)
    assert _params_allclose(agent_state.student_train_state.run_stats,
                            loaded_state.student_train_state.run_stats)
    assert _params_allclose(agent_state.teacher_params, loaded_state.teacher_params)
    assert _params_allclose(agent_state.teacher_run_stats, loaded_state.teacher_run_stats)
    # Replay buffer contents are not persisted — loaded buffer is a fresh,
    # empty buffer at the configured capacity (keeps pytree shape stable
    # so the first train_fn call doesn't recompile).
    assert loaded_state.replay_buffer is not None
    assert int(loaded_state.replay_buffer.size) == 0
    assert loaded_state.rollout_state is not None


def test_VanillaDagger_bc_only_minimal_buffer(vanilla_dagger_config):
    """With critic learning disabled, both `next_obs` and `value_target`
    should be zero-sized in the short-term and long-term buffers."""
    config = OmegaConf.create(OmegaConf.to_container(vanilla_dagger_config, resolve=True))
    with open_dict(config.experiment):
        config.experiment.use_critic_learning = False

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

    agent_conf = VanillaDaggerJax.init_agent_conf(env, config)
    rng = jax.random.PRNGKey(0)
    agent_state = VanillaDaggerJax.init_agent_state(env, agent_conf, rng)

    buf = agent_state.replay_buffer
    assert buf.store_next_obs is False
    assert buf.next_obs.shape == (0, 0)
    assert buf.store_value_target is False
    assert buf.value_target.shape == (0,)

    long_buf = agent_state.long_term_buffer
    assert long_buf.store_value_target is False
    assert long_buf.value_target.shape == (0,)

    train_fn = jax.jit(VanillaDaggerJax.build_train_fn(env, agent_conf))
    out = train_fn(rng, agent_state, traj)
    new_buf = out["agent_state"].replay_buffer
    assert int(new_buf.size) > 0
    assert new_buf.next_obs.shape == (0, 0)
    assert new_buf.value_target.shape == (0,)
    assert float(jnp.max(out["training_metrics"].mean_critic_loss)) == 0.0


def test_VanillaDagger_critic_distill_allocates_value_target(vanilla_dagger_config):
    """With critic learning enabled (default), the short-term and long-term
    buffers both allocate a value_target column and drop next_obs."""
    config = vanilla_dagger_config

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

    agent_conf = VanillaDaggerJax.init_agent_conf(env, config)
    rng = jax.random.PRNGKey(0)
    agent_state = VanillaDaggerJax.init_agent_state(env, agent_conf, rng)

    buf = agent_state.replay_buffer
    assert buf.store_value_target is True
    assert buf.value_target.shape == (int(config.experiment.buffer_size),)
    assert buf.store_next_obs is False
    assert buf.next_obs.shape == (0, 0)

    long_buf = agent_state.long_term_buffer
    assert long_buf.store_value_target is True
    assert long_buf.value_target.shape == (int(config.experiment.long_term_buffer_size),)

    train_fn = jax.jit(VanillaDaggerJax.build_train_fn(env, agent_conf))
    out = train_fn(rng, agent_state, traj)
    new_buf = out["agent_state"].replay_buffer
    assert int(new_buf.size) > 0
    assert jnp.any(new_buf.value_target != 0.0)


def test_VanillaDagger_long_term_reservoir_fills(vanilla_dagger_config):
    """The long-term reservoir should accumulate transitions across training
    and its `total_seen` counter should match the number of env steps."""
    config = vanilla_dagger_config

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

    agent_conf = VanillaDaggerJax.init_agent_conf(env, config)
    rng = jax.random.PRNGKey(0)
    agent_state = VanillaDaggerJax.init_agent_state(env, agent_conf, rng)

    # Baseline: empty reservoir
    assert int(agent_state.long_term_buffer.total_seen) == 0
    assert int(agent_state.long_term_buffer.size) == 0

    train_fn = jax.jit(VanillaDaggerJax.build_train_fn(env, agent_conf))
    out = train_fn(rng, agent_state, traj)

    exp = config.experiment
    expected_transitions = int(exp.num_updates) * int(exp.num_envs)
    long_buf = out["agent_state"].long_term_buffer
    assert int(long_buf.total_seen) == expected_transitions
    # Size caps at capacity
    assert int(long_buf.size) == min(expected_transitions, int(exp.long_term_buffer_size))
    assert jnp.any(long_buf.obs != 0.0)


def test_VanillaDagger_long_term_min_include_prob(vanilla_dagger_config):
    """A positive min_include_prob floors the reservoir's inclusion rate.
    Verify it's propagated as static metadata and training runs cleanly."""
    config = OmegaConf.create(OmegaConf.to_container(vanilla_dagger_config, resolve=True))
    with open_dict(config.experiment):
        config.experiment.long_term_min_include_prob = 0.5

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

    agent_conf = VanillaDaggerJax.init_agent_conf(env, config)
    rng = jax.random.PRNGKey(0)
    agent_state = VanillaDaggerJax.init_agent_state(env, agent_conf, rng)
    # Flag survives as static metadata on the reservoir
    assert agent_state.long_term_buffer.min_include_prob == 0.5

    train_fn = jax.jit(VanillaDaggerJax.build_train_fn(env, agent_conf))
    out = train_fn(rng, agent_state, traj)
    long_buf = out["agent_state"].long_term_buffer
    # Same flag on the output (static didn't get dropped)
    assert long_buf.min_include_prob == 0.5
    # Training collected transitions into the reservoir (fill phase OK).
    assert int(long_buf.total_seen) > 0
    assert jnp.any(long_buf.obs != 0.0)


def test_VanillaDagger_long_term_disabled(vanilla_dagger_config):
    """`long_term_buffer_size: 0` disables the reservoir — no long sampling
    branch at trace, long buffer stays zero-sized."""
    config = OmegaConf.create(OmegaConf.to_container(vanilla_dagger_config, resolve=True))
    with open_dict(config.experiment):
        config.experiment.long_term_buffer_size = 0

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

    agent_conf = VanillaDaggerJax.init_agent_conf(env, config)
    rng = jax.random.PRNGKey(0)
    agent_state = VanillaDaggerJax.init_agent_state(env, agent_conf, rng)
    assert agent_state.long_term_buffer.obs.shape == (0, int(agent_conf.config.experiment.obs_dim))

    train_fn = jax.jit(VanillaDaggerJax.build_train_fn(env, agent_conf))
    out = train_fn(rng, agent_state, traj)
    assert int(out["agent_state"].replay_buffer.size) > 0
    assert int(out["agent_state"].long_term_buffer.size) == 0


def test_VanillaDagger_eval_fn(vanilla_dagger_config):
    """Standalone eval_fn should jit, run, and produce finite metrics without
    touching the agent's training state."""
    config = vanilla_dagger_config

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

    agent_conf = VanillaDaggerJax.init_agent_conf(env, config)
    rng = jax.random.PRNGKey(0)
    agent_state = VanillaDaggerJax.init_agent_state(env, agent_conf, rng)

    eval_fn = VanillaDaggerJax.build_eval_fn(env, agent_conf)
    eval_fn = jax.jit(eval_fn)

    try:
        jaxpr = make_jaxpr(eval_fn)(rng, agent_state, traj)
        assert jaxpr is not None
    except Exception as e:
        pytest.fail(f"JAX eval function compilation failed: {e}")

    out = eval_fn(rng, agent_state, traj)
    assert "eval_summary" in out
    assert "validation_metrics" in out
    # finite-ness — training may not have happened yet, but the metric
    # should at least evaluate without nan/inf.
    assert jnp.isfinite(out["eval_summary"].mean_episode_return)
    assert jnp.isfinite(out["eval_summary"].mean_episode_length)


def test_VanillaDagger_buffer_survives_chunk_swap(vanilla_dagger_config):
    """The whole point of VanillaDagger: the replay buffer must survive a
    teacher swap + env reset between training chunks."""
    config = OmegaConf.create(OmegaConf.to_container(vanilla_dagger_config, resolve=True))

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)
    agent_conf = VanillaDaggerJax.init_agent_conf(env, config)
    rng = jax.random.PRNGKey(0)
    agent_state = VanillaDaggerJax.init_agent_state(env, agent_conf, rng)
    train_fn = jax.jit(VanillaDaggerJax.build_train_fn(env, agent_conf))

    # Chunk 1: builds up the buffer.
    out1 = train_fn(rng, agent_state, traj)
    state1 = out1["agent_state"]
    size_after_chunk1 = int(state1.replay_buffer.size)
    assert size_after_chunk1 > 0, "buffer should have collected transitions"

    # Simulate a teacher + env swap: null env_state/last_obs (forces fresh reset
    # inside _train_fn) and replace teacher params with themselves (stand-in
    # for loading a different pretrained teacher). Buffer stays.
    swapped = state1.replace(env_state=None, last_obs=None,
                              teacher_params=state1.teacher_params,
                              teacher_run_stats=state1.teacher_run_stats)
    assert swapped.replay_buffer is not None
    assert int(swapped.replay_buffer.size) == size_after_chunk1

    # Chunk 2: buffer continues to grow from the preserved state.
    out2 = train_fn(rng, swapped, traj)
    state2 = out2["agent_state"]
    assert int(state2.replay_buffer.size) >= size_after_chunk1, \
        "buffer size must not drop across a traj/teacher swap"
