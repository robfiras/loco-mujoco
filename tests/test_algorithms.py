import pytest
from jax import make_jaxpr
from omegaconf import open_dict

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax, GAILJax, AMPJax
from loco_mujoco.algorithms.experimental import S2PGPPOJax, BPTTPPOJax, HistoryPPOJax
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
