"""Vanilla DAgger with swappable student / teacher / trajectory.

Rollout uses a sticky per-env mixture: each env independently picks either
the student or the teacher with probability `teacher_beta` and commits to
that choice for a uniformly-random number of steps in
`[sticky_min, sticky_max]`. Whoever acts, the transition is *labeled with
the teacher's action on that observation* — that's the DAgger trick.

Training is supervised: minibatch from the replay buffer and minimise the
NLL of the student Gaussian at the stored teacher action.

Swap semantics (all parts of the JAX pytree):
  - traj:           passed as arg to `train_fn(rng, agent_state, traj)`.
  - teacher:        `agent_state.replace(teacher_params=..., teacher_run_stats=...)`.
  - student:        `agent_state.replace(student_train_state=...)`.
  - replay buffer:  preserved by default across chunks (this is the whole
                    point — data collected under one teacher stays useful
                    after a teacher/traj swap). Reset with `.replace(
                    replay_buffer=None)` if you want a clean slate.
"""
import ast
from dataclasses import dataclass
from typing import Any
import warnings

import jax
import jax.numpy as jnp
from flax import struct
import flax
import optax
from omegaconf import DictConfig, OmegaConf, ListConfig

from loco_mujoco.algorithms import (JaxRLAlgorithmBase, AgentConfBase, AgentStateBase,
                                    ActorCritic, TrainState, ReplayBuffer,
                                    MetricHandlerTransition)
from loco_mujoco.core.wrappers import LogWrapper, LogEnvState, VecEnv, SummaryMetrics
from loco_mujoco.utils import MetricsHandler, ValidationSummary


# ---------------------------------------------------------------------------
# Metrics + rollout state
# ---------------------------------------------------------------------------

@struct.dataclass
class DaggerSummaryMetrics(SummaryMetrics):
    mean_bc_loss: float = 0.0       # actor NLL (what gets minimised)
    mean_mse_action: float = 0.0    # diagnostic: ||pi.mean - a_teacher||^2
    mean_critic_loss: float = 0.0   # TD(0) MSE, 0 when critic learning is off
    buffer_size: int = 0
    frac_teacher_steps: float = 0.0


@struct.dataclass
class DaggerRolloutState:
    """Per-env sticky mixture state.

    using_teacher[i]    : current env i is rolling out with the teacher.
    steps_remaining[i]  : steps left before resampling the mixture for env i.
    """
    using_teacher: jnp.ndarray
    steps_remaining: jnp.ndarray

    @classmethod
    def create(cls, num_envs: int, rng, beta: float, h_min: int, h_max: int):
        k1, k2 = jax.random.split(rng)
        using_teacher = jax.random.uniform(k1, (num_envs,)) < beta
        steps_remaining = jax.random.randint(k2, (num_envs,), h_min, h_max + 1)
        return cls(using_teacher=using_teacher, steps_remaining=steps_remaining)


# ---------------------------------------------------------------------------
# Agent conf / state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VanillaDaggerAgentConf(AgentConfBase):
    config: DictConfig
    student_net: ActorCritic
    teacher_net: ActorCritic
    student_tx: Any

    def serialize(self):
        conf_dict = OmegaConf.to_container(self.config, resolve=True, throw_on_missing=True)
        return {
            "config": conf_dict,
            "student_net": flax.serialization.to_state_dict(self.student_net),
            "teacher_net": flax.serialization.to_state_dict(self.teacher_net),
        }

    @classmethod
    def from_dict(cls, d):
        config = OmegaConf.create(d["config"])
        student_tx = VanillaDaggerJax._get_optimizer(config)
        student_net = flax.serialization.from_state_dict(
            VanillaDaggerJax._build_net(config, role="student"), d["student_net"]
        )
        teacher_net = flax.serialization.from_state_dict(
            VanillaDaggerJax._build_net(config, role="teacher"), d["teacher_net"]
        )
        return cls(config=config, student_net=student_net,
                   teacher_net=teacher_net, student_tx=student_tx)


@struct.dataclass
class VanillaDaggerAgentState(AgentStateBase):
    """Dynamic agent state. All three swappable entities live here:
    `student_train_state`, `teacher_params`/`teacher_run_stats`, and the
    trajectory (passed separately to `_train_fn`)."""
    student_train_state: TrainState
    teacher_params: Any
    teacher_run_stats: Any
    replay_buffer: Any = None      # ReplayBuffer; lazily initialised in _train_fn
    rollout_state: Any = None      # DaggerRolloutState; lazily initialised
    env_state: Any = None
    last_obs: Any = None

    def serialize(self):
        return {
            "student_train_state": flax.serialization.to_state_dict(self.student_train_state),
            "teacher_params": flax.serialization.to_state_dict(self.teacher_params),
            "teacher_run_stats": flax.serialization.to_state_dict(self.teacher_run_stats),
        }

    @classmethod
    def from_dict(cls, d, agent_conf):
        student_net = agent_conf.student_net
        train_state = TrainState.create(
            apply_fn=student_net.apply,
            tx=agent_conf.student_tx,
            params=d["student_train_state"]["params"],
            run_stats=d["student_train_state"]["run_stats"],
        )
        opt_state = flax.serialization.from_state_dict(
            train_state.opt_state, d["student_train_state"]["opt_state"]
        )
        train_state = train_state.replace(opt_state=opt_state)
        # Rebuild buffer + rollout_state at the configured shape so the
        # loaded agent has the same pytree structure as one produced by
        # init_agent_state — avoids a recompile on the first train_fn call.
        exp = agent_conf.config.experiment
        replay_buffer = ReplayBuffer.create(
            int(exp.obs_dim), int(exp.action_dim), int(exp.buffer_size)
        )
        rollout_state = DaggerRolloutState.create(
            int(exp.num_envs), jax.random.PRNGKey(0),
            beta=float(getattr(exp, "teacher_beta", 0.5)),
            h_min=int(getattr(exp, "sticky_min", 5)),
            h_max=int(getattr(exp, "sticky_max", 50)),
        )
        return cls(
            student_train_state=train_state,
            teacher_params=d["teacher_params"],
            teacher_run_stats=d["teacher_run_stats"],
            replay_buffer=replay_buffer,
            rollout_state=rollout_state,
        )


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------

class VanillaDaggerJax(JaxRLAlgorithmBase):
    """DAgger with a frozen teacher and a trainable student, both ActorCritic.

    See the module docstring for swap semantics. The training loop interleaves
    one env step (per vmapped env) with `gradient_steps` BC updates, once the
    buffer contains at least `learning_starts` transitions.
    """

    _agent_conf = VanillaDaggerAgentConf
    _agent_state = VanillaDaggerAgentState

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    @classmethod
    def _build_net(cls, config: DictConfig, role: str) -> ActorCritic:
        """Build an ActorCritic for `role` ∈ {student, teacher}.

        Reads `config.experiment.<role>.*` if present, else falls back to
        `config.experiment.*` — so a minimal config with only shared fields
        works out of the box.

        `actor_obs_ind` / `critic_obs_ind` (lists of ints) are set by
        `init_agent_conf` after resolving `actor_obs_group` / `critic_obs_group`
        against `env.obs_container`. They survive save/load because they
        live on the config, so `from_dict` doesn't need env.
        """
        exp = config.experiment
        sub = getattr(exp, role, None)

        def field(name, default=None):
            if sub is not None and name in sub:
                return sub[name]
            return getattr(exp, name, default)

        hidden_layers = field("hidden_layers", [512, 256])
        if not isinstance(hidden_layers, (list, ListConfig)):
            hidden_layers = ast.literal_eval(hidden_layers)

        kwargs = dict(
            action_dim=int(exp.action_dim),
            activation=str(field("activation", "tanh")),
            init_std=float(field("init_std", 1.0)),
            learnable_std=bool(field("learnable_std", True)),
            hidden_layer_dims=tuple(hidden_layers),
        )
        actor_ind = field("actor_obs_ind", None)
        critic_ind = field("critic_obs_ind", None)
        if actor_ind is not None:
            kwargs["actor_obs_ind"] = jnp.asarray(list(actor_ind), dtype=jnp.int32)
        if critic_ind is not None:
            kwargs["critic_obs_ind"] = jnp.asarray(list(critic_ind), dtype=jnp.int32)
        return ActorCritic(**kwargs)

    @classmethod
    def _get_optimizer(cls, config: DictConfig):
        exp = config.experiment
        max_grad = float(getattr(exp, "max_grad_norm", 0.5))
        lr = float(exp.lr)
        weight_decay = float(getattr(exp, "weight_decay", 0.0))
        return optax.chain(
            optax.clip_by_global_norm(max_grad),
            optax.adamw(lr, weight_decay=weight_decay, eps=1e-5),
        )

    # ------------------------------------------------------------------
    # Conf / state init
    # ------------------------------------------------------------------

    @classmethod
    def init_agent_conf(cls, env, config: DictConfig):
        from omegaconf import open_dict, OmegaConf
        obs_dim = int(env.info.observation_space.shape[0])

        def _resolve_group(group):
            """Group name → list[int] observation indices, or None if no filter."""
            if group is None:
                return None
            return [int(i) for i in env.obs_container.get_obs_ind_by_group(group)]

        with open_dict(config.experiment):
            config.experiment.num_updates = int(config.experiment.total_timesteps) // int(config.experiment.num_envs)
            config.experiment.obs_dim = obs_dim
            config.experiment.action_dim = int(env.info.action_space.shape[0])

            # Optional top-level filters (fallback if a role doesn't specify its own).
            top_actor_group = config.experiment.get("actor_obs_group", None)
            top_critic_group = config.experiment.get("critic_obs_group", None)

            # Resolve per-role obs filters against the env and bake the
            # resulting index lists into the config. This keeps from_dict
            # env-independent — the loaded conf rebuilds the same network.
            for role in ("student", "teacher"):
                if config.experiment.get(role, None) is None:
                    config.experiment[role] = OmegaConf.create({})
                sub = config.experiment[role]
                actor_group = sub.get("actor_obs_group", top_actor_group)
                critic_group = sub.get("critic_obs_group", top_critic_group)
                actor_ind = _resolve_group(actor_group)
                critic_ind = _resolve_group(critic_group)
                if actor_ind is not None:
                    sub.actor_obs_ind = actor_ind
                if critic_ind is not None:
                    sub.critic_obs_ind = critic_ind

        student_net = cls._build_net(config, role="student")
        teacher_net = cls._build_net(config, role="teacher")
        student_tx = cls._get_optimizer(config)
        return cls._agent_conf(config=config, student_net=student_net,
                               teacher_net=teacher_net, student_tx=student_tx)

    @classmethod
    def init_agent_state(cls, env, agent_conf: VanillaDaggerAgentConf, rng,
                         teacher_params=None, teacher_run_stats=None):
        """Initialise the agent state.

        `teacher_params`/`teacher_run_stats` should be passed from a
        pretrained checkpoint (e.g. a PPO `TrainState`). If omitted, the
        teacher is randomly initialised — useful only as a placeholder.
        """
        exp = agent_conf.config.experiment
        rng, rng_s, rng_t = jax.random.split(rng, 3)
        init_x = jnp.zeros((1, int(exp.obs_dim)))

        student_init = agent_conf.student_net.init(rng_s, init_x)
        student_train_state = TrainState.create(
            apply_fn=agent_conf.student_net.apply,
            params=student_init["params"],
            run_stats=student_init["run_stats"],
            tx=agent_conf.student_tx,
        )

        if teacher_params is None or teacher_run_stats is None:
            warnings.warn(
                "VanillaDagger: teacher params/run_stats not provided — using "
                "random init. Call `.replace(teacher_params=..., teacher_run_stats=...)` "
                "with a pretrained checkpoint before training."
            )
            teacher_init = agent_conf.teacher_net.init(rng_t, init_x)
            teacher_params = teacher_init["params"]
            teacher_run_stats = teacher_init["run_stats"]

        # Eagerly build the replay buffer and rollout-mixture state here rather
        # than lazily inside _train_fn. Keeps the agent_state pytree structure
        # stable across chunks so the jitted train_fn doesn't recompile when
        # the user passes a state with a populated buffer on chunk 2+.
        rng, rng_r = jax.random.split(rng)
        replay_buffer = ReplayBuffer.create(
            int(exp.obs_dim), int(exp.action_dim), int(exp.buffer_size)
        )
        rollout_state = DaggerRolloutState.create(
            int(exp.num_envs), rng_r,
            beta=float(getattr(exp, "teacher_beta", 0.5)),
            h_min=int(getattr(exp, "sticky_min", 5)),
            h_max=int(getattr(exp, "sticky_max", 50)),
        )

        return cls._agent_state(
            student_train_state=student_train_state,
            teacher_params=teacher_params,
            teacher_run_stats=teacher_run_stats,
            replay_buffer=replay_buffer,
            rollout_state=rollout_state,
        )

    # ------------------------------------------------------------------
    # Env wrap
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_env(env, config):
        env = LogWrapper(env)
        env = VecEnv(env)
        # DAgger doesn't use reward, so no reward normalization.
        return env

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @classmethod
    def _train_fn(cls, rng, env,
                  agent_conf: VanillaDaggerAgentConf,
                  agent_state: VanillaDaggerAgentState = None,
                  traj=None,
                  mh: MetricsHandler = None):
        if agent_state is None:
            raise ValueError(
                "VanillaDagger._train_fn requires an agent_state with teacher "
                "params. Build it via init_agent_state(..., teacher_params=..., "
                "teacher_run_stats=...)."
            )

        exp = agent_conf.config.experiment
        student_net = agent_conf.student_net
        teacher_net = agent_conf.teacher_net
        env = cls._wrap_env(env, exp)

        # Reattach apply_fn (not serializable through flax save)
        student_ts = agent_state.student_train_state.replace(apply_fn=student_net.apply)
        teacher_params = agent_state.teacher_params
        teacher_run_stats = agent_state.teacher_run_stats

        # Hyperparameters
        num_envs = int(exp.num_envs)
        batch_size = int(exp.batch_size)
        learning_starts = int(getattr(exp, "learning_starts", batch_size))
        gradient_steps = int(getattr(exp, "gradient_steps", 1))
        teacher_beta = float(getattr(exp, "teacher_beta", 0.5))
        sticky_min = int(getattr(exp, "sticky_min", 5))
        sticky_max = int(getattr(exp, "sticky_max", 50))
        # Critic learning (for downstream RL warm-start). TD(0) self-bootstrap
        # with stop_gradient on the target — no target network.
        use_critic_learning = bool(getattr(exp, "use_critic_learning", True))
        critic_loss_coef = float(getattr(exp, "critic_loss_coef", 0.5))
        gamma = float(getattr(exp, "gamma", 0.99))

        # ENV (reuse carry if present)
        if agent_state.env_state is not None:
            env_state = agent_state.env_state
            last_obs = agent_state.last_obs
        else:
            rng, _rng = jax.random.split(rng)
            last_obs, env_state = env.reset(jax.random.split(_rng, num_envs), traj)

        # REPLAY BUFFER (preserve across chunks by default — this is the whole
        # point: data gathered under old teacher/traj stays useful after swap).
        if agent_state.replay_buffer is not None:
            buf = agent_state.replay_buffer
        else:
            buf = ReplayBuffer.create(int(exp.obs_dim), int(exp.action_dim),
                                      int(exp.buffer_size))

        # ROLLOUT MIXTURE STATE
        if agent_state.rollout_state is not None:
            rollout_st = agent_state.rollout_state
        else:
            rng, _rng = jax.random.split(rng)
            rollout_st = DaggerRolloutState.create(
                num_envs, _rng, teacher_beta, sticky_min, sticky_max
            )

        # --------------------------------------------------------------
        # Helpers
        # --------------------------------------------------------------

        def _teacher_mean(obs):
            # Frozen teacher. RunningMeanStd always writes, so we pass
            # mutable=["run_stats"] to keep Flax happy but drop the updates —
            # teacher_run_stats stays put. (The ephemeral per-batch Welford
            # update is negligible against the teacher's converged stats.)
            (pi, _), _ = teacher_net.apply(
                {"params": teacher_params, "run_stats": teacher_run_stats}, obs,
                mutable=["run_stats"],
            )
            return pi.mean()

        def _student_sample(s_ts, obs, rng_s):
            (pi, _), updates = student_net.apply(
                {"params": s_ts.params, "run_stats": s_ts.run_stats}, obs,
                mutable=["run_stats"],
            )
            s_ts = s_ts.replace(run_stats=updates["run_stats"])
            return pi.sample(seed=rng_s), s_ts

        def _resample_expired(rollout_st, rng_r):
            expired = rollout_st.steps_remaining <= 0
            k1, k2 = jax.random.split(rng_r)
            new_teacher = jax.random.uniform(k1, rollout_st.using_teacher.shape) < teacher_beta
            new_remaining = jax.random.randint(
                k2, rollout_st.steps_remaining.shape, sticky_min, sticky_max + 1
            )
            return DaggerRolloutState(
                using_teacher=jnp.where(expired, new_teacher, rollout_st.using_teacher),
                steps_remaining=jnp.where(expired, new_remaining, rollout_st.steps_remaining),
            )

        def _collect_transition(s_ts, buf, rollout_st, env_state, last_obs, rng):
            # Compute student + teacher actions in parallel
            rng, rng_s = jax.random.split(rng)
            a_student, s_ts = _student_sample(s_ts, last_obs, rng_s)
            a_teacher = _teacher_mean(last_obs)

            # Per-env sticky pick
            mask = rollout_st.using_teacher[:, None]
            acting = jnp.where(mask, a_teacher, a_student)

            # Step env with the acting policy
            next_obs, reward, absorbing, done, info, env_state = env.step(
                env_state, acting, traj
            )

            # Label with teacher action regardless of who acted
            buf = buf.add_batch(last_obs, next_obs, a_teacher, reward,
                                done.astype(jnp.float32))

            # Sticky countdown + resample expired slots
            rollout_st = rollout_st.replace(
                steps_remaining=rollout_st.steps_remaining - 1
            )
            rng, rng_r = jax.random.split(rng)
            rollout_st = _resample_expired(rollout_st, rng_r)

            return s_ts, buf, rollout_st, env_state, next_obs, rng

        # --------------------------------------------------------------
        # Update (actor BC + optional critic TD(0))
        # --------------------------------------------------------------

        def _update_loss(params, s_ts, obs_b, next_obs_b, act_b, reward_b, done_b):
            # Forward on obs (for actor NLL and V(s))
            (pi, v_s), u1 = student_net.apply(
                {"params": params, "run_stats": s_ts.run_stats}, obs_b,
                mutable=["run_stats"],
            )
            actor_loss = -jnp.mean(pi.log_prob(act_b))
            mse_action = jnp.mean((pi.mean() - act_b) ** 2)

            if use_critic_learning:
                # Chain run_stats through to keep RunningMeanStd updating on
                # both obs and next_obs (both are observations the student sees).
                (_, v_s_next), u2 = student_net.apply(
                    {"params": params, "run_stats": u1["run_stats"]}, next_obs_b,
                    mutable=["run_stats"],
                )
                # TD(0) target, stop_gradient so we only backprop through v_s
                target = reward_b + gamma * v_s_next * (1.0 - done_b)
                target = jax.lax.stop_gradient(target)
                critic_loss = jnp.mean((v_s - target) ** 2)
                total_loss = actor_loss + critic_loss_coef * critic_loss
                new_run_stats = u2["run_stats"]
            else:
                critic_loss = jnp.array(0.0)
                total_loss = actor_loss
                new_run_stats = u1["run_stats"]

            return total_loss, (new_run_stats, actor_loss, critic_loss, mse_action)

        def _single_update(carry, _):
            s_ts, buf, rng = carry
            rng, rng_sample = jax.random.split(rng)
            obs_b, next_obs_b, act_b, reward_b, done_b = buf.sample(rng_sample, batch_size)
            (_, aux), grads = jax.value_and_grad(_update_loss, has_aux=True)(
                s_ts.params, s_ts, obs_b, next_obs_b, act_b, reward_b, done_b
            )
            new_rs, actor_loss, critic_loss, mse_action = aux
            s_ts = s_ts.apply_gradients(grads=grads)
            s_ts = s_ts.replace(run_stats=new_rs)
            return (s_ts, buf, rng), (actor_loss, critic_loss, mse_action)

        def _do_updates(s_ts, buf, rng):
            (s_ts, _, _), aux = jax.lax.scan(
                _single_update, (s_ts, buf, rng), jnp.arange(gradient_steps)
            )
            return s_ts, jnp.mean(aux[0]), jnp.mean(aux[1]), jnp.mean(aux[2])

        def _skip_updates(s_ts, buf, rng):
            return s_ts, jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)

        # --------------------------------------------------------------
        # Outer step
        # --------------------------------------------------------------

        def _update_step(runner_state, unused):
            s_ts, buf, rollout_st, env_state, last_obs, rng = runner_state

            s_ts, buf, rollout_st, env_state, next_obs, rng = _collect_transition(
                s_ts, buf, rollout_st, env_state, last_obs, rng
            )

            rng, rng_upd = jax.random.split(rng)
            s_ts, bc_loss, critic_loss, mse_action = jax.lax.cond(
                buf.size >= learning_starts,
                lambda args: _do_updates(*args),
                lambda args: _skip_updates(*args),
                (s_ts, buf, rng_upd),
            )

            # Metrics
            logged = env_state.find(LogEnvState).metrics
            done_count = jnp.maximum(jnp.sum(logged.done), 1)
            metric = DaggerSummaryMetrics(
                mean_episode_return=jnp.sum(
                    jnp.where(logged.done, logged.returned_episode_returns, 0.0)
                ) / done_count,
                mean_episode_length=jnp.sum(
                    jnp.where(logged.done, logged.returned_episode_lengths, 0.0)
                ) / done_count,
                max_timestep=jnp.max(logged.timestep * num_envs),
                mean_bc_loss=bc_loss,
                mean_mse_action=mse_action,
                mean_critic_loss=critic_loss,
                buffer_size=buf.size,
                frac_teacher_steps=jnp.mean(rollout_st.using_teacher.astype(jnp.float32)),
            )

            runner_state = (s_ts, buf, rollout_st, env_state, next_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (student_ts, buf, rollout_st, env_state, last_obs, _rng)
        runner_state, training_metrics = jax.lax.scan(
            _update_step, runner_state, None, int(exp.num_updates)
        )
        student_ts, buf, rollout_st, env_state, last_obs, _ = runner_state

        agent_state_out = cls._agent_state(
            student_train_state=student_ts,
            teacher_params=teacher_params,
            teacher_run_stats=teacher_run_stats,
            replay_buffer=buf,
            rollout_state=rollout_st,
            env_state=env_state,
            last_obs=last_obs,
        )
        return {
            "agent_state": agent_state_out,
            "training_metrics": training_metrics,
            "validation_metrics": ValidationSummary(),
        }

    # ------------------------------------------------------------------
    # Evaluation (standalone, same signature as train_fn)
    # ------------------------------------------------------------------

    @classmethod
    def _eval_fn(cls, rng, env,
                 agent_conf: VanillaDaggerAgentConf,
                 agent_state: VanillaDaggerAgentState,
                 traj=None,
                 mh: MetricsHandler = None):
        """Deterministic student-only rollout for evaluation.

        Resets `config.eval_num_envs` (falls back to `num_envs`) fresh envs
        and runs the student policy with its mean action for
        `config.eval_num_steps` steps (default 500). Returns an episode
        summary plus the `mh` validation summary if provided. Does not
        touch the teacher, the replay buffer, or the training state.
        """
        exp = agent_conf.config.experiment
        student_net = agent_conf.student_net
        env = cls._wrap_env(env, exp)

        student_ts = agent_state.student_train_state.replace(apply_fn=student_net.apply)

        eval_num_envs = int(getattr(exp, "eval_num_envs", exp.num_envs))
        eval_num_steps = int(getattr(exp, "eval_num_steps", 500))

        def _eval_step(carry, unused):
            s_ts, env_state, last_obs, rng = carry
            (pi, _), updates = student_net.apply(
                {"params": s_ts.params, "run_stats": s_ts.run_stats}, last_obs,
                mutable=["run_stats"],
            )
            s_ts = s_ts.replace(run_stats=updates["run_stats"])
            action = pi.mean()  # deterministic

            obs, reward, absorbing, done, info, env_state = env.step(
                env_state, action, traj
            )
            logged = env_state.find(LogEnvState).metrics
            transition = MetricHandlerTransition(env_state, logged)
            return (s_ts, env_state, obs, rng), transition

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, eval_num_envs)
        last_obs, env_state = env.reset(reset_rng, traj)

        (_, _, _, _), traj_batch = jax.lax.scan(
            _eval_step, (student_ts, env_state, last_obs, rng),
            None, eval_num_steps,
        )

        logged = traj_batch.logged_metrics
        done_count = jnp.maximum(jnp.sum(logged.done), 1)
        summary = SummaryMetrics(
            mean_episode_return=jnp.sum(
                jnp.where(logged.done, logged.returned_episode_returns, 0.0)
            ) / done_count,
            mean_episode_length=jnp.sum(
                jnp.where(logged.done, logged.returned_episode_lengths, 0.0)
            ) / done_count,
            max_timestep=jnp.max(logged.timestep * eval_num_envs),
        )

        if mh is None:
            validation_metrics = ValidationSummary()
        else:
            validation_metrics = mh(traj_batch.env_state)

        return {"eval_summary": summary, "validation_metrics": validation_metrics}

    # ------------------------------------------------------------------
    # Play
    # ------------------------------------------------------------------

    @classmethod
    def play_policy(cls, env,
                    agent_conf: VanillaDaggerAgentConf,
                    agent_state: VanillaDaggerAgentState,
                    n_envs: int, n_steps=None, render=True,
                    record=False, rng=None, deterministic=False,
                    use_mujoco=False, wrap_env=True,
                    use_teacher=False, traj=None):
        """Roll out either the student (default) or the teacher.

        Set `use_teacher=True` to sanity-check the teacher; otherwise the
        student is played.
        """
        import numpy as np

        if use_mujoco:
            assert n_envs == 1, "Only one mujoco env can be run at a time."

        if wrap_env and not use_mujoco:
            env = cls._wrap_env(env, agent_conf.config.experiment)

        if rng is None:
            rng = jax.random.key(0)

        keys = jax.random.split(rng, n_envs + 1)
        rng, env_keys = keys[0], keys[1:]

        student_net = agent_conf.student_net
        teacher_net = agent_conf.teacher_net

        s_params = agent_state.student_train_state.params
        s_run_stats = agent_state.student_train_state.run_stats
        t_params = agent_state.teacher_params
        t_run_stats = agent_state.teacher_run_stats

        if deterministic and not use_teacher:
            # Collapse the Gaussian to its mean.
            s_params = dict(s_params)
            s_params["log_std"] = np.ones_like(s_params["log_std"]) * -np.inf

        @jax.jit
        def _act(obs, rng_a):
            if use_teacher:
                (pi, _), _ = teacher_net.apply(
                    {"params": t_params, "run_stats": t_run_stats}, obs,
                    mutable=["run_stats"],
                )
            else:
                (pi, _), _ = student_net.apply(
                    {"params": s_params, "run_stats": s_run_stats}, obs,
                    mutable=["run_stats"],
                )
            return pi.sample(seed=rng_a)

        # reset env
        if use_mujoco:
            obs = env.reset()
            env_state = None
        else:
            obs, env_state = env.reset(env_keys, traj)

        if n_steps is None:
            n_steps = np.iinfo(np.int32).max

        for _ in range(n_steps):
            rng, rng_a = jax.random.split(rng)
            action = _act(obs, rng_a)
            action = jnp.atleast_2d(action)

            if use_mujoco:
                obs, reward, absorbing, done, info = env.step(action)
                env.render(record=True)
                if done:
                    obs = env.reset()
            else:
                obs, reward, absorbing, done, info, env_state = env.step(
                    env_state, action, traj
                )
                env.mjx_render(env_state, record=record)

        env.stop()
