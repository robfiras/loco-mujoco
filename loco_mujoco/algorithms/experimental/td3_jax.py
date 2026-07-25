"""TD3 (Twin Delayed Deep Deterministic policy gradient) on top of OffPolicyBase.

Differences from SAC (everything else is shared via OffPolicyBase):
  - Deterministic actor: actor(s) -> a in (-1, 1)  (tanh-squashed Dense head)
  - Exploration noise added at env-collect time:  a = clip( a + N(0, σ_act), -1, 1 )
  - Target policy smoothing: a' = clip( target_actor(s') + clip(N(0, σ_tgt), -c, c), -1, 1 )
  - No entropy bonus in Q-target (next_q_bonus = 0)
  - Actor loss = -E[ Q1(s, actor(s)) ]   (no min, just Q1)
  - Delayed actor + target updates: every `policy_delay` critic steps
"""

import ast
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import flax
import optax
from flax import struct
from omegaconf import DictConfig, OmegaConf, ListConfig, open_dict

from loco_mujoco.algorithms import (AgentConfBase, TrainState)
from loco_mujoco.algorithms.common.networks import (RunningMeanStd,
                                                    get_activation_fn)
from loco_mujoco.core.wrappers import SummaryMetrics
from loco_mujoco.algorithms.experimental.offpolicy_base import (
    OffPolicyBase, OffPolicyCriticNet, ReplayBuffer, OffPolicyAgentState,
    _normalize_dense_kernels, _c51_project_target,
)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class TD3ActorNet(nn.Module):
    """Deterministic actor.

    Default (output_squash="none") emits RAW (unbounded) logits — the env / actor
    loss clips and penalises OOB. Bounding via tanh saturates gradients when
    logits drift large, so unconstrained outputs preserve useful gradient inside
    [-1, 1] at the cost of needing the OOB penalty to keep them there.

    output_squash="tanh" applies jnp.tanh to the final Dense output, bounding the
    mean to (-1, 1) before exploration / target-policy noise is added. Use this
    when the actor is drifting OOB enough that the critic extrapolation is
    destabilising training (raw-actor failure mode on the kin G1 env).
    """
    action_dim: int
    hidden_layer_dims: tuple = (256, 256)
    activation: str = "relu"
    use_obs_norm: bool = True
    output_squash: str = "none"
    obs_ind: jnp.ndarray = None

    @nn.compact
    def __call__(self, x):
        activation_fn = get_activation_fn(self.activation)
        if self.obs_ind is not None:
            x = x[..., self.obs_ind]
        if self.use_obs_norm:
            x = RunningMeanStd()(x)
        for dim in self.hidden_layer_dims:
            x = nn.Dense(dim, kernel_init=orthogonal(jnp.sqrt(2)),
                         bias_init=constant(0.0))(x)
            x = activation_fn(x)
        a = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01),
                     bias_init=constant(0.0))(x)
        if self.output_squash == "tanh":
            a = jnp.tanh(a)
        elif self.output_squash != "none":
            raise ValueError(f"Unknown output_squash: {self.output_squash!r} "
                             "(expected 'none' or 'tanh')")
        return a


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

@struct.dataclass
class TD3SummaryMetrics(SummaryMetrics):
    mean_critic_loss: float = 0.0
    mean_actor_loss: float = 0.0
    mean_action_norm: float = 0.0
    mean_oob_penalty: float = 0.0
    mean_critic_grad_norm: float = 0.0
    mean_actor_grad_norm: float = 0.0
    mean_action_smoothness: float = 0.0
    mean_curriculum_cur_max: float = 0.0
    mean_global_step: float = 0.0
    buffer_size: int = 0


# ---------------------------------------------------------------------------
# Conf
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TD3AgentConf(AgentConfBase):
    config: DictConfig
    actor_net: TD3ActorNet
    critic_net: OffPolicyCriticNet
    actor_tx: Any
    critic_tx: Any

    def serialize(self):
        conf_dict = OmegaConf.to_container(self.config, resolve=True, throw_on_missing=True)
        return {
            "config": conf_dict,
            "actor_net": flax.serialization.to_state_dict(self.actor_net),
            "critic_net": flax.serialization.to_state_dict(self.critic_net),
        }

    @classmethod
    def from_dict(cls, d):
        config = OmegaConf.create(d["config"])
        exp = config.experiment
        actor_net = flax.serialization.from_state_dict(TD3ActorNet, d["actor_net"])
        critic_net = flax.serialization.from_state_dict(OffPolicyCriticNet, d["critic_net"])

        max_grad_norm = float(getattr(exp, 'max_grad_norm', 0.5))
        actor_tx = optax.chain(
            optax.clip_by_global_norm(float(getattr(exp, 'max_actor_grad_norm', max_grad_norm))),
            optax.adam(float(exp.lr_actor)),
        )
        critic_tx = optax.chain(
            optax.clip_by_global_norm(float(getattr(exp, 'max_critic_grad_norm', max_grad_norm))),
            optax.adam(float(exp.lr_critic)),
        )
        return cls(config=config, actor_net=actor_net, critic_net=critic_net,
                   actor_tx=actor_tx, critic_tx=critic_tx)


# ---------------------------------------------------------------------------
# Extra state (target actor + step counter for policy delay)
# ---------------------------------------------------------------------------

@struct.dataclass
class _TD3Extra:
    target_actor_params: Any
    target_actor_run_stats: Any


@struct.dataclass
class TD3AgentState(OffPolicyAgentState):
    """TD3 agent state — extra_state carries `_TD3Extra` (target actor)."""

    @classmethod
    def from_dict(cls, d, agent_conf):
        exp = agent_conf.config.experiment
        obs_dim = int(exp.obs_dim)
        action_dim = int(exp.action_dim)
        actor_net = agent_conf.actor_net
        critic_net = agent_conf.critic_net

        rng = jax.random.PRNGKey(0)

        actor_init = actor_net.init(rng, jnp.zeros((1, obs_dim)))
        actor_state = TrainState.create(
            apply_fn=actor_net.apply,
            params=actor_init["params"],
            run_stats=actor_init.get("run_stats", {}),
            tx=agent_conf.actor_tx,
        )
        actor_state = flax.serialization.from_state_dict(actor_state, d["actor_state"])
        actor_state = actor_state.replace(apply_fn=actor_net.apply)

        critic_init = critic_net.init(rng, jnp.zeros((1, obs_dim)),
                                      jnp.zeros((1, action_dim)))
        critic_state = TrainState.create(
            apply_fn=critic_net.apply,
            params=critic_init["params"],
            run_stats=critic_init.get("run_stats", {}),
            tx=agent_conf.critic_tx,
        )
        critic_state = flax.serialization.from_state_dict(critic_state, d["critic_state"])
        critic_state = critic_state.replace(apply_fn=critic_net.apply)

        target_dict = flax.serialization.from_state_dict(
            {"params": critic_init["params"],
             "run_stats": critic_init.get("run_stats", {})},
            d["target_critic"],
        )

        template_extra = _TD3Extra(
            target_actor_params=actor_init["params"],
            target_actor_run_stats=actor_init.get("run_stats", {}),
        )
        extra_state = flax.serialization.from_state_dict(template_extra, d["extra_state"])

        replay_buffer = ReplayBuffer.create(
            obs_dim, action_dim, int(exp.buffer_size),
            add_subsample=bool(getattr(exp, 'buffer_add_subsample', False)),
            add_prob=float(getattr(exp, 'buffer_add_prob', 1.0)),
        )

        return cls(
            actor_state=actor_state,
            critic_state=critic_state,
            target_critic_params=target_dict["params"],
            target_critic_run_stats=target_dict["run_stats"],
            extra_state=extra_state,
            replay_buffer=replay_buffer,
        )


# ---------------------------------------------------------------------------
# TD3 implementation
# ---------------------------------------------------------------------------

class TD3Jax(OffPolicyBase):
    _agent_conf = TD3AgentConf
    _agent_state = TD3AgentState

    # ---------- Conf init -----------------------------------------------
    @classmethod
    def init_agent_conf(cls, env, config):
        obs_dim = int(env.info.observation_space.shape[0])
        action_dim = int(env.info.action_space.shape[0])

        # Resolve actor/critic obs groups -> obs_ind arrays (mirrors ppo_jax).
        actor_obs_group = getattr(config.experiment, "actor_obs_group", None)
        critic_obs_group = getattr(config.experiment, "critic_obs_group", None)
        if actor_obs_group is not None:
            actor_obs_ind = jnp.asarray(
                env.obs_container.get_obs_ind_by_group(str(actor_obs_group)),
                dtype=jnp.int32,
            )
        else:
            actor_obs_ind = None
        if critic_obs_group is not None:
            critic_obs_ind = jnp.asarray(
                env.obs_container.get_obs_ind_by_group(str(critic_obs_group)),
                dtype=jnp.int32,
            )
        else:
            critic_obs_ind = None

        with open_dict(config.experiment):
            config.experiment.obs_dim = obs_dim
            config.experiment.action_dim = action_dim
            config.experiment.num_updates = (
                config.experiment.total_timesteps // config.experiment.num_envs
            )

        exp = config.experiment
        hidden_layers = (exp.hidden_layers
                         if isinstance(exp.hidden_layers, (list, ListConfig))
                         else ast.literal_eval(exp.hidden_layers))

        actor_net = TD3ActorNet(
            action_dim=action_dim,
            hidden_layer_dims=tuple(hidden_layers),
            activation=str(exp.activation),
            use_obs_norm=bool(getattr(exp, 'use_obs_norm', False)),
            output_squash=str(getattr(exp, 'output_squash', 'none')),
            obs_ind=actor_obs_ind,
        )
        critic_net = cls._build_critic_net(exp, obs_ind=critic_obs_ind)
        actor_tx, critic_tx = cls._build_optimisers(exp)

        return cls._agent_conf(
            config=config,
            actor_net=actor_net,
            critic_net=critic_net,
            actor_tx=actor_tx,
            critic_tx=critic_tx,
        )

    @classmethod
    def init_agent_state(cls, env, agent_conf, rng):
        exp = agent_conf.config.experiment
        obs_dim = int(exp.obs_dim)
        action_dim = int(exp.action_dim)

        rng, rng_a, rng_c = jax.random.split(rng, 3)
        actor_params = agent_conf.actor_net.init(rng_a, jnp.zeros((1, obs_dim)))
        actor_state = TrainState.create(
            apply_fn=agent_conf.actor_net.apply,
            params=actor_params["params"],
            run_stats=actor_params.get("run_stats", {}),
            tx=agent_conf.actor_tx,
        )
        critic_params = agent_conf.critic_net.init(
            rng_c, jnp.zeros((1, obs_dim)), jnp.zeros((1, action_dim))
        )
        # `critic_state.run_stats` is treated as the FULL mutable-variables
        # bundle for the critic (everything except `params`). Today it holds
        # `{"run_stats": ...}` when RunningMeanStd is in the trunk; with BN on,
        # it also holds `{"batch_stats": ...}`. Bundling avoids threading a
        # second carry field through every apply / target-update site.
        # `log_probs_collection` is a sow output, not a long-lived state, so
        # we exclude it from the bundle.
        _BUNDLE_EXCLUDE = {"params", "log_probs_collection"}
        critic_bundle = {k: v for k, v in critic_params.items() if k not in _BUNDLE_EXCLUDE}
        critic_state = TrainState.create(
            apply_fn=agent_conf.critic_net.apply,
            params=critic_params["params"],
            run_stats=critic_bundle,
            tx=agent_conf.critic_tx,
        )
        target_critic_params = critic_params["params"]
        target_critic_run_stats = critic_bundle
        extra_state = _TD3Extra(
            target_actor_params=actor_params["params"],
            target_actor_run_stats=actor_params.get("run_stats", {}),
        )
        replay_buffer = ReplayBuffer.create(
            obs_dim, action_dim, int(exp.buffer_size),
            add_subsample=bool(getattr(exp, 'buffer_add_subsample', False)),
            add_prob=float(getattr(exp, 'buffer_add_prob', 1.0)),
        )

        return TD3AgentState(
            actor_state=actor_state,
            critic_state=critic_state,
            target_critic_params=target_critic_params,
            target_critic_run_stats=target_critic_run_stats,
            extra_state=extra_state,
            replay_buffer=replay_buffer,
        )

    # ---------- OffPolicyBase hooks -------------------------------------
    @classmethod
    def _init_extra_state(cls, rng, exp):
        # Used only when called from base class; init_agent_state above does it
        # explicitly with proper shapes. This path is the "new run" path in
        # OffPolicyBase._train_fn. Init zeros — base class uses it only to
        # populate via params; production path uses init_agent_state.
        # In practice init_agent_state is always called first, so this is a stub.
        return _TD3Extra(target_actor_params=None, target_actor_run_stats=None)

    @classmethod
    def _select_action(cls, actor_state, obs, rng, exp,
                       extra_state, deterministic=False):
        action, updates = actor_state.apply_fn(
            {"params": actor_state.params, "run_stats": actor_state.run_stats},
            obs, mutable=["run_stats"],
        )
        actor_state = actor_state.replace(run_stats=updates["run_stats"])
        if not deterministic:
            sigma_act = float(getattr(exp, 'exploration_noise', 0.1))
            noise = sigma_act * jax.random.normal(rng, action.shape)
            # Pass tanh(mean) + noise to the env un-clipped. With output_squash=tanh
            # the mean is already in (-1, 1) so noise can push the action slightly
            # OOB — that's what the reward-side OOB term and the kin control's
            # input clip are there to handle. With output_squash=none the actor is
            # raw and OOB is unbounded; in that case the reward / actor-loss OOB
            # plumbing is the only restoring force.
            action = action + noise
        return action, actor_state

    @classmethod
    def _next_action_and_q_bonus(cls, actor_state, next_obs, rng, exp, extra_state):
        # Target policy smoothing using TARGET actor (extra_state.target_actor_*)
        sigma_tgt = float(getattr(exp, 'target_noise', 0.2))
        noise_clip = float(getattr(exp, 'noise_clip', 0.5))
        target_a, _ = actor_state.apply_fn(
            {"params": extra_state.target_actor_params,
             "run_stats": extra_state.target_actor_run_stats},
            next_obs, mutable=["run_stats"],
        )
        noise = jnp.clip(sigma_tgt * jax.random.normal(rng, target_a.shape),
                         -noise_clip, noise_clip)
        next_action = jnp.clip(target_a + noise, -1.0, 1.0)
        return next_action, jnp.zeros(next_action.shape[:-1]), {}

    @classmethod
    def _actor_loss(cls, actor_params, actor_apply_fn, actor_run_stats,
                    critic_st, critic_apply_fn,
                    obs_b, rng, exp, extra_state, nobs_b=None):
        action_raw, updates = actor_apply_fn(
            {"params": actor_params, "run_stats": actor_run_stats},
            obs_b, mutable=["run_stats"],
        )
        # Optional stochastic actor loss (Motivo-style): query Q on
        # tanh(mean) + clip(N(0, target_noise), -noise_clip, noise_clip)
        # instead of the deterministic mean. Smooths the objective by penalizing
        # actions that sit on narrow Q peaks.
        stochastic_actor_loss = bool(getattr(exp, 'stochastic_actor_loss', False))
        if stochastic_actor_loss:
            sigma_tgt = float(getattr(exp, 'target_noise', 0.05))
            noise_clip = float(getattr(exp, 'noise_clip', 0.15))
            noise = jnp.clip(sigma_tgt * jax.random.normal(rng, action_raw.shape),
                             -noise_clip, noise_clip)
            action_for_Q = jnp.clip(action_raw + noise, -1.0, 1.0)
        else:
            # When the actor-side OOB penalty is off (lambda_oob == 0.0), route Q's
            # gradient through the RAW action so the critic itself pulls OOB logits
            # back toward in-bounds regions. This relies on the reward-side OOB
            # term (LmjMimicReward.action_out_of_bounds_coeff) teaching the critic
            # that OOB actions are bad. When lambda_oob > 0.0, keep clipping so the
            # critic's extrapolated Q on OOB queries cannot feed back into the actor
            # and the actor-side penalty is the sole restoring force on OOB channels.
            lambda_oob_branch = float(getattr(exp, 'lambda_oob', 0.1))
            if lambda_oob_branch == 0.0:
                action_for_Q = action_raw
            else:
                action_for_Q = jnp.clip(action_raw, -1.0, 1.0)
        # Actor's critic query uses the SAME bundle as the critic loss (the
        # `critic_st.run_stats` field holds run_stats + optional batch_stats).
        # `training=False` so any BN inside the critic uses running averages
        # (the small per-actor-step batch would otherwise corrupt them).
        actor_bundle = critic_st.run_stats
        actor_mutables = list(actor_bundle.keys())
        (q1, q2), _ = critic_apply_fn(
            {"params": critic_st.params, **actor_bundle},
            obs_b, action_for_Q, mutable=actor_mutables, training=False,
        )
        # TD3 uses Q1 for the policy gradient
        q_loss = -jnp.mean(q1)

        # Out-of-bounds quadratic penalty on the RAW (unclipped) action:
        # 0 inside (-1, 1), (|a|-1)^2 outside. This is the only gradient the
        # actor receives for OOB logits.
        oob = jnp.maximum(jnp.abs(action_raw) - 1.0, 0.0) ** 2
        oob_penalty = jnp.mean(oob)

        # Scale by mean(|Q|) so penalty and Q loss are comparable magnitude.
        q_scale = jnp.mean(jnp.abs(q1))
        lambda_oob = float(getattr(exp, 'lambda_oob', 0.1))
        penalty_loss = lambda_oob * jax.lax.stop_gradient(q_scale) * oob_penalty

        # Optional action-smoothness regularizer: penalize the L2 distance
        # between the (tanh-squashed) actor output on consecutive states.
        # Discourages bang-bang policies. With output_squash="tanh" the
        # actor returns post-tanh actions, so action_raw is already in (-1,1)
        # and we compare it directly to actor(next_obs).
        lambda_smooth = float(getattr(exp, 'lambda_action_smooth', 0.0))
        if lambda_smooth > 0.0 and nobs_b is not None:
            next_action, _ = actor_apply_fn(
                {"params": actor_params, "run_stats": updates["run_stats"]},
                nobs_b, mutable=["run_stats"],
            )
            smoothness = jnp.mean((action_raw - next_action) ** 2)
            smooth_loss = lambda_smooth * smoothness
        else:
            smoothness = jnp.array(0.0)
            smooth_loss = jnp.array(0.0)

        loss = q_loss + penalty_loss + smooth_loss
        return loss, {
            "run_stats": updates["run_stats"],
            "action_norm": jnp.mean(jnp.linalg.norm(action_raw, axis=-1)),
            "oob_penalty": oob_penalty,
            "q_scale": q_scale,
            "action_smoothness": smoothness,
        }

    @classmethod
    def _update_extra(cls, extra_state, actor_aux, rng, exp):
        # Soft-update the target actor (Polyak) when actor was updated
        # We need the *online* actor params. The base class doesn't pass them
        # here, so we cannot soft-update target actor in this hook.
        # → handled by overriding _should_update_actor + a wrapper. But this
        # design works around it: target actor is updated in tandem with the
        # online actor via custom logic in _train_fn. To keep the design clean,
        # we update the target actor here using a closure trick: actor_aux
        # carries the new online params (added below in _actor_loss override).
        # NOTE: simpler alternative — override _train_fn slightly.
        # For this implementation we leave the target_actor update for the
        # overridden _train_fn (see below).
        return extra_state, {}

    @classmethod
    def _should_update_actor(cls, step_count, exp):
        policy_delay = int(getattr(exp, 'policy_delay', 2))
        return jnp.equal(jnp.mod(step_count, policy_delay), 0)

    @classmethod
    def _extra_metrics(cls, actor_state, obs, rng, extra_state, exp):
        return {}

    @classmethod
    def _build_summary_metric(cls, base_kwargs, extra):
        return TD3SummaryMetrics(
            mean_action_norm=extra.get("action_norm", 0.0),
            **base_kwargs,
        )

    # ---------- Override _train_fn to soft-update target actor ----------
    # The shared base updates only target_critic. TD3 also needs target_actor
    # updates with the same tau, after each actor update. We override the
    # outer loop minimally to add this.
    @classmethod
    def _train_fn(cls, rng, env, agent_conf, agent_state=None, traj=None,
                  mh=None):
        from loco_mujoco.algorithms.experimental.offpolicy_base import (
            OffPolicyAgentState as _S,
        )
        from loco_mujoco.core.wrappers import LogEnvState
        from loco_mujoco.core.mujoco_mjx import MjxState
        from loco_mujoco.utils import ValidationSummary

        exp = agent_conf.config.experiment
        actor_net = agent_conf.actor_net
        critic_net = agent_conf.critic_net

        env = cls._wrap_env(env, exp)

        if agent_state is None:
            agent_state = cls.init_agent_state(env, agent_conf, rng)
        actor_state = agent_state.actor_state.replace(apply_fn=actor_net.apply)
        critic_state = agent_state.critic_state.replace(apply_fn=critic_net.apply)
        target_critic_params = agent_state.target_critic_params
        target_critic_run_stats = agent_state.target_critic_run_stats
        extra_state = agent_state.extra_state
        replay_buffer = agent_state.replay_buffer

        if agent_state.env_state is not None:
            env_state = agent_state.env_state
            last_obs = agent_state.last_obs
        else:
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, exp.num_envs)
            last_obs, env_state = env.reset(reset_rng, traj)

        learning_starts = int(getattr(exp, 'learning_starts', exp.batch_size))
        batch_size = int(exp.batch_size)
        tau = float(getattr(exp, 'tau', 0.005))
        gamma = float(exp.gamma)
        gradient_steps = int(getattr(exp, 'gradient_steps', 1))

        def _critic_forward_target(obs, action, params, run_stats):
            # `run_stats` here is the full mutable-variables bundle for the
            # critic (run_stats + optional batch_stats).
            #
            # Non-joint forward suffices when BN is off; this is the legacy
            # path. When BN is on the caller should instead use
            # _critic_forward_target_joint so BN sees both the bootstrap and
            # current-state batches together.
            mutables = list(run_stats.keys())
            (q1, q2), updates = critic_net.apply(
                {"params": params, **run_stats},
                obs, action,
                mutable=mutables,
                training=False,
            )
            new_bundle = {k: updates.get(k, run_stats[k]) for k in run_stats}
            return q1, q2, new_bundle

        def _critic_forward_target_joint(obs, nobs, act, next_act,
                                         params, run_stats):
            """Joint forward: concat (obs, nobs)/(act, next_act) so BN sees
            both halves, then split. Returns (q1_curr, q2_curr, q1_next,
            q2_next, new_bundle)."""
            mutables = list(run_stats.keys())
            cat_obs = jnp.concatenate([obs, nobs], axis=0)
            cat_act = jnp.concatenate([act, next_act], axis=0)
            (q1c, q2c), updates = critic_net.apply(
                {"params": params, **run_stats},
                cat_obs, cat_act,
                mutable=mutables,
                training=True,
            )
            q1_curr, q1_next = jnp.split(q1c, 2)
            q2_curr, q2_next = jnp.split(q2c, 2)
            new_bundle = {k: updates.get(k, run_stats[k]) for k in run_stats}
            return q1_curr, q2_curr, q1_next, q2_next, new_bundle

        def _collect_transition(actor_state, replay_buffer, env_state,
                                last_obs, rng, extra_state):
            rng, rng_act = jax.random.split(rng)
            action, actor_state = cls._select_action(
                actor_state, last_obs, rng_act, exp, extra_state, deterministic=False
            )
            next_obs, reward, absorbing, done, info, env_state = env.step(
                env_state, action, traj
            )
            if replay_buffer.add_subsample:
                rng, rng_buf = jax.random.split(rng)
                replay_buffer = replay_buffer.add_batch(
                    last_obs, next_obs, action, reward,
                    absorbing.astype(jnp.float32), rng=rng_buf,
                )
            else:
                replay_buffer = replay_buffer.add_batch(
                    last_obs, next_obs, action, reward, absorbing.astype(jnp.float32)
                )
            return actor_state, replay_buffer, env_state, next_obs, rng

        def _single_gradient_update(carry, step_idx):
            (actor_st, critic_st, tgt_p, tgt_rs, ex_st, buf, rng_up) = carry

            rng_up, rng_sample = jax.random.split(rng_up)
            obs_b, nobs_b, act_b, rew_b, done_b = buf.sample(rng_sample, batch_size)

            # critic
            rng_up, rng_next = jax.random.split(rng_up)
            next_action, next_q_bonus, _ = cls._next_action_and_q_bonus(
                actor_st, nobs_b, rng_next, exp, ex_st
            )

            critic_loss_kind = str(getattr(exp, 'critic_loss', 'mse'))
            use_bn = bool(getattr(exp, 'use_batch_norm', False))
            num_atoms = int(getattr(exp, 'num_atoms', 1))
            min_v = float(getattr(exp, 'min_v', -5.0))
            max_v = float(getattr(exp, 'max_v', 5.0))

            # Compute the next-state target. For the MSE path we use the
            # scalar Q values returned by the critic; for categorical we also
            # need the per-atom target log-probs which the critic sows.
            if critic_loss_kind == 'categorical':
                tgt_mutables = list(tgt_rs.keys()) + ["log_probs_collection"]
                (q1n_v, q2n_v), tgt_updates = critic_net.apply(
                    {"params": tgt_p, **tgt_rs},
                    nobs_b, next_action,
                    mutable=tgt_mutables, training=False,
                )
                # log_probs_collection: {"q1": {"log_probs": (sown tuple,)},
                # "q2": ...}. `sow` wraps the value in a tuple by default, so
                # `[0]` extracts the actual log_probs array.
                tgt_lp_coll = tgt_updates.get("log_probs_collection", {})
                tgt_log_probs_q1 = tgt_lp_coll["q1"]["log_probs"][0]
                tgt_log_probs_q2 = tgt_lp_coll["q2"]["log_probs"][0]
                new_tgt_rs = {k: tgt_updates.get(k, tgt_rs[k]) for k in tgt_rs}
                # Pick the twin with the smaller scalar Q (per-sample); use its
                # log-probs for the target distribution. Matches XQC.
                pick_q1 = (q1n_v <= q2n_v)
                tgt_log_probs = jnp.where(
                    pick_q1[:, None], tgt_log_probs_q1, tgt_log_probs_q2,
                )
            else:
                q1n, q2n, new_tgt_rs = _critic_forward_target(
                    nobs_b, next_action, tgt_p, tgt_rs
                )
                # Q-target aggregation across twins. Default is standard TD3
                # `min(Q1, Q2)`. If `pessimism_penalty` is set, use Motivo-style
                # ensemble pessimism: `mean(Q) - k * |Q1 - Q2|`.
                pessimism_penalty = getattr(exp, "pessimism_penalty", None)
                if pessimism_penalty is None:
                    q_next = jnp.minimum(q1n, q2n) + next_q_bonus
                else:
                    k = float(pessimism_penalty)
                    q_mean = 0.5 * (q1n + q2n)
                    q_unc = jnp.abs(q1n - q2n)
                    q_next = q_mean - k * q_unc + next_q_bonus
                q_target = rew_b + gamma * (1.0 - done_b) * q_next
                q_target = jax.lax.stop_gradient(q_target)

            def _critic_loss_fn(params):
                run_stats = critic_st.run_stats
                mutables = list(run_stats.keys())
                if critic_loss_kind == 'categorical':
                    # Shift bin centers by the bootstrap target and project
                    # the next-state distribution onto the support, then CE.
                    bin_values = jnp.linspace(min_v, max_v, num_atoms).reshape(1, -1)
                    target_bin_values = (
                        rew_b.reshape(-1, 1)
                        + gamma * (bin_values + next_q_bonus.reshape(-1, 1))
                          * (1.0 - done_b.reshape(-1, 1))
                    )
                    target_probs = jax.lax.stop_gradient(
                        _c51_project_target(
                            tgt_log_probs, target_bin_values,
                            num_atoms, min_v, max_v,
                        )
                    )
                    mutables_online = mutables + ["log_probs_collection"]
                    (_q1, _q2), updates = critic_net.apply(
                        {"params": params, **run_stats},
                        obs_b, act_b,
                        mutable=mutables_online, training=use_bn,
                    )
                    pred_lp_coll = updates.get("log_probs_collection", {})
                    pred_lp_q1 = pred_lp_coll["q1"]["log_probs"][0]
                    pred_lp_q2 = pred_lp_coll["q2"]["log_probs"][0]
                    loss = -jnp.mean(jnp.sum(target_probs * pred_lp_q1, axis=-1))
                    loss = loss - jnp.mean(jnp.sum(target_probs * pred_lp_q2, axis=-1))
                    new_bundle = {k: updates.get(k, run_stats[k]) for k in run_stats}
                    return loss, new_bundle
                if use_bn:
                    # XQC / CrossQ joint forward: run the online critic on
                    # concat([obs, nobs], [act, next_act]) so BN sees both
                    # batches together, then split.
                    cat_obs = jnp.concatenate([obs_b, nobs_b], axis=0)
                    cat_act = jnp.concatenate([act_b, next_action], axis=0)
                    (q1c, q2c), updates = critic_net.apply(
                        {"params": params, **run_stats},
                        cat_obs, cat_act,
                        mutable=mutables, training=True,
                    )
                    q1, _ = jnp.split(q1c, 2)
                    q2, _ = jnp.split(q2c, 2)
                else:
                    (q1, q2), updates = critic_net.apply(
                        {"params": params, **run_stats},
                        obs_b, act_b,
                        mutable=mutables, training=False,
                    )
                loss = jnp.mean((q1 - q_target) ** 2) + jnp.mean((q2 - q_target) ** 2)
                new_bundle = {k: updates.get(k, run_stats[k]) for k in run_stats}
                return loss, new_bundle

            (critic_loss, new_critic_rs), critic_grads = jax.value_and_grad(
                _critic_loss_fn, has_aux=True
            )(critic_st.params)
            critic_grad_norm = optax.global_norm(critic_grads)
            critic_st = critic_st.apply_gradients(grads=critic_grads)
            critic_st = critic_st.replace(run_stats=new_critic_rs)

            # Post-step weight normalization (XQC / Salimans-Kingma).
            # When `use_weight_norm=True`, renormalize each Dense kernel after
            # the gradient step: W <- W / ||W||_axis=-2 (per-output-unit norm).
            # `normalize_last_layer=True` also renormalizes the predictor head.
            # No-op when the flag is unset, so the TD3 baseline is unchanged.
            if bool(getattr(exp, 'use_weight_norm', False)):
                critic_st = critic_st.replace(
                    params=_normalize_dense_kernels(
                        critic_st.params,
                        normalize_last_layer=bool(getattr(exp, 'normalize_last_layer', True)),
                    )
                )

            # actor (delayed)
            rng_up, rng_actor = jax.random.split(rng_up)
            do_actor_update = cls._should_update_actor(step_idx, exp)

            def _actor_loss_fn(params):
                return cls._actor_loss(
                    params, actor_st.apply_fn, actor_st.run_stats,
                    critic_st, critic_net.apply,
                    obs_b, rng_actor, exp, ex_st, nobs_b=nobs_b,
                )

            (actor_loss, actor_aux), actor_grads = jax.value_and_grad(
                _actor_loss_fn, has_aux=True
            )(actor_st.params)
            # raw pre-clip grad norm; reported only on steps where the actor
            # actually updates (zero on policy-delay skip steps).
            actor_grad_norm_raw = optax.global_norm(actor_grads)
            actor_grad_norm = jnp.where(do_actor_update, actor_grad_norm_raw, 0.0)

            def _apply_actor(args):
                a_st, ex_st_in, t_p, t_rs = args
                a_st = a_st.apply_gradients(grads=actor_grads)
                # always advance run_stats
                a_st = a_st.replace(run_stats=actor_aux["run_stats"])
                # soft-update target actor and target critic together
                new_target_actor_params = jax.tree.map(
                    lambda tp, cp: tau * cp + (1.0 - tau) * tp,
                    ex_st_in.target_actor_params, a_st.params,
                )
                new_target_actor_rs = a_st.run_stats
                new_ex = ex_st_in.replace(
                    target_actor_params=new_target_actor_params,
                    target_actor_run_stats=new_target_actor_rs,
                )
                return a_st, new_ex, t_p, t_rs

            def _skip_actor(args):
                a_st, ex_st_in, t_p, t_rs = args
                a_st = a_st.replace(run_stats=actor_aux["run_stats"])
                return a_st, ex_st_in, t_p, t_rs

            actor_st, ex_st, _, _ = jax.lax.cond(
                do_actor_update, _apply_actor, _skip_actor,
                (actor_st, ex_st, tgt_p, tgt_rs),
            )

            # critic target soft-update (every step). Params via Polyak. Also
            # soft-update the bundle of mutable variables (RunningMeanStd's
            # run_stats and, when BN is on, batch_stats) the same way XQC does.
            # `jax.tree.map` over an empty dict is a no-op, so this is safe
            # for the baseline configuration where the bundle is empty.
            new_tgt_p = jax.tree.map(
                lambda tp, cp: tau * cp + (1.0 - tau) * tp,
                tgt_p, critic_st.params,
            )
            new_tgt_rs = jax.tree.map(
                lambda tp, cp: tau * cp + (1.0 - tau) * tp,
                new_tgt_rs, critic_st.run_stats,
            )

            new_carry = (actor_st, critic_st, new_tgt_p, new_tgt_rs,
                         ex_st, buf, rng_up)
            return new_carry, (critic_loss, actor_loss,
                                actor_aux["action_norm"],
                                actor_aux["oob_penalty"],
                                critic_grad_norm,
                                actor_grad_norm,
                                actor_aux["action_smoothness"])

        def _do_updates(actor_st, critic_st, tgt_p, tgt_rs, ex_st, buf, rng_up):
            carry = (actor_st, critic_st, tgt_p, tgt_rs, ex_st, buf, rng_up)
            carry, losses = jax.lax.scan(
                _single_gradient_update, carry, jnp.arange(gradient_steps)
            )
            actor_st, critic_st, tgt_p, tgt_rs, ex_st, _, _ = carry
            # losses[5] is actor_grad_norm, zeroed on policy-delay skip steps.
            # Sum then divide by the number of *actual* actor updates to get
            # a mean over only the steps where the actor moved.
            n_actor_updates = jnp.maximum(jnp.sum(losses[5] != 0.0), 1)
            return (actor_st, critic_st, tgt_p, tgt_rs, ex_st,
                    jnp.mean(losses[0]), jnp.mean(losses[1]),
                    jnp.mean(losses[2]), jnp.mean(losses[3]),
                    jnp.mean(losses[4]),
                    jnp.sum(losses[5]) / n_actor_updates,
                    jnp.mean(losses[6]))

        def _skip_updates(actor_st, critic_st, tgt_p, tgt_rs, ex_st, buf, rng_up):
            return (actor_st, critic_st, tgt_p, tgt_rs, ex_st,
                    jnp.array(0.0), jnp.array(0.0), jnp.array(0.0),
                    jnp.array(0.0), jnp.array(0.0), jnp.array(0.0),
                    jnp.array(0.0))

        def _update_step(runner_state, unused):
            (actor_state, critic_state, tgt_params, tgt_run_stats,
             ex_state, replay_buffer, env_state, last_obs, rng) = runner_state

            actor_state, replay_buffer, env_state, next_obs, rng = \
                _collect_transition(actor_state, replay_buffer, env_state,
                                    last_obs, rng, ex_state)

            rng, rng_update = jax.random.split(rng)
            result = jax.lax.cond(
                replay_buffer.size >= learning_starts,
                lambda args: _do_updates(*args),
                lambda args: _skip_updates(*args),
                (actor_state, critic_state, tgt_params, tgt_run_stats,
                 ex_state, replay_buffer, rng_update),
            )
            (actor_state, critic_state, tgt_params, tgt_run_stats,
             ex_state, critic_loss, actor_loss, action_norm, oob_penalty,
             critic_grad_norm, actor_grad_norm, action_smoothness) = result

            log_env_state = env_state.find(LogEnvState)
            logged_metrics = log_env_state.metrics
            # `returned_episode_returns` / `returned_episode_lengths` carry the
            # last-completed episode return/length for each env, held constant
            # until that env finishes another one. Averaging across envs gives
            # a smooth running estimate.
            base_kwargs = dict(
                mean_episode_return=jnp.mean(logged_metrics.returned_episode_returns),
                mean_episode_length=jnp.mean(logged_metrics.returned_episode_lengths),
                max_timestep=jnp.max(logged_metrics.timestep * exp.num_envs),
                mean_critic_loss=critic_loss,
                mean_actor_loss=actor_loss,
                buffer_size=replay_buffer.size,
            )
            # Curriculum-progress metrics. Read from MjxState.additional_carry
            # via env.find_attr(...), which recurses through the wrapper chain.
            additional_carry = env.find_attr(env_state, "additional_carry")
            global_step_mean = jnp.mean(
                additional_carry.global_step.astype(jnp.float32)
            )
            curriculum_cur_max = jnp.mean(
                additional_carry.curriculum_cur_max.astype(jnp.float32)
            )
            metric = TD3SummaryMetrics(mean_action_norm=action_norm,
                                        mean_oob_penalty=oob_penalty,
                                        mean_critic_grad_norm=critic_grad_norm,
                                        mean_actor_grad_norm=actor_grad_norm,
                                        mean_action_smoothness=action_smoothness,
                                        mean_curriculum_cur_max=curriculum_cur_max,
                                        mean_global_step=global_step_mean,
                                        **base_kwargs)

            runner_state = (actor_state, critic_state, tgt_params, tgt_run_stats,
                            ex_state, replay_buffer, env_state, next_obs, rng)
            return runner_state, metric

        # ----- log-interval wrapper: run `log_every` inner steps and emit one
        # aggregated metric row. Reduces wandb log volume and smooths noise.
        log_every = max(1, int(getattr(exp, 'log_every', 100)))
        num_outer = max(1, int(exp.num_updates) // log_every)

        def _logged_step(runner_state, unused):
            runner_state, inner_metrics = jax.lax.scan(
                _update_step, runner_state, None, log_every
            )
            aggregated = jax.tree.map(lambda x: jnp.mean(x, axis=0), inner_metrics)
            # monotone counters: take the value at the end of the window
            aggregated = aggregated.replace(
                max_timestep=jnp.max(inner_metrics.max_timestep),
                buffer_size=inner_metrics.buffer_size[-1],
            )
            return runner_state, aggregated

        rng, _rng = jax.random.split(rng)
        runner_state = (
            actor_state, critic_state, target_critic_params, target_critic_run_stats,
            extra_state, replay_buffer, env_state, last_obs, _rng,
        )
        runner_state, training_metrics = jax.lax.scan(
            _logged_step, runner_state, None, num_outer
        )
        (actor_state, critic_state, tgt_params, tgt_run_stats,
         ex_state, replay_buffer, env_state, last_obs, _) = runner_state

        agent_state_out = TD3AgentState(
            actor_state=actor_state,
            critic_state=critic_state,
            target_critic_params=tgt_params,
            target_critic_run_stats=tgt_run_stats,
            extra_state=ex_state,
            replay_buffer=replay_buffer,
            env_state=env_state,
            last_obs=last_obs,
        )
        return {
            "agent_state": agent_state_out,
            "training_metrics": training_metrics,
            "validation_metrics": ValidationSummary(),
        }

    # ---------- play_policy --------------------------------------------
    @classmethod
    def play_policy(cls, env, agent_conf, agent_state,
                    n_envs: int, n_steps=None, render=True,
                    record=False, rng=None, deterministic=True,
                    use_mujoco=False, wrap_env=True,
                    train_state_seed=None, traj=None):
        import numpy as np

        exp = agent_conf.config.experiment
        actor_net = agent_conf.actor_net
        actor_state = agent_state.actor_state.replace(apply_fn=actor_net.apply)

        if wrap_env and not use_mujoco:
            env = cls._wrap_env(env, exp)

        if rng is None:
            rng = jax.random.key(0)

        keys = jax.random.split(rng, n_envs + 1)
        rng, env_keys = keys[0], keys[1:]

        @jax.jit
        def _act(actor_st, obs, _rng):
            action, updates = actor_net.apply(
                {"params": actor_st.params, "run_stats": actor_st.run_stats},
                obs, mutable=["run_stats"],
            )
            actor_st = actor_st.replace(run_stats=updates["run_stats"])
            if not deterministic:
                sigma = float(getattr(exp, 'exploration_noise', 0.1))
                noise = sigma * jax.random.normal(_rng, action.shape)
                action = jnp.clip(action + noise, -1.0, 1.0)
            return action, actor_st

        if use_mujoco:
            obs = env.reset()
            env_state = None
        else:
            obs, env_state = env.reset(env_keys, traj)

        if n_steps is None:
            n_steps = np.iinfo(np.int32).max

        for _ in range(n_steps):
            rng, _rng = jax.random.split(rng)
            action, actor_state = _act(actor_state, obs, _rng)
            action = jnp.atleast_2d(action)
            if use_mujoco:
                obs, _, _, done, _ = env.step(action)
            else:
                obs, _, _, done, _, env_state = env.step(env_state, action, traj)
            if use_mujoco:
                env.render(record=True)
                if done:
                    obs = env.reset()
            else:
                env.mjx_render(env_state, record=record)
        env.stop()
