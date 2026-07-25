"""
Off-policy actor-critic base class shared by SAC and TD3.

Owns:
  - ReplayBuffer
  - Twin Q-network architecture (`OffPolicyCriticNet`)
  - Target critic with soft (Polyak) updates
  - Environment collection / replay-buffer add
  - Outer scan over env steps (one transition per env per step + N gradient updates)
  - Conditional learning-starts logic
  - Metric struct skeleton

Subclasses provide algorithm-specific pieces via hook methods:
  - `_build_actor_net(action_dim, exp)`
  - `_init_extra_state(rng, exp)`                         # optional (alpha for SAC, none for TD3)
  - `_select_action(actor_state, obs, rng, exp, ...)`     # for env interaction
  - `_next_action_and_q_bonus(actor_state, next_obs, rng, exp, extra_state)`
        returns (next_action, next_bonus)
        target_q = reward + gamma * (1-done) * (min(Q1, Q2)(next_obs, next_action) + next_bonus)
        SAC: next_bonus = -alpha * log_pi(next_action)
        TD3: next_bonus = 0  (action is target-smoothed)
  - `_actor_loss(actor_params, ..., extra_state)`         # alpha*log_pi-Q for SAC, -Q for TD3
  - `_update_extra(extra_state, aux, rng, exp)`           # alpha update for SAC
  - `_should_update_actor(step_count, exp)`               # for TD3 policy delay (default: always)
  - `_extra_metrics(actor_state, obs, rng, extra_state, exp)`
  - `_build_summary_metric(base_kwargs, extra)`
"""

from typing import Any
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import flax
import optax
from flax import struct
from omegaconf import DictConfig

from loco_mujoco.algorithms import (JaxRLAlgorithmBase, AgentConfBase,
                                    AgentStateBase, TrainState, ReplayBuffer)
from loco_mujoco.algorithms.common.networks import (RunningMeanStd,
                                                    get_activation_fn)
from loco_mujoco.core.wrappers import (LogWrapper, LogEnvState, VecEnv)
from loco_mujoco.utils import MetricsHandler, ValidationSummary


# ---------------------------------------------------------------------------
# Networks (shared)
# ---------------------------------------------------------------------------

class _QNet(nn.Module):
    """One Q-network sub-module (s,a) -> q.

    Optional pieces (all gated; defaults preserve the original MLP critic):
      - `use_batch_norm`: insert BatchNorm into each hidden block. With
        `pre_activation_bn=True`: `Dense(no bias) -> BN -> activation`.
        With `pre_activation_bn=False`: `Dense -> activation -> BN`.
      - `num_atoms > 1`: replace the scalar head with a categorical head
        of `num_atoms` logits. Returns the scalar value `sum(softmax * z)`
        where `z = linspace(min_v, max_v, num_atoms)`. The per-atom
        log-probs are stashed via `self.sow("log_probs_collection", ...)`
        so the categorical CE loss can retrieve them.
    """

    hidden_layer_dims: tuple = (256, 256)
    activation: str = "tanh"
    use_batch_norm: bool = False
    pre_activation_bn: bool = True
    num_atoms: int = 1
    min_v: float = -5.0
    max_v: float = 5.0

    @nn.compact
    def __call__(self, x, *, training=False):
        activation_fn = get_activation_fn(self.activation)
        for dim in self.hidden_layer_dims:
            if self.use_batch_norm:
                if self.pre_activation_bn:
                    # XQC pre-activation: Dense(no bias) -> BN -> act
                    x = nn.Dense(dim, use_bias=False,
                                 kernel_init=orthogonal(jnp.sqrt(2)))(x)
                    x = nn.BatchNorm(use_running_average=not training,
                                     momentum=0.99, epsilon=0.001)(x)
                    x = activation_fn(x)
                else:
                    # CrossQ post-activation: Dense -> act -> BN
                    x = nn.Dense(dim, kernel_init=orthogonal(jnp.sqrt(2)),
                                 bias_init=constant(0.0))(x)
                    x = activation_fn(x)
                    x = nn.BatchNorm(use_running_average=not training,
                                     momentum=0.99, epsilon=0.001)(x)
            else:
                x = nn.Dense(dim, kernel_init=orthogonal(jnp.sqrt(2)),
                             bias_init=constant(0.0))(x)
                x = activation_fn(x)

        # Head
        n_out = max(1, int(self.num_atoms))
        head = nn.Dense(n_out, kernel_init=orthogonal(1.0),
                        bias_init=constant(0.0))(x)
        if n_out == 1:
            return jnp.squeeze(head, axis=-1)
        # Categorical: convert logits -> log_probs over `num_atoms` bins,
        # extract scalar value via support.
        log_probs = nn.log_softmax(head, axis=-1)
        # Stash log-probs so the CE loss can find them via mutable collection.
        self.sow("log_probs_collection", "log_probs", log_probs)
        bin_values = jnp.linspace(self.min_v, self.max_v, n_out)
        value = jnp.sum(jnp.exp(log_probs) * bin_values, axis=-1)
        return value


class OffPolicyCriticNet(nn.Module):
    """Twin Q-networks, optionally with input observation normalisation."""

    hidden_layer_dims: tuple = (256, 256)
    activation: str = "tanh"
    use_obs_norm: bool = True
    use_batch_norm: bool = False
    pre_activation_bn: bool = True
    num_atoms: int = 1
    min_v: float = -5.0
    max_v: float = 5.0

    @nn.compact
    def __call__(self, obs, action, *, training=False):
        if self.use_obs_norm:
            obs = RunningMeanStd()(obs)
        x = jnp.concatenate([obs, action], axis=-1)
        kwargs = dict(
            hidden_layer_dims=self.hidden_layer_dims,
            activation=self.activation,
            use_batch_norm=self.use_batch_norm,
            pre_activation_bn=self.pre_activation_bn,
            num_atoms=self.num_atoms,
            min_v=self.min_v,
            max_v=self.max_v,
        )
        q1 = _QNet(**kwargs, name="q1")(x, training=training)
        q2 = _QNet(**kwargs, name="q2")(x, training=training)
        return q1, q2


# ReplayBuffer has moved to loco_mujoco.algorithms.common.dataclasses;
# re-imported above and re-exported here for back-compat.


# ---------------------------------------------------------------------------
# Weight-normalization helper (Salimans-Kingma / XQC-style post-step renorm)
# ---------------------------------------------------------------------------


def _normalize_dense_kernels(params, normalize_last_layer: bool = True):
    """Renormalize every Dense layer's kernel (and bias if present) so that
    ``||kernel||_axis=-2 = 1``. Mirrors ``xqc/networks/common.py::norm_network``.

    Args:
        params: Flax params PyTree (FrozenDict).
        normalize_last_layer: If False, skip the final Dense in each twin's
            ``_QNet`` (matches the XQC ``normalize_last_layer`` flag).

    Returns:
        New params PyTree with the same structure.
    """
    flat = flax.traverse_util.flatten_dict(params, sep="/")

    dense_paths = sorted({
        "/".join(k.split("/")[:-1])
        for k in flat
        if k.endswith("/kernel") and "Dense" in k
    })

    if not normalize_last_layer and dense_paths:
        # Drop the last Dense under each q1/q2 (or other) prefix — that's the
        # predictor head.
        from collections import defaultdict
        last_per_prefix = defaultdict(lambda: ("", -1))
        for path in dense_paths:
            parts = path.split("/")
            prefix = "/".join(parts[:-1])
            dense_name = parts[-1]
            if dense_name.startswith("Dense_"):
                try:
                    idx = int(dense_name[len("Dense_"):])
                except ValueError:
                    continue
                if idx > last_per_prefix[prefix][1]:
                    last_per_prefix[prefix] = (path, idx)
        skip = {p for (p, _) in last_per_prefix.values()}
        dense_paths = [p for p in dense_paths if p not in skip]

    for path in dense_paths:
        kernel_key = f"{path}/kernel"
        bias_key = f"{path}/bias"
        kernel = flat[kernel_key]
        if bias_key in flat:
            bias = flat[bias_key]
            w = jnp.concatenate([kernel, jnp.expand_dims(bias, -2)], axis=-2)
            norm = jnp.linalg.norm(w, axis=-2, keepdims=True) + 1e-12
            flat[kernel_key] = kernel / norm
            flat[bias_key] = bias / norm.squeeze(-2)
        else:
            norm = jnp.linalg.norm(kernel, axis=-2, keepdims=True) + 1e-12
            flat[kernel_key] = kernel / norm

    return flax.traverse_util.unflatten_dict(flat, sep="/")


def _c51_project_target(target_log_probs, target_bin_values, num_atoms,
                        min_v, max_v):
    """Project the next-state log-probabilities onto the C51 support after
    shifting the bin centers by the bootstrapped target ``r + γ·(z-α·log π)``.

    Mirrors ``xqc.agents.xqc.critic.categorical_td_loss``:
      - ``target_bin_values``: (B, num_atoms) bin centers AFTER the
        ``reward + gamma * (bin - actor_entropy) * (1 - done)`` shift.
      - Distribute each old bin's probability mass between the two nearest
        new bins using floor/ceil weights.

    Returns the target distribution (B, num_atoms) ready for the cross-entropy
    against the predicted log-probabilities.
    """
    target_bin_values = jnp.clip(target_bin_values, min_v, max_v)

    # `b` indexes new-bin positions for each shifted old bin center.
    b = (target_bin_values - min_v) / ((max_v - min_v) / (num_atoms - 1))
    l = jnp.floor(b)
    u = jnp.ceil(b)
    l_mask = jax.nn.one_hot(l.reshape(-1), num_atoms).reshape(
        (-1, num_atoms, num_atoms)
    )
    u_mask = jax.nn.one_hot(u.reshape(-1), num_atoms).reshape(
        (-1, num_atoms, num_atoms)
    )

    target_probs_old = jnp.exp(target_log_probs)
    # When floor == ceil (target value lands exactly on a bin center), put all
    # mass on that bin; else split (u-b) to lower and (b-l) to upper.
    m_l = (target_probs_old * (u + (l == u).astype(jnp.float32) - b)).reshape(
        (-1, num_atoms, 1)
    )
    m_u = (target_probs_old * (b - l)).reshape((-1, num_atoms, 1))
    target_probs = jnp.sum(m_l * l_mask + m_u * u_mask, axis=1)
    return target_probs


# ---------------------------------------------------------------------------
# Base agent conf / state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OffPolicyAgentConf(AgentConfBase):
    """Common conf fields for off-policy algos. Subclasses can extend."""
    config: DictConfig
    actor_net: Any
    critic_net: Any
    actor_tx: Any
    critic_tx: Any


@struct.dataclass
class OffPolicyAgentState(AgentStateBase):
    actor_state: TrainState
    critic_state: TrainState
    target_critic_params: Any
    target_critic_run_stats: Any
    extra_state: Any                # algorithm-specific (e.g. log_alpha for SAC)
    replay_buffer: ReplayBuffer
    env_state: Any = None
    last_obs: Any = None

    def serialize(self):
        return {
            "actor_state": flax.serialization.to_state_dict(self.actor_state),
            "critic_state": flax.serialization.to_state_dict(self.critic_state),
            "target_critic": flax.serialization.to_state_dict(
                {"params": self.target_critic_params,
                 "run_stats": self.target_critic_run_stats}
            ),
            "extra_state": flax.serialization.to_state_dict(self.extra_state),
        }


# ---------------------------------------------------------------------------
# OffPolicyBase
# ---------------------------------------------------------------------------

class OffPolicyBase(JaxRLAlgorithmBase):
    """Shared off-policy actor-critic engine. Subclasses must override the hooks
    listed in this module's docstring."""

    # ---------- Hooks (override in subclasses) ---------------------------
    @classmethod
    def _build_actor_net(cls, action_dim: int, exp):
        raise NotImplementedError

    @classmethod
    def _init_extra_state(cls, rng, exp):
        """Init extra (algorithm-specific) state. Returns pytree."""
        return {}

    @classmethod
    def _select_action(cls, actor_state, obs, rng, exp,
                       extra_state, deterministic=False):
        """Returns (action, updated_actor_state). Used at env-collect time."""
        raise NotImplementedError

    @classmethod
    def _next_action_and_q_bonus(cls, actor_state, next_obs, rng, exp,
                                 extra_state):
        """Returns (next_action, next_q_bonus, aux_dict).
        Q-target: r + γ(1-d) · ( min(Q1,Q2)(next_obs, next_action) + next_q_bonus ).
        SAC: next_q_bonus = -alpha · log_pi(next_action)
        TD3: next_q_bonus = 0  (action is target-policy smoothed)
        aux_dict can carry log_pi or any signal needed elsewhere."""
        raise NotImplementedError

    @classmethod
    def _actor_loss(cls, actor_params, actor_apply_fn, actor_run_stats,
                    critic_st, critic_apply_fn,
                    obs_b, rng, exp, extra_state):
        """Compute actor loss and any aux (e.g. log_pi for SAC)."""
        raise NotImplementedError

    @classmethod
    def _update_extra(cls, extra_state, actor_aux, rng, exp):
        """Update algorithm-specific state (e.g. alpha). Returns
        (new_extra_state, aux_metrics_dict)."""
        return extra_state, {}

    @classmethod
    def _should_update_actor(cls, step_count, exp):
        """For TD3 policy delay; default = always update."""
        return jnp.array(True)

    @classmethod
    def _extra_metrics(cls, actor_state, obs, rng, extra_state, exp):
        """Compute extra metrics for logging (e.g. mean_alpha, mean_entropy)."""
        return {}

    @classmethod
    def _build_summary_metric(cls, base_kwargs, extra):
        """Combine base metric kwargs and extra into the algo's summary struct."""
        raise NotImplementedError

    # ---------- Optimisers (override if you need separate optimisers) ----
    @classmethod
    def _build_optimisers(cls, exp):
        max_grad = float(getattr(exp, 'max_grad_norm', 0.5))
        actor_tx = optax.chain(
            optax.clip_by_global_norm(float(getattr(exp, 'max_actor_grad_norm', max_grad))),
            optax.adam(float(exp.lr_actor)),
        )
        critic_tx = optax.chain(
            optax.clip_by_global_norm(float(getattr(exp, 'max_critic_grad_norm', max_grad))),
            optax.adam(float(exp.lr_critic)),
        )
        return actor_tx, critic_tx

    # ---------- Conf helpers (subclasses can override) -------------------
    @classmethod
    def _build_critic_net(cls, exp):
        import ast
        from omegaconf import ListConfig
        hidden = (exp.hidden_layers if isinstance(exp.hidden_layers, (list, ListConfig))
                  else ast.literal_eval(exp.hidden_layers))
        return OffPolicyCriticNet(
            hidden_layer_dims=tuple(hidden),
            activation=str(exp.activation),
            use_obs_norm=bool(getattr(exp, 'use_obs_norm', False)),
            use_batch_norm=bool(getattr(exp, 'use_batch_norm', False)),
            pre_activation_bn=bool(getattr(exp, 'pre_activation_bn', True)),
            num_atoms=int(getattr(exp, 'num_atoms', 1)),
            min_v=float(getattr(exp, 'min_v', -5.0)),
            max_v=float(getattr(exp, 'max_v', 5.0)),
        )

    # ---------- Training loop (the shared engine) ------------------------
    @classmethod
    def _train_fn(cls, rng, env,
                  agent_conf,
                  agent_state=None,
                  traj=None,
                  mh: MetricsHandler = None):

        exp = agent_conf.config.experiment
        actor_net = agent_conf.actor_net
        critic_net = agent_conf.critic_net

        env = cls._wrap_env(env, exp)

        # ----- restore or init -----
        if agent_state is not None:
            actor_state = agent_state.actor_state.replace(apply_fn=actor_net.apply)
            critic_state = agent_state.critic_state.replace(apply_fn=critic_net.apply)
            target_critic_params = agent_state.target_critic_params
            target_critic_run_stats = agent_state.target_critic_run_stats
            extra_state = agent_state.extra_state
            replay_buffer = agent_state.replay_buffer
        else:
            rng, rng_a, rng_c, rng_e = jax.random.split(rng, 4)
            obs_dim = int(exp.obs_dim)
            action_dim = int(exp.action_dim)

            actor_params = actor_net.init(rng_a, jnp.zeros((1, obs_dim)))
            actor_state = TrainState.create(
                apply_fn=actor_net.apply,
                params=actor_params["params"],
                run_stats=actor_params.get("run_stats", {}),
                tx=agent_conf.actor_tx,
            )
            critic_params = critic_net.init(
                rng_c, jnp.zeros((1, obs_dim)), jnp.zeros((1, action_dim))
            )
            # Bundle every non-params variable collection (run_stats, batch_stats)
            # into `critic_state.run_stats` so subsequent apply / target-update
            # sites can plumb a single object regardless of whether BN is on.
            # `log_probs_collection` is a sow output, not long-lived state.
            _BUNDLE_EXCLUDE = {"params", "log_probs_collection"}
            critic_bundle = {k: v for k, v in critic_params.items() if k not in _BUNDLE_EXCLUDE}
            critic_state = TrainState.create(
                apply_fn=critic_net.apply,
                params=critic_params["params"],
                run_stats=critic_bundle,
                tx=agent_conf.critic_tx,
            )
            target_critic_params = critic_params["params"]
            target_critic_run_stats = critic_bundle
            extra_state = cls._init_extra_state(rng_e, exp)
            replay_buffer = ReplayBuffer.create(
                int(exp.obs_dim), int(exp.action_dim), int(exp.buffer_size)
            )

        # ----- env init -----
        if agent_state is not None and agent_state.env_state is not None:
            env_state = agent_state.env_state
            last_obs = agent_state.last_obs
        else:
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, exp.num_envs)
            last_obs, env_state = env.reset(reset_rng, traj)

        # ----- helpers / config -----
        learning_starts = int(getattr(exp, 'learning_starts', exp.batch_size))
        batch_size = int(exp.batch_size)
        tau = float(getattr(exp, 'tau', 0.005))
        gamma = float(exp.gamma)
        gradient_steps = int(getattr(exp, 'gradient_steps', 1))

        def _critic_forward_target(obs, action, params, run_stats):
            # `run_stats` is the full mutable-variables bundle for the critic
            # (run_stats + optional batch_stats). Splatting `**run_stats` lets
            # us handle both old (only run_stats) and BN (run_stats + batch_stats)
            # critics without conditional plumbing through every call site.
            mutables = list(run_stats.keys())
            (q1, q2), updates = critic_net.apply(
                {"params": params, **run_stats},
                obs, action, mutable=mutables, training=False,
            )
            new_bundle = {k: updates.get(k, run_stats[k]) for k in run_stats}
            return q1, q2, new_bundle

        # ----- collect transition -----
        def _collect_transition(actor_state, replay_buffer, env_state,
                                last_obs, rng, extra_state):
            rng, rng_act = jax.random.split(rng)
            action, actor_state = cls._select_action(
                actor_state, last_obs, rng_act, exp, extra_state, deterministic=False
            )
            next_obs, reward, absorbing, done, info, env_state = env.step(
                env_state, action, traj
            )
            replay_buffer = replay_buffer.add_batch(
                last_obs, next_obs, action, reward, done.astype(jnp.float32)
            )
            return actor_state, replay_buffer, env_state, next_obs, rng

        # ----- single gradient update -----
        def _single_gradient_update(carry, step_idx):
            (actor_st, critic_st, tgt_p, tgt_rs,
             ex_st, buf, rng_up) = carry

            rng_up, rng_sample = jax.random.split(rng_up)
            obs_b, nobs_b, act_b, rew_b, done_b = buf.sample(rng_sample, batch_size)

            # -- critic loss --
            rng_up, rng_next = jax.random.split(rng_up)
            next_action, next_q_bonus, _ = cls._next_action_and_q_bonus(
                actor_st, nobs_b, rng_next, exp, ex_st
            )
            q1_next, q2_next, new_tgt_rs = _critic_forward_target(
                nobs_b, next_action, tgt_p, tgt_rs
            )
            # Q-target aggregation across twins. Default is the standard TD3/SAC
            # `min(Q1, Q2)` (worst-case pessimism). If `pessimism_penalty` is set,
            # use Motivo-style ensemble pessimism: `mean(Q) - k * |Q1 - Q2|`,
            # which treats the inter-twin disagreement as an uncertainty estimate.
            pessimism_penalty = getattr(exp, "pessimism_penalty", None)
            if pessimism_penalty is None:
                q_next = jnp.minimum(q1_next, q2_next) + next_q_bonus
            else:
                k = float(pessimism_penalty)
                q_mean = 0.5 * (q1_next + q2_next)
                q_unc = jnp.abs(q1_next - q2_next)
                q_next = q_mean - k * q_unc + next_q_bonus
            q_target = rew_b + gamma * (1.0 - done_b) * q_next
            q_target = jax.lax.stop_gradient(q_target)

            def _critic_loss_fn(params):
                run_stats = critic_st.run_stats
                mutables = list(run_stats.keys())
                (q1, q2), updates = critic_net.apply(
                    {"params": params, **run_stats},
                    obs_b, act_b, mutable=mutables, training=False,
                )
                loss = jnp.mean((q1 - q_target) ** 2) + jnp.mean((q2 - q_target) ** 2)
                new_bundle = {k: updates.get(k, run_stats[k]) for k in run_stats}
                return loss, new_bundle

            (critic_loss, new_critic_rs), critic_grads = jax.value_and_grad(
                _critic_loss_fn, has_aux=True
            )(critic_st.params)
            critic_st = critic_st.apply_gradients(grads=critic_grads)
            critic_st = critic_st.replace(run_stats=new_critic_rs)

            # Post-step weight normalization (XQC / Salimans-Kingma).
            # When `use_weight_norm=True`, renormalize each Dense kernel after
            # the gradient step: W <- W / ||W||_axis=-2 (per-output-unit norm).
            # `normalize_last_layer=True` also renormalizes the predictor head.
            # No-op when the flag is unset, so SAC/TD3 baselines are unchanged.
            if bool(getattr(exp, 'use_weight_norm', False)):
                critic_st = critic_st.replace(
                    params=_normalize_dense_kernels(
                        critic_st.params,
                        normalize_last_layer=bool(getattr(exp, 'normalize_last_layer', True)),
                    )
                )

            # -- actor loss + extra update (delayed for TD3) --
            rng_up, rng_actor = jax.random.split(rng_up)
            do_actor_update = cls._should_update_actor(step_idx, exp)

            def _actor_loss_fn(params):
                return cls._actor_loss(
                    params, actor_st.apply_fn, actor_st.run_stats,
                    critic_st, critic_net.apply,
                    obs_b, rng_actor, exp, ex_st,
                )

            (actor_loss, actor_aux), actor_grads = jax.value_and_grad(
                _actor_loss_fn, has_aux=True
            )(actor_st.params)

            def _apply_actor(args):
                a_st, ex_st_in = args
                a_st = a_st.apply_gradients(grads=actor_grads)
                # update extra (e.g. alpha) only when we updated actor
                rng_inner = jax.random.fold_in(rng_actor, step_idx)
                new_ex_st, _ = cls._update_extra(ex_st_in, actor_aux, rng_inner, exp)
                return a_st, new_ex_st

            def _skip_actor(args):
                return args

            actor_st, ex_st = jax.lax.cond(
                do_actor_update, _apply_actor, _skip_actor, (actor_st, ex_st)
            )
            # if actor_run_stats produced inside loss aux, reapply (for SAC RunningMeanStd)
            new_actor_rs = actor_aux.get("run_stats", None) if isinstance(actor_aux, dict) else None
            if new_actor_rs is not None:
                actor_st = actor_st.replace(run_stats=new_actor_rs)

            # -- soft target update -- (params + the bundle of mutable
            # variables; jax.tree.map over an empty dict is a no-op, so the
            # bundle update is safe when BN/obs_norm are off).
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
            losses = (critic_loss, actor_loss)
            return new_carry, losses

        # ----- N gradient updates -----
        def _do_updates(actor_st, critic_st, tgt_p, tgt_rs, ex_st, buf, rng_up):
            carry = (actor_st, critic_st, tgt_p, tgt_rs, ex_st, buf, rng_up)
            carry, losses = jax.lax.scan(
                _single_gradient_update, carry, jnp.arange(gradient_steps)
            )
            actor_st, critic_st, tgt_p, tgt_rs, ex_st, _, _ = carry
            critic_loss = jnp.mean(losses[0])
            actor_loss = jnp.mean(losses[1])
            return (actor_st, critic_st, tgt_p, tgt_rs, ex_st,
                    critic_loss, actor_loss)

        def _skip_updates(actor_st, critic_st, tgt_p, tgt_rs, ex_st, buf, rng_up):
            return (actor_st, critic_st, tgt_p, tgt_rs, ex_st,
                    jnp.array(0.0), jnp.array(0.0))

        # ----- outer step -----
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
             ex_state, critic_loss, actor_loss) = result

            log_env_state = env_state.find(LogEnvState)
            logged_metrics = log_env_state.metrics
            # `returned_episode_returns` / `returned_episode_lengths` carry the
            # last-completed episode return/length for each env, held constant
            # until that env finishes another one. Averaging across envs gives
            # a smooth running estimate. (Slots are 0 until each env has
            # completed its first episode — biases the very early curve only.)
            base_kwargs = dict(
                mean_episode_return=jnp.mean(logged_metrics.returned_episode_returns),
                mean_episode_length=jnp.mean(logged_metrics.returned_episode_lengths),
                max_timestep=jnp.max(logged_metrics.timestep * exp.num_envs),
                mean_critic_loss=critic_loss,
                mean_actor_loss=actor_loss,
                buffer_size=replay_buffer.size,
            )
            rng, rng_metrics = jax.random.split(rng)
            extra = cls._extra_metrics(actor_state, next_obs, rng_metrics,
                                       ex_state, exp)
            metric = cls._build_summary_metric(base_kwargs, extra)

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

        # ----- main scan -----
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

        agent_state_out = cls._agent_state(
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

    # ---------- env wrap (subclasses can override) -----------------------
    @staticmethod
    def _wrap_env(env, config):
        env = LogWrapper(env)
        env = VecEnv(env)
        # Reward normalization no longer lives on the env wrapper; off-policy
        # algorithms would need to track stats on the agent state if desired.
        return env
