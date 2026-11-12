"""SuperMimic: whole-body tracking with one differentiable world model.

Follows "SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World
Model" (arXiv:2610.28480), Algorithm A.1:

  1. collect one vectorised env step with the deterministic policy plus Gaussian action noise,
  2. fit a world model on k-step windows of the replay buffer (autoregressive loss, Eq. 1),
  3. improve the policy by backpropagating through H-step imagined rollouts of the frozen
     world model (Eq. 2), with a TD(lambda) critic bootstrapping the tail (Eq. 3).

The paper's world model predicts the simulator state and computes observations and rewards
*analytically* from it through a differentiable forward kinematics of the G1. Here the same
role is played by the environment itself: an imagined step is an ordinary ``mjx_step`` in
which the simulation is replaced by the learned dynamics (``physics_fn``) followed by
``mjx.kinematics`` / ``com_pos`` / ``com_vel``. Observations (including observation history
and domain-randomisation noise), rewards, trajectory bookkeeping and absorbing checks are the
environment's own JAX functions, so the algorithm is agnostic to the robot, the observation
and the reward. Only kinematic quantities are recomputed: terms that read contacts
(``data.contact``) or actuator forces (``data.qfrc_actuator``) see the values of the
imagination start state for the whole horizon (with MjWarp: none at all) and pass no gradient.

World-model state: MuJoCo's own ``(qpos, qvel)``. The model must have a free root joint
followed by hinge/slide joints only (true for all floating-base robots in LocoMuJoCo). The
network sees heading-invariant features and predicts velocity increments plus position
corrections on top of explicit integration; its last layer is zero-initialised, so at init it
extrapolates at constant velocity.

Termination: the terminal state handler's ``mjx_absorbing_prob`` if it has one, else a learned
head on (a group of) the next observation, trained on the env's ``absorbing`` label. The
probability gives the soft survival weight of the imagined rewards, so the policy also gets a
gradient away from termination.

Domain randomisation: when the env randomises (any randomizer other than
``NoDomainRandomization``), the per-episode randomizer and control-function states are
flattened into a context vector ``xi`` that the dynamics and the critic receive. Otherwise
``xi`` is empty.

Requires the JAX MJX backend (``use_mjwarp=False``): MjWarp is not differentiable.
"""

import ast
import functools
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax import struct, traverse_util
from flax.linen.initializers import constant, orthogonal, variance_scaling, zeros
import mujoco
from mujoco import mjx
import optax
from omegaconf import DictConfig, OmegaConf, ListConfig, open_dict

from loco_mujoco.algorithms import JaxRLAlgorithmBase, AgentConfBase, AgentStateBase, TrainState
from loco_mujoco.algorithms.common.networks import get_activation_fn
from loco_mujoco.algorithms.experimental.td3_jax import TD3ActorNet
from loco_mujoco.core import MjxState
from loco_mujoco.core.domain_randomizer import NoDomainRandomization
from loco_mujoco.core.utils.flat_kinematics import FlatKinematics
from loco_mujoco.core.utils.math import take_columns
from loco_mujoco.core.wrappers import SummaryMetrics
from loco_mujoco.core.wrappers.mjx import BaseWrapper, VECENV_AXIS_NAME
from loco_mujoco.utils import ValidationSummary


# ---------------------------------------------------------------------------
# Quaternion helpers (MuJoCo convention: w, x, y, z)
# ---------------------------------------------------------------------------

def _safe_norm(v):
    # sqrt has an infinite derivative at 0; the epsilon keeps gradients finite at zero rotation.
    return jnp.sqrt(jnp.sum(v * v, axis=-1, keepdims=True) + 1e-12)


def _quat_mul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return jnp.stack([aw * bw - ax * bx - ay * by - az * bz,
                      aw * bx + ax * bw + ay * bz - az * by,
                      aw * by - ax * bz + ay * bw + az * bx,
                      aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def _quat_conj(q):
    return q * jnp.array([1.0, -1.0, -1.0, -1.0])


def _quat_rotate(q, v):
    qv, w = q[..., 1:], q[..., :1]
    t = 2.0 * jnp.cross(qv, v)
    return v + w * t + jnp.cross(qv, t)


def _quat_rotate_inv(q, v):
    return _quat_rotate(_quat_conj(q), v)


def _quat_normalize(q):
    return q / _safe_norm(q)


def _heading_quat(q):
    """Rotation about world z by the yaw of the body x-axis."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    yaw = jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    zero = jnp.zeros_like(yaw)
    return jnp.stack([jnp.cos(0.5 * yaw), zero, zero, jnp.sin(0.5 * yaw)], axis=-1)


def _quat_from_rotvec(v):
    th = _safe_norm(v)
    return jnp.concatenate([jnp.cos(0.5 * th), v * (jnp.sin(0.5 * th) / th)], axis=-1)


def _rotvec_from_quat(q):
    q = jnp.where(q[..., :1] < 0.0, -q, q)
    s = _safe_norm(q[..., 1:])
    angle = 2.0 * jnp.arctan2(s, q[..., :1])
    return q[..., 1:] * (angle / s)


def _rot6d_from_quat(q):
    """First two columns of the rotation matrix (continuous rotation representation)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    c0 = jnp.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], axis=-1)
    c1 = jnp.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], axis=-1)
    return jnp.concatenate([c0, c1], axis=-1)


# ---------------------------------------------------------------------------
# World-model state representation
# ---------------------------------------------------------------------------

class WorldModelSpec:
    """Feature map, integration step and error metric of the world model on (qpos, qvel).

    qpos = [root_pos (3), root_quat (4), joint_pos (nj)],  qvel = [root_lin_vel_w (3),
    root_ang_vel_b (3), joint_vel (nj)]  (MuJoCo free-joint layout).

    Features (heading-invariant):  [root_z, rot6d(heading^-1 * root_quat), lin_vel in heading
    frame, ang_vel_b, joint_pos, joint_vel] / in_scale.

    Network output delta = [dv_h (3), dw_b (3), dqd (nj), dp_h (3), dth_b (3), dq (nj)] * out_scale:
        v' = v + R_h dv_h,   w' = w + dw_b,   qd' = qd + dqd
        p' = p + dt v' + R_h dp_h,   q' = q * exp(dt w' + dth_b),   qj' = qj + dt qd' + dq
    """

    def __init__(self, nj, dt, in_scale, out_scale, err_scale, drift=1.0):
        self.nj = int(nj)
        self.dt = float(dt)
        self.drift = float(drift)  # weight of the new velocity in the position update
        rep = lambda scales, sizes: jnp.concatenate(  # noqa: E731
            [jnp.full((n,), float(s)) for s, n in zip(scales, sizes)])
        self.in_scale = rep(in_scale, (1, 6, 3, 3, self.nj, self.nj))
        self.out_scale = rep(out_scale, (3, 3, self.nj, 3, 3, self.nj))
        self.err_scale = rep(err_scale, (3, 3, 3, 3, self.nj, self.nj))

    @property
    def feat_dim(self):
        return 13 + 2 * self.nj

    @property
    def delta_dim(self):
        return 12 + 2 * self.nj

    @staticmethod
    def _split(qpos, qvel):
        return qpos[..., 0:3], _quat_normalize(qpos[..., 3:7]), qpos[..., 7:], \
            qvel[..., 0:3], qvel[..., 3:6], qvel[..., 6:]

    def features(self, qpos, qvel):
        p, q, qj, v, w, qd = self._split(qpos, qvel)
        hq = _heading_quat(q)
        q_local = _quat_mul(_quat_conj(hq), q)
        feat = jnp.concatenate([p[..., 2:3], _rot6d_from_quat(q_local), _quat_rotate_inv(hq, v),
                                w, qj, qd], axis=-1)
        return feat / self.in_scale

    def integrate(self, qpos, qvel, delta):
        nj = self.nj
        d = delta * self.out_scale
        dv_h, dw_b, dqd = d[..., 0:3], d[..., 3:6], d[..., 6:6 + nj]
        dp_h, dth_b, dq = d[..., 6 + nj:9 + nj], d[..., 9 + nj:12 + nj], d[..., 12 + nj:]
        p, q, qj, v, w, qd = self._split(qpos, qvel)
        hq = _heading_quat(q)
        v_new = v + _quat_rotate(hq, dv_h)
        w_new = w + dw_b
        qd_new = qd + dqd
        c_old, c_new = self.dt * (1.0 - self.drift), self.dt * self.drift
        p_new = p + c_old * v + c_new * v_new + _quat_rotate(hq, dp_h)
        q_new = _quat_normalize(_quat_mul(q, _quat_from_rotvec(c_old * w + c_new * w_new + dth_b)))
        qj_new = qj + c_old * qd + c_new * qd_new + dq
        return (jnp.concatenate([p_new, q_new, qj_new], axis=-1),
                jnp.concatenate([v_new, w_new, qd_new], axis=-1))

    def targets(self, qpos, qvel, qpos_next, qvel_next):
        """The network output that makes ``integrate`` go from one state to the next."""
        p, q, qj, v, w, qd = self._split(qpos, qvel)
        p1, q1, qj1, v1, w1, qd1 = self._split(qpos_next, qvel_next)
        hq = _heading_quat(q)
        c_old, c_new = self.dt * (1.0 - self.drift), self.dt * self.drift
        delta = jnp.concatenate([_quat_rotate_inv(hq, v1 - v), w1 - w, qd1 - qd,
                                 _quat_rotate_inv(hq, p1 - p - c_old * v - c_new * v1),
                                 _rotvec_from_quat(_quat_mul(_quat_conj(q), q1)) - c_old * w - c_new * w1,
                                 qj1 - qj - c_old * qd - c_new * qd1], axis=-1)
        return delta / self.out_scale

    def error(self, qpos_pred, qvel_pred, qpos_true, qvel_true):
        """Normalised error vector [dp, rotvec, dv, dw, dqj, dqd] / err_scale."""
        p1, q1, qj1, v1, w1, qd1 = self._split(qpos_pred, qvel_pred)
        p2, q2, qj2, v2, w2, qd2 = self._split(qpos_true, qvel_true)
        rot = _rotvec_from_quat(_quat_mul(_quat_conj(q2), q1))
        err = jnp.concatenate([p1 - p2, rot, v1 - v2, w1 - w2, qj1 - qj2, qd1 - qd2], axis=-1)
        return err / self.err_scale


def check_floating_base(model):
    """Raise unless qpos/qvel are [free root joint | hinge/slide joints]."""
    if model.njnt == 0 or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or model.jnt_qposadr[0] != 0:
        raise ValueError("SuperMimic needs a free root joint as the first joint of the model.")
    others = model.jnt_type[1:]
    ok = np.isin(others, (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE))
    if not np.all(ok):
        raise ValueError("SuperMimic supports only hinge/slide joints besides the free root joint.")
    if model.nq - 7 != model.nv - 6:
        raise ValueError(f"Unexpected joint layout: nq={model.nq}, nv={model.nv}.")
    return model.nq - 7


# ---------------------------------------------------------------------------
# Domain-randomisation context
# ---------------------------------------------------------------------------

def uses_domain_randomization(env):
    return not isinstance(env._domain_randomizer, NoDomainRandomization)


def context_from_carry(carry, enabled, index=None):
    """Flatten the per-episode randomizer and control-function states into one vector, or its entries ``index``."""
    if not enabled:
        return jnp.zeros((0,))
    leaves = jax.tree.leaves((carry.domain_randomizer_state, carry.control_func_state))
    leaves = [jnp.ravel(jnp.asarray(x)) for x in leaves]
    leaves = [x.astype(jnp.float32) for x in leaves if jnp.issubdtype(x.dtype, jnp.floating)]
    xi = jnp.concatenate(leaves) if leaves else jnp.zeros((0,))
    return xi if index is None else xi[jnp.array(index, dtype=int)]


def context_index(env, n_samples=32):
    """Entries of the context that differ between episodes: the others tell the networks nothing."""
    if not uses_domain_randomization(env):
        return []
    @jax.jit  # compiled: op by op, every primitive of the reset is its own compilation
    def contexts(keys):
        states = jax.vmap(env.mjx_reset, axis_name=VECENV_AXIS_NAME)(keys)
        return jax.vmap(lambda c: context_from_carry(c, True))(states.additional_carry)

    xi = contexts(jax.random.split(jax.random.PRNGKey(0), n_samples))
    return np.flatnonzero(np.asarray(jnp.std(xi, axis=0)) > 1e-6).tolist()


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    hidden_layer_dims: tuple
    out_dim: int
    activation: str = "silu"
    zero_last: bool = False
    compute_dtype: str = "float32"  # matmul dtype; the parameters and the output stay float32
    layer_norm: bool = False
    last_gain: float = 1.0
    torch_init: bool = False  # torch.nn.Linear's init: a third of the default variance, uniform biases

    @nn.compact
    def __call__(self, x):
        act = get_activation_fn(self.activation)

        def init(fan_in):
            if not self.torch_init:
                return {}
            bound = fan_in ** -0.5
            return dict(kernel_init=variance_scaling(1 / 3, "fan_in", "uniform"),
                        bias_init=lambda key, shape, dtype=jnp.float32: jax.random.uniform(key, shape, dtype, -bound, bound))

        for dim in self.hidden_layer_dims:
            x = nn.Dense(dim, dtype=self.compute_dtype, **init(x.shape[-1]))(x)
            if self.layer_norm:
                x = nn.LayerNorm(dtype=jnp.float32)(x)
            x = act(x)
        last = dict(kernel_init=zeros if self.zero_last else orthogonal(self.last_gain), bias_init=constant(0.0))
        if self.torch_init and not self.zero_last:
            kernel = variance_scaling(self.last_gain ** 2 / 3, "fan_in", "uniform")
            last = {**init(x.shape[-1]), "kernel_init": kernel}
            if self.last_gain != 1.0:
                last["bias_init"] = constant(0.0)
        x = nn.Dense(self.out_dim, dtype=self.compute_dtype, **last)(x)
        return x.astype(jnp.float32)


class SuperMimicWorldModelNet(nn.Module):
    """Dynamics MLP (zero-initialised output) and termination head."""
    delta_dim: int
    hidden_layer_dims: tuple = (2048, 2048, 2048)
    term_hidden_layer_dims: tuple = (64, 64)
    activation: str = "silu"
    compute_dtype: str = "float32"
    layer_norm: bool = False
    torch_init: bool = False

    def setup(self):
        self.dynamics_net = _MLP(self.hidden_layer_dims, self.delta_dim, self.activation, zero_last=True,
                                 compute_dtype=self.compute_dtype, layer_norm=self.layer_norm,
                                 torch_init=self.torch_init)
        self.termination_net = _MLP(self.term_hidden_layer_dims, 1, self.activation, compute_dtype=self.compute_dtype)

    def __call__(self, feat, action, xi, term_obs):
        return self.dynamics(feat, action, xi), self.termination(term_obs)

    def dynamics(self, feat, action, xi, norm=None):
        x = jnp.concatenate([feat, action, xi], axis=-1)
        return self.dynamics_net(x if norm is None else norm(x))

    def termination(self, term_obs):
        return self.termination_net(term_obs)[..., 0]


class SuperMimicPolicyNet(nn.Module):
    """a = dec(tanh(enc(goal)), history): the encoder sees the reference only, the decoder the recent observations."""
    action_dim: int
    enc_hidden_dims: tuple = (2048, 1024, 512, 256)
    dec_hidden_dims: tuple = (2048, 1024, 512, 256)
    latent_dim: int = 32
    activation: str = "elu"
    compute_dtype: str = "float32"
    torch_init: bool = False

    @nn.compact
    def __call__(self, goal, history):
        z = _MLP(self.enc_hidden_dims, self.latent_dim, self.activation, compute_dtype=self.compute_dtype,
                 torch_init=self.torch_init, name="enc")(goal)
        dec = _MLP(self.dec_hidden_dims, self.action_dim, self.activation, compute_dtype=self.compute_dtype,
                   last_gain=0.01, torch_init=self.torch_init, name="dec")
        return dec(jnp.concatenate([jnp.tanh(z), history], axis=-1))


@functools.lru_cache(maxsize=None)
def _slim_layer(activation, dtype, layer_norm):
    """
    Dense (+ LayerNorm) + activation of ``_MLP`` with a hand-written backward pass that keeps one array per layer
    (the output, or the normalized pre-activation) instead of every intermediate of the activation and the norm.

    """
    assert (activation, layer_norm) in (("elu", False), ("relu", False), ("tanh", False), ("silu", True)), \
        f"imag_compact has no backward pass for {activation} with layer_norm={layer_norm}"
    act = get_activation_fn(activation)

    def pre_activation(x, dense):
        kernel = dense["kernel"].astype(dtype)
        return x @ kernel + dense["bias"].astype(dtype), kernel

    def dense_bwd(x, kernel, d_pre):
        return d_pre @ kernel.T, {"kernel": (x.T @ d_pre).astype(jnp.float32),
                                  "bias": jnp.sum(d_pre.astype(jnp.float32), axis=0)}

    def fwd(x, dense, norm):
        pre, kernel = pre_activation(x, dense)
        if not layer_norm:
            y = act(pre)
            return y, (x, kernel, y)
        pre = pre.astype(jnp.float32)
        mean = jnp.mean(pre, axis=-1, keepdims=True)
        var = jnp.maximum(jnp.mean(pre * pre, axis=-1, keepdims=True) - mean * mean, 0.0)
        inv_std = jax.lax.rsqrt(var + 1e-6)
        unit = (pre - mean) * inv_std
        return act(unit * norm["scale"] + norm["bias"]), (x, kernel, unit, inv_std, norm["scale"], norm["bias"])

    def bwd(res, ct):
        if not layer_norm:
            x, kernel, y = res
            slope = {"elu": lambda: jnp.where(y > 0, 1.0, y + 1.0), "relu": lambda: y > 0,
                     "tanh": lambda: 1.0 - y * y}[activation]()
            dx, d_dense = dense_bwd(x, kernel, ct * slope.astype(ct.dtype))
            return dx, d_dense, None
        x, kernel, unit, inv_std, scale, bias = res
        z = unit * scale + bias
        sig = jax.nn.sigmoid(z)
        dz = ct * (sig * (1.0 + z * (1.0 - sig)))
        d_unit = dz * scale
        d_pre = inv_std * (d_unit - jnp.mean(d_unit, axis=-1, keepdims=True)
                           - unit * jnp.mean(d_unit * unit, axis=-1, keepdims=True))
        dx, d_dense = dense_bwd(x, kernel, d_pre.astype(dtype))
        return dx, d_dense, {"scale": jnp.sum(dz * unit, axis=0), "bias": jnp.sum(dz, axis=0)}

    layer = jax.custom_vjp(lambda x, dense, norm: fwd(x, dense, norm)[0])
    layer.defvjp(fwd, bwd)
    return layer


def slim_mlp_apply(layers, x, activation, dtype):
    """``_MLP`` on a batch ``x`` (B, D) from its parameters, with ``_slim_layer`` for the hidden layers."""
    n = sum(k.startswith("Dense_") for k in layers)
    for i in range(n - 1):
        norm = layers.get(f"LayerNorm_{i}")
        x = _slim_layer(activation, dtype, norm is not None)(x.astype(dtype), layers[f"Dense_{i}"], norm)
    last = layers[f"Dense_{n - 1}"]
    return (x.astype(dtype) @ last["kernel"].astype(dtype) + last["bias"].astype(dtype)).astype(jnp.float32)


# smallest error scale of the world-model loss, per entry of WorldModelSpec.error: [p, rot, v, w] then [qj, qd]
WM_ERR_FLOOR = ((3e-3, 5e-3, 2e-2, 5e-2), (3e-3, 5e-2))


@struct.dataclass
class RunningNorm:
    """Running mean and variance of the columns of x, frozen after ``max_count`` rows."""
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray

    @classmethod
    def create(cls, dim):
        return cls(mean=jnp.zeros(dim), var=jnp.ones(dim), count=jnp.zeros(()))

    def update(self, x, max_count=2e8):
        # centred on the running mean: both moments come from one pass over x
        x = x.reshape(-1, x.shape[-1]) - self.mean
        n = x.shape[0]
        delta = jnp.mean(x, axis=0)
        v = jnp.maximum(jnp.mean(x * x, axis=0) - delta ** 2, 0.0)
        total = self.count + n
        live = self.count < max_count
        return RunningNorm(mean=jnp.where(live, self.mean + delta * n / total, self.mean),
                           var=jnp.where(live, (self.var * self.count + v * n + delta ** 2 * self.count * n / total) / total,
                                         self.var),
                           count=jnp.where(live, total, self.count))

    def std(self, eps=1e-4):
        return jnp.sqrt(self.var + eps ** 2)

    def __call__(self, x, index=None, eps=1e-2):
        mean, var = (self.mean, self.var) if index is None else (self.mean[index], self.var[index])
        return (x - mean) / jnp.sqrt(var + eps ** 2)


class SuperMimicValueNet(nn.Module):
    """V(obs[obs_ind], xi)."""
    hidden_layer_dims: tuple = (256, 128)
    activation: str = "silu"
    obs_ind: jnp.ndarray = None
    compute_dtype: str = "float32"
    torch_init: bool = False

    @nn.compact
    def __call__(self, obs, xi):
        if self.obs_ind is not None:
            obs = obs[..., self.obs_ind]
        net = _MLP(self.hidden_layer_dims, 1, self.activation, compute_dtype=self.compute_dtype,
                   torch_init=self.torch_init)
        return net(jnp.concatenate([obs, xi], axis=-1))[..., 0]


# ---------------------------------------------------------------------------
# Buffers
# ---------------------------------------------------------------------------

def state_is_valid(qpos, qvel, max_abs_qvel):
    """False for diverged simulator states (non-finite, or velocities beyond any physical motion).
    MJX occasionally blows up an env before its episode ends; such states must not reach the world
    model or seed imagined rollouts."""
    return (jnp.all(jnp.isfinite(qpos), axis=-1) & jnp.all(jnp.isfinite(qvel), axis=-1)
            & (jnp.max(jnp.abs(jnp.nan_to_num(qvel, nan=jnp.inf)), axis=-1) < max_abs_qvel))


def _zero_where_invalid(valid, x):
    return jnp.where(valid.reshape(valid.shape + (1,) * (x.ndim - valid.ndim)), x, jnp.zeros_like(x))


@struct.dataclass
class WindowBuffer:
    """Time-major [T, E] ring of transitions for world-model fitting. All envs step together, so
    one write pointer serves every env. ``next_qpos`` / ``next_qvel`` hold the true successor
    state also when the episode ended there (before the auto-reset). Invalid transitions (see
    ``state_is_valid``) are stored zeroed with ``valid=False``."""
    qpos: jnp.ndarray
    qvel: jnp.ndarray
    next_qpos: jnp.ndarray
    next_qvel: jnp.ndarray
    action: jnp.ndarray
    xi: jnp.ndarray
    next_term_obs: jnp.ndarray
    absorbing: jnp.ndarray
    done: jnp.ndarray
    valid: jnp.ndarray
    ptr: jnp.ndarray
    count: jnp.ndarray

    @classmethod
    def create(cls, length, n_envs, nq, nv, action_dim, xi_dim, term_obs_dim):
        f = lambda *shape: jnp.zeros((length, n_envs) + shape)  # noqa: E731
        return cls(qpos=f(nq), qvel=f(nv), next_qpos=f(nq), next_qvel=f(nv), action=f(action_dim),
                   xi=f(xi_dim), next_term_obs=f(term_obs_dim), absorbing=f(),
                   done=jnp.zeros((length, n_envs), dtype=bool), valid=jnp.zeros((length, n_envs), dtype=bool),
                   ptr=jnp.array(0, dtype=jnp.int32), count=jnp.array(0, dtype=jnp.int32))

    def add(self, valid, **rows):
        new = {k: getattr(self, k).at[self.ptr].set(_zero_where_invalid(valid, v)) for k, v in rows.items()}
        length = self.qpos.shape[0]
        return self.replace(**new, valid=self.valid.at[self.ptr].set(valid), ptr=(self.ptr + 1) % length,
                            count=jnp.minimum(self.count + 1, length))

    def sample_windows(self, rng, n, k):
        """n windows of k consecutive transitions. Returns (k, n, ...) arrays and an (n,) mask that
        is False for windows with an episode boundary before their last transition or with an
        invalid transition; their entries are zeroed."""
        length, n_envs = self.qpos.shape[:2]
        rng_t, rng_e = jax.random.split(rng)
        # age 0 is the newest row; a window starting at age a covers ages a .. a-k+1
        age = jax.random.randint(rng_t, (n,), k - 1, jnp.maximum(self.count, k))
        t0 = (self.ptr - 1 - age) % length
        e = jax.random.randint(rng_e, (n,), 0, n_envs)
        tt = (t0[None, :] + jnp.arange(k)[:, None]) % length
        ee = jnp.broadcast_to(e[None, :], (k, n))
        mask = jnp.all(~self.done[tt[:-1], ee[:-1]], axis=0) & jnp.all(self.valid[tt, ee], axis=0)
        get = lambda x: _zero_where_invalid(jnp.broadcast_to(mask, (k, n)), x[tt, ee])  # noqa: E731
        return dict(qpos=get(self.qpos), qvel=get(self.qvel), next_qpos=get(self.next_qpos),
                    next_qvel=get(self.next_qvel), action=get(self.action),
                    xi=_zero_where_invalid(mask, self.xi[t0, e]),
                    next_term_obs=get(self.next_term_obs), absorbing=get(self.absorbing)), mask


@struct.dataclass
class StartState:
    """What an imagined rollout needs from a real state (``data`` is rebuilt from qpos/qvel)."""
    qpos: jnp.ndarray
    qvel: jnp.ndarray
    observation: jnp.ndarray
    info: Any
    carry: Any
    history: Any = None


@struct.dataclass
class StartBuffer:
    """[S, E] ring of the most recent full env states, the start states of imagination. Only valid
    states (see ``state_is_valid``) are sampled."""
    states: StartState
    valid: jnp.ndarray
    ptr: jnp.ndarray
    count: jnp.ndarray

    @classmethod
    def create(cls, length, env_state, history=None):
        start = _start_state(env_state).replace(history=history)
        states = jax.tree.map(lambda x: jnp.zeros((length,) + x.shape, x.dtype), start)
        valid = jnp.zeros((length, start.qpos.shape[0]), dtype=bool)
        return cls(states=states, valid=valid, ptr=jnp.array(0, dtype=jnp.int32), count=jnp.array(0, dtype=jnp.int32))

    def add(self, env_state, valid, advance=True, history=None):
        """Writes the current row. With ``advance`` False the next add overwrites it, so a ring of S rows
        advanced every n steps spans S * n steps."""
        start = _start_state(env_state).replace(history=history)
        states = jax.tree.map(lambda b, x: b.at[self.ptr].set(x), self.states, start)
        length = self.states.qpos.shape[0]
        step = jnp.asarray(advance, dtype=jnp.int32)
        return self.replace(states=states, valid=self.valid.at[self.ptr].set(valid),
                            ptr=(self.ptr + step) % length, count=jnp.minimum(self.count + step, length))

    def sample(self, rng, n):
        # uniform over the valid entries; categorical would draw n noise values per buffer entry
        rank = jnp.cumsum(self.valid.reshape(-1))
        idx = jnp.searchsorted(rank, jax.random.randint(rng, (n,), 0, jnp.maximum(rank[-1], 1)), side="right")
        idx = jnp.minimum(idx, rank.shape[0] - 1)
        n_envs = self.valid.shape[1]
        return jax.tree.map(lambda x: x[idx // n_envs, idx % n_envs], self.states)


def _start_state(env_state):
    return StartState(qpos=env_state.data.qpos, qvel=env_state.data.qvel,
                      observation=env_state.observation, info=env_state.info,
                      carry=env_state.additional_carry)


# ---------------------------------------------------------------------------
# Summary metrics / conf / state
# ---------------------------------------------------------------------------

@struct.dataclass
class SuperMimicSummaryMetrics(SummaryMetrics):
    mean_wm_loss: float = 0.0
    mean_wm_nse_1step: float = 0.0
    mean_wm_nse_kstep: float = 0.0
    mean_term_bce: float = 0.0
    mean_term_acc: float = 0.0
    mean_imag_return: float = 0.0
    mean_imag_survival: float = 0.0
    mean_imag_alive: float = 0.0
    mean_critic_loss: float = 0.0
    mean_actor_grad_norm: float = 0.0
    mean_action_norm: float = 0.0
    num_updates: int = 0


class _Packed:
    """Packs the leaves of a pytree into one flat array per dtype, and back.

    A scan whose carry and outputs are packed writes one or two arrays per step instead of one per leaf. XLA cannot
    put these per-step writes into CUDA graphs, so each of them is a separate kernel launch.
    """

    def __init__(self, tree):
        leaves, self.treedef = jax.tree.flatten(tree)
        self.shapes = [tuple(l.shape) for l in leaves]
        self.groups, self.rest = {}, []
        for i, l in enumerate(leaves):
            if jax.dtypes.issubdtype(l.dtype, jax.dtypes.extended):  # typed PRNG keys cannot be concatenated
                self.rest.append(i)
            else:
                self.groups.setdefault(jnp.dtype(l.dtype), []).append(i)

    def pack(self, tree):
        leaves = jax.tree.leaves(tree)
        flat = tuple(jnp.concatenate([leaves[i].reshape(-1) for i in idx]) for idx in self.groups.values())
        return flat, tuple(leaves[i] for i in self.rest)

    def unpack(self, packed, lead=()):
        """``lead``: the leading axes a scan stacked onto the packed arrays."""
        flat, rest = packed
        leaves = [None] * len(self.shapes)
        for f, idx in zip(flat, self.groups.values()):
            off = 0
            for i in idx:
                size = int(np.prod(self.shapes[i], dtype=np.int64))
                leaves[i] = f[..., off:off + size].reshape(tuple(lead) + self.shapes[i])
                off += size
        for l, i in zip(rest, self.rest):
            leaves[i] = l
        return jax.tree.unflatten(self.treedef, leaves)


def _tuple(x):
    return tuple(x if isinstance(x, (list, tuple, ListConfig)) else ast.literal_eval(x))


def scale_by_muon(momentum=0.95, ns_steps=5, eps=1e-7):
    """Muon (Jordan et al. 2024) on 2D parameters, scaled to Adam's update RMS (Liu et al. 2025), so the
    Adam learning rate carries over. Nesterov momentum, Newton-Schulz in bfloat16."""
    a, b, c = 3.4445, -4.775, 2.0315

    def orthogonalize(g):
        x = g.astype(jnp.bfloat16)
        wide = g.shape[0] > g.shape[1]
        x = x.T if wide else x
        x = x / (jnp.linalg.norm(x.astype(jnp.float32)).astype(jnp.bfloat16) + eps)
        for _ in range(ns_steps):
            gram = x @ x.T
            x = a * x + (b * gram + c * gram @ gram) @ x
        x = x.T if wide else x
        return x.astype(g.dtype) * (0.2 * max(g.shape) ** 0.5)

    def init_fn(params):
        return jax.tree.map(jnp.zeros_like, params)

    def update_fn(updates, state, params=None):
        del params
        state = jax.tree.map(lambda m, g: momentum * m + (1 - momentum) * g, state, updates)
        return jax.tree.map(lambda m, g: orthogonalize((1 - momentum) * g + momentum * m), state, updates), state

    return optax.GradientTransformation(init_fn, update_fn)


def muon_labels(params, root=None):
    """"muon" for the hidden kernels of each MLP (under ``root`` if given), "adam" for the first and last
    layers and all vectors."""
    flat = traverse_util.flatten_dict(params)
    dense = {}
    for path in flat:
        if path[-1] == "kernel" and path[-2].startswith("Dense_") and (root is None or root in path):
            dense.setdefault(path[:-2], []).append(int(path[-2].split("_")[1]))
    labels = {path: "muon" if path[-1] == "kernel" and path[:-2] in dense
              and min(dense[path[:-2]]) < int(path[-2].split("_")[1]) < max(dense[path[:-2]]) else "adam"
              for path in flat}
    return traverse_util.unflatten_dict(labels)


def _build_nets(env, config):
    """Networks and optimisers from the config. Needs ``config.experiment`` to hold the dims."""
    exp = config.experiment
    compute_dtype = str(getattr(exp, "net_dtype", "float32"))
    torch_init = bool(getattr(exp, "torch_init", False))
    if int(getattr(exp, "history_len", 0)) > 0:
        actor_net = SuperMimicPolicyNet(
            action_dim=int(exp.action_dim),
            enc_hidden_dims=_tuple(exp.hidden_layers),
            dec_hidden_dims=_tuple(exp.hidden_layers),
            activation=str(exp.activation),
            compute_dtype=compute_dtype,
            torch_init=torch_init,
        )
    else:
        actor_net = TD3ActorNet(
            action_dim=int(exp.action_dim),
            hidden_layer_dims=_tuple(exp.hidden_layers),
            activation=str(exp.activation),
            use_obs_norm=False,
            output_squash="none",
            obs_ind=_obs_ind(env, getattr(exp, "actor_obs_group", None), "actor_obs_group"),
            compute_dtype=compute_dtype,
        )
    critic_net = SuperMimicValueNet(
        hidden_layer_dims=_tuple(exp.critic_hidden_layers),
        activation=str(exp.activation),
        obs_ind=_obs_ind(env, getattr(exp, "critic_obs_group", None), "critic_obs_group"),
        compute_dtype=compute_dtype,
        torch_init=torch_init,
    )
    wm_net = SuperMimicWorldModelNet(
        delta_dim=12 + 2 * int(exp.n_joints),
        hidden_layer_dims=_tuple(exp.wm_hidden_layers),
        term_hidden_layer_dims=_tuple(exp.wm_term_hidden_layers),
        activation=str(exp.wm_activation),
        compute_dtype=compute_dtype,
        layer_norm=bool(getattr(exp, "wm_layer_norm", False)),
        torch_init=torch_init,
    )

    # Cosine decay to lr_final_ratio * lr over the first lr_decay_fraction of training (paper A.5).
    n_updates = max(1, int(exp.num_iterations * float(exp.updates_per_env_step)))
    frac = float(exp.lr_decay_fraction)
    alpha = float(exp.lr_final_ratio)
    sched = lambda lr, steps: optax.cosine_decay_schedule(float(lr), max(1, int(frac * steps)), alpha)  # noqa: E731
    # A non-finite gradient skips its step rather than turning the parameters into NaN for good.
    guard = lambda tx: apply_if_finite_select(tx, max_consecutive_errors=10 ** 9)  # noqa: E731

    def optimiser(name, lr, labels, **adam):
        if name == "adam":
            return optax.adam(lr, **adam)
        muon = optax.chain(scale_by_muon(), optax.scale_by_learning_rate(lr))
        return optax.multi_transform({"muon": muon, "adam": optax.adam(lr, **adam)}, labels)

    actor_tx = optax.chain(optax.clip_by_global_norm(float(exp.actor_max_grad_norm)),
                           optax.clip(float(exp.actor_max_grad_value)),
                           optimiser(str(getattr(exp, "actor_optimizer", "adam")), sched(exp.lr_actor, n_updates),
                                     muon_labels, b1=0.9, b2=0.95))
    spike_factor = float(getattr(exp, "actor_grad_spike_factor", 0.0) or 0.0)
    if spike_factor > 0.0:
        actor_tx = skip_gradient_spikes(actor_tx, spike_factor)
    actor_tx = guard(actor_tx)
    n_critic = n_updates * int(exp.critic_epochs) * int(exp.critic_minibatches)
    critic_tx = guard(optax.chain(optax.clip_by_global_norm(float(exp.critic_max_grad_norm)),
                                  optax.adam(sched(exp.lr_critic, n_critic))))
    n_wm = n_updates * int(exp.wm_batches_per_update)
    wm_tx = guard(optax.chain(optax.clip_by_global_norm(float(exp.wm_max_grad_norm)),
                              optimiser(str(getattr(exp, "wm_optimizer", "adam")), sched(exp.lr_wm, n_wm),
                                        lambda p: muon_labels(p, "dynamics_net"))))
    return actor_net, critic_net, wm_net, actor_tx, critic_tx, wm_tx


def apply_if_finite_select(inner, max_consecutive_errors):
    """``optax.apply_if_finite`` with a select instead of ``lax.cond``.

    On the GPU, XLA runs a ``lax.cond`` from the host: the host waits for the predicate, so every optimizer step
    drains the GPU's queue of launched kernels. Here the inner update always runs and the rejected result is
    discarded, which gives the same updates and the same state (``optax.ApplyIfFiniteState``).
    """
    inner = optax.with_extra_args_support(inner)
    increment = lambda x: x + (x < jnp.iinfo(jnp.int32).max).astype(jnp.int32)  # noqa: E731

    def init_fn(params):
        return optax.ApplyIfFiniteState(notfinite_count=jnp.zeros([], jnp.int32), last_finite=jnp.array(True),
                                        total_notfinite=jnp.zeros([], jnp.int32), inner_state=inner.init(params))

    def update_fn(updates, state, params=None, **extra_args):
        isfinite = jnp.all(jnp.array([jnp.all(jnp.isfinite(u)) for u in jax.tree.leaves(updates)]))
        notfinite_count = jnp.where(isfinite, jnp.zeros([], jnp.int32), increment(state.notfinite_count))
        apply = isfinite | (notfinite_count > max_consecutive_errors)
        new_updates, new_inner = inner.update(updates, state.inner_state, params, **extra_args)
        new_updates = jax.tree.map(lambda u: jnp.where(apply, u, jnp.zeros_like(u)), new_updates)
        new_inner = jax.tree.map(lambda n, o: jnp.where(apply, n, o), new_inner, state.inner_state)
        return new_updates, optax.ApplyIfFiniteState(
            notfinite_count=notfinite_count, last_finite=isfinite,
            total_notfinite=jnp.where(isfinite, state.total_notfinite, increment(state.total_notfinite)),
            inner_state=new_inner)

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


class SpikeGuardState(NamedTuple):
    inner_state: Any
    mean_norm: jnp.ndarray
    count: jnp.ndarray
    skipped: jnp.ndarray


def skip_gradient_spikes(inner, factor, warmup=200, decay=0.99):
    """Skips updates whose gradient norm exceeds `factor` times its running mean.

    Backpropagating through a long rollout occasionally explodes. Clipping bounds the size of such a
    step but not its direction, and a run of them can destroy a good policy. The running mean takes
    skipped gradients in at their limit, so a real rise of the norm is followed rather than blocked.
    """

    def init_fn(params):
        return SpikeGuardState(inner.init(params), jnp.zeros((), jnp.float32), jnp.zeros((), jnp.int32),
                               jnp.zeros((), jnp.int32))

    def update_fn(grads, state, params=None):
        norm = optax.global_norm(grads)
        spike = (state.count >= warmup) & (norm > factor * state.mean_norm)
        updates, inner_state = inner.update(grads, state.inner_state, params)
        updates = jax.tree.map(lambda u: jnp.where(spike, jnp.zeros_like(u), u), updates)
        inner_state = jax.tree.map(lambda old, new: jnp.where(spike, old, new), state.inner_state, inner_state)
        seen = jnp.where(state.count >= warmup, jnp.minimum(norm, factor * state.mean_norm), norm)
        rate = jnp.maximum(1.0 - decay, 1.0 / (state.count + 1.0))
        mean_norm = state.mean_norm + rate * (seen - state.mean_norm)
        return updates, SpikeGuardState(inner_state, mean_norm, state.count + 1,
                                        state.skipped + spike.astype(jnp.int32))

    return optax.GradientTransformation(init_fn, update_fn)


def _obs_ind(env, group, what):
    if group is None:
        return None
    ind = jnp.asarray(env.obs_container.get_obs_ind_by_group(group), dtype=jnp.int32)
    if ind.size == 0:
        raise ValueError(f"{what}={group!r} selected no observations. Check the group name(s) "
                         f"against the observation spec.")
    return ind


@dataclass(frozen=True)
class SuperMimicAgentConf(AgentConfBase):
    config: DictConfig
    actor_net: TD3ActorNet
    critic_net: SuperMimicValueNet
    wm_net: SuperMimicWorldModelNet
    actor_tx: Any
    critic_tx: Any
    wm_tx: Any
    term_obs_ind: Any = None
    goal_ind: Any = None  # with history_len > 0: the observation entries of the policy's encoder and history
    history_ind: Any = None

    def serialize(self):
        return {"config": OmegaConf.to_container(self.config, resolve=True, throw_on_missing=True),
                "goal_ind": None if self.goal_ind is None else np.asarray(self.goal_ind),
                "history_ind": None if self.history_ind is None else np.asarray(self.history_ind),
                "actor_net": flax.serialization.to_state_dict(self.actor_net),
                "critic_net": flax.serialization.to_state_dict(self.critic_net),
                "wm_net": flax.serialization.to_state_dict(self.wm_net),
                "term_obs_ind": None if self.term_obs_ind is None else np.asarray(self.term_obs_ind)}

    @classmethod
    def from_dict(cls, d):
        config = OmegaConf.create(d["config"])
        actor_net = d["actor_net"]
        critic_net = flax.serialization.from_state_dict(SuperMimicValueNet, d["critic_net"])
        wm_net = flax.serialization.from_state_dict(SuperMimicWorldModelNet, d["wm_net"])
        # Optimisers only need the config, not the env (obs_ind lives on the restored nets).
        _, _, _, actor_tx, critic_tx, wm_tx = _build_nets(_NoGroupEnv(), _strip_groups(config))
        term = d.get("term_obs_ind")
        return cls(config=config, actor_net=actor_net, critic_net=critic_net, wm_net=wm_net,
                   actor_tx=actor_tx, critic_tx=critic_tx, wm_tx=wm_tx,
                   term_obs_ind=None if term is None else jnp.asarray(term),
                   goal_ind=None if d.get("goal_ind") is None else jnp.asarray(d["goal_ind"]),
                   history_ind=None if d.get("history_ind") is None else jnp.asarray(d["history_ind"]))


class _NoGroupEnv:
    obs_container = None


def _strip_groups(config):
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    with open_dict(config.experiment):
        for k in ("actor_obs_group", "critic_obs_group"):
            config.experiment[k] = None
    return config


@struct.dataclass
class SuperMimicAgentState(AgentStateBase):
    actor_state: TrainState
    critic_state: TrainState
    wm_state: TrainState
    xi_mean: jnp.ndarray
    xi_std: jnp.ndarray
    xi_initialized: jnp.ndarray
    num_updates: jnp.ndarray
    # Env steps taken over all train calls: drives the UTD pacing and the logged timestep, so a run can be
    # continued in chunks (the learning-rate schedules live in the optimiser states).
    iteration: jnp.ndarray
    # Run state (not serialised); created by the first call of the train function.
    window_buffer: Any = None
    start_buffer: Any = None
    env_state: Any = None
    episode_stats: Any = None
    bin_resets: Any = None
    # Polyak average of the actor parameters, the policy to deploy (None without policy_ema).
    actor_ema: Any = None
    lr_decay_start: Any = None  # update at which the policy learning-rate boost starts to decay, -1 before
    actor_norm: Any = None  # RunningNorm of the observations, of the imagined observations for the critic
    critic_norm: Any = None
    obs_history: Any = None  # (n_envs, history_len, history_dim), run state
    wm_norm: Any = None  # dict of RunningNorm: "input", "output" and "error" of the world model

    @property
    def deployed_params(self):
        return self.actor_state.params if self.actor_ema is None else self.actor_ema

    def serialize(self):
        return {"actor_state": flax.serialization.to_state_dict(self.actor_state),
                "actor_ema": None if self.actor_ema is None else flax.serialization.to_state_dict(self.actor_ema),
                "lr_decay_start": None if self.lr_decay_start is None else np.asarray(self.lr_decay_start),
                "actor_norm": None if self.actor_norm is None else jax.tree.map(np.asarray, self.actor_norm),
                "critic_norm": None if self.critic_norm is None else jax.tree.map(np.asarray, self.critic_norm),
                "wm_norm": None if self.wm_norm is None else jax.tree.map(np.asarray, self.wm_norm),
                "critic_state": flax.serialization.to_state_dict(self.critic_state),
                "wm_state": flax.serialization.to_state_dict(self.wm_state),
                "xi": {"mean": np.asarray(self.xi_mean), "std": np.asarray(self.xi_std),
                       "initialized": np.asarray(self.xi_initialized)},
                "num_updates": np.asarray(self.num_updates),
                "iteration": np.asarray(self.iteration)}

    @classmethod
    def from_dict(cls, d, agent_conf):
        template = SuperMimicJax._init_state_from_conf(agent_conf, jax.random.PRNGKey(0))
        restore = lambda st, key: flax.serialization.from_state_dict(st, d[key])  # noqa: E731
        conf = agent_conf
        return template.replace(
            actor_state=restore(template.actor_state, "actor_state").replace(apply_fn=conf.actor_net.apply),
            critic_state=restore(template.critic_state, "critic_state").replace(apply_fn=conf.critic_net.apply),
            wm_state=restore(template.wm_state, "wm_state").replace(apply_fn=conf.wm_net.apply),
            xi_mean=jnp.asarray(d["xi"]["mean"]), xi_std=jnp.asarray(d["xi"]["std"]),
            xi_initialized=jnp.asarray(d["xi"]["initialized"]),
            actor_ema=None if d.get("actor_ema") is None else
            flax.serialization.from_state_dict(template.actor_state.params, d["actor_ema"]),
            lr_decay_start=None if d.get("lr_decay_start") is None else jnp.asarray(d["lr_decay_start"]),
            actor_norm=None if d.get("actor_norm") is None else jax.tree.map(jnp.asarray, d["actor_norm"]),
            critic_norm=None if d.get("critic_norm") is None else jax.tree.map(jnp.asarray, d["critic_norm"]),
            wm_norm=None if d.get("wm_norm") is None else jax.tree.map(jnp.asarray, d["wm_norm"]),
            num_updates=jnp.asarray(d["num_updates"]),
            iteration=jnp.asarray(d.get("iteration", 0), dtype=jnp.int32))


class BinResets:
    """
    Chooses episode starts by the time bin of the dataset they fall in: uniformly, towards bins with low returns,
    and, with share ``fail_share``, one bin before recent failures. The share fades with the failures themselves.
    """

    def __init__(self, split_points, n_samples, bin_frames, max_return, fail_share):
        """
        Args:
            split_points (jnp.ndarray): Start of each clip, then the end of the last one. Padded with -1.
            n_samples (int): Static upper bound of the number of frames.
            bin_frames (int): Frames per bin.
            max_return (float): Scale of the discounted returns.
            fail_share (float): Share of the resets sent before failures.
        """
        n_clips = split_points.shape[0] - 1
        self.n_bins = n_samples // bin_frames + n_clips
        self.bin_frames, self.max_return, self.fail_share = bin_frames, max_return, fail_share
        lengths = jnp.maximum(split_points[1:] - split_points[:-1], 0)
        lengths = jnp.where(split_points[1:] >= 0, lengths, 0)
        self.clip_len = lengths
        self.first_bin = jnp.concatenate((jnp.zeros(1, dtype=int), jnp.cumsum(-(-lengths // bin_frames))))
        bins = jnp.arange(self.n_bins)
        self.bin_clip = jnp.clip(jnp.searchsorted(self.first_bin, bins, side="right") - 1, 0, n_clips - 1)
        self.valid = bins < self.first_bin[-1]
        self.prev_bin = jnp.where(bins > self.first_bin[self.bin_clip], bins - 1, bins)

    def init_state(self):
        w = self.valid / jnp.sum(self.valid)
        return BinResetState(ret=jnp.zeros(self.n_bins), avg_focus=w, fail=jnp.zeros(self.n_bins))

    def bin_of(self, traj_no, step):
        return self.first_bin[traj_no] + step // self.bin_frames

    def probs(self, state):
        n = jnp.sum(self.valid)
        logits = jnp.where(self.valid, -25.0 * state.ret / self.max_return, -jnp.inf)
        focus = jax.nn.softmax(logits)
        p = 0.5 * self.valid / n + 0.4 * focus + 0.1 * state.avg_focus / jnp.sum(state.avg_focus)
        fail = jnp.zeros_like(state.fail).at[self.prev_bin].add(state.fail * self.valid)
        total = jnp.sum(fail)
        share = self.fail_share * jnp.minimum(total, 1.0)  # no failures left: back to the base distribution
        return (1 - share) * p + share * fail / jnp.maximum(total, 1e-6), focus

    def sample(self, state, rng, n):
        """Returns the clips, frames and bins of n episode starts."""
        p, _ = self.probs(state)
        rng_bin, rng_frame = jax.random.split(rng)
        cdf = jnp.cumsum(p)
        bins = jnp.searchsorted(cdf, jax.random.uniform(rng_bin, (n,)) * cdf[-1])
        bins = jnp.minimum(bins, jnp.sum(self.valid) - 1)
        clip = self.bin_clip[bins]
        start = (bins - self.first_bin[clip]) * self.bin_frames
        end = jnp.minimum(start + self.bin_frames, self.clip_len[clip] - 1)  # at least one step before the clip ends
        span = jnp.maximum(end - start, 1)
        return clip, start + jax.random.randint(rng_frame, (n,), 0, 1 << 30) % span, bins

    def update(self, state, done, term, start_bin, end_bin, disc_ret):
        """Updates the bins with the episodes that ended, given their start and end bins and discounted returns."""
        w = done.astype(jnp.float32)

        def ema(stat, bins, x, old):
            count = jnp.zeros_like(stat).at[bins].add(w)
            mean = jnp.zeros_like(stat).at[bins].add(x * w) / jnp.maximum(count, 1.0)
            return jnp.where(count > 0, old * stat + (1 - old) * mean, stat)

        any_done = jnp.any(done)
        _, focus = self.probs(state)
        return state.replace(
            ret=ema(state.ret, start_bin, disc_ret, 0.8),
            avg_focus=jnp.where(any_done, 0.999 * state.avg_focus + 0.001 * focus, state.avg_focus),
            fail=ema(jnp.where(any_done, 0.999 * state.fail, state.fail), end_bin, term.astype(jnp.float32), 0.9))


@struct.dataclass
class BinResetState:
    ret: jnp.ndarray  # discounted return of the episodes starting in each bin
    avg_focus: jnp.ndarray  # running average of the return-based focus
    fail: jnp.ndarray  # termination rate of the episodes ending in each bin


@struct.dataclass
class EpisodeStats:
    ret: jnp.ndarray
    length: jnp.ndarray
    last_ret: jnp.ndarray
    last_length: jnp.ndarray
    disc_ret: jnp.ndarray
    start_bin: jnp.ndarray


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------

class SuperMimicJax(JaxRLAlgorithmBase):
    _agent_conf = SuperMimicAgentConf
    _agent_state = SuperMimicAgentState

    # ---------- conf / state --------------------------------------------
    @classmethod
    def init_agent_conf(cls, env, config):
        nj = check_floating_base(env._model)
        exp = config.experiment
        term_group = getattr(exp, "termination_obs_group", None)
        term_obs_ind = _obs_ind(env, term_group, "termination_obs_group")
        if not bool(getattr(exp, "wm_term_head", True)):  # the buffer then keeps one unused column
            term_obs_ind = jnp.zeros(1, dtype=jnp.int32)
        obs_dim = int(env.info.observation_space.shape[0])
        with open_dict(config.experiment):
            exp.obs_dim = obs_dim
            exp.action_dim = int(env.info.action_space.shape[0])
            exp.n_joints = int(nj)
            exp.nq = int(env._model.nq)
            exp.nv = int(env._model.nv)
            exp.xi_index = context_index(env)
            exp.xi_dim = len(exp.xi_index)
            exp.term_obs_dim = obs_dim if term_obs_ind is None else int(term_obs_ind.size)
            exp.dt = float(env.dt)
            exp.num_iterations = int(float(exp.total_timesteps) // int(exp.num_envs))
        actor_net, critic_net, wm_net, actor_tx, critic_tx, wm_tx = _build_nets(env, config)
        use_history = int(getattr(exp, "history_len", 0)) > 0
        return cls._agent_conf(config=config, actor_net=actor_net, critic_net=critic_net, wm_net=wm_net,
                               actor_tx=actor_tx, critic_tx=critic_tx, wm_tx=wm_tx,
                               term_obs_ind=term_obs_ind,
                               goal_ind=_obs_ind(env, exp.goal_obs_group, "goal_obs_group") if use_history else None,
                               history_ind=_obs_ind(env, exp.history_obs_group, "history_obs_group")
                               if use_history else None)

    @classmethod
    def init_agent_state(cls, env, agent_conf, rng):
        return cls._init_state_from_conf(agent_conf, rng)

    @classmethod
    def _init_state_from_conf(cls, agent_conf, rng):
        exp = agent_conf.config.experiment
        obs_dim, action_dim, xi_dim = int(exp.obs_dim), int(exp.action_dim), int(exp.xi_dim)
        feat_dim = 13 + 2 * int(exp.n_joints)
        rng_a, rng_c, rng_w = jax.random.split(rng, 3)
        history_len = int(getattr(exp, "history_len", 0))
        # compiled: op by op, every primitive of an initializer is its own compilation
        init = lambda net, key, *dims: jax.jit(  # noqa: E731
            lambda k: net.init(k, *(jnp.zeros((1, d)) for d in dims))["params"])(key)
        if history_len > 0:
            actor_params = init(agent_conf.actor_net, rng_a, agent_conf.goal_ind.size,
                                history_len * agent_conf.history_ind.size)
        else:
            actor_params = init(agent_conf.actor_net, rng_a, obs_dim)
        critic_params = init(agent_conf.critic_net, rng_c, obs_dim, xi_dim)
        wm_params = init(agent_conf.wm_net, rng_w, feat_dim, action_dim, xi_dim, int(exp.term_obs_dim))
        mk = lambda net, params, tx: TrainState.create(apply_fn=net.apply, params=params,  # noqa: E731
                                                       run_stats={}, tx=tx)
        return SuperMimicAgentState(
            actor_state=mk(agent_conf.actor_net, actor_params, agent_conf.actor_tx),
            critic_state=mk(agent_conf.critic_net, critic_params, agent_conf.critic_tx),
            wm_state=mk(agent_conf.wm_net, wm_params, agent_conf.wm_tx),
            xi_mean=jnp.zeros(xi_dim), xi_std=jnp.ones(xi_dim),
            xi_initialized=jnp.array(False), num_updates=jnp.array(0, dtype=jnp.int32),
            iteration=jnp.array(0, dtype=jnp.int32),
            actor_ema=jax.tree.map(jnp.copy, actor_params) if float(getattr(exp, "policy_ema", 0.0)) > 0 else None,
            lr_decay_start=jnp.array(-1, dtype=jnp.int32),
            actor_norm=RunningNorm.create(obs_dim) if bool(getattr(exp, "obs_norm", False)) else None,
            critic_norm=RunningNorm.create(obs_dim if agent_conf.critic_net.obs_ind is None
                                           else agent_conf.critic_net.obs_ind.size)
            if bool(getattr(exp, "obs_norm", False)) else None,
            wm_norm=dict(input=RunningNorm.create(feat_dim + action_dim + xi_dim),
                         output=RunningNorm.create(feat_dim - 1), error=RunningNorm.create(feat_dim - 1))
            if bool(getattr(exp, "wm_running_norm", False)) else None)

    # ---------- training ------------------------------------------------
    @classmethod
    def build_init_fn(cls, env, agent_conf):
        """``init(rng, agent_state, traj)`` fills the run state, so that calling the train function in chunks
        compiles it once instead of a second time for the filled state."""
        return lambda rng_key, agent_state, traj: cls._train_fn(rng_key, env, agent_conf, agent_state, traj=traj,
                                                                init_only=True)

    @classmethod
    def _train_fn(cls, rng, env, agent_conf, agent_state=None, traj=None, mh=None, init_only=False):
        exp = agent_conf.config.experiment
        if isinstance(env, BaseWrapper):
            env = env.unwrapped()
        actor_net, critic_net, wm_net = agent_conf.actor_net, agent_conf.critic_net, agent_conf.wm_net
        term_obs_ind = agent_conf.term_obs_ind
        use_xi = int(exp.xi_dim) > 0

        spec = WorldModelSpec(int(exp.n_joints), float(exp.dt), _tuple(exp.wm_in_scale),
                              _tuple(exp.wm_out_scale), _tuple(exp.wm_err_scale),
                              float(getattr(exp, "wm_drift", 1.0)))
        n_envs = int(exp.num_envs)
        gamma, lam = float(exp.gamma), float(exp.gae_lambda)
        horizon = int(exp.imag_horizon)
        imag_batch = int(exp.imag_batch_size)
        window = int(exp.wm_window)
        wm_batch = int(exp.wm_batch_size)
        wm_batches = int(exp.wm_batches_per_update)
        critic_epochs, critic_mb = int(exp.critic_epochs), int(exp.critic_minibatches)
        utd = float(exp.updates_per_env_step)
        warmup = max(int(exp.warmup_steps), window + 1)
        action_noise = float(exp.action_noise_std)
        max_abs_qvel = float(getattr(exp, "max_abs_qvel", 100.0))
        remat = bool(getattr(exp, "remat", True))
        remat_keep_dots = bool(getattr(exp, "remat_keep_dots", False))
        # imag_pack: the imagined rollout carries and stacks one array per dtype instead of one per leaf
        imag_pack = bool(getattr(exp, "imag_pack", False))
        actor_loss_sum = bool(getattr(exp, "actor_loss_sum", False))
        analytic_only_truncates = bool(getattr(exp, "imag_soft_termination_only", False))
        policy_ema = float(getattr(exp, "policy_ema", 0.0))
        start_stride = int(getattr(exp, "start_buffer_stride", 1))
        # Policy learning rate x lr_mult, ramping to x1 over lr_decay_len updates once the training return
        # reaches lr_decay_return, at update lr_decay_by at the latest.
        lr_mult = float(getattr(exp, "policy_lr_mult", 1.0))
        lr_decay_len = int(getattr(exp, "lr_decay_len", 300))
        lr_decay_return = float(getattr(exp, "lr_decay_return", 0.75))
        lr_decay_by = int(getattr(exp, "lr_decay_by", 1400))
        max_step_reward = float(getattr(exp, "max_step_reward", 0.1))

        if agent_state is None:
            rng, _rng = jax.random.split(rng)
            agent_state = cls.init_agent_state(env, agent_conf, _rng)

        term_obs = (lambda o: o) if term_obs_ind is None else (lambda o: o[..., term_obs_ind])  # noqa: E731
        xi_index = list(exp.xi_index) if use_xi else None
        xi_of = lambda carry: context_from_carry(carry, use_xi, xi_index)  # noqa: E731

        actor_apply = cls.policy_fn(agent_conf)
        history_ind = agent_conf.history_ind
        history_len = int(getattr(exp, "history_len", 0))

        def push_history(history, obs):
            return None if history is None else jnp.concatenate([history[:, 1:], take_columns(obs, history_ind)[:, None]], axis=1)

        def fresh_history(obs):
            return None if history_ind is None else jnp.repeat(obs[:, None, history_ind], history_len, axis=1)

        # The stored rollouts hold the critic's columns only, which it reads several times per update.
        critic_ind, critic_core = critic_net.obs_ind, critic_net.clone(obs_ind=None)
        critic_obs = (lambda o: o) if critic_ind is None else (lambda o: take_columns(o, critic_ind))  # noqa: E731

        def critic_apply(params, obs, xi_n, norm):
            return critic_core.apply({"params": params}, obs if norm is None else norm(obs), xi_n)

        def wm_dynamics(wm, feat, action, xi_n):
            params, norm = wm
            y = wm_net.apply({"params": params}, feat, action, xi_n, None if norm is None else norm["input"],
                             method=SuperMimicWorldModelNet.dynamics)
            return y if norm is None else y * norm["output"].std() + norm["output"].mean

        def wm_termination(params, obs_t):
            return wm_net.apply({"params": params}, obs_t, method=SuperMimicWorldModelNet.termination)

        # ----- env (vmapped, no wrappers: the terminal state is needed before the reset) -----
        def _env_step(state, action):
            nstate = env.mjx_step(state, action, traj, auto_reset=False)
            reset = jax.lax.cond(nstate.done, lambda s: env._mjx_reset_in_step(s, traj), lambda s: s, nstate)
            return nstate, reset

        v_env_step = jax.vmap(_env_step, axis_name=VECENV_AXIS_NAME)
        v_reset = jax.vmap(lambda k: env.mjx_reset(k, traj), axis_name=VECENV_AXIS_NAME)

        use_bin_resets = bool(getattr(exp, "bin_resets", False))
        bins = BinResets(traj.data.split_points, traj.data.qpos.shape[0], max(1, round(float(exp.bin_seconds) / env.dt)),
                         float(exp.max_step_reward) / (1 - gamma), float(exp.fail_reset)) if use_bin_resets else None

        # ----- run state -----
        if agent_state.env_state is None:
            rng, _rng = jax.random.split(rng)
            env_state = v_reset(jax.random.split(_rng, n_envs))
            xi0 = jax.vmap(xi_of)(env_state.additional_carry)
            xi_mean = jnp.where(agent_state.xi_initialized, agent_state.xi_mean, jnp.mean(xi0, axis=0))
            xi_std = jnp.where(agent_state.xi_initialized, agent_state.xi_std, jnp.std(xi0, axis=0))
            agent_state = agent_state.replace(
                xi_mean=xi_mean, xi_std=xi_std, xi_initialized=jnp.array(True), env_state=env_state,
                obs_history=fresh_history(env_state.observation),
                window_buffer=WindowBuffer.create(int(exp.buffer_len), n_envs, int(exp.nq), int(exp.nv),
                                                  int(exp.action_dim), int(exp.xi_dim), int(exp.term_obs_dim)),
                start_buffer=StartBuffer.create(int(exp.start_buffer_len), env_state,
                                                fresh_history(env_state.observation)),
                episode_stats=EpisodeStats(*(jnp.zeros(n_envs) for _ in range(5)), jnp.zeros(n_envs, dtype=int)),
                bin_resets=bins.init_state() if use_bin_resets else None)
            if use_bin_resets:
                ts = env_state.additional_carry.traj_state
                agent_state = agent_state.replace(episode_stats=agent_state.episode_stats.replace(
                    start_bin=bins.bin_of(ts.traj_no, ts.subtraj_step_no)))
        # weak types differ between the input and output state, which would force a recompile per chunk
        strong_types = lambda tree: jax.tree.map(lambda x: jnp.asarray(x, jnp.asarray(x).dtype), tree)  # noqa: E731
        if init_only:
            return strong_types(agent_state)

        xi_mean = agent_state.xi_mean
        # Dimensions that do not vary across envs (not randomised) are mapped to 0.
        xi_inv_std = jnp.where(agent_state.xi_std > 1e-6, 1.0 / jnp.maximum(agent_state.xi_std, 1e-6), 0.0)
        xi_norm = lambda xi: (xi - xi_mean) * xi_inv_std  # noqa: E731

        # ----- world-model fitting (Eq. 1) -----
        err_floor = jnp.concatenate([jnp.full((n,), f) for f, n in zip(
            WM_ERR_FLOOR[0] + WM_ERR_FLOOR[1], (3, 3, 3, 3, int(exp.n_joints), int(exp.n_joints)))])

        # without the head, the terminal state handler must provide a smooth termination for imagination
        term_head = bool(getattr(exp, "wm_term_head", True))

        def wm_loss_fn(params, batch, mask, norm):
            xi_n = xi_norm(batch["xi"])

            def step(s, j):
                qpos, qvel = s
                delta = wm_dynamics((params, norm), spec.features(qpos, qvel), batch["action"][j], xi_n)
                qpos, qvel = spec.integrate(qpos, qvel, delta)
                e = spec.error(qpos, qvel, batch["next_qpos"][j], batch["next_qvel"][j])
                if norm is not None:
                    e = e * spec.err_scale / jnp.maximum(norm["error"].std(), err_floor)
                return (qpos, qvel), jnp.sum(e * e, axis=-1)

            _, sq = jax.lax.scan(step, (batch["qpos"][0], batch["qvel"][0]), jnp.arange(window),  # (k, n)
                                 unroll=window if bool(getattr(exp, "wm_unroll", False)) else 1)
            w = mask.astype(jnp.float32)
            w_sum = jnp.maximum(jnp.sum(w), 1.0)
            state_loss = jnp.sum(jnp.mean(sq, axis=0) * w) / w_sum
            term_loss, acc = jnp.zeros(()), jnp.zeros(())
            if term_head:
                logits = wm_termination(params, batch["next_term_obs"])  # (k, n)
                bce = optax.sigmoid_binary_cross_entropy(logits, batch["absorbing"])
                term_loss = jnp.sum(jnp.mean(bce, axis=0) * w) / w_sum
                acc = jnp.sum(jnp.mean((logits > 0) == (batch["absorbing"] > 0.5), axis=0) * w) / w_sum
            # mean normalised squared error per dimension, at each step of the window
            nse = jnp.sum(sq * w[None], axis=1) / w_sum / spec.delta_dim
            loss = state_loss + float(exp.wm_term_loss_weight) * term_loss
            return loss, dict(wm_loss=loss, term_bce=term_loss, term_acc=acc, nse_1=nse[0], nse_k=nse[-1])

        def wm_update(wm_state, buf, rng, norm):
            def body(st, r):
                batch, mask = buf.sample_windows(r, wm_batch, window)
                (_, m), g = jax.value_and_grad(wm_loss_fn, has_aux=True)(st.params, batch, mask, norm)
                return st.apply_gradients(grads=g), m

            wm_state, m = jax.lax.scan(body, wm_state, jax.random.split(rng, wm_batches))
            return wm_state, jax.tree.map(jnp.mean, m)

        # ----- imagination (Eq. 2) -----
        # The gradients run through the kinematics, which the warp kernels cannot differentiate, so with
        # MjWarp only the real env steps use warp and imagination gets a JAX copy of the model.
        use_warp = getattr(env, "_use_mjwarp", False)
        kin_sys = mjx.put_model(env._model, impl="jax") if use_warp else None
        first_data = mjx.make_data(env._model, impl="jax") if use_warp else env._first_data

        # needs observations and rewards that only read the fields FlatKinematics updates
        flat_kin = FlatKinematics(env._model) if bool(getattr(exp, "flat_kinematics", False)) else None

        def imag_kinematics(sys, data):
            if flat_kin is not None:
                return flat_kin(data)
            sys = kin_sys if use_warp else sys
            return mjx.com_vel(sys, mjx.com_pos(sys, mjx.kinematics(sys, data)))

        # imag_lean: the rollout computes only what the policy reads, and the critic inputs come afterwards.
        # Needs a smooth termination, and rewards that do not read the previous observation.
        lean = bool(getattr(exp, "imag_lean", False))
        assert not lean or (analytic_only_truncates and agent_conf.history_ind is not None)
        # imag_unroll: no XLA while loop around the rollout (fewer, larger kernels; compilation takes about twice
        # as long)
        imag_unroll = horizon if bool(getattr(exp, "imag_unroll", False)) else 1
        # imag_compact: the rollout step is only the policy decoder, the world model and the policy's history
        # columns of the next state. What does not depend on the imagined state comes from a forward-only pass over
        # the env's carry before it (goals, on which the policy encoder runs once for all steps, observation noise,
        # carries and infos), and kinematics, reward, termination and critic inputs are computed afterwards for all
        # states of the rollout at once. The networks get hand-written backward passes (see _slim_layer). Few enough
        # values per step to store them (remat=false) instead of recomputing the step.
        # Needs, on top of imag_lean: an env step that changes the simulated state only in the physics (no pushes);
        # a carry and info that do not depend on the simulated state or the action, apart from ``last_action``;
        # policy observations that read only qpos, qvel and the trajectory, with noise that is added without
        # reading them (no observation delay); and a reward whose only state is ``reward_state.last_action``.
        compact = bool(getattr(exp, "imag_compact", False))
        assert not compact or (lean and not imag_pack and isinstance(agent_conf.actor_net, SuperMimicPolicyNet)), \
            "imag_compact needs imag_lean, a SuperMimicPolicyNet actor and no imag_pack"
        goal_ind, net_actor = agent_conf.goal_ind, agent_conf.actor_net
        # an imagined env step leaves out the stateful observations the policy does not read
        step_obs_ind = None
        if lean:
            read = np.union1d(goal_ind, agent_conf.history_ind)
            unread = [o.obs_ind for o in env.obs_container.list_all_stateful() if not np.isin(o.obs_ind, read).any()]
            if unread:
                step_obs_ind = np.setdiff1d(np.arange(env.info.observation_space.shape[0]), np.concatenate(unread))

        def imag_env_step(state, action, wm_params):
            def physics(sys, data, carry, a):
                delta = wm_dynamics(wm_params, spec.features(data.qpos, data.qvel), a, xi_norm(xi_of(carry)))
                qpos, qvel = spec.integrate(data.qpos, data.qvel, delta)
                return imag_kinematics(sys, data.replace(qpos=qpos, qvel=qvel)), carry
            return env.mjx_step(state, action, traj, auto_reset=False, physics_fn=physics, obs_ind=step_obs_ind)

        v_imag_env_step = jax.vmap(imag_env_step, in_axes=(0, 0, None))
        # the env step with the state left as it is: advances the carry and computes only the goal, which leaves
        # in the other entries of the observation what the env adds to it (noise)
        v_goal_env_step = jax.vmap(lambda state, action: env.mjx_step(
            state, action, traj, auto_reset=False, physics_fn=lambda sys, data, carry, a: (data, carry),
            obs_ind=np.asarray(goal_ind), skip_reward=True))

        def slim_wm_delta(wm, qpos, qvel, xi_n, action):
            """The world model's output for a batch of states, as ``physics`` in ``imag_env_step`` computes it."""
            params, norm = wm
            x = jnp.concatenate([spec.features(qpos, qvel), action, xi_n], axis=-1)
            y = slim_mlp_apply(params["dynamics_net"], x if norm is None else norm["input"](x), wm_net.activation,
                               wm_net.compute_dtype)
            return y if norm is None else y * norm["output"].std() + norm["output"].mean

        v_absorbing_prob = jax.vmap(lambda s: env._mjx_absorbing_prob(s.observation, s.info, s.data,
                                                                      s.additional_carry, traj))

        def has_horizon_left(st):
            """Whether a full imagined rollout fits before the end of the clip."""
            if traj is None:
                return jnp.ones(st.done.shape, dtype=bool)
            ts = st.additional_carry.traj_state
            points = traj.data.split_points
            return ts.subtraj_step_no + horizon < points[ts.traj_no + 1] - points[ts.traj_no]

        obs_sys = kin_sys if use_warp else env.sys

        def critic_inputs(qpos, qvel, carry):
            """The critic's observation columns of a batch of imagined states, without observation noise."""
            def one(qp, qv, c):
                data = imag_kinematics(env.sys, first_data.replace(qpos=qp, qvel=qv))
                return env._mjx_create_observation(obs_sys, data, c, traj, critic_ind)[0]
            return jax.vmap(one)(qpos, qvel, carry)

        def deferred_outputs(qpos, qvel, carry, info, action, prev_action):
            """Critic observation, reward and termination probability of imagined states, each reached with
            ``action``. The observation is the critic's training input and takes no gradient."""
            def one(qp, qv, c, i, a, a_prev):
                data = imag_kinematics(env.sys, first_data.replace(qpos=qp, qvel=qv))
                critic_in = env._mjx_create_observation(obs_sys, *jax.lax.stop_gradient((data, c)), traj,
                                                        critic_ind)[0]
                # only computed if the reward or the termination read it
                obs = env._mjx_create_observation(obs_sys, data, c, traj)[0]
                d_hat = env._mjx_absorbing_prob(obs, i, data, c, traj)
                # the carry as the reward sees it in mjx_step: before the step counters advance
                c = c.replace(cur_step_in_episode=c.cur_step_in_episode - 1, global_step=c.global_step - 1)
                if hasattr(c.reward_state, "last_action"):
                    c = c.replace(reward_state=c.reward_state.replace(last_action=a_prev))
                absorbing, c = env._mjx_is_absorbing(obs, i, data, c, traj)
                reward = env._mjx_reward(jnp.zeros_like(obs), a, obs, absorbing, i, env._model, data, c, traj)[0]
                return critic_in, reward, d_hat
            return jax.vmap(jax.vmap(one))(qpos, qvel, carry, info, action, prev_action)

        def imag_scan(step, c0, xs=None):
            """``horizon`` steps of ``step``, rematerialized in the backward pass if ``remat``."""
            def wrap(f):
                if not remat:
                    return f
                # keeps the network matmuls, so the backward pass only recomputes the env functions
                return jax.checkpoint(f, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable
                                      if remat_keep_dots else None)

            if not imag_pack:
                return jax.lax.scan(wrap(step), c0, xs, horizon, unroll=imag_unroll)
            c_shape, o_shape = jax.eval_shape(step, c0, None)
            pc, po = _Packed(c_shape), _Packed(o_shape)

            def packed_step(c, _):
                c, o = step(pc.unpack(c), None)
                return pc.pack(c), po.pack(o)

            cH, ro = jax.lax.scan(wrap(packed_step), pc.pack(c0), None, horizon)
            return pc.unpack(cH), po.unpack(ro, (horizon,))

        def imagine_lean(actor_params, wm_params, critic_params, start, rng, actor_norm, critic_norm):
            """``imagine`` for imag_lean: the critic inputs of the rollout are left to ``with_critic_inputs``."""
            n = start.qpos.shape[0]
            carry0 = start.carry.replace(key=jax.random.split(rng, n),
                                         cur_step_in_episode=jnp.ones_like(start.carry.cur_step_in_episode))
            data_template = jax.vmap(lambda qp, qv: first_data.replace(qpos=qp, qvel=qv))(start.qpos, start.qvel)
            no_obs = jnp.zeros_like(start.observation)

            def step(c, _):
                qpos, qvel, goal, history, info, carry, surv = c
                hist = actor_norm(history, agent_conf.history_ind)
                a = net_actor.apply({"params": actor_params}, actor_norm(goal, goal_ind),
                                    hist.reshape(hist.shape[:-2] + (-1,)))
                ns = v_imag_env_step(MjxState(data=data_template.replace(qpos=qpos, qvel=qvel), observation=no_obs,
                                              reward=jnp.zeros(n), absorbing=jnp.zeros(n, dtype=bool),
                                              done=jnp.zeros(n, dtype=bool), info=info, additional_carry=carry),
                                     a, wm_params)
                d_hat = v_absorbing_prob(ns)
                out = dict(reward=ns.reward, surv=surv, d_hat=d_hat, action_norm=jnp.linalg.norm(a, axis=-1),
                           state=jax.lax.stop_gradient((ns.data.qpos, ns.data.qvel, ns.additional_carry)))
                return (ns.data.qpos, ns.data.qvel, take_columns(ns.observation, goal_ind),
                        push_history(history, ns.observation), ns.info, ns.additional_carry,
                        surv * (1.0 - d_hat)), out

            ones = jnp.ones(n)
            c0 = (start.qpos, start.qvel, take_columns(start.observation, goal_ind), start.history, start.info,
                  carry0, ones)
            cH, ro = imag_scan(step, c0)
            xi_H = jax.vmap(xi_of)(cH[5])
            obs_H = critic_inputs(cH[0], cH[1], cH[5])
            v_H = critic_apply(critic_params, obs_H, xi_norm(xi_H), critic_norm)
            ro.update(alive=jnp.ones_like(ro["surv"]), done=jnp.zeros_like(ro["surv"]),
                      start=jax.lax.stop_gradient((start.qpos, start.qvel, carry0)))
            return ro, (obs_H, xi_H, cH[6], ones, v_H)

        def imagine_compact(actor_params, wm_params, critic_params, start, rng, actor_norm, critic_norm):
            """``imagine`` for imag_compact: the rollout comes with the critic inputs of all its states."""
            n = start.qpos.shape[0]
            carry0 = start.carry.replace(key=jax.random.split(rng, n),
                                         cur_step_in_episode=jnp.ones_like(start.carry.cur_step_in_episode))
            data_template = jax.vmap(lambda qp, qv: first_data.replace(qpos=qp, qvel=qv))(start.qpos, start.qvel)
            no_obs, ones, hist_ind = jnp.zeros_like(start.observation), jnp.ones(n), agent_conf.history_ind
            stack = lambda s, x: jnp.concatenate([s[None], x])  # noqa: E731

            def normalize(norm, x, ind):
                return x if norm is None else norm(x, ind)

            # forward-only pass over the env's carry: goal, observation noise, carry and info of s_1 .. s_H
            def ahead_step(c, _):
                state = MjxState(data=data_template, observation=no_obs, reward=jnp.zeros(n),
                                 absorbing=jnp.zeros(n, dtype=bool), done=jnp.zeros(n, dtype=bool),
                                 info=c[0], additional_carry=c[1])
                ns = v_goal_env_step(state, jnp.zeros((n, env.info.action_space.shape[0])))
                return (ns.info, ns.additional_carry), dict(
                    goal=take_columns(ns.observation, goal_ind), noise=take_columns(ns.observation, hist_ind),
                    state=(ns.additional_carry, ns.info))

            ahead = jax.lax.stop_gradient(
                jax.lax.scan(ahead_step, (start.info, carry0), None, horizon, unroll=imag_unroll)[1])
            carries, infos = ahead["state"]
            # the policy encoder once, on the goals of s_0 .. s_{H-1}
            goals = stack(take_columns(start.observation, goal_ind), ahead["goal"][:-1])
            z = slim_mlp_apply(actor_params["enc"],
                               normalize(actor_norm, goals.reshape((-1, goals.shape[-1])), goal_ind),
                               net_actor.activation, net_actor.compute_dtype)
            z = jnp.tanh(z).reshape((horizon, n, -1))
            xi_n = xi_norm(jax.vmap(jax.vmap(xi_of))(jax.tree.map(lambda s, x: stack(s, x[:-1]), carry0, carries)))
            # the history is carried normalized, so each observation is normalized once, when it is produced
            hist_norm = None if actor_norm is None else actor_norm.replace(
                mean=actor_norm.mean[hist_ind], var=actor_norm.var[hist_ind])

            def step(c, x):
                qpos, qvel, history = c
                z_t, xi_t, next_carry, noise = x
                a = slim_mlp_apply(actor_params["dec"], jnp.concatenate([z_t, history.reshape((n, -1))], axis=-1),
                                   net_actor.activation, net_actor.compute_dtype)
                qpos, qvel = spec.integrate(qpos, qvel, slim_wm_delta(wm_params, qpos, qvel, xi_t, a))
                new = jax.vmap(lambda qp, qv, nc, a_: env._mjx_create_observation(
                    obs_sys, first_data.replace(qpos=qp, qvel=qv), nc.replace(last_action=a_), traj,
                    hist_ind)[0])(qpos, qvel, next_carry, a)
                # as the env step finishes its observation
                new = normalize(hist_norm, jnp.nan_to_num(new + noise, nan=0.0), None)
                return ((qpos, qvel, jnp.concatenate([history[:, 1:], new[:, None]], axis=1)),
                        dict(action=a, qpos=qpos, qvel=qvel))

            _, ro = imag_scan(step, (start.qpos, start.qvel, normalize(actor_norm, start.history, hist_ind)),
                              (z, xi_n, carries, ahead["noise"]))
            act = ro["action"]
            # states s_0 .. s_H in one pass; s_0 is not reached by an action and only gives critic inputs.
            # The carry gets the actions with their gradient: the reward and the bootstrap value read them.
            carry, info = jax.tree.map(stack, (carry0, start.info), (carries.replace(last_action=act), infos))
            last = getattr(carry0.reward_state, "last_action", jnp.zeros_like(act[0]))
            obs, reward, d_hat = deferred_outputs(
                stack(start.qpos, ro["qpos"]), stack(start.qvel, ro["qvel"]), carry, info,
                stack(jnp.zeros_like(act[0]), act), jnp.concatenate([jnp.zeros_like(act[:1]), last[None], act[:-1]]))
            reward, d_hat = reward[1:], d_hat[1:]
            surv = jnp.cumprod(1.0 - d_hat, axis=0)
            # the bootstrap value is the only reader of an observation with its gradient
            carry_H = jax.tree.map(lambda x: x[-1], carry)
            xi_H = jax.vmap(xi_of)(carry_H)
            obs_H = critic_inputs(ro["qpos"][-1], ro["qvel"][-1], carry_H)
            v_H = critic_apply(critic_params, obs_H, xi_norm(xi_H), critic_norm)
            ro = dict(reward=reward, surv=stack(ones, surv[:-1]), d_hat=d_hat,
                      action_norm=jnp.linalg.norm(act, axis=-1), alive=jnp.ones_like(reward),
                      done=jnp.zeros_like(reward), obs=obs,
                      xi=jax.lax.stop_gradient(jax.vmap(jax.vmap(xi_of))(carry)))
            return ro, (obs_H, xi_H, surv[-1], ones, v_H)

        def with_critic_inputs(ro, obs_H, xi_H):
            """The rollout with the critic inputs of all its states s_0 .. s_H."""
            if compact:  # has them from its pass over all states
                return ro
            ro = dict(ro)
            if lean:
                start, states = ro.pop("start"), ro.pop("state")
                qpos, qvel, carry = jax.tree.map(lambda s, x: jnp.concatenate([s[None], x[:-1]]), start, states)
                ro.update(obs=jax.vmap(critic_inputs)(qpos, qvel, carry), xi=jax.vmap(jax.vmap(xi_of))(carry))
            return dict(ro, obs=jnp.concatenate([ro["obs"], obs_H[None]]), xi=jnp.concatenate([ro["xi"], xi_H[None]]))

        def imagine(actor_params, wm_params, critic_params, start, rng, actor_norm, critic_norm):
            """H-step rollout; returns per-step tensors (H, B) and the critic inputs (H+1, B, .)."""
            keys = jax.random.split(rng, start.qpos.shape[0])
            # fresh keys, so replayed start states do not repeat the env's noise samples
            # and the episode time limit is not part of the imagined MDP: the critic cannot see it
            carry0 = start.carry.replace(key=keys, cur_step_in_episode=jnp.ones_like(start.carry.cur_step_in_episode))
            data_template = jax.vmap(lambda qp, qv: first_data.replace(qpos=qp, qvel=qv))(start.qpos, start.qvel)

            def make_state(qpos, qvel, obs, info, carry):
                return MjxState(data=data_template.replace(qpos=qpos, qvel=qvel), observation=obs,
                                reward=jnp.zeros(qpos.shape[0]), absorbing=jnp.zeros(qpos.shape[0], dtype=bool),
                                done=jnp.zeros(qpos.shape[0], dtype=bool), info=info, additional_carry=carry)

            def step(c, _):
                qpos, qvel, obs, history, info, carry, surv, alive = c
                a = actor_apply(actor_params, actor_norm, obs, history)
                ns = v_imag_env_step(make_state(qpos, qvel, obs, info, carry), a, wm_params)
                d_hat = v_absorbing_prob(ns)
                if d_hat is None:  # the handler has no smooth version
                    assert term_head
                    d_hat = jax.nn.sigmoid(wm_termination(wm_params[0], term_obs(ns.observation)))
                # with a smooth termination the survival weight already covers the absorbing states
                ended = ns.done if analytic_only_truncates is False else ns.done & ~ns.absorbing
                done = jax.lax.stop_gradient(ended.astype(jnp.float32))
                out = dict(reward=ns.reward, surv=surv, alive=alive, d_hat=d_hat, done=done,
                           action_norm=jnp.linalg.norm(a, axis=-1),
                           obs=critic_obs(obs), xi=jax.vmap(xi_of)(carry))
                nc = (ns.data.qpos, ns.data.qvel, ns.observation, push_history(history, ns.observation), ns.info,
                      ns.additional_carry,
                      surv * (1.0 - d_hat), alive * (1.0 - done))
                return nc, out

            ones = jnp.ones(start.qpos.shape[0])
            c0 = (start.qpos, start.qvel, start.observation, start.history, start.info, carry0, ones, ones)
            cH, ro = imag_scan(step, c0)
            obs_H, carry_H, surv_H, alive_H = cH[2], cH[5], cH[6], cH[7]
            xi_H = jax.vmap(xi_of)(carry_H)
            obs_H = critic_obs(obs_H)
            v_H = critic_apply(critic_params, obs_H, xi_norm(xi_H), critic_norm)
            return ro, (obs_H, xi_H, surv_H, alive_H, v_H)

        def actor_loss_fn(actor_params, wm_params, critic_params, start, rng, actor_norm, critic_norm):
            rollout = imagine_compact if compact else imagine_lean if lean else imagine
            ro, (obs_H, xi_H, surv_H, alive_H, v_H) = rollout(
                actor_params, wm_params, critic_params, start, rng, actor_norm, critic_norm)
            disc = gamma ** jnp.arange(horizon)
            J = jnp.sum(disc[:, None] * ro["surv"] * ro["alive"] * ro["reward"], axis=0)
            J = J + gamma ** horizon * surv_H * alive_H * v_H
            loss = -jnp.mean(J) / (1.0 if actor_loss_sum else horizon)
            return loss, (ro, obs_H, xi_H, alive_H)

        def critic_update(critic_state, ro, rng, norm):
            """Fits the critic to the lambda-returns of a rollout with the critic inputs of its states s_0 .. s_H."""
            obs = (ro["obs"] if norm is None else norm(ro["obs"])).astype(critic_net.compute_dtype)
            # one array of inputs, so that a minibatch is one gather
            X = jnp.concatenate([obs, xi_norm(ro["xi"]).astype(obs.dtype)], axis=-1)  # (H+1, B, .)
            value = lambda params, x: critic_apply(params, x, x[..., :0], None)  # noqa: E731
            values = value(critic_state.params, X)
            cont = (1.0 - ro["d_hat"]) * (1.0 - ro["done"])  # continuation into s_{t+1}

            def back(R, t):
                R = ro["reward"][t] + gamma * cont[t] * ((1.0 - lam) * values[t + 1] + lam * R)
                return R, R

            _, targets = jax.lax.scan(back, values[horizon], jnp.arange(horizon - 1, -1, -1))
            Y = jax.lax.stop_gradient(targets[::-1]).reshape(-1)
            W = ro["alive"].reshape(-1)
            n = Y.shape[0]
            X = X.reshape((-1, X.shape[-1]))  # the minibatches only index its first n rows, s_0 .. s_{H-1}
            mb = n // critic_mb

            def mb_step(st, idx):
                def loss_fn(params):
                    err = value(params, X[idx]) - Y[idx]
                    return jnp.sum(W[idx] * err ** 2) / jnp.maximum(jnp.sum(W[idx]), 1.0)

                loss, g = jax.value_and_grad(loss_fn)(st.params)
                return st.apply_gradients(grads=g), loss

            # the minibatches of all epochs: one loop
            perms = jnp.concatenate([jax.random.permutation(r, n)[:mb * critic_mb].reshape((critic_mb, mb))
                                     for r in jax.random.split(rng, critic_epochs)])
            critic_state, losses = jax.lax.scan(mb_step, critic_state, perms)
            return critic_state, losses[-1]

        def policy_update(actor_state, critic_state, wm_params, start_buffer, rng, mult, actor_norm, critic_norm):
            rng_s, rng_i, rng_c = jax.random.split(rng, 3)
            start = start_buffer.sample(rng_s, imag_batch)
            (loss, (ro, obs_H, xi_H, alive_H)), g = jax.value_and_grad(actor_loss_fn, has_aux=True)(
                actor_state.params, wm_params, critic_state.params, start, rng_i, actor_norm, critic_norm)
            old_params = actor_state.params
            actor_state = actor_state.apply_gradients(grads=g)
            actor_state = actor_state.replace(params=jax.tree.map(lambda o, n: o + mult * (n - o), old_params,
                                                                  actor_state.params))
            ro = with_critic_inputs(*jax.lax.stop_gradient((ro, obs_H, xi_H)))
            if critic_norm is not None:
                critic_norm = critic_norm.update(ro["obs"][:-1])
            with jax.named_scope("critic_update"):
                critic_state, critic_loss = critic_update(critic_state, ro, rng_c, critic_norm)
            m = dict(imag_return=jnp.mean(jnp.sum(ro["surv"] * ro["alive"] * ro["reward"], axis=0)),
                     imag_survival=jnp.mean(ro["surv"][-1]), imag_alive=jnp.mean(ro["alive"][-1]),
                     critic_loss=critic_loss, actor_grad_norm=optax.global_norm(g),
                     action_norm=jnp.mean(ro["action_norm"]))
            return actor_state, critic_state, critic_norm, m

        def one_update(_, c):
            agent_state, rng, _ = c
            rng, rng_w, rng_p = jax.random.split(rng, 3)
            with jax.named_scope("wm_update"):
                wm_state, m_wm = wm_update(agent_state.wm_state, agent_state.window_buffer, rng_w, agent_state.wm_norm)
            decay_at = agent_state.lr_decay_start
            ramp = jnp.where(decay_at >= 0, jnp.clip((agent_state.num_updates - decay_at) / lr_decay_len, 0.0, 1.0), 0.0)
            with jax.named_scope("policy_update"):
                actor_state, critic_state, critic_norm, m_pi = policy_update(
                    agent_state.actor_state, agent_state.critic_state, (wm_state.params, agent_state.wm_norm),
                    agent_state.start_buffer, rng_p,
                    lr_mult + (1.0 - lr_mult) * ramp, agent_state.actor_norm, agent_state.critic_norm)
            if policy_ema > 0:
                agent_state = agent_state.replace(actor_ema=jax.tree.map(
                    lambda e, p: policy_ema * e + (1.0 - policy_ema) * p, agent_state.actor_ema, actor_state.params))
            agent_state = agent_state.replace(wm_state=wm_state, actor_state=actor_state, critic_norm=critic_norm,
                                              critic_state=critic_state, num_updates=agent_state.num_updates + 1)
            return agent_state, rng, {**m_wm, **m_pi}

        metric_keys = ("wm_loss", "term_bce", "term_acc", "nse_1", "nse_k", "imag_return", "imag_survival",
                       "imag_alive", "critic_loss", "actor_grad_norm", "action_norm")

        # ----- one env step + updates at the configured update-to-data ratio -----
        def iteration(c, it):
            agent_state, rng, last_m = c
            rng, rng_a = jax.random.split(rng)
            st = agent_state.env_state
            actor_norm, history = agent_state.actor_norm, agent_state.obs_history
            if actor_norm is not None:
                actor_norm = actor_norm.update(st.observation)
            action = actor_apply(agent_state.actor_state.params, actor_norm, st.observation, history)
            action = action + action_noise * jax.random.normal(rng_a, action.shape)
            start_ok = state_is_valid(st.data.qpos, st.data.qvel, max_abs_qvel) & \
                jnp.all(jnp.isfinite(st.observation), axis=-1)
            start_buffer = agent_state.start_buffer.add(st, start_ok & has_horizon_left(st),
                                                        (it + 1) % start_stride == 0, history)
            next_bin = None
            if use_bin_resets:
                rng, rng_b = jax.random.split(rng)
                next_clip, next_frame, next_bin = bins.sample(agent_state.bin_resets, rng_b, n_envs)
                carry = st.additional_carry
                st = st.replace(additional_carry=carry.replace(traj_state=carry.traj_state.replace(
                    next_traj_no=next_clip, next_subtraj_step_no=next_frame)))
            with jax.named_scope("real_step"):
                nstate, reset_state = v_env_step(st, action)
            next_obs_t = term_obs(nstate.observation)
            transition_ok = (start_ok & state_is_valid(nstate.data.qpos, nstate.data.qvel, max_abs_qvel)
                             & jnp.all(jnp.isfinite(action), axis=-1) & jnp.all(jnp.isfinite(nstate.observation), axis=-1))
            window_buffer = agent_state.window_buffer.add(
                transition_ok,
                qpos=st.data.qpos, qvel=st.data.qvel, next_qpos=nstate.data.qpos, next_qvel=nstate.data.qvel,
                action=action, xi=jax.vmap(xi_of)(st.additional_carry),
                next_term_obs=next_obs_t, absorbing=nstate.absorbing.astype(jnp.float32), done=nstate.done)
            wm_norm = agent_state.wm_norm
            if wm_norm is not None:
                def masked_update(norm, x):
                    return norm.update(jnp.where(transition_ok[:, None], x, norm.mean))

                wm_in = jnp.concatenate([spec.features(st.data.qpos, st.data.qvel), action,
                                         xi_norm(jax.vmap(xi_of)(st.additional_carry))], axis=-1)
                wm_norm = dict(input=masked_update(wm_norm["input"], wm_in),
                               output=masked_update(wm_norm["output"], spec.targets(
                                   st.data.qpos, st.data.qvel, nstate.data.qpos, nstate.data.qvel)),
                               error=masked_update(wm_norm["error"], spec.error(
                                   nstate.data.qpos, nstate.data.qvel, st.data.qpos, st.data.qvel) * spec.err_scale))
            es = agent_state.episode_stats
            ret, length = es.ret + nstate.reward, es.length + 1
            disc_ret = es.disc_ret + gamma ** es.length * nstate.reward
            bin_resets, start_bin = agent_state.bin_resets, es.start_bin
            if use_bin_resets:
                # truncated episodes bootstrap with their mean reward
                boot = jnp.where(nstate.absorbing, 0.0, gamma ** length * (ret / length) / (1 - gamma))
                ts = nstate.additional_carry.traj_state
                bin_resets = bins.update(bin_resets, nstate.done, nstate.absorbing, es.start_bin,
                                         bins.bin_of(ts.traj_no, ts.subtraj_step_no), disc_ret + boot)
                start_bin = jnp.where(nstate.done, next_bin, es.start_bin)
            es = EpisodeStats(ret=jnp.where(nstate.done, 0.0, ret), length=jnp.where(nstate.done, 0.0, length),
                              last_ret=jnp.where(nstate.done, ret, es.last_ret),
                              last_length=jnp.where(nstate.done, length, es.last_length),
                              disc_ret=jnp.where(nstate.done, 0.0, disc_ret), start_bin=start_bin)
            step_return = jnp.mean(es.last_ret / jnp.maximum(es.last_length, 1.0)) / max_step_reward
            trigger = (step_return >= lr_decay_return) | (agent_state.num_updates >= lr_decay_by)
            decay_at = jnp.where((agent_state.lr_decay_start < 0) & trigger, agent_state.num_updates,
                                 agent_state.lr_decay_start)
            if history is not None:
                history = jnp.where(nstate.done[:, None, None], fresh_history(reset_state.observation),
                                    push_history(history, reset_state.observation))
            agent_state = agent_state.replace(env_state=reset_state, start_buffer=start_buffer, lr_decay_start=decay_at,
                                              actor_norm=actor_norm, obs_history=history, wm_norm=wm_norm,
                                              window_buffer=window_buffer, episode_stats=es, bin_resets=bin_resets)

            # Warm-up waits for the window buffer, which starts empty in every new process.
            ready = window_buffer.count >= min(warmup + 1, int(exp.buffer_len))
            n_up = jnp.where(ready, jnp.floor((it + 1) * utd) - jnp.floor(it * utd), 0).astype(jnp.int32)
            agent_state, rng, last_m = jax.lax.fori_loop(0, n_up, one_update, (agent_state, rng, last_m))

            metric = SuperMimicSummaryMetrics(
                mean_episode_return=jnp.mean(es.last_ret), mean_episode_length=jnp.mean(es.last_length),
                max_timestep=(it + 1) * n_envs,
                mean_wm_loss=last_m["wm_loss"], mean_wm_nse_1step=last_m["nse_1"], mean_wm_nse_kstep=last_m["nse_k"],
                mean_term_bce=last_m["term_bce"], mean_term_acc=last_m["term_acc"],
                mean_imag_return=last_m["imag_return"], mean_imag_survival=last_m["imag_survival"],
                mean_imag_alive=last_m["imag_alive"], mean_critic_loss=last_m["critic_loss"],
                mean_actor_grad_norm=last_m["actor_grad_norm"], mean_action_norm=last_m["action_norm"],
                num_updates=agent_state.num_updates)
            return (agent_state, rng, last_m), metric

        log_every = max(1, int(getattr(exp, "log_every", 100)))
        # One call runs `iterations_per_call` env steps (default: the whole run) from agent_state.iteration on.
        n_call = int(getattr(exp, "iterations_per_call", None) or exp.num_iterations)
        num_outer = max(1, n_call // log_every)
        start = agent_state.iteration

        def logged(c, outer):
            c, ms = jax.lax.scan(iteration, c, start + outer * log_every + jnp.arange(log_every))
            agg = jax.tree.map(lambda x: jnp.mean(x, axis=0), ms)
            agg = agg.replace(max_timestep=ms.max_timestep[-1], num_updates=ms.num_updates[-1])
            return c, agg

        last_m0 = {k: jnp.zeros((), jnp.float32) for k in metric_keys}
        rng, _rng = jax.random.split(rng)
        (agent_state, _, _), training_metrics = jax.lax.scan(logged, (agent_state, _rng, last_m0),
                                                             jnp.arange(num_outer))
        agent_state = agent_state.replace(iteration=start + num_outer * log_every)
        return {"agent_state": strong_types(agent_state), "training_metrics": training_metrics,
                "validation_metrics": ValidationSummary()}

    @classmethod
    def policy_fn(cls, agent_conf):
        """``act(params, norm, obs, history) -> action`` for batched observations; history is None without
        history_len."""
        goal_ind, history_ind, net = agent_conf.goal_ind, agent_conf.history_ind, agent_conf.actor_net

        def act(params, norm, obs, history=None):
            if history_ind is None:
                return net.apply({"params": params}, obs if norm is None else norm(obs))
            goal, hist = take_columns(obs, goal_ind), history
            if norm is not None:
                goal, hist = norm(goal, goal_ind), norm(hist, history_ind)
            return net.apply({"params": params}, goal, hist.reshape(hist.shape[:-2] + (-1,)))

        return act

    # ---------- play ----------------------------------------------------
    @classmethod
    def play_policy(cls, env, agent_conf, agent_state, n_envs: int, n_steps=None, render=True,
                    record=False, rng=None, deterministic=True, use_mujoco=False, traj=None):
        from loco_mujoco.core.wrappers import LogWrapper, VecEnv

        exp = agent_conf.config.experiment
        assert agent_conf.history_ind is None, "play_policy does not keep an observation history."
        policy = cls.policy_fn(agent_conf)
        params = agent_state.deployed_params
        if not use_mujoco:
            env = VecEnv(LogWrapper(env))
        rng = jax.random.key(0) if rng is None else rng
        keys = jax.random.split(rng, n_envs + 1)
        rng, env_keys = keys[0], keys[1:]

        @jax.jit
        def _act(obs, _rng):
            action = policy(params, agent_state.actor_norm, obs)
            if not deterministic:
                action = action + float(exp.action_noise_std) * jax.random.normal(_rng, action.shape)
            return action

        if use_mujoco:
            obs, env_state = env.reset(), None
        else:
            obs, env_state = env.reset(env_keys, traj)
        n_steps = np.iinfo(np.int32).max if n_steps is None else n_steps
        for _ in range(n_steps):
            rng, _rng = jax.random.split(rng)
            action = jnp.atleast_2d(_act(obs, _rng))
            if use_mujoco:
                obs, _, _, done, _ = env.step(action)
                env.render(record=record)
                if done:
                    obs = env.reset()
            else:
                obs, _, _, done, _, env_state = env.step(env_state, action, traj)
                env.mjx_render(env_state, record=record)
        env.stop()
