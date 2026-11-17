"""Evaluate a SuperMimic checkpoint on every clip of its dataset, without noise or domain randomization.

Each clip plays once from its first frame. A clip fails if the rollout terminates before its end. The normalized
return is the undiscounted reward sum divided by the maximum attainable one, averaged over clips.

usage: python eval_clips.py RUN_DIR CHECKPOINT [--import=MODULE]
RUN_DIR holds the config.yaml of the run. MODULE registers external envs and components.
"""
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf, open_dict

args = [a for a in sys.argv[1:] if not a.startswith("--")]
for a in sys.argv[1:]:
    if a.startswith("--import="):
        __import__(a.split("=", 1)[1])

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import SuperMimicJax

run, ckpt = Path(args[0]), args[1]
exp = OmegaConf.load(run / "config.yaml").experiment
with open_dict(exp):
    exp.env_params.horizon = 10 ** 8
    exp.env_params.pop("domain_randomization_type", None)
    exp.env_params.pop("domain_randomization_params", None)
    exp.env_params.pop("init_state_params", None)
env, traj = TaskFactory.get_factory_cls(exp.task_factory.name).make(**exp.env_params, **exp.task_factory.params)
names = list(exp.task_factory.params.lafan1_dataset_conf.dataset_name)
agent_conf, state = SuperMimicJax.load_agent(run / ckpt)
params, policy, history_ind = state.deployed_params, SuperMimicJax.policy_fn(agent_conf), agent_conf.history_ind

# reward of a step on the reference, to normalize the return
rp = exp.env_params.get("reward_params") or {}
if hasattr(env._reward_function, "max_reward"):
    max_step_reward = env._reward_function.max_reward
elif exp.env_params.get("reward_type", "MimicReward") == "MimicReward":
    # the site terms count only with sites; the velocity weight covers the linear and the angular term
    max_step_reward = float(rp.get("qpos_w_sum", 0.0) + rp.get("qvel_w_sum", 0.0) + rp.get("rpos_w_sum", 0.5)
                            + rp.get("rquat_w_sum", 0.3) + 2 * rp.get("rvel_w_sum", 0.0))
else:
    # tracking rewards defined outside of LocoMuJoCo that follow the root / body weight names
    weights = ("w_root_pos", "w_root_ori", "w_body_pos", "w_body_ori", "w_body_lin_vel", "w_body_ang_vel")
    missing = [w for w in weights if w not in rp]
    if missing:
        raise ValueError(f"Cannot normalize the return of {exp.env_params.reward_type}: give it a `max_reward` "
                         f"property, or set {missing} in reward_params.")
    max_step_reward = float(sum(rp[w] for w in weights)) * env.dt
n_steps = jnp.array([int(env.th.len_trajectory(i, traj.data)) - 1 for i in range(len(names))])

# reset outside of jit: per-env start frames set inside jit do not reach the reset correctly
env.th.random_start, env.th.use_fixed_start = False, True


def reset(key, clip):
    env.th.fixed_start_conf = (clip, 0)
    return env.mjx_reset(key, traj)


env_state = jax.vmap(reset, axis_name="vecenv")(jax.random.split(jax.random.PRNGKey(0), len(names)), jnp.arange(len(names)))
ts = env_state.additional_carry.traj_state
assert np.array_equal(np.asarray(ts.traj_no), np.arange(len(names))) and not np.asarray(ts.subtraj_step_no).any()
ref_q = jax.vmap(lambda c: traj.data.get(c, 0, jnp).qpos[7:])(jnp.arange(len(names)))
assert float(jnp.max(jnp.abs(env_state.data.qpos[:, 7:] - ref_q))) < 1e-4
vstep = jax.vmap(lambda s, a: env.mjx_step(s, a, traj, auto_reset=False), axis_name="vecenv")


@jax.jit
def rollout(c):
    def step(c, _):
        s, history, ret, steps, alive, failed = c
        n = vstep(s, policy(params, state.actor_norm, s.observation, history))
        if history is not None:
            history = jnp.concatenate([history[:, 1:], n.observation[:, None, history_ind]], axis=1)
        ret = ret + jnp.where(alive, n.reward, 0.0)
        steps = steps + alive
        failed = failed | (alive & n.absorbing)
        return (n, history, ret, steps, alive & ~n.absorbing & (steps < n_steps), failed), None
    return jax.lax.scan(step, c, None, 500)[0]


zeros = jnp.zeros(len(names))
history0 = None if history_ind is None else jnp.repeat(
    env_state.observation[:, None, history_ind], int(exp.history_len), axis=1)
c = (env_state, history0, zeros, zeros.astype(int), jnp.ones(len(names), dtype=bool), jnp.zeros(len(names), dtype=bool))
while bool(c[4].any()):
    c = rollout(c)
_, _, ret, steps, _, failed = jax.tree.map(np.asarray, c)
norm = ret / (np.asarray(n_steps) * max_step_reward)
per_clip = {n: dict(norm_return=round(float(r), 4), steps=int(s), of=int(t), failed=bool(f))
            for n, r, s, t, f in zip(names, norm, steps, np.asarray(n_steps), failed)}
print(json.dumps(per_clip, indent=1))
print(f"{ckpt}: norm return {norm.mean():.4f}, failed {int(failed.sum())}/{len(names)}, "
      f"completed frames {steps.sum() / np.asarray(n_steps).sum():.3f}", flush=True)
