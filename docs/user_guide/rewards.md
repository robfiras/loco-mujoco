# Rewards

The reward of an environment is computed by a reward module. It is selected with `reward_type`, the name of
a registered reward class, and configured with `reward_params`, a dictionary of keyword arguments for that
class.

```python
import numpy as np
from loco_mujoco import ImitationFactory

env, traj = ImitationFactory.make(
    "UnitreeH1",
    default_dataset_conf=dict(task="walk"),
    goal_type="GoalTrajMimic",
    reward_type="MimicReward",
    reward_params=dict(rpos_w_sum=0.5, rquat_w_sum=0.3, qvel_w_sum=0.2),
)

obs = env.reset()
action = np.zeros(env.info.action_space.shape[0])
obs, reward, absorbing, done, info = env.step(action)
```

The same arguments are used in the `env_params` section of the
[training configurations](../training/configuration.md).

## Overview

| `reward_type` | Purpose | Needs |
|---|---|---|
| `NoReward` | Always 0. The default of the environment, for replay and for algorithms that learn their own reward such as GAIL and AMP. | nothing |
| `TargetXVelocityReward` | Move forward at a fixed speed | nothing |
| `TargetVelocityGoalReward` | Track a commanded root velocity. The default of {class}`~loco_mujoco.task_factories.rl_factory.RLFactory`. | the goal `GoalRandomRootVelocity` |
| `LocomotionReward` | Velocity tracking with the usual regularization terms of legged locomotion | the goal `GoalRandomRootVelocity` |
| `TargetVelocityTrajReward` | Track the root velocity of a reference trajectory | a trajectory |
| `MimicReward` | DeepMimic-style tracking of a reference motion in joint space and in the space of the mimic sites | a trajectory |
| `SiteTrackingReward` | BeyondMimic-style tracking of the mimic sites in the world frame | a trajectory |

Rewards that need a trajectory require an environment created with
{class}`~loco_mujoco.task_factories.imitation_factory.ImitationFactory`. They compare the simulation state to
the frame of the reference trajectory that belongs to the current step. An episode of such an environment
ends when the reference clip ends.

```{note}
{class}`~loco_mujoco.task_factories.imitation_factory.ImitationFactory` does not choose a reward. Without
`reward_type` the environment uses `NoReward`, which is what GAIL and AMP expect. For tracking with a
reward, set `reward_type="MimicReward"` together with `goal_type="GoalTrajMimic"`.
```

## Velocity rewards

### TargetXVelocityReward

{class}`~loco_mujoco.core.reward.default.TargetXVelocityReward` rewards a fixed forward speed. With the
velocity $v_x$ of the root along the world x-axis and the required parameter `target_velocity` $v^*$:

$$
r = \exp\!\left(-(v_x - v^*)^2\right)
$$

### TargetVelocityGoalReward

{class}`~loco_mujoco.core.reward.default.TargetVelocityGoalReward` tracks the command that the goal
`GoalRandomRootVelocity` samples at every reset: a forward velocity, a lateral velocity and a yaw rate. Let
$v_{xy}$ be the horizontal velocity of the root in its own frame, $\omega_z$ its yaw rate, and
$g = (g_x, g_y, g_\text{yaw})$ the command:

$$
r = w^\Sigma_{xy} \exp\!\left(-w^{e}_{xy}\, \tfrac{1}{2}\lVert v_{xy} - g_{xy} \rVert^2\right)
  + w^\Sigma_\text{yaw} \exp\!\left(-w^{e}_\text{yaw}\, (\omega_z - g_\text{yaw})^2\right)
$$

| Parameter | Default | Meaning |
|---|---|---|
| `tracking_w_exp_xy` | 10.0 | $w^e_{xy}$, sharpness of the linear velocity term |
| `tracking_w_exp_yaw` | 10.0 | $w^e_\text{yaw}$, sharpness of the yaw rate term |
| `tracking_w_sum_xy` | 1.0 | $w^\Sigma_{xy}$, weight of the linear velocity term |
| `tracking_w_sum_yaw` | 1.0 | $w^\Sigma_\text{yaw}$, weight of the yaw rate term |

### LocomotionReward

{class}`~loco_mujoco.core.reward.default.LocomotionReward` adds regularization terms to the tracking reward
above and clips the sum at zero:

$$
r = \max\!\Big(0,\; r_\text{tracking} - \sum_k c_k\, p_k + c_\text{air}\, b_\text{air}\Big)
$$

It accepts the four tracking parameters of `TargetVelocityGoalReward` and one coefficient $c_k$ per term. A
coefficient of 0 turns the term off. All terms are penalties $p_k$ except the air time term $b_\text{air}$,
which is a bonus.

| Parameter | Default | Term |
|---|---|---|
| `z_vel_coeff` | 2.0 | Squared vertical velocity of the root, in the root frame |
| `roll_pitch_vel_coeff` | 0.05 | Squared roll and pitch rates of the root |
| `roll_pitch_pos_coeff` | 0.2 | Squared roll and pitch angles of the root |
| `nominal_joint_pos_coeff` | 0.0 | Squared deviation of the joints from the default pose of the model (`qpos0`). `nominal_joint_pos_names` restricts the joints, by default all joints with limits are used. |
| `joint_position_limit_coeff` | 10.0 | Amount by which joints exceed their limits |
| `joint_vel_coeff` | 0.0 | Squared joint velocities |
| `joint_acc_coeff` | 2e-7 | Squared change of the joint velocities since the last step, divided by the control time step |
| `joint_torque_coeff` | 2e-5 | Squared actuator forces (`qfrc_actuator`) |
| `action_rate_coeff` | 0.01 | Squared change of the action since the last step |
| `energy_coeff` | 0.0 | Sum of the absolute joint powers, $\sum_i \lvert \dot q_i \rvert\, \lvert \tau_i \rvert$ |
| `air_time_coeff`, `air_time_max` | 0.0, 0.0 | Air time bonus $b_\text{air}$: for every foot that touches the floor, the time since its last touchdown minus `air_time_max` |
| `symmetry_air_coeff` | 0.0 | Number of foot pairs (first and second foot, third and fourth foot) that are in the air at the same time. Meant for quadrupeds. |

The two foot terms use the geoms that the robot lists in its info property `foot_geom_names` and their
contacts with the geom named `floor`. This reward keeps the previous joint velocities, the previous action
and the air times in `carry.reward_state`.

## Trajectory rewards

### TargetVelocityTrajReward

{class}`~loco_mujoco.core.reward.trajectory_based.TargetVelocityTrajReward` is the counterpart of
`TargetVelocityGoalReward` for a velocity that comes from a reference trajectory instead of a sampled
command. With $u = (v_x, v_y, \omega_z)$ of the robot and $\hat u$ of the current reference frame, both in
the frame of the respective root:

$$
r = \exp\!\left(-w\, \tfrac{1}{3}\lVert u - \hat u \rVert^2\right)
$$

The only parameter is `w_exp` ($w$, default 10.0). It pairs with the goal `GoalTrajRootVelocity`.

### MimicReward

{class}`~loco_mujoco.core.reward.trajectory_based.MimicReward` is a DeepMimic-style tracking reward
(Peng et al., 2018) and the original imitation reward of LocoMuJoCo. It is a weighted sum of exponential
kernels over five tracking errors, plus optional penalties:

$$
r = \max\!\Big(0,\; \sum_{i} w^\Sigma_i \exp\!\left(-w^e_i\, d_i\right) + \max(-1,\, r_\text{penalty})\Big)
$$

| Term | Error $d_i$ (mean over the entries) | Sharpness $w^e_i$ (default) | Weight $w^\Sigma_i$ (default) |
|---|---|---|---|
| Joint positions | Squared error of `qpos`, plus the angular distance of the root quaternion | `qpos_w_exp` (10.0) | `qpos_w_sum` (0.0) |
| Joint velocities | Squared error of `qvel` | `qvel_w_exp` (2.0) | `qvel_w_sum` (0.0) |
| Site positions | Squared error of the positions of the mimic sites relative to the main site | `rpos_w_exp` (100.0) | `rpos_w_sum` (0.5) |
| Site orientations | Squared error of the rotation vectors of the mimic sites relative to the main site | `rquat_w_exp` (10.0) | `rquat_w_sum` (0.3) |
| Site velocities | Squared error of the relative angular and linear site velocities, as two terms with the same parameters | `rvel_w_exp` (0.1) | `rvel_w_sum` (0.0) |

With the defaults only the two site pose terms are active and the reward of perfect tracking is 0.8.

| Further parameter | Default | Meaning |
|---|---|---|
| `sites_for_mimic` | `None` | Names of the sites to track. `None` uses the mimic sites of the robot (info property `sites_for_mimic`). With fewer than two sites the three site terms are dropped. |
| `main_site_name` | `None` | Site that the relative quantities refer to. `None` uses the first site of the list. |
| `joints_for_mimic` | `None` | Names of the joints that enter the joint terms. `None` uses all joints, including the free joint of the root. |
| `action_out_of_bounds_coeff` | 0.01 | Penalty on the mean squared amount by which the action leaves the action space |
| `joint_acc_coeff` | 0.0 | Penalty on the change of the joint velocities, as in `LocomotionReward` |
| `joint_torque_coeff` | 0.0 | Penalty on the squared actuator forces |
| `action_rate_coeff` | 0.0 | Penalty on the squared change of the action |

```{note}
In the current implementation the coefficients `joint_acc_coeff`, `joint_torque_coeff` and
`action_rate_coeff` of `MimicReward` are applied twice, so these three penalties scale with the square of
their coefficient.
```

All site quantities are relative to the main site of the robot, so the site terms do not depend on where
the robot is in the world or which way it faces. The policy is rewarded for reproducing the pose and the
motion of the limbs, not for staying on the path of the reference. Use the same `sites_for_mimic` and
`main_site_name` for the reward and for the goal `GoalTrajMimic` so that the policy observes what it is
rewarded for.

### SiteTrackingReward

{class}`~loco_mujoco.core.reward.trajectory_based.SiteTrackingReward` is a BeyondMimic-style tracking reward
(Liao et al., 2025) that is computed on the mimic sites instead of on a robot-specific list of bodies, so
it applies to every robot. It has no joint-space terms. Each of its six tracking terms has the form
$w_k \exp(-e_k / \sigma_k^2)$:

| Term | Error $e_k$ | Width $\sigma_k$ (default) | Weight $w_k$ (default) |
|---|---|---|---|
| Main site position | Squared distance to the reference, in the world | `main_pos_std` (0.3) | `w_main_pos` (0.5) |
| Main site orientation | Squared rotation angle to the reference, in the world | `main_ori_std` (0.4) | `w_main_ori` (0.5) |
| Site positions | Mean squared distance of all sites to the aligned reference | `site_pos_std` (0.3) | `w_site_pos` (1.0) |
| Site orientations | Mean squared rotation angle of all sites to the aligned reference | `site_ori_std` (0.4) | `w_site_ori` (1.0) |
| Site linear velocities | Mean squared error, in the world | `site_lin_vel_std` (1.0) | `w_site_lin_vel` (1.0) |
| Site angular velocities | Mean squared error, in the world | `site_ang_vel_std` (3.14) | `w_site_ang_vel` (1.0) |

The main site is the first mimic site. For the two site pose terms the reference is first moved to the
horizontal position and the heading of the robot, while it keeps its own height, roll and pitch. A robot
that drifts away from the reference path therefore only loses reward in the two main site terms, not in
all of them.

$$
r = s \,\Big(\sum_{k=1}^{6} w_k \exp\!\left(-e_k / \sigma_k^2\right)
      + w_\text{rate}\, \lVert a_t - a_{t-1} \rVert^2
      + w_\text{limit} \sum_j \text{excess}_j \Big)
$$

| Further parameter | Default | Meaning |
|---|---|---|
| `sites_for_mimic` | `None` | Sites to track, the main site first. `None` uses the mimic sites of the robot. |
| `w_action_rate` | -0.1 | $w_\text{rate}$, weight of the squared change of the action |
| `w_joint_limit` | -2.0 | $w_\text{limit}$, weight of the amount by which joints leave their soft limits |
| `soft_joint_pos_limit_factor` | 0.9 | Part of each joint range, around its center, that is not penalized |
| `scale_by_dt` | `True` | If set, $s$ is the control time step of the environment, otherwise 1 |

The reward is not clipped and can be negative. Its value for a step exactly on the reference, without
penalties, is available as the property `max_reward` of the reward object.

### Choosing between MimicReward and SiteTrackingReward

| | `MimicReward` | `SiteTrackingReward` |
|---|---|---|
| Frame of the site terms | Relative to the main site of the robot | World frame for the main site and the velocities, aligned reference for the site poses |
| Global position and heading | Not rewarded by the site terms. Only the joint position term sees the root pose, and its weight is 0 by default. | Rewarded through the main site terms |
| Joint-space terms | Yes, optional | No |
| Range | Between 0 and the sum of the weights | Unbounded below, scaled by the time step by default |
| Regularization | Optional penalties, off by default except for out-of-bounds actions | Action rate and soft joint limit penalties, on by default |

`MimicReward` is the reward that the tracking examples in `examples/training_examples` are configured with,
and it is the safer first choice: it is bounded, and it does not punish a policy that tracks the motion
well but slowly drifts from the reference path. Choose `SiteTrackingReward` when the robot has to stay on
the path of the reference in the world, for example when the clips are placed on a terrain, or when you
want the reward formulation of BeyondMimic.

## The reward interface

A reward derives from {class}`~loco_mujoco.core.reward.base.Reward` and implements `__call__`, which returns
the reward and the carry:

```python
def __call__(self, state, action, next_state, absorbing, info, env, model, data, carry, backend, traj=None):
    ...
    return reward, carry
```

`state` and `next_state` are the observations before and after the step, `data` is the simulation state
after the step, and `traj` is the reference trajectory if there is one. A reward that keeps state between
steps returns it from `init_state` and updates `carry.reward_state`. A reward that needs a trajectory
returns `True` from the property `requires_trajectory`. See
[Writing custom modules](custom_modules.md#rewards) for a complete example.

## See also

- [Observations](observations.md#goals) for the goals that the rewards pair with.
- [Initial and terminal states](initial_terminal_states.md) for the termination conditions used with tracking rewards.
- [DeepMimic-style training](../training/deepmimic.md) and [evaluation](../training/evaluation.md).
- API: {mod}`loco_mujoco.core.reward.default`, {mod}`loco_mujoco.core.reward.trajectory_based`.
