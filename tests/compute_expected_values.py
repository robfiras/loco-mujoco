"""
Compute current expected values for tests (rewards, initial states) using the
current physics engine (mujoco/mjx) and jax/numpy. Run from project root:
  python tests/compute_expected_values.py

Use the printed values to update the test files with new target results.
"""
import sys
import os
import numpy as np
import jax
import jax.numpy as jnp
import numpy.random as nr
import jax.random as jr

# Run from project root; add tests to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'tests')

jax.config.update('jax_platform_name', 'cpu')

# Apply mock_random before any env/trajectory creation (same as pytest fixture)
def _apply_mock_random():
    _orig_jr_normal = jr.normal
    _orig_jr_uniform = jr.uniform
    _orig_jr_randint = jr.randint
    _orig_nr_normal = nr.normal
    _orig_nr_uniform = nr.uniform
    _orig_nr_randint = nr.randint
    _orig_nr_randn = nr.randn

    jr.normal = lambda key, shape=(), dtype=jnp.float32: jnp.full(shape, 0.5, dtype=dtype)
    jr.uniform = lambda key, shape=(), dtype=jnp.float32, minval=0.0, maxval=1.0: jnp.full(
        shape, minval + (maxval - minval) * 0.3, dtype=dtype
    )
    jr.randint = lambda key, shape, minval, maxval, dtype=jnp.int32: jnp.full(
        shape, minval + (maxval - minval) // 2, dtype=dtype
    )
    nr.normal = lambda loc=0.0, scale=1.0, size=None: np.full(
        size if size is not None else (), loc + 0.5 * scale
    )
    nr.uniform = lambda low=0.0, high=1.0, size=None: np.full(
        (size if size is not None else ()) if (np.isscalar(low) and np.isscalar(high))
        else np.broadcast(low, high).size,
        np.asarray(low) + (np.asarray(high) - np.asarray(low)) * 0.3,
    )
    nr.randint = lambda low, high=None, size=None, dtype=int: np.full(
        size if size else (),
        (low + (high - 1)) // 2 if high is not None else low,
        dtype=dtype
    )
    nr.randn = lambda *args: np.full(args if args else (), 0.2)


def main():
    _apply_mock_random()

    from test_conf import (
        DummyHumamoidEnv,
        generate_test_trajectories,
        Trajectory,
        TrajectoryInfo,
        TrajectoryData,
        TrajectoryModel,
    )
    from loco_mujoco.core.trajectory import Trajectory as TrajClass
    import mujoco

    DEFAULTS = {"horizon": 1000, "gamma": 0.99, "n_envs": 1}
    DEFAULTS_TRAJ = {"horizon": 1000, "gamma": 0.99, "n_envs": 1,
                     "th_params": {"name": "FixedStartTrajectoryHandler", "start_conf": (0, 0)}}

    def make_standing_trajectory():
        N_steps = 1000
        mjx_env = DummyHumamoidEnv(enable_mjx=False,
                                   terminal_state_type="RootPoseTrajTerminalStateHandler",
                                   **DEFAULTS)
        key = jax.random.PRNGKey(0)
        mjx_env.reset(key)
        model = mjx_env.model
        data = mjx_env.data
        qpos = np.tile(data.qpos.copy(), (N_steps, 1))
        qvel = np.tile(data.qvel.copy(), (N_steps, 1))
        xpos = np.tile(data.xpos.copy(), (N_steps, 1, 1))
        xquat = np.tile(data.xquat.copy(), (N_steps, 1, 1))
        cvel = np.tile(data.cvel.copy(), (N_steps, 1, 1))
        subtree_com = np.tile(data.subtree_com.copy(), (N_steps, 1, 1))
        site_xpos = np.tile(data.site_xpos.copy(), (N_steps, 1, 1))
        site_xmat = np.tile(data.site_xmat.copy(), (N_steps, 1, 1))
        njnt = model.njnt
        jnt_type = model.jnt_type.copy()
        jnt_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(njnt)]
        traj_info = TrajectoryInfo(jnt_names, model=TrajectoryModel(njnt, jnp.array(jnt_type)), frequency=1 / mjx_env.dt)
        traj_data = TrajectoryData(jnp.array(qpos), jnp.array(qvel), jnp.array(xpos), jnp.array(xquat), jnp.array(cvel),
                                   jnp.array(subtree_com), jnp.array(site_xpos), jnp.array(site_xmat),
                                   split_points=jnp.array([0, N_steps]))
        return TrajClass(traj_info, traj_data)

    def make_falling_trajectory():
        N_steps = 1000
        mjx_env = DummyHumamoidEnv(enable_mjx=False, **DEFAULTS)
        key = jax.random.PRNGKey(0)
        mjx_env.reset(key)
        action_dim = mjx_env.info.action_space.shape[0]
        qpos, qvel, xpos, xquat, cvel, subtree_com, site_xpos, site_xmat = [], [], [], [], [], [], [], []
        for i in range(N_steps):
            action = np.zeros(action_dim)
            mjx_env.step(action)
            data = mjx_env.get_data()
            qpos.append(data.qpos)
            qvel.append(data.qvel)
            xpos.append(data.xpos)
            xquat.append(data.xquat)
            cvel.append(data.cvel)
            subtree_com.append(data.subtree_com)
            site_xpos.append(data.site_xpos)
            site_xmat.append(data.site_xmat)
        model = mjx_env.get_model()
        qpos = np.stack(qpos)
        qvel = np.stack(qvel)
        xpos = np.stack(xpos)
        xquat = np.stack(xquat)
        cvel = np.stack(cvel)
        subtree_com = np.stack(subtree_com)
        site_xpos = np.stack(site_xpos)
        site_xmat = np.stack(site_xmat)
        njnt = model.njnt
        jnt_type = model.jnt_type
        jnt_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(njnt)]
        traj_info = TrajectoryInfo(jnt_names, model=TrajectoryModel(njnt, jnp.array(jnt_type)), frequency=1 / mjx_env.dt)
        traj_data = TrajectoryData(jnp.array(qpos), jnp.array(qvel), jnp.array(xpos), jnp.array(xquat), jnp.array(cvel),
                                   jnp.array(subtree_com), jnp.array(site_xpos), jnp.array(site_xmat),
                                   split_points=jnp.array([0, N_steps]))
        return TrajClass(traj_info, traj_data)

    print("=" * 60)
    print("REWARD TESTS - New expected values (use atol=1e-7, rtol=0)")
    print("=" * 60)

    standing_trajectory = make_standing_trajectory()
    falling_trajectory = make_falling_trajectory()

    reward_configs = [
        ("TargetXVelocityReward", {"reward_type": "TargetXVelocityReward", "reward_params": dict(target_velocity=1.0)}),
        ("TargetVelocityGoalReward", {"goal_type": "GoalRandomRootVelocity", "reward_type": "TargetVelocityGoalReward"}),
        ("LocomotionReward", {"goal_type": "GoalRandomRootVelocity", "reward_type": "LocomotionReward", "reward_params": dict(joint_position_limit_coeff=1.0)}),
        ("TargetVelocityTrajReward", {"reward_type": "TargetVelocityTrajReward"}),
        ("MimicReward", {"reward_type": "MimicReward"}),
    ]

    for name, kwargs in reward_configs:
        for backend in ["numpy", "jax"]:
            _apply_mock_random()
            if backend == "numpy":
                expert_traj = make_standing_trajectory()
                nominal_traj = make_falling_trajectory()
                expert_traj.data = expert_traj.data.to_numpy()
                nominal_traj.data = nominal_traj.data.to_numpy()
            else:
                expert_traj = make_standing_trajectory()
                nominal_traj = make_falling_trajectory()
            transitions = generate_test_trajectories(expert_traj, nominal_traj, backend, horizon=100, **kwargs)
            reward_sum = float(np.sum(transitions.rewards)) if backend == "numpy" else float(jnp.sum(transitions.rewards))
            reward_42 = float(transitions.rewards[42]) if backend == "numpy" else float(transitions.rewards[42])
            print(f"\n{name} [{backend}]:")
            print(f"  reward_sum = {reward_sum}")
            print(f"  reward_42  = {reward_42}")

    print("\n" + "=" * 60)
    print("INITIAL STATE HANDLER - DefaultInitialStateHandler")
    print("=" * 60)

    from loco_mujoco.core.utils import mj_jntname2qposid

    def _create_env(backend, init_state_type, init_state_params=None, trajectory=None):
        mjx_env = DummyHumamoidEnv(enable_mjx=backend == "jax",
                                   init_state_type=init_state_type,
                                   init_state_params=init_state_params,
                                   **DEFAULTS)
        if trajectory is not None:
            if backend == "numpy":
                trajectory.data = trajectory.data.to_numpy()
            mjx_env.process_trajectory(trajectory)
        return mjx_env

    for backend in ["numpy", "jax"]:
        seed = 0
        key = jax.random.PRNGKey(seed)
        mjx_env_1 = _create_env(backend, "DefaultInitialStateHandler")
        qpos_init = np.zeros(18)
        qpos_init[:7] = np.array([1.5, 1.2, 0.3, 1., 0, 0., 0.])
        j_id = mj_jntname2qposid("abdomen_z", mjx_env_1._model)
        qpos_init[j_id] = 0.3
        qvel_init = 0.1 * np.ones(17)
        init_state_params = dict(qpos_init=qpos_init, qvel_init=qvel_init)
        mjx_env_2 = _create_env(backend, "DefaultInitialStateHandler", init_state_params=init_state_params)

        if backend == "numpy":
            state_0 = mjx_env_1.reset(key)
            state_1 = mjx_env_2.reset(key)
            print(f"\nDefaultInitialStateHandler [numpy] state_0 = np.array({np.array(state_0).tolist()})")
            print(f"DefaultInitialStateHandler [numpy] state_1 = np.array({np.array(state_1).tolist()})")
        else:
            state_0 = mjx_env_1.mjx_reset(key)
            state_1 = mjx_env_2.mjx_reset(key)
            o0 = np.array(state_0.observation)
            o1 = np.array(state_1.observation)
            print(f"\nDefaultInitialStateHandler [jax] state_0.observation = np.array({o0.tolist()})")
            print(f"DefaultInitialStateHandler [jax] state_1.observation = np.array({o1.tolist()})")

    print("\n" + "=" * 60)
    print("INITIAL STATE HANDLER - TrajInitialStateHandler")
    print("=" * 60)

    for backend in ["numpy", "jax"]:
        _apply_mock_random()
        nr.seed(0)
        falling_traj = make_falling_trajectory()  # fresh trajectory per backend
        if backend == "numpy":
            falling_traj.data = falling_traj.data.to_numpy()
        seed = 0
        key = jax.random.PRNGKey(seed)
        mjx_env = _create_env(backend, "TrajInitialStateHandler", trajectory=falling_traj)
        if backend == "numpy":
            state_0 = mjx_env.reset(key)
            state_1 = mjx_env.reset(key)
            state_2 = mjx_env.reset(key)
            print(f"\nTrajInitialStateHandler [numpy] state_0 = np.array({np.array(state_0).tolist()})")
            print(f"TrajInitialStateHandler [numpy] state_1 = np.array({np.array(state_1).tolist()})")
            print(f"TrajInitialStateHandler [numpy] state_2 = np.array({np.array(state_2).tolist()})")
        else:
            state_0 = mjx_env.mjx_reset(key)
            state_1 = mjx_env.mjx_reset(jax.random.PRNGKey(seed + 1))
            state_2 = mjx_env.mjx_reset(jax.random.PRNGKey(seed + 2))
            print(f"\nTrajInitialStateHandler [jax] state_0 = np.array({np.array(state_0.observation).tolist()})")
            print(f"TrajInitialStateHandler [jax] state_1 = np.array({np.array(state_1.observation).tolist()})")
            print(f"TrajInitialStateHandler [jax] state_2 = np.array({np.array(state_2.observation).tolist()})")

    print("\nDone. Use the values above to update test_reward.py and test_initial_state_handler.py with atol=1e-7.")


if __name__ == "__main__":
    main()
