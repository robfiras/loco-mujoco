"""
Throughput benchmark for the three LocoMuJoCo simulation backends.

Measures full environment steps per second (physics + control + observation +
reward + termination + auto-reset) under uniformly random actions for

    * ``mujoco`` -- CPU MuJoCo, one environment per process,
    * ``mjx``    -- MJX (JAX) on the accelerator, ``n_envs`` vmapped environments,
    * ``mjwarp`` -- MuJoCo Warp through the MJX API, ``n_envs`` vmapped environments,

and three collision models

    * ``reduced`` -- the ``Mjx<Env>`` model: only the foot-floor contact pairs,
    * ``full``    -- the ``Mjx<Env>`` model with the contact reduction switched off;
                     same timestep and solver settings as ``reduced``. MJX does not
                     implement all geom pairs of every robot, in which case the
                     configuration is reported as failed,
    * ``native``  -- the plain CPU ``<Env>`` model as shipped, with full collisions and
                     the CPU timestep and solver settings (``mujoco`` backend only).

Every configuration runs in a fresh subprocess, so a crash or an out-of-memory
error in one configuration is recorded and does not affect the others.

Examples:

    python speed_test.py                                   # default sweep
    python speed_test.py --env UnitreeH1 --n-envs 1024 4096
    python speed_test.py --backends mjwarp --models full --n-envs 4096
    python speed_test.py --plot results.csv                # plot an existing CSV
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

RESULT_TAG = "SPEED_TEST_RESULT "
FIELDS = ["backend", "model", "env", "n_envs", "status", "steps_per_sec", "steps_per_sec_std",
          "realtime_factor", "compile_time_s", "env_dt", "n_measurements", "error"]


def _make_env(env_name, backend, model, n_envs, ncon_per_env, njmax):
    """Builds the environment for one benchmark configuration."""
    from loco_mujoco import RLFactory
    from loco_mujoco.environments.base import LocoEnv

    kwargs = dict(headless=True)
    if backend == "mjwarp":
        # naconmax bounds the contacts of *all* worlds, njmax the constraints of one world
        kwargs.update(use_mjwarp=True, naconmax=ncon_per_env * n_envs, njmax=njmax)

    if model == "native":
        return RLFactory.make(env_name, **kwargs)[0]

    mjx_name = "Mjx" + env_name
    if model == "full":
        base_cls = LocoEnv.registered_envs[mjx_name]
        attrs = {"_modify_spec_for_mjx": lambda self, spec: spec}
        if backend == "mujoco":
            # plain CPU environment that keeps the Mjx variant's timestep and solver settings
            attrs["mjx_enabled"] = False
        full_cls = type(mjx_name + "FullContact", (base_cls,), attrs)
        full_cls.register()
        mjx_name = full_cls.__name__
    return RLFactory.make(mjx_name, **kwargs)[0]


def _cpu_process(args, barrier, queue):
    """One CPU process: builds an environment and steps it ``n_steps`` times."""
    try:
        queue.put(_cpu_measure(args, barrier))
    except Exception as e:   # noqa: BLE001 -- report instead of leaving the parent waiting
        barrier.abort()
        queue.put(f"{type(e).__name__}: {e}")


def _cpu_measure(args, barrier):
    import numpy as np
    env = _make_env(args.env, "mujoco", args.model, 1, None, None)
    low, high = env.info.action_space.low, env.info.action_space.high
    rng = np.random.default_rng(os.getpid())

    def run(n):
        for _ in range(n):
            _, _, absorbing, done, _ = env.step(rng.uniform(low, high))
            if absorbing or done:
                env.reset()

    env.reset()
    run(50)   # warm-up
    barrier.wait()
    times = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        run(args.n_steps)
        times.append(time.perf_counter() - start)
    return times, float(env.dt)


def _run_cpu(args):
    import multiprocessing as mp
    import numpy as np
    ctx = mp.get_context("spawn")
    barrier, queue = ctx.Barrier(args.n_envs), ctx.Queue()
    procs = [ctx.Process(target=_cpu_process, args=(args, barrier, queue)) for _ in range(args.n_envs)]
    for p in procs:
        p.start()
    out = [queue.get(timeout=args.timeout) for _ in procs]
    for p in procs:
        p.join()
    errors = [o for o in out if isinstance(o, str)]
    if errors:
        raise RuntimeError(errors[0])
    # all processes start together; the slowest one defines the wall-clock time of a repetition
    wall = np.max(np.array([o[0] for o in out]), axis=0)
    sps = args.n_envs * args.n_steps / wall
    return sps, 0.0, out[0][1]


def _run_gpu(args):
    import jax
    import numpy as np
    env = _make_env(args.env, args.backend, args.model, args.n_envs, args.ncon_per_env, args.njmax)
    n_envs = args.n_envs

    def step(state, key):
        actions = jax.vmap(env.sample_action_space)(jax.random.split(key, n_envs))
        return jax.vmap(env.mjx_step)(state, actions), None

    @jax.jit
    def rollout(state, key):
        return jax.lax.scan(step, state, jax.random.split(key, args.n_steps))[0]

    key = jax.random.key(0)
    state = jax.jit(jax.vmap(env.mjx_reset))(jax.random.split(key, n_envs))
    jax.block_until_ready(state.observation)

    # The first call compiles; the second can compile again, because the state returned by
    # the rollout need not have the exact types of the freshly reset one. Both are warm-up.
    n_warmup = 2
    times = []
    for i in range(n_warmup + args.repeats):
        start = time.perf_counter()
        state = rollout(state, jax.random.fold_in(key, i))
        jax.block_until_ready(state.observation)
        times.append(time.perf_counter() - start)
    steady = np.array(times[n_warmup:])
    compile_time = max(sum(times[:n_warmup]) - n_warmup * steady.mean(), 0.0)
    return n_envs * args.n_steps / steady, compile_time, float(env.dt)


def worker(args):
    """Runs one configuration and prints the result as a tagged JSON line."""
    if args.backend == "mujoco":
        os.environ["JAX_PLATFORMS"] = "cpu"   # keep the CPU baseline off the GPU
    result = dict(backend=args.backend, model=args.model, env=args.env, n_envs=args.n_envs)
    try:
        import numpy as np
        sps, compile_time, dt = (_run_cpu if args.backend == "mujoco" else _run_gpu)(args)
        result.update(status="ok", steps_per_sec=float(np.mean(sps)), steps_per_sec_std=float(np.std(sps)),
                      realtime_factor=float(np.mean(sps) * dt), compile_time_s=float(compile_time),
                      env_dt=dt, n_measurements=len(sps))
    except Exception as e:   # noqa: BLE001 -- any failure is a benchmark result
        result.update(status="failed", error=f"{type(e).__name__}: {str(e)[:300]}")
    print(RESULT_TAG + json.dumps(result), flush=True)


def _configurations(args):
    for backend in args.backends:
        for model in args.models:
            if model == "native" and backend != "mujoco":
                continue
            for n in (args.cpu_procs if backend == "mujoco" else args.n_envs):
                yield backend, model, n


def orchestrate(args):
    rows = []
    for backend, model, n in _configurations(args):
        n_steps = args.cpu_steps if backend == "mujoco" else args.n_steps
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--backend", backend, "--model", model,
               "--env", args.env, "--n-envs", str(n), "--n-steps", str(n_steps), "--repeats", str(args.repeats),
               "--ncon-per-env", str(args.ncon_per_env), "--njmax", str(args.njmax),
               "--timeout", str(args.timeout)]
        row = dict(backend=backend, model=model, env=args.env, n_envs=n)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
            tagged = [line for line in proc.stdout.splitlines() if line.startswith(RESULT_TAG)]
            if tagged:
                row = json.loads(tagged[-1][len(RESULT_TAG):])
            else:
                tail = (proc.stderr.strip().splitlines() or ["no output"])[-1]
                row.update(status="failed", error=f"exit code {proc.returncode}: {tail[:300]}")
        except subprocess.TimeoutExpired:
            row.update(status="failed", error=f"timeout after {args.timeout}s")
        rows.append(row)
        if row["status"] == "ok":
            print(f"{backend:7s} {model:8s} n={n:<6d} {row['steps_per_sec']:>12,.0f} steps/s "
                  f"(+-{row['steps_per_sec_std']:,.0f}, {row['realtime_factor']:,.0f}x real time, "
                  f"compile {row['compile_time_s']:.0f}s)", flush=True)
        else:
            print(f"{backend:7s} {model:8s} n={n:<6d} FAILED: {row['error']}", flush=True)
        with open(args.output, "w", newline="") as f:   # rewrite after every run: partial sweeps stay usable
            writer = csv.DictWriter(f, fieldnames=FIELDS, restval="")
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nResults written to {args.output}")


def plot(csv_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(csv_path) as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "ok"]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    series = sorted({(r["backend"], r["model"]) for r in rows})
    for backend, model in series:
        pts = sorted((int(r["n_envs"]), float(r["steps_per_sec"])) for r in rows
                     if (r["backend"], r["model"]) == (backend, model))
        ax.plot(*zip(*pts), marker="o", ms=3, label=f"{backend} ({model})",
                linestyle="--" if backend == "mujoco" else "-")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("parallel environments (CPU: processes)")
    ax.set_ylabel("environment steps / s")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    base = os.path.splitext(csv_path)[0]
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png", dpi=200)
    print(f"Plot written to {base}.pdf and {base}.png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default="UnitreeG1", help="environment name without the 'Mjx' prefix")
    p.add_argument("--backends", nargs="+", default=["mujoco", "mjx", "mjwarp"],
                   choices=["mujoco", "mjx", "mjwarp"])
    p.add_argument("--models", nargs="+", default=["reduced", "full", "native"],
                   choices=["reduced", "full", "native"])
    p.add_argument("--n-envs", nargs="+", type=int, default=[1, 16, 64, 256, 1024, 2048, 4096, 8192],
                   help="parallel environments for mjx / mjwarp (a single value in --worker mode)")
    p.add_argument("--cpu-procs", nargs="+", type=int, default=[1, 4, 8, 16],
                   help="parallel processes for the mujoco backend")
    p.add_argument("--n-steps", type=int, default=100, help="environment steps per measurement (mjx / mjwarp)")
    p.add_argument("--cpu-steps", type=int, default=2000, help="environment steps per measurement (mujoco)")
    p.add_argument("--repeats", type=int, default=5, help="measurements per configuration")
    p.add_argument("--ncon-per-env", type=int, default=64, help="mjwarp: contact buffer size per environment")
    p.add_argument("--njmax", type=int, default=256, help="mjwarp: constraint buffer size per environment")
    p.add_argument("--timeout", type=int, default=1800, help="seconds before a configuration is abandoned")
    p.add_argument("--output", default="speed_test_results.csv")
    p.add_argument("--plot", metavar="CSV", help="plot an existing result file and exit")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--backend", help=argparse.SUPPRESS)
    p.add_argument("--model", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.plot:
        plot(args.plot)
    elif args.worker:
        args.n_envs = args.n_envs[0]
        worker(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
