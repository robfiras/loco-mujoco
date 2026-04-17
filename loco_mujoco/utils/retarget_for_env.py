"""
CLI: loco-mujoco-retarget-for-new-env

Retargets every default + LAFAN1 mocap clip available on the HuggingFace
dataset repo for a single target environment. Useful when you add a new
custom env and want to avoid the per-run "retarget-on-first-access" cost
(which also makes GPU memory management annoying — see the JAX/PyTorch
preallocation issue in retargeting).

The source envs are read from the env's `default_dataset_source_env` /
`lafan1_dataset_source_env` info properties, with module-level fallbacks.

Retargeted clips land in LOCOMUJOCO_CONVERTED_DEFAULT_PATH / LOCOMUJOCO_CONVERTED_LAFAN1_PATH,
same layout that `ImitationFactory.make(...)` reads from.
"""
# JAX's default 75% GPU pre-allocation starves the PyTorch-based SMPL fitting.
# Must be set before any JAX import.
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse


HF_REPO_ID = "robfiras/loco-mujoco-datasets"


def _list_default_tasks(source_env: str, dataset_type: str = "mocap"):
    """List all task npz files published on HF for a given source env."""
    from huggingface_hub import HfApi
    api = HfApi()
    files = api.list_repo_files(HF_REPO_ID, repo_type="dataset")
    prefix = f"DefaultDatasets/{dataset_type}/{source_env}/"
    return sorted(os.path.basename(f).replace(".npz", "") for f in files if f.startswith(prefix) and f.endswith(".npz"))


def _list_lafan1_clips(source_env: str):
    """List all LAFAN1 clips published on HF for a given source env."""
    from huggingface_hub import HfApi
    api = HfApi()
    files = api.list_repo_files(HF_REPO_ID, repo_type="dataset")
    prefix = f"Lafan1/mocap/{source_env}/"
    return sorted(os.path.basename(f).replace(".npz", "") for f in files if f.startswith(prefix) and f.endswith(".npz"))


def _get_cached_default_path(dataset_type: str, env_name: str, task: str):
    import loco_mujoco
    base = loco_mujoco.get_variable("LOCOMUJOCO_CONVERTED_DEFAULT_PATH")
    if not base:
        raise RuntimeError("LOCOMUJOCO_CONVERTED_DEFAULT_PATH is not set. Run "
                           "loco-mujoco-set-all-caches or set it manually first.")
    return os.path.join(base, dataset_type, env_name, f"{task}.npz")


def _get_cached_lafan1_path(env_name: str, clip: str):
    import loco_mujoco
    base = loco_mujoco.get_variable("LOCOMUJOCO_CONVERTED_LAFAN1_PATH")
    if not base:
        raise RuntimeError("LOCOMUJOCO_CONVERTED_LAFAN1_PATH is not set. Run "
                           "loco-mujoco-set-all-caches or set it manually first.")
    return os.path.join(base, env_name, f"{clip}.npz")


def _resolve_source_envs(env_name_target: str):
    """Read default/lafan1 source envs off the target env class's info properties."""
    from loco_mujoco.environments.base import LocoEnv
    from loco_mujoco.task_factories.imitation_factory import (
        _DEFAULT_RETARGET_SOURCE_ENV, _LAFAN1_RETARGET_SOURCE_ENV,
    )
    env_cls = LocoEnv.registered_envs.get(env_name_target)
    if env_cls is None:
        raise RuntimeError(
            f"Environment '{env_name_target}' is not registered. "
            "Import the package that defines it before invoking this CLI.")
    # Access info properties without instantiating the env — the defaults on
    # BaseRobotHumanoid are class-level methods, so we need an instance for @property access.
    # Instantiate minimally; most LocoEnvs can be built with just defaults.
    try:
        env = env_cls()
        default_src = getattr(env, "default_dataset_source_env", _DEFAULT_RETARGET_SOURCE_ENV)
        lafan1_src = getattr(env, "lafan1_dataset_source_env", _LAFAN1_RETARGET_SOURCE_ENV)
    except Exception:
        # If the env can't be built without args, fall back to the module defaults.
        default_src = _DEFAULT_RETARGET_SOURCE_ENV
        lafan1_src = _LAFAN1_RETARGET_SOURCE_ENV
    return default_src, lafan1_src


def retarget_all_default(env_name_target: str, source_env: str,
                         dataset_type: str = "mocap", skip_existing: bool = True):
    from huggingface_hub import hf_hub_download
    from loco_mujoco.core.trajectory import Trajectory
    from loco_mujoco.smpl.retargeting import retarget_traj_from_robot_to_robot

    tasks = _list_default_tasks(source_env, dataset_type)
    print(f"[retarget] default / {source_env}: {len(tasks)} tasks -> {env_name_target}")
    for task in tasks:
        cached = _get_cached_default_path(dataset_type, env_name_target, task)
        if skip_existing and os.path.exists(cached):
            print(f"  skip (cached): {task}")
            continue
        print(f"  retarget: {task}")
        source_file = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=f"DefaultDatasets/{dataset_type}/{source_env}/{task}.npz",
            repo_type="dataset",
        )
        traj_source = Trajectory.load(source_file)
        traj_target = retarget_traj_from_robot_to_robot(source_env, traj_source, env_name_target)
        os.makedirs(os.path.dirname(cached), exist_ok=True)
        traj_target.save(cached)


def retarget_all_lafan1(env_name_target: str, source_env: str, skip_existing: bool = True):
    from loco_mujoco.datasets.humanoids.LAFAN1 import load_lafan1_trajectory
    from loco_mujoco.smpl.retargeting import retarget_traj_from_robot_to_robot

    clips = _list_lafan1_clips(source_env)
    print(f"[retarget] lafan1 / {source_env}: {len(clips)} clips -> {env_name_target}")
    for clip in clips:
        cached = _get_cached_lafan1_path(env_name_target, clip)
        if skip_existing and os.path.exists(cached):
            print(f"  skip (cached): {clip}")
            continue
        print(f"  retarget: {clip}")
        traj_source = load_lafan1_trajectory(source_env, [clip])
        traj_target = retarget_traj_from_robot_to_robot(source_env, traj_source, env_name_target)
        os.makedirs(os.path.dirname(cached), exist_ok=True)
        traj_target.save(cached)


def retarget_for_new_env_cli():
    parser = argparse.ArgumentParser(
        description="Retarget all HuggingFace default + LAFAN1 clips to a new target env.")
    parser.add_argument("--env", type=str, required=True,
                        help="Name of the target (registered) environment.")
    parser.add_argument("--dataset-types", type=str, nargs="+",
                        default=["default", "lafan1"],
                        choices=["default", "lafan1"],
                        help="Which dataset types to retarget.")
    parser.add_argument("--default-dataset-type", type=str, default="mocap",
                        help="Sub-type for default datasets (e.g. 'mocap').")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-retarget clips even if a cached file already exists.")
    parser.add_argument("--import-module", type=str, nargs="+", default=[],
                        dest="import_modules",
                        help="Python module(s) to import before resolving the env name "
                             "(needed when the target env lives in an external package, "
                             "e.g. --import-module lmj_real).")
    args = parser.parse_args()

    # Import external packages first so their envs register with LocoEnv
    import importlib
    for mod in args.import_modules:
        print(f"Importing {mod} to register custom environments...")
        importlib.import_module(mod)

    default_src, lafan1_src = _resolve_source_envs(args.env)
    print(f"Target env: {args.env}")
    print(f"Default source env: {default_src}")
    print(f"LAFAN1 source env:  {lafan1_src}")

    skip_existing = not args.overwrite
    if "default" in args.dataset_types:
        retarget_all_default(args.env, default_src,
                             dataset_type=args.default_dataset_type,
                             skip_existing=skip_existing)
    if "lafan1" in args.dataset_types:
        retarget_all_lafan1(args.env, lafan1_src, skip_existing=skip_existing)

    print("\nDone.")
