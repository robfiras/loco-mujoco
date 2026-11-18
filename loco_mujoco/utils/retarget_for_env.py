"""
Command-line tool ``loco-mujoco-retarget-for-new-env``.

Retargets every default and LAFAN1 motion capture clip available on the HuggingFace dataset repository for a single
target environment. Useful when you add a new custom environment and want to avoid the cost of retargeting on
first access in every run (which also makes GPU memory management difficult, since the PyTorch-based SMPL fitting
competes with the memory pre-allocation of JAX).

The source environments are read from the ``default_dataset_source_env`` and ``lafan1_dataset_source_env`` info
properties of the target environment, with module-level fallbacks.

Retargeted clips are stored in ``LOCOMUJOCO_CONVERTED_DEFAULT_PATH`` and ``LOCOMUJOCO_CONVERTED_LAFAN1_PATH``, in
the same layout that :meth:`~loco_mujoco.task_factories.imitation_factory.ImitationFactory.make` reads from.
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
    """
    Path of the cached (retargeted) default dataset file of a task for an environment.

    """
    import loco_mujoco
    base = loco_mujoco.get_variable("LOCOMUJOCO_CONVERTED_DEFAULT_PATH")
    if not base:
        raise RuntimeError("LOCOMUJOCO_CONVERTED_DEFAULT_PATH is not set. Run "
                           "loco-mujoco-set-all-caches or set it manually first.")
    return os.path.join(base, dataset_type, env_name, f"{task}.npz")


def _get_cached_lafan1_path(env_name: str, clip: str):
    """
    Path of the cached (retargeted) LAFAN1 clip for an environment.

    """
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


def _extend_if_needed(traj, env_name_target):
    """
    Fill in xpos/xquat/cvel/site_xpos/... so the cached file is complete and later loads do not extend it again.

    """
    from loco_mujoco.smpl.retargeting import load_robot_conf_file
    from loco_mujoco.datasets.humanoids.LAFAN1.load import extend_motion
    if traj.data.is_complete:
        return traj
    robot_conf = load_robot_conf_file(env_name_target)
    return extend_motion(env_name_target, robot_conf, traj,
                         replace_qvel_with_finite_diff=False)


def retarget_all_default(env_name_target: str, source_env: str,
                         dataset_type: str = "mocap", skip_existing: bool = True):
    """
    Retarget all default dataset tasks of a source environment to a target environment.

    The tasks available for the source environment are listed on the HuggingFace dataset repository, downloaded,
    retargeted with :func:`~loco_mujoco.smpl.retargeting.retarget_traj_from_robot_to_robot`, extended with the
    model-specific quantities (body and site poses, velocities), and saved in ``LOCOMUJOCO_CONVERTED_DEFAULT_PATH``.

    Args:
        env_name_target (str): Name of the registered target environment.
        source_env (str): Name of the environment whose datasets are used as the source.
        dataset_type (str): Type of the default datasets; only ``"mocap"`` is available.
        skip_existing (bool): If True, tasks for which a cached file already exists are skipped.

    Raises:
        RuntimeError: If ``LOCOMUJOCO_CONVERTED_DEFAULT_PATH`` is not set.

    """
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
        traj_target = _extend_if_needed(traj_target, env_name_target)
        os.makedirs(os.path.dirname(cached), exist_ok=True)
        traj_target.save(cached)


def retarget_all_lafan1(env_name_target: str, source_env: str, skip_existing: bool = True):
    """
    Retarget all LAFAN1 clips of a source environment to a target environment.

    The clips available for the source environment are listed on the HuggingFace dataset repository, loaded,
    retargeted with :func:`~loco_mujoco.smpl.retargeting.retarget_traj_from_robot_to_robot`, extended with the
    model-specific quantities (body and site poses, velocities), and saved in ``LOCOMUJOCO_CONVERTED_LAFAN1_PATH``.

    Args:
        env_name_target (str): Name of the registered target environment.
        source_env (str): Name of the environment whose LAFAN1 datasets are used as the source.
        skip_existing (bool): If True, clips for which a cached file already exists are skipped.

    Raises:
        RuntimeError: If ``LOCOMUJOCO_CONVERTED_LAFAN1_PATH`` is not set.

    """
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
        traj_target = _extend_if_needed(traj_target, env_name_target)
        os.makedirs(os.path.dirname(cached), exist_ok=True)
        traj_target.save(cached)


def retarget_for_new_env_cli():
    """
    Entry point of the command-line tool ``loco-mujoco-retarget-for-new-env``.

    Arguments:

    - ``--env``: Name of the registered target environment (required).
    - ``--dataset-types``: Which datasets to retarget, ``default`` and/or ``lafan1`` (both by default).
    - ``--default-dataset-type``: Type of the default datasets (``mocap``).
    - ``--overwrite``: Retarget clips again even if a cached file exists.
    - ``--import-module``: Python module(s) to import first, so that environments defined in an external package
      are registered.

    """
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
