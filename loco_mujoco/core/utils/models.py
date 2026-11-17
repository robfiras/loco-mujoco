"""Locates the robot model files (XML, meshes, textures) and downloads them on first use.

The model files live in the repository https://github.com/robfiras/loco-mujoco-models and are too
large to ship inside the Python package. They are looked up in this order:

1. the directory given by ``LOCOMUJOCO_ROBOT_MODELS_PATH`` (environment variable, or a variable set
   with ``loco-mujoco-add-variable``), for a local checkout or a machine without network access;
2. an installed ``loco_mujoco_models`` package
   (``pip install git+https://github.com/robfiras/loco-mujoco-models.git``);
3. the cache in ``~/.loco-mujoco-caches/robot_models``, which is filled by downloading the pinned
   revision of the repository the first time a model is needed.
"""
import argparse
import importlib.util
import os
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Optional

import yaml
from filelock import FileLock

MODELS_REPOSITORY = "robfiras/loco-mujoco-models"
# Commit of the models repository this version of LocoMuJoCo was tested with.
MODELS_REVISION = "1aeea8b21cb160ff9f24d9ee8198661c9591f245"
MODELS_URL = f"https://github.com/{MODELS_REPOSITORY}/archive/{MODELS_REVISION}.tar.gz"
MODELS_PATH_VARIABLE = "LOCOMUJOCO_ROBOT_MODELS_PATH"

_PACKAGE_NAME = "loco_mujoco_models"
_COMPLETE_MARKER = ".complete"
_models_dir: Optional[Path] = None   # resolved once per process


def default_cache_dir() -> Path:
    """Directory that holds the downloaded model files, one sub-directory per revision."""
    return Path.home() / ".loco-mujoco-caches" / "robot_models"


def _configured_path() -> Optional[Path]:
    """The user-defined models directory, from the environment or LOCOMUJOCO_VARIABLES.yaml."""
    path = os.environ.get(MODELS_PATH_VARIABLE)
    if not path:
        import loco_mujoco
        try:
            with open(loco_mujoco.PATH_TO_VARIABLES) as f:
                path = (yaml.safe_load(f) or {}).get(MODELS_PATH_VARIABLE)
        except FileNotFoundError:
            path = None
    return Path(os.path.expanduser(str(path))) if path else None


def _installed_package_dir() -> Optional[Path]:
    """The directory of an installed ``loco_mujoco_models`` package, if there is one."""
    spec = importlib.util.find_spec(_PACKAGE_NAME)
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0])


def get_models_dir(download: bool = True) -> Path:
    """
    Returns the directory that contains one folder per robot.

    Args:
        download (bool): Whether to download the models if they are found nowhere.

    Returns:
        Path: The models directory.

    Raises:
        FileNotFoundError: If the configured directory does not exist, or if the models are not
            available locally and ``download`` is False.
    """
    global _models_dir
    if _models_dir is not None:
        return _models_dir

    configured = _configured_path()
    if configured is not None:
        if not configured.is_dir():
            raise FileNotFoundError(f"{MODELS_PATH_VARIABLE} points to '{configured}', which is not a directory.")
        _models_dir = configured
        return _models_dir

    installed = _installed_package_dir()
    if installed is not None:
        _models_dir = installed
        return _models_dir

    cached = default_cache_dir() / MODELS_REVISION[:12]
    if not (cached / _COMPLETE_MARKER).exists():
        if not download:
            raise FileNotFoundError(f"The robot models are not available in '{cached}'. "
                                    "Run `loco-mujoco-download-models` to fetch them.")
        download_models(cached)
    _models_dir = cached
    return _models_dir


def download_models(target: Optional[Path] = None, url: str = MODELS_URL, force: bool = False) -> Path:
    """
    Downloads the pinned revision of the robot models and unpacks it into ``target``.

    Safe to call from several processes at once: a file lock makes the others wait, and the
    directory only appears under its final name once it is complete.

    Args:
        target (Path, optional): Destination directory. Defaults to the cache of the pinned revision.
        url (str): Archive to download.
        force (bool): Download again even if ``target`` is already complete.

    Returns:
        Path: The directory that contains one folder per robot.

    Raises:
        RuntimeError: If the archive cannot be downloaded or holds no model files.
    """
    target = Path(target) if target is not None else default_cache_dir() / MODELS_REVISION[:12]
    target.parent.mkdir(parents=True, exist_ok=True)

    with FileLock(str(target) + ".lock"):
        if (target / _COMPLETE_MARKER).exists() and not force:
            return target

        print(f"[LocoMuJoCo] Downloading the robot models (about 200 MB, only once) to {target} ...", flush=True)
        work_dir = Path(tempfile.mkdtemp(prefix=".download-", dir=target.parent))
        try:
            archive = work_dir / "models.tar.gz"
            try:
                with urllib.request.urlopen(url, timeout=60) as response, open(archive, "wb") as f:
                    shutil.copyfileobj(response, f, length=1 << 20)
            except (urllib.error.URLError, OSError) as e:
                raise RuntimeError(
                    f"Could not download the robot models from {url} ({e}). On a machine without network "
                    f"access, copy the '{_PACKAGE_NAME}' folder of https://github.com/{MODELS_REPOSITORY} "
                    f"to this machine and set {MODELS_PATH_VARIABLE} to it.") from e

            unpacked = work_dir / "models"
            n_files = _extract_models(archive, unpacked)
            if n_files == 0:
                raise RuntimeError(f"The archive at {url} contains no '{_PACKAGE_NAME}' folder.")
            (unpacked / _COMPLETE_MARKER).write_text(f"{url}\n")

            if target.exists():
                shutil.rmtree(target)
            os.replace(unpacked, target)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
        print(f"[LocoMuJoCo] Robot models ready ({n_files} files).", flush=True)
    return target


def _extract_models(archive: Path, destination: Path) -> int:
    """
    Unpacks the ``loco_mujoco_models`` folder of a repository archive into ``destination``.

    Only regular files below ``<top-level directory>/loco_mujoco_models/`` are written, each to a
    path built from its own name parts, so an archive cannot place files outside ``destination``.

    Returns:
        int: The number of files written.
    """
    n_files = 0
    with tarfile.open(archive) as tar:
        for member in tar:
            parts = PurePosixPath(member.name).parts
            if (not member.isfile() or len(parts) < 3 or parts[1] != _PACKAGE_NAME
                    or member.name.startswith("/") or ".." in parts):
                continue
            out = destination.joinpath(*parts[2:])
            out.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
            n_files += 1
    return n_files


def download_models_cli():
    """Entry point of ``loco-mujoco-download-models``."""
    parser = argparse.ArgumentParser(
        description="Download the robot models used by LocoMuJoCo. This happens automatically the first "
                    "time an environment is created; run it ahead of time on clusters or in containers.")
    parser.add_argument("--force", action="store_true", help="download again even if the models are present")
    args = parser.parse_args()

    if not args.force:
        for source, path in ((MODELS_PATH_VARIABLE, _configured_path()),
                             (f"the installed {_PACKAGE_NAME} package", _installed_package_dir())):
            if path is not None:
                print(f"The robot models are taken from {source}: {path}. Nothing to download.")
                return
    print(f"Robot models available in {download_models(force=args.force)}")
