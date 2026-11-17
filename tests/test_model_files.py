"""Tests for locating the robot model files and downloading them on first use.

No network is used: the "download" is a small archive on disk, built with the layout of a GitHub
repository archive, and served through a file:// URL.
"""
import importlib.util
import io
import tarfile
import urllib.error

import pytest
import yaml

import loco_mujoco
from loco_mujoco.core import Mujoco
from loco_mujoco.core.utils import models

_TOP = "loco-mujoco-models-0123abc"
_real_installed_package_dir = models._installed_package_dir


def _add(tar, name, content=b"x", type_=tarfile.REGTYPE, linkname=""):
    info = tarfile.TarInfo(name)
    info.type, info.linkname, info.size = type_, linkname, (len(content) if type_ == tarfile.REGTYPE else 0)
    tar.addfile(info, io.BytesIO(content) if type_ == tarfile.REGTYPE else None)


@pytest.fixture
def archive_url(tmp_path):
    """A repository archive with two robots plus entries that must not be unpacked."""
    path = tmp_path / "src" / "models.tar.gz"
    path.parent.mkdir()
    with tarfile.open(path, "w:gz") as tar:
        _add(tar, f"{_TOP}/README.md")                                         # outside the package folder
        _add(tar, f"{_TOP}/loco_mujoco_models/__init__.py", b"")
        _add(tar, f"{_TOP}/loco_mujoco_models/robot_a/a.xml", b"<mujoco/>")
        _add(tar, f"{_TOP}/loco_mujoco_models/robot_b/assets/mesh.stl", b"solid")
        _add(tar, f"{_TOP}/loco_mujoco_models/../../escaped.txt")              # path traversal
        _add(tar, f"{_TOP}/loco_mujoco_models/link", type_=tarfile.SYMTYPE, linkname="/etc/passwd")
    return path.as_uri()


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """No configured path, no installed package, an empty cache and a fresh per-process result."""
    variables = tmp_path / "LOCOMUJOCO_VARIABLES.yaml"
    variables.write_text(yaml.dump({"SOME_OTHER_KEY": "x"}))
    monkeypatch.setattr(loco_mujoco, "PATH_TO_VARIABLES", variables)
    monkeypatch.delenv(models.MODELS_PATH_VARIABLE, raising=False)
    monkeypatch.setattr(models, "_installed_package_dir", lambda: None)
    monkeypatch.setattr(models, "default_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(models, "_models_dir", None)
    return variables


def test_download_unpacks_only_the_model_files(tmp_path, archive_url):
    target = models.download_models(tmp_path / "out", url=archive_url)

    files = sorted(str(p.relative_to(target)) for p in target.rglob("*") if p.is_file())
    assert files == [".complete", "__init__.py", "robot_a/a.xml", "robot_b/assets/mesh.stl"]
    assert (target / "robot_a" / "a.xml").read_bytes() == b"<mujoco/>"
    assert not (tmp_path / "escaped.txt").exists()
    assert not list(tmp_path.glob(".download-*"))   # the working directory is removed


def test_first_use_downloads_once_and_then_reuses_the_cache(tmp_path, archive_url, monkeypatch):
    monkeypatch.setattr(models, "MODELS_URL", archive_url)
    calls = []
    real_download = models.download_models
    monkeypatch.setattr(models, "download_models",
                        lambda target: calls.append(target) or real_download(target, url=archive_url))

    first = models.get_models_dir()
    assert first == tmp_path / "cache" / models.MODELS_REVISION[:12]
    assert (first / "robot_a" / "a.xml").is_file()
    assert Mujoco.get_model_path("robot_a", "a.xml") == str(first / "robot_a" / "a.xml")

    monkeypatch.setattr(models, "_models_dir", None)   # as in a new process
    assert models.get_models_dir() == first
    assert len(calls) == 1


def test_download_again_only_when_forced(tmp_path, archive_url):
    target = models.download_models(tmp_path / "out", url=archive_url)
    (target / "robot_a" / "a.xml").write_text("edited")

    models.download_models(target, url=archive_url)
    assert (target / "robot_a" / "a.xml").read_text() == "edited"
    models.download_models(target, url=archive_url, force=True)
    assert (target / "robot_a" / "a.xml").read_text() == "<mujoco/>"


def test_configured_directory_takes_precedence(tmp_path, isolated, monkeypatch):
    local = tmp_path / "my_models"
    local.mkdir()
    monkeypatch.setattr(models, "_installed_package_dir", lambda: tmp_path / "installed")
    monkeypatch.setattr(models, "download_models", lambda *a, **k: pytest.fail("must not download"))

    isolated.write_text(yaml.dump({models.MODELS_PATH_VARIABLE: str(local)}))
    assert models.get_models_dir() == local

    # the environment variable overrides the variables file
    other = tmp_path / "env_models"
    other.mkdir()
    monkeypatch.setattr(models, "_models_dir", None)
    monkeypatch.setenv(models.MODELS_PATH_VARIABLE, str(other))
    assert models.get_models_dir() == other


def test_configured_directory_must_exist(tmp_path, monkeypatch):
    monkeypatch.setenv(models.MODELS_PATH_VARIABLE, str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="not a directory"):
        models.get_models_dir()


def test_installed_package_is_used_without_downloading(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "_installed_package_dir", lambda: tmp_path / "installed")
    monkeypatch.setattr(models, "download_models", lambda *a, **k: pytest.fail("must not download"))
    assert models.get_models_dir() == tmp_path / "installed"


def test_missing_models_without_download_names_the_command():
    with pytest.raises(FileNotFoundError, match="loco-mujoco-download-models"):
        models.get_models_dir(download=False)


def test_failed_download_explains_the_offline_setup(tmp_path, monkeypatch):
    def offline(*args, **kwargs):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(models.urllib.request, "urlopen", offline)
    with pytest.raises(RuntimeError, match=models.MODELS_PATH_VARIABLE):
        models.download_models(tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".download-*"))


def test_archive_without_models_is_rejected(tmp_path):
    path = tmp_path / "empty.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        _add(tar, f"{_TOP}/README.md")
    with pytest.raises(RuntimeError, match="contains no"):
        models.download_models(tmp_path / "out", url=path.as_uri())


def test_installed_package_lookup_matches_importlib():
    """The real lookup (the fixture replaces it with a stub) agrees with what Python can import."""
    spec = importlib.util.find_spec("loco_mujoco_models")
    assert (_real_installed_package_dir() is None) == (spec is None)
