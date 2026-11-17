"""Tests for the observation summaries of MujocoBase.

``create_observation_summary`` writes an HTML summary of an environment's
observation container to a local file and optionally opens it in the browser;
``print_observation_summary`` renders the same tables in the terminal. The
browser is monkeypatched so nothing is opened while testing.
"""
from pathlib import Path

import jax
import pytest

from test_conf import DummyHumamoidEnv
from loco_mujoco.core import mujoco_base

DEFAULTS = {"horizon": 1000, "gamma": 0.99, "n_envs": 1}

jax.config.update("jax_platform_name", "cpu")


@pytest.fixture
def env():
    return DummyHumamoidEnv(enable_mjx=False, **DEFAULTS)


def test_observation_summary_writes_file(env, tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(mujoco_base.webbrowser, "open", lambda url, *_a, **_k: opened.append(url))

    out_file = tmp_path / "obs_table.html"
    path = env.create_observation_summary(filename=out_file.as_posix(), open_in_browser=False)

    assert Path(path) == out_file.resolve()
    assert not opened
    html = out_file.read_text()
    assert "<html>" in html and "</html>" in html
    assert "<table" in html
    # every observation is listed
    for obs in env.obs_container.values():
        assert obs.name in html


def test_observation_summary_default_path_and_browser(env, tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(mujoco_base.tempfile, "gettempdir", lambda: tmp_path.as_posix())
    monkeypatch.setattr(mujoco_base.webbrowser, "open",
                        lambda url, *_a, **_k: opened.append(url) or True)

    path = Path(env.create_observation_summary(open_in_browser=True))

    assert path.is_file() and path.parent == tmp_path.resolve()
    assert opened == [path.as_uri()]


def test_observation_summary_falls_back_to_terminal(env, tmp_path, monkeypatch, capsys):
    # no browser available, e.g. on a headless machine
    monkeypatch.setattr(mujoco_base.webbrowser, "open", lambda *_a, **_k: False)

    env.create_observation_summary(filename=(tmp_path / "obs.html").as_posix())

    out = capsys.readouterr().out
    assert "Could not open a browser" in out
    assert "All observations" in out


def test_print_observation_summary(env, capsys):
    env.print_observation_summary()

    out = capsys.readouterr().out
    assert "Observation Summary" in out
    assert "Obs Indices" in out


def test_print_obs_summary_on_creation(capsys):
    # printed by default ...
    DummyHumamoidEnv(enable_mjx=False, **DEFAULTS)
    assert "Observation Summary" in capsys.readouterr().out

    # ... unless disabled
    DummyHumamoidEnv(enable_mjx=False, print_obs_summary=False, **DEFAULTS)
    assert "Observation Summary" not in capsys.readouterr().out
