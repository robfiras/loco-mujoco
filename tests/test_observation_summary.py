"""Tests for MujocoBase.create_observation_summary.

This method builds a (large) HTML summary of an environment's observation
container and uploads it to 0x0.st. The upload is the only networked part, so
we monkeypatch ``requests.post`` (and ``webbrowser.open``) to exercise the
whole HTML-generation path offline -- both the success branch (returns the
uploaded URL, optionally writes a local file) and the failure branch.
"""
import jax
import pytest

from test_conf import DummyHumamoidEnv
from loco_mujoco.core import mujoco_base

DEFAULTS = {"horizon": 1000, "gamma": 0.99, "n_envs": 1}

jax.config.update("jax_platform_name", "cpu")


class _FakeResponse:
    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text


@pytest.fixture
def env():
    return DummyHumamoidEnv(enable_mjx=False, **DEFAULTS)


def test_observation_summary_success_writes_file(env, tmp_path, monkeypatch):
    posted = {}

    def fake_post(url, files=None, headers=None):
        posted["url"] = url
        posted["html"] = files["file"][1]
        return _FakeResponse(200, "https://0x0.st/abc.html\n")

    monkeypatch.setattr(mujoco_base.requests, "post", fake_post)
    # open_in_browser=False avoids webbrowser, but guard it anyway
    monkeypatch.setattr(mujoco_base.webbrowser, "open", lambda *_a, **_k: None)

    out_file = tmp_path / "obs_table.html"
    url = env.create_observation_summary(filename=out_file.as_posix(),
                                         open_in_browser=False)

    assert url == "https://0x0.st/abc.html"
    assert posted["url"] == "https://0x0.st"
    # a non-trivial HTML document was generated and saved
    assert out_file.is_file()
    html = out_file.read_text()
    assert "<html>" in html and "</html>" in html
    assert "<table" in html


def test_observation_summary_opens_browser_when_requested(env, monkeypatch):
    opened = {}
    monkeypatch.setattr(mujoco_base.requests, "post",
                        lambda *a, **k: _FakeResponse(200, "https://0x0.st/xyz"))
    monkeypatch.setattr(mujoco_base.webbrowser, "open",
                        lambda url, *a, **k: opened.setdefault("url", url))

    url = env.create_observation_summary(open_in_browser=True)
    assert url == "https://0x0.st/xyz"
    assert opened["url"] == "https://0x0.st/xyz"


def test_observation_summary_upload_failure_raises(env, monkeypatch):
    monkeypatch.setattr(mujoco_base.requests, "post",
                        lambda *a, **k: _FakeResponse(500, "server error"))
    with pytest.raises(Exception, match="Upload failed"):
        env.create_observation_summary(open_in_browser=False)
