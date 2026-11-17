"""Tests for loco_mujoco.core.wrappers.gymnasium.GymnasiumWrapper.

The wrapper is a thin adapter around a LocoMuJoCo environment, so instead of
building a (heavy) real environment we substitute a lightweight fake inner env
and monkeypatch the task factories. This keeps the tests fast + offline while
still exercising every branch of the wrapper (render-mode handling, factory
routing, space conversion, and the delegating methods).
"""
import types

import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")
from gymnasium.spaces import Box  # noqa: E402

from loco_mujoco.core.wrappers import gymnasium as gym_wrapper  # noqa: E402
from loco_mujoco.task_factories import RLFactory, ImitationFactory  # noqa: E402

GymnasiumWrapper = gym_wrapper.GymnasiumWrapper


class _FakeSpace:
    def __init__(self, low, high, shape):
        self.low = np.asarray(low)
        self.high = np.asarray(high)
        self.shape = shape


class _FakeInfo:
    def __init__(self):
        self.observation_space = _FakeSpace([-1.0, -2.0], [1.0, 2.0], (2,))
        self.action_space = _FakeSpace([-1.0], [1.0], (1,))


class _FakeEnv:
    """Records calls so the wrapper's delegation can be asserted."""

    def __init__(self):
        self.dt = 0.02
        self.info = _FakeInfo()
        self.calls = []

    def step(self, action):
        self.calls.append(("step", action))
        return "obs", 1.0, False, False, {"k": "v"}

    def reset(self, key):
        self.calls.append(("reset", key))
        return "reset_obs"

    def render(self, *args):
        self.calls.append(("render", args))
        return np.zeros((4, 3, 3), dtype=np.uint8)

    def stop(self):
        self.calls.append(("stop",))

    def create_dataset(self, **kwargs):
        self.calls.append(("create_dataset", kwargs))
        return {"states": 1}

    def play_trajectory(self, **kwargs):
        self.calls.append(("play_trajectory", kwargs))
        return "played"

    def play_trajectory_from_velocity(self, **kwargs):
        self.calls.append(("play_trajectory_from_velocity", kwargs))
        return "played_vel"


@pytest.fixture
def patched_factories(monkeypatch):
    """Route both factories to a fake env, capturing the kwargs they receive."""
    captured = {"rl": None, "imitation": None, "env": None}

    def _rl_make(env_name, **kwargs):
        env = _FakeEnv()
        captured["rl"] = (env_name, kwargs)
        captured["env"] = env
        return env, None

    def _imitation_make(env_name, **kwargs):
        env = _FakeEnv()
        captured["imitation"] = (env_name, kwargs)
        captured["env"] = env
        return env, None

    monkeypatch.setattr(RLFactory, "make", staticmethod(_rl_make))
    monkeypatch.setattr(ImitationFactory, "make", staticmethod(_imitation_make))
    return captured


# --------------------------- construction / routing ---------------------------

def test_rl_factory_routing_and_spaces(patched_factories):
    env = GymnasiumWrapper("DummyEnv")
    # no dataset conf -> RLFactory used, not ImitationFactory
    assert patched_factories["rl"] is not None
    assert patched_factories["imitation"] is None
    # headless defaults to True (no render mode)
    assert patched_factories["rl"][1]["headless"] is True
    assert isinstance(env.observation_space, Box)
    assert isinstance(env.action_space, Box)
    assert env.observation_space.shape == (2,)
    assert env.action_space.shape == (1,)
    assert env.metadata["render_fps"] == pytest.approx(1.0 / 0.02)


@pytest.mark.parametrize("conf", [
    "default_dataset_conf", "amass_dataset_conf",
    "lafan1_dataset_conf", "custom_dataset_conf",
])
def test_imitation_factory_routing(patched_factories, conf):
    GymnasiumWrapper("DummyEnv", **{conf: {"task": "balance"}})
    assert patched_factories["imitation"] is not None
    assert patched_factories["rl"] is None


def test_human_render_mode_sets_headless_false(patched_factories):
    GymnasiumWrapper("DummyEnv", render_mode="human")
    assert patched_factories["rl"][1]["headless"] is False


def test_headless_kwarg_rejected(patched_factories):
    with pytest.raises(AssertionError, match="headless"):
        GymnasiumWrapper("DummyEnv", headless=True)


def test_invalid_render_mode_rejected(patched_factories):
    with pytest.raises(AssertionError, match="Unsupported render mode"):
        GymnasiumWrapper("DummyEnv", render_mode="not_a_mode")


# --------------------------- delegation ---------------------------

def test_step_delegates(patched_factories):
    env = GymnasiumWrapper("DummyEnv")
    out = env.step(np.array([0.5]))
    assert out == ("obs", 1.0, False, False, {"k": "v"})
    assert patched_factories["env"].calls[0][0] == "step"


def test_reset_with_seed_is_deterministic(patched_factories):
    env = GymnasiumWrapper("DummyEnv")
    obs, info = env.reset(seed=123)
    assert obs == "reset_obs"
    assert info == {}


def test_reset_without_seed(patched_factories):
    env = GymnasiumWrapper("DummyEnv")
    obs, info = env.reset()
    assert obs == "reset_obs"
    assert info == {}


def test_render_human_calls_inner_render(patched_factories):
    env = GymnasiumWrapper("DummyEnv", render_mode="human")
    assert env.render() is None
    assert any(c[0] == "render" for c in patched_factories["env"].calls)


def test_render_rgb_array_swaps_axes(patched_factories):
    env = GymnasiumWrapper("DummyEnv", render_mode="rgb_array")
    img = env.render()
    # inner env returns (4, 3, 3); wrapper swaps axes 0 and 1 -> (3, 4, 3)
    assert img.shape == (3, 4, 3)


def test_render_none_mode_returns_none(patched_factories):
    env = GymnasiumWrapper("DummyEnv")
    assert env.render() is None


def test_close_stops_inner(patched_factories):
    env = GymnasiumWrapper("DummyEnv")
    env.close()
    assert ("stop",) in patched_factories["env"].calls


def test_create_dataset_and_play_delegation(patched_factories):
    env = GymnasiumWrapper("DummyEnv")
    assert env.create_dataset(foo=1) == {"states": 1}
    assert env.play_trajectory(n_episodes=1) == "played"
    assert env.play_trajectory_from_velocity(n_episodes=1) == "played_vel"


def test_unwrapped_returns_inner(patched_factories):
    env = GymnasiumWrapper("DummyEnv")
    assert env.unwrapped is patched_factories["env"]


def test_convert_space_scalar_bounds():
    space = _FakeSpace([-3.0, -1.0], [2.0, 5.0], (2,))
    box = GymnasiumWrapper._convert_space(space)
    assert isinstance(box, Box)
    assert box.shape == (2,)
    # low/high are the global min/max of the source bounds
    assert float(box.low.min()) == -3.0
    assert float(box.high.max()) == 5.0
