"""Smoke tests that construct every registered environment offline.

Each robot lives in its own module under ``environments/humanoids`` /
``environments/quadrupeds`` and defines its observation/action spec, default
xml, PD gains, and (for the GPU backend) an ``_modify_spec_for_mjx`` hook.
None of that is exercised by the ``DummyHumamoidEnv`` used elsewhere, so this
file builds each real robot via ``RLFactory`` (no dataset / network needed) and:

* CPU variants  -> construct + ``reset`` + ``step`` (covers the per-robot spec
  code plus the numpy control/observation path);
* Mjx variants  -> construct only (covers each robot's ``_modify_spec_for_mjx``);
* one Mjx env   -> a full ``mjx_reset``/``mjx_step`` round-trip to drive the
  shared ``mujoco_mjx`` step/reset path once (jit-compiled, so kept to one env).

``MyoSkeleton`` is skipped: it requires a license-gated model download that is
unavailable in CI.
"""
import numpy as np
import jax
import pytest

from loco_mujoco.task_factories import RLFactory
from loco_mujoco.environments.base import LocoEnv

jax.config.update("jax_platform_name", "cpu")

# needs a license-gated asset download that CI can't provide
_SKIP = {"MyoSkeleton", "MjxMyoSkeleton"}

_ALL = [n for n in LocoEnv.registered_envs if n not in _SKIP]
_CPU_ENVS = sorted(n for n in _ALL if not n.startswith("Mjx"))
_MJX_ENVS = sorted(n for n in _ALL if n.startswith("Mjx"))

# robots whose constructor supports trimming arms / the back joint
_DISABLE_ARMS_ENVS = ["Atlas", "UnitreeH1", "UnitreeG1"]


@pytest.mark.parametrize("name", _CPU_ENVS)
def test_construct_reset_step_cpu(name):
    env, extra = RLFactory.make(name)
    assert extra is None

    obs_dim = env.info.observation_space.shape[0]
    assert obs_dim > 0

    obs = env.reset(jax.random.PRNGKey(0))
    assert obs.shape == env.info.observation_space.shape

    action = np.zeros(env.info.action_space.shape, dtype=np.float32)
    obs2, reward, absorbing, done, info = env.step(action)

    assert obs2.shape == env.info.observation_space.shape
    assert np.ndim(reward) == 0
    assert isinstance(bool(absorbing), bool)
    assert isinstance(bool(done), bool)
    assert isinstance(info, dict)


@pytest.mark.parametrize("name", _MJX_ENVS)
def test_construct_mjx(name):
    # constructing the Mjx variant runs each robot's _modify_spec_for_mjx hook
    env, _ = RLFactory.make(name)
    assert env.info.observation_space.shape[0] > 0
    assert env.info.action_space.shape[0] > 0


@pytest.mark.parametrize("name", _DISABLE_ARMS_ENVS)
def test_disable_arms_shrinks_spec(name):
    full, _ = RLFactory.make(name)
    trimmed, _ = RLFactory.make(name, disable_arms=True)
    # removing the arm joints must drop both observations and actuators
    assert trimmed.info.observation_space.shape[0] < full.info.observation_space.shape[0]
    assert trimmed.info.action_space.shape[0] < full.info.action_space.shape[0]


def test_mjx_reset_step_roundtrip():
    # single jit-compiled round-trip to cover the shared mujoco_mjx step/reset
    env, _ = RLFactory.make("MjxUnitreeG1")
    state = env.mjx_reset(jax.random.PRNGKey(0))
    action = np.zeros(env.info.action_space.shape, dtype=np.float32)
    state2 = env.mjx_step(state, action)
    jax.block_until_ready(state2.observation)
    assert state2.observation.shape == state.observation.shape


# ---------------------------------------------------------------------------
# Error and deprecation messages
# ---------------------------------------------------------------------------

def test_myoskeleton_without_model_raises(monkeypatch, tmp_path):
    """A missing MyoSkeleton model raises an error instead of ending the process with exit()."""
    from loco_mujoco.environments.humanoids.myoskeleton import MyoSkeleton
    monkeypatch.setattr(MyoSkeleton, "get_default_xml_file_path",
                        classmethod(lambda cls: str(tmp_path / "missing" / "myoskeleton.xml")))
    with pytest.raises(FileNotFoundError, match="loco-mujoco-myomodel-init"):
        MyoSkeleton()


@pytest.mark.parametrize("deprecated, replacement", [("HumanoidTorque", "SkeletonTorque"),
                                                     ("HumanoidMuscle", "SkeletonMuscle")])
def test_deprecated_skeleton_names_point_to_the_new_class(monkeypatch, deprecated, replacement):
    from loco_mujoco.environments.humanoids import skeletons
    # the warning is emitted before the environment is built, so the build can be skipped
    monkeypatch.setattr(getattr(skeletons, replacement), "__init__", lambda self, *args, **kwargs: None)
    with pytest.warns(DeprecationWarning, match=f"{deprecated} is deprecated.*use {replacement} instead"):
        getattr(skeletons, deprecated)()


def test_missing_default_xml_names_the_environment():
    with pytest.raises(NotImplementedError, match="in the LocoEnv environment"):
        LocoEnv.get_default_xml_file_path()


def test_mjx_has_no_dead_n_envs_property():
    """Mjx.n_envs returned an attribute that was never set; the number of environments is the batch size of
    the state."""
    from loco_mujoco.core.mujoco_mjx import Mjx
    assert not hasattr(Mjx, "n_envs")
