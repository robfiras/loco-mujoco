"""Smoke test of an installed LocoMuJoCo package, as a new user would use it.

Run it from outside the repository, so that the installed package is imported and not the source tree:

    cd /tmp && JAX_PLATFORMS=cpu python /path/to/install_smoke_test.py

It checks the paths of the getting-started guide on the CPU: a MuJoCo environment, an MJX environment, and an
imitation environment with a clip of the default dataset (downloaded from Hugging Face). The robot models are
downloaded on first use. The script exits with a non-zero status if any check fails.
"""

import platform
import sys
import traceback

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

import loco_mujoco
from loco_mujoco import ImitationFactory, RLFactory


def cpu_environment():
    """Create a CPU environment and step it with random actions."""
    env, _ = RLFactory.make("UnitreeH1", print_obs_summary=False)
    env.reset(jax.random.key(0))
    for _ in range(50):
        env.step(np.random.randn(env.info.action_space.shape[0]))


def mjx_environment():
    """Create an MJX environment and step a batch of it with jit and vmap."""
    env, _ = RLFactory.make("MjxUnitreeH1", print_obs_summary=False)
    n_envs = 8
    state = jax.jit(jax.vmap(env.mjx_reset))(jax.random.split(jax.random.key(0), n_envs))
    step = jax.jit(jax.vmap(env.mjx_step))
    action = jnp.zeros((n_envs, env.info.action_space.shape[0]))
    for _ in range(10):
        state = step(state, action)
    jax.block_until_ready(state.observation)
    assert not bool(jnp.isnan(state.observation).any()), "NaN in the observation"


def imitation_environment():
    """Create an imitation environment with a default-dataset clip and reset it with the trajectory."""
    env, traj = ImitationFactory.make("MjxUnitreeH1", default_dataset_conf=dict(task="walk"),
                                      print_obs_summary=False)
    reset = jax.jit(jax.vmap(env.mjx_reset, in_axes=(0, None)))
    state = reset(jax.random.split(jax.random.key(1), 4), traj)
    jax.block_until_ready(state.observation)


def main() -> int:
    print(f"Python {platform.python_version()}, LocoMuJoCo {loco_mujoco.__version__} from {loco_mujoco.__file__}")
    print(f"MuJoCo {mujoco.__version__}, JAX {jax.__version__}, devices {jax.devices()}")
    checks = [("CPU environment (MuJoCo)", cpu_environment),
              ("MJX environment", mjx_environment),
              ("ImitationFactory with the default walk clip", imitation_environment)]
    failed = []
    for name, check in checks:
        try:
            check()
            print(f"PASS  {name}")
        except Exception:
            print(f"FAIL  {name}")
            traceback.print_exc()
            failed.append(name)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
