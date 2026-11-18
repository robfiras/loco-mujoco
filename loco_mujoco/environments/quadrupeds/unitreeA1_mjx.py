"""
Mjx (GPU-parallel) version of the Unitree A1 environment.
"""

import mujoco
from mujoco import MjSpec

from .unitreeA1 import UnitreeA1


class MjxUnitreeA1(UnitreeA1):
    """
    Mjx version of the Unitree A1 environment.

    Same robot, observation space and action space as
    :class:`~loco_mujoco.environments.quadrupeds.unitreeA1.UnitreeA1`, but with ``mjx_enabled = True``, so that the
    environment can be simulated in parallel on the GPU through the functional JAX interface
    (``mjx_reset``/``mjx_step``). Mjx handles many complex contacts poorly. Collisions are reduced to the four foot
    geoms against the floor (see :meth:`_modify_spec_for_mjx`), and a cheaper solver configuration is used by default.

    """

    mjx_enabled = True

    def __init__(self, timestep=0.002, n_substeps=5, **kwargs):
        """
        Constructor.

        Args:
            timestep (float): Simulation timestep in seconds.
            n_substeps (int): Number of simulation steps per environment step. With the defaults, one environment
                step takes 0.01 s.
            **kwargs: Additional arguments passed to
                :class:`~loco_mujoco.environments.quadrupeds.unitreeA1.UnitreeA1`. If ``model_option_conf`` is not
                given, the MuJoCo options are set to ``iterations=8``, ``ls_iterations=12``, a pyramidal friction
                cone, ``impratio=1`` and Euler damping disabled.

        """
        if "model_option_conf" not in kwargs.keys():
            model_option_conf = dict(iterations=8, ls_iterations=12,
                                     cone=mujoco.mjtCone.mjCONE_PYRAMIDAL,
                                     impratio=1,
                                     disableflags=mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
        else:
            model_option_conf = kwargs["model_option_conf"]
            del kwargs["model_option_conf"]
        super().__init__(timestep=timestep, n_substeps=n_substeps, model_option_conf=model_option_conf, **kwargs)

    def _modify_spec_for_mjx(self, spec: MjSpec) -> MjSpec:
        """
        Simplifies the contacts of the model to speed up the simulation in Mjx.

        All geoms are excluded from collision detection (``contype`` and ``conaffinity`` set to 0). Explicit contact
        pairs are then added between the floor and the four foot geoms (``RL_foot``, ``RR_foot``, ``FL_foot`` and
        ``FR_foot``), so that only foot-floor contacts are simulated.

        Args:
            spec (MjSpec): MuJoCo specification of the model. It is modified in place.

        Returns:
            MjSpec: The modified MuJoCo specification.

        """

        # --- 1. Make all geoms have contype and conaffinity of 0 ---
        for g in spec.geoms:
            g.contype = 0
            g.conaffinity = 0

        # --- 2. Define specific contact pairs ---
        spec.add_pair(geomname1="floor", geomname2="RL_foot")
        spec.add_pair(geomname1="floor", geomname2="RR_foot")
        spec.add_pair(geomname1="floor", geomname2="FL_foot")
        spec.add_pair(geomname1="floor", geomname2="FR_foot")

        return spec
