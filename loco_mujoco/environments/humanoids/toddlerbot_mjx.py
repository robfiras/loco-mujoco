"""
Mjx (GPU-parallel) version of the ToddlerBot environment.
"""

import mujoco
from mujoco import MjSpec

from .toddlerbot import ToddlerBot


class MjxToddlerBot(ToddlerBot):
    """
    Mjx version of the ToddlerBot environment.

    Same robot, observation space and action space as
    :class:`~loco_mujoco.environments.humanoids.toddlerbot.ToddlerBot`, but with ``mjx_enabled = True``, so that the
    environment can be simulated in parallel on the GPU through the functional JAX interface
    (``mjx_reset``/``mjx_step``). Mjx handles many complex contacts poorly. Collisions are by default reduced to the
    two foot collision geoms against the floor and against each other (see :meth:`_modify_spec_for_mjx`), and a
    cheaper solver configuration is used by default.

    Pass ``mjx_reduced_contacts=False`` to keep the collision geometry of the original model instead.

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
                :class:`~loco_mujoco.environments.humanoids.toddlerbot.ToddlerBot`. If ``model_option_conf`` is not
                given, the MuJoCo options are set to ``iterations=2``, ``ls_iterations=4`` and Euler damping disabled.

        """
        if "model_option_conf" not in kwargs.keys():
            model_option_conf = dict(iterations=2, ls_iterations=4, disableflags=mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
        else:
            model_option_conf = kwargs["model_option_conf"]
            del kwargs["model_option_conf"]
        super().__init__(timestep=timestep, n_substeps=n_substeps, model_option_conf=model_option_conf, **kwargs)

    def _modify_spec_for_mjx(self, spec: MjSpec):
        """
        Simplifies the contacts of the model to speed up the simulation in Mjx.

        All geoms are excluded from collision detection (``contype`` and ``conaffinity`` set to 0). Explicit contact
        pairs are then added between the floor and the two foot collision geoms (``ank_roll_link_collision`` and
        ``ank_roll_link_2_collision``) and between the two feet, so that only these contacts are simulated.

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
        spec.add_pair(geomname1="floor", geomname2="ank_roll_link_collision")
        spec.add_pair(geomname1="floor", geomname2="ank_roll_link_2_collision")
        spec.add_pair(geomname1="ank_roll_link_collision", geomname2="ank_roll_link_2_collision")

        return spec
