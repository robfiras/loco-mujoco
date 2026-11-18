"""
Mjx (GPU-parallel) version of the Unitree G1 environment.
"""

import mujoco
from mujoco import MjSpec

from .unitreeG1 import UnitreeG1


class MjxUnitreeG1(UnitreeG1):
    """
    Mjx version of the Unitree G1 environment.

    Same robot, observation space and action space as
    :class:`~loco_mujoco.environments.humanoids.unitreeG1.UnitreeG1`, but with ``mjx_enabled = True``, so that the
    environment can be simulated in parallel on the GPU through the functional JAX interface
    (``mjx_reset``/``mjx_step``). Mjx handles many complex contacts poorly. Collisions are by default reduced to the
    eight foot collision geoms against the floor (see :meth:`_modify_spec_for_mjx`), and a cheaper solver
    configuration is used by default.

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
                :class:`~loco_mujoco.environments.humanoids.unitreeG1.UnitreeG1`. If ``model_option_conf`` is not
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
        pairs are then added between the floor and the eight foot collision geoms (``right_foot_<i>_col`` and
        ``left_foot_<i>_col`` for ``i = 1..4``), so that only foot-floor contacts are simulated.

        Args:
            spec (MjSpec): MuJoCo specification of the model. It is modified in place.

        Returns:
            MjSpec: The modified MuJoCo specification.

        """

        foot_geoms = ["right_foot_1_col", "right_foot_2_col", "right_foot_3_col", "right_foot_4_col",
                      "left_foot_1_col", "left_foot_2_col", "left_foot_3_col", "left_foot_4_col"]

        # --- Make all geoms have contype and conaffinity of 0 ---
        for g in spec.geoms:
            g.contype = 0
            g.conaffinity = 0

        # --- Define specific contact pairs ---
        for g_name in foot_geoms:
            spec.add_pair(geomname1="floor", geomname2=g_name)

        return spec
