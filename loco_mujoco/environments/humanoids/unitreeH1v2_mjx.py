"""
Mjx (GPU-parallel) version of the Unitree H1v2 environment.
"""

import mujoco
from mujoco import MjSpec

from .unitreeH1v2 import UnitreeH1v2


class MjxUnitreeH1v2(UnitreeH1v2):
    """
    Mjx version of the Unitree H1v2 environment.

    Same robot, observation space and action space as
    :class:`~loco_mujoco.environments.humanoids.unitreeH1v2.UnitreeH1v2`, but with ``mjx_enabled = True``, so that the
    environment can be simulated in parallel on the GPU through the functional JAX interface
    (``mjx_reset``/``mjx_step``). Mjx handles many complex contacts poorly. One box per foot is by default added as
    collision geometry, and collisions are reduced to these boxes against the floor and against each other (see
    :meth:`_modify_spec_for_mjx`), and a cheaper solver configuration is used by default.

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
                :class:`~loco_mujoco.environments.humanoids.unitreeH1v2.UnitreeH1v2`. If ``model_option_conf`` is not
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

        The following changes are applied:

        1. All geoms are excluded from collision detection (``contype`` and ``conaffinity`` set to 0).
        2. One box geom per foot (``right_foot1`` and ``left_foot1``) is added to the ankle roll links.
        3. Explicit contact pairs are added between the floor and each box, and between the two boxes.

        Args:
            spec (MjSpec): MuJoCo specification of the model. It is modified in place.

        Returns:
            MjSpec: The modified MuJoCo specification.

        """

        # --- 1. Make all geoms have contype and conaffinity of 0 ---
        for g in spec.geoms:
            g.contype = 0
            g.conaffinity = 0

        # --- 2. add primitive foot shapes ---
        foot_attr = dict(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.04, 0.0, -0.02],
                         size=[0.12, 0.04, 0.025], rgba=[1.0, 1.0, 1.0, 0.2], contype=0, conaffinity=0)

        r_foot_b = [body for body in spec.bodies if body.name == "right_ankle_roll_link"][0]
        r_foot_b.add_geom(name="right_foot1", **foot_attr)

        l_foot_b = [body for body in spec.bodies if body.name == "left_ankle_roll_link"][0]
        l_foot_b.add_geom(name="left_foot1", **foot_attr)

        # --- 3. Define specific contact pairs ---
        spec.add_pair(geomname1="floor", geomname2="right_foot1")
        spec.add_pair(geomname1="floor", geomname2="left_foot1")
        spec.add_pair(geomname1="right_foot1", geomname2="left_foot1")

        return spec
