"""
Mjx (GPU-parallel) version of the Unitree H1 environment.
"""

import mujoco
from mujoco import MjSpec

from .unitreeH1 import UnitreeH1


class MjxUnitreeH1(UnitreeH1):
    """
    Mjx version of the Unitree H1 environment.

    Same robot, observation space and action space as
    :class:`~loco_mujoco.environments.humanoids.unitreeH1.UnitreeH1`, but with ``mjx_enabled = True``, so that the
    environment can be simulated in parallel on the GPU through the functional JAX interface
    (``mjx_reset``/``mjx_step``). Mjx handles many complex contacts poorly. The foot meshes are by default replaced by
    two capsules per foot, and collisions are reduced to these capsules against the floor and against the capsules of
    the other foot (see :meth:`_modify_spec_for_mjx`), and a cheaper solver configuration is used by default.

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
                :class:`~loco_mujoco.environments.humanoids.unitreeH1.UnitreeH1`. If ``model_option_conf`` is not
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

        1. The foot mesh geoms (``right_foot`` and ``left_foot``) are deleted.
        2. All remaining geoms are excluded from collision detection (``contype`` and ``conaffinity`` set to 0).
        3. Each foot is replaced by two capsules (``right_foot1``, ``right_foot2``, ``left_foot1``, ``left_foot2``)
           attached to the ankle links.
        4. Explicit contact pairs are added between the floor and each capsule, and between each capsule of the
           right foot and each capsule of the left foot.

        Args:
            spec (MjSpec): MuJoCo specification of the model. It is modified in place.

        Returns:
            MjSpec: The modified MuJoCo specification.

        """

        # --- 1. remove old feet and add new ones ---
        # remove original foot meshes
        for g in spec.geoms:
            if g.name in ["right_foot", "left_foot"]:
                spec.delete(g)

        # --- 2. Make all geoms have contype and conaffinity of 0 ---
        for g in spec.geoms:
            g.contype = 0
            g.conaffinity = 0

        # --- 3. add primitive foot shapes ---
        back_foot_attr = dict(type=mujoco.mjtGeom.mjGEOM_CAPSULE, quat=[1.0, 0.0, 1.0, 0.0], pos=[-0.03, 0.0, -0.05],
                              size=[0.015, 0.025, 0.0], rgba=[1.0, 1.0, 1.0, 0.2], contype=0, conaffinity=0)
        front_foot_attr = dict(type=mujoco.mjtGeom.mjGEOM_CAPSULE, quat=[1.0, 1.0, 0.0, 0.0], pos=[0.15, 0.0, -0.054],
                               size=[0.02, 0.025, 0.0], rgba=[1.0, 1.0, 1.0, 0.2], contype=0, conaffinity=0)

        r_foot_b = [body for body in spec.bodies if body.name == "right_ankle_link"][0]
        r_foot_b.add_geom(name="right_foot1", **back_foot_attr)
        r_foot_b.add_geom(name="right_foot2", **front_foot_attr)

        l_foot_b = [body for body in spec.bodies if body.name == "left_ankle_link"][0]
        l_foot_b.add_geom(name="left_foot1", **back_foot_attr)
        l_foot_b.add_geom(name="left_foot2", **front_foot_attr)

        # --- 4. Define specific contact pairs ---
        spec.add_pair(geomname1="floor", geomname2="right_foot1")
        spec.add_pair(geomname1="floor", geomname2="right_foot2")
        spec.add_pair(geomname1="floor", geomname2="left_foot1")
        spec.add_pair(geomname1="floor", geomname2="left_foot2")
        spec.add_pair(geomname1="right_foot1", geomname2="left_foot1")
        spec.add_pair(geomname1="right_foot1", geomname2="left_foot2")
        spec.add_pair(geomname1="right_foot2", geomname2="left_foot1")
        spec.add_pair(geomname1="right_foot2", geomname2="left_foot2")

        return spec
