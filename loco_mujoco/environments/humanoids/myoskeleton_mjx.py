"""
Mjx version of the MyoSkeleton environment.
"""

import mujoco
from mujoco import MjSpec

from .myoskeleton import MyoSkeleton


class MjxMyoSkeleton(MyoSkeleton):
    """
    Mjx version of the :class:`~loco_mujoco.environments.humanoids.myoskeleton.MyoSkeleton` environment.

    The model, the observation space and the action space are the same as in the CPU environment, but
    ``mjx_enabled`` is set so that the environment can be simulated with the JAX-based backend
    (functional ``mjx_reset``/``mjx_step`` interface that can be jit-compiled and vectorized). Compared to the
    CPU environment, the default timestep, the number of substeps and the solver options are changed, and by
    default (``mjx_reduced_contacts=True``) all contacts are disabled
    except the ones between the collision geometries of the feet and the floor.

    """

    mjx_enabled = True

    def __init__(self, timestep=0.002, n_substeps=5, **kwargs):
        """
        Constructor.

        Args:
            timestep (float): Simulation timestep in seconds.
            n_substeps (int): Number of simulation steps per environment step.
            **kwargs: Additional arguments passed to :class:`~loco_mujoco.environments.humanoids.myoskeleton.MyoSkeleton`. If no
                ``model_option_conf`` is given, the solver is configured with 2 iterations and 4 line-search
                iterations, and the implicit joint damping of the Euler integrator is disabled.

        """
        if "model_option_conf" not in kwargs.keys():
            model_option_conf = dict(iterations=2, ls_iterations=4, disableflags=mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
        else:
            model_option_conf = kwargs["model_option_conf"]
            del kwargs["model_option_conf"]
        super().__init__(timestep=timestep, n_substeps=n_substeps, model_option_conf=model_option_conf, **kwargs)

    def _modify_spec_for_mjx(self, spec: MjSpec):
        """
        Reduces the contacts of the model to speed up the simulation with Mjx.

        Mjx is slow in handling many complex contacts. Therefore, all contacts are disabled except the ones
        between the collision geometries of the feet (three per foot and two per forefoot) and the floor.

        Args:
            spec (MjSpec): Mujoco specification.

        Returns:
            MjSpec: Modified Mujoco specification.

        """

        # --- 1. Make all geoms have contype and conaffinity of 0 ---
        for g in spec.geoms:
            g.contype = 0
            g.conaffinity = 0

        # --- 2. Define specific contact pairs ---
        spec.add_pair(geomname1="floor", geomname2="foot1_r_coll")
        spec.add_pair(geomname1="floor", geomname2="foot2_r_coll")
        spec.add_pair(geomname1="floor", geomname2="foot3_r_coll")
        spec.add_pair(geomname1="floor", geomname2="bofoot1_r_coll")
        spec.add_pair(geomname1="floor", geomname2="bofoot2_r_coll")
        spec.add_pair(geomname1="floor", geomname2="foot1_l_coll")
        spec.add_pair(geomname1="floor", geomname2="foot2_l_coll")
        spec.add_pair(geomname1="floor", geomname2="foot3_l_coll")
        spec.add_pair(geomname1="floor", geomname2="bofoot1_l_coll")
        spec.add_pair(geomname1="floor", geomname2="bofoot2_l_coll")

        return spec
