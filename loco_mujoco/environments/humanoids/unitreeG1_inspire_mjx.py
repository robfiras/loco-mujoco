import mujoco
from mujoco import MjSpec

from .unitreeG1_inspire import UnitreeG1Inspire


class MjxUnitreeG1Inspire(UnitreeG1Inspire):

    mjx_enabled = True

    def __init__(self, timestep=0.002, n_substeps=5, **kwargs):
        if "model_option_conf" not in kwargs.keys():
            model_option_conf = dict(iterations=2, ls_iterations=4, disableflags=mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
        else:
            model_option_conf = kwargs["model_option_conf"]
            del kwargs["model_option_conf"]
        super().__init__(timestep=timestep, n_substeps=n_substeps, model_option_conf=model_option_conf, njmax=500, **kwargs)

    def _modify_spec_for_mjx(self, spec: MjSpec):
        """
        Mjx is bad in handling many complex contacts. To speed-up simulation significantly we apply
        some changes to the XML:
            1. Replace the complex foot meshes with primitive shapes. Here, one foot mesh is replaced with
               two capsules.
            2. Disable all contacts except the ones between feet and the floor.

        Args:
            spec: Handle to Mujoco XML.

        Returns:
            Mujoco XML handle.

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
