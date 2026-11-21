"""Base class of terrains that write their heightfield at runtime."""

from loco_mujoco.core.terrain import Terrain


class DynamicTerrain(Terrain):
    """
    Dynamic terrain class inheriting from Terrain. This class should not be used directly, but should be inherited by
    other terrain classes that are dynamic (e.g., rough terrain).

    """

    @property
    def is_dynamic(self) -> bool:
        """
        Check if the terrain is dynamic.

        Returns:
            bool: True, as this terrain is dynamic.
        """
        return True
