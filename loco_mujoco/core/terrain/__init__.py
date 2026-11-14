from .base import Terrain
from .static import StaticTerrain
from .dynamic import DynamicTerrain
from .rough import RoughTerrain
from .shared import SharedTerrain

# register all terrains
StaticTerrain.register()
RoughTerrain.register()
SharedTerrain.register()
