from .base import Terrain
from .static import StaticTerrain
from .dynamic import DynamicTerrain
from .rough import RoughTerrain
from .shared import SharedTerrain
from .heightmap import HeightmapTerrain

# register all terrains
StaticTerrain.register()
RoughTerrain.register()
SharedTerrain.register()
HeightmapTerrain.register()
