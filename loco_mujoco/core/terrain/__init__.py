from .base import Terrain
from .static import StaticTerrain, FlatTerrain
from .dynamic import DynamicTerrain
from .rough import RoughTerrain
from .heightmap import HeightmapTerrain

# register all terrains
StaticTerrain.register()
FlatTerrain.register()
RoughTerrain.register()
HeightmapTerrain.register()
