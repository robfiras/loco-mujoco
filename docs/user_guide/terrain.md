# Terrain

The terrain is the ground that the robot moves on. It is a module of the environment like the observations
or the reward: it is chosen with `terrain_type` and configured with `terrain_params`, and everything else
about the task stays as it is.

```python
from loco_mujoco import ImitationFactory

env, traj = ImitationFactory.make("UnitreeG1", default_dataset_conf=dict(task="walk"),
                                  terrain_type="RoughTerrain",
                                  terrain_params=dict(random_min_height=-0.05, random_max_height=0.05))
```

```{image} ../../imgs/terrains.gif
:align: center
:alt: The flat terrain, the rough terrain and a heightmap terrain
```

*The three built-in terrains: a `StaticTerrain` never changes after construction, a `DynamicTerrain` writes
a heightfield of its own for every environment at run time. The map of the `HeightmapTerrain` is an
example. The terrain loads any height map.*

## Which terrain to use

| Terrain | What it is | Where an episode starts | Backends |
|---|---|---|---|
| `FlatTerrain` (default) | The floor of the model, a plane at height 0. A `StaticTerrain`; `terrain_type="StaticTerrain"` selects the same terrain. | At the world origin | MuJoCo, MJX, MuJoCo Warp |
| `RoughTerrain` | A random uneven heightfield of 8 m x 8 m with a flat platform in the middle. **Every environment has its own**, and it is generated anew on every reset. A `DynamicTerrain`. | At the world origin, on the platform | MuJoCo, MJX. Not MuJoCo Warp, which keeps one heightfield for all worlds. |
| `HeightmapTerrain` | **One** static terrain for all environments, loaded from a height map file. It is aligned with the world frame, for datasets whose clips are already placed on the map, such as motions on stairs. A `StaticTerrain`. | Where its clip is on the map | MuJoCo, MJX, MuJoCo Warp |

In short:

- Use `RoughTerrain` for robustness to uneven ground on MuJoCo and MJX. Each environment sees a new random
  surface on every reset.
- Use `HeightmapTerrain` when the reference motions belong to a particular piece of terrain, and for any
  uneven terrain on MuJoCo Warp. The terrain comes from a file instead of being generated and never changes
  after construction, so thousands of environments share it, and the robot starts where the clip starts
  instead of at the origin.

## Rough terrain

{class}`~loco_mujoco.core.terrain.rough.RoughTerrain` replaces the floor of the model with a heightfield of
80 x 80 samples that covers 8 m x 8 m. On every reset it draws random heights on a coarse grid, interpolates
them to the full resolution, and flattens a square platform around the origin so that the robot starts on
level ground.

```{literalinclude} ../../examples/tutorials/09_terrain.py
:language: python
```

| Parameter | Default | Meaning |
|---|---|---|
| `random_min_height` | -0.05 | Lowest height of the random surface in meters |
| `random_max_height` | 0.05 | Highest height of the random surface in meters |
| `random_step` | 0.005 | Step between the height values that are drawn, in meters |
| `random_downsampled_scale` | 0.4 | Resolution of the coarse grid relative to the heightfield. Smaller values give smoother terrain. |
| `inner_platform_size_in_meters` | 1.0 | Edge length of the flat platform at the origin |

The heightfield is finite. When the root of the robot comes within 0.5 m of its border, the horizontal
position of the root is set back to the origin. All other parts of the simulation state stay as they are,
so a policy can walk on indefinitely.

The heightfield of an environment is part of its state (`carry.terrain_state`) and is written into the
model before every step. The viewer uploads the heightfield again on every rendered frame, which makes
rendering slower than on a static terrain. When many MJX environments are rendered together, the surface of
the first environment is shown for all of them.

## Heightmap terrain

{class}`~loco_mujoco.core.terrain.heightmap.HeightmapTerrain` loads a terrain from an `.npz` file with the
following arrays:

| Array | Shape | Meaning |
|---|---|---|
| `h` | (ny, nx) | Heights in meters. Cell (i, j) is centered at `origin + (j * cell, i * cell)`. |
| `cell` | scalar | Cell size in meters |
| `origin` | (2,) | World position of cell (0, 0). Optional, the default is the world origin. |

The following example writes a map of 10 m x 5 m with a step of 15 cm that starts 3 m in front of the origin,
and loads it into an environment without a reference trajectory:

```python
import numpy as np
from loco_mujoco import RLFactory

h = np.zeros((100, 200), dtype=np.float32)
h[:, 100:] = 0.15
np.savez("step.npz", h=h, cell=0.05, origin=np.array([-2.0, -2.5]))

env, _ = RLFactory.make("UnitreeG1", terrain_type="HeightmapTerrain", terrain_params=dict(file="step.npz"))
```

```{important}
With a reference trajectory, the clips have to be placed on the map already. A `HeightmapTerrain` does not
move trajectories to the origin (see [Where episodes start](#where-episodes-start)), so a clip that was
recorded elsewhere in the world, such as the walking clips of the default dataset, starts wherever its root
is in the data, which is usually outside the map.
```

| Parameter | Default | Meaning |
|---|---|---|
| `file` | required | Path of the height map |
| `contact` | `"hfield"` | Collision mode, see below |
| `contact_geoms` | `None` | Names of the geoms that collide with the terrain, for example the feet. `None` lets all geoms collide. A fine heightfield creates many contacts per touching geom, so restricting the geoms makes the simulation faster. |
| `ground_clearance` | 0.002 | On reset the robot is lifted until its colliding geoms are at least this far above the terrain, in meters. Reference motions and reset noise can otherwise place feet inside steps. `None` turns the correction off. |
| `tile_size` | 8.0 | Edge length in meters of the square tiles that the map is split into. `None` keeps one heightfield. Tiles keep collisions accurate on large maps on MuJoCo Warp. |
| `base_depth` | 1.0 | Depth of the base under the lowest point of each tile, in meters |
| `box_quant` | 0.01 | Height quantization in meters for the box collision modes |
| `pool_file` | `None` | Table of boxes per trajectory, required for `contact="pool"` |
| `cache_dir` | `~/.cache/loco_mujoco/hfields` | Where the tiles are written in MuJoCo's binary heightfield format. They are created once per map. |

The three collision modes trade accuracy on step edges against speed:

| `contact` | Collision surface | Use it for |
|---|---|---|
| `"hfield"` | The heightfield tiles | Smooth or irregular terrain |
| `"boxes"` | Axis-aligned boxes that cover the map on a ground plane. Cells of equal height are merged into rectangles. | Stairs and other terrain with sharp edges. A heightfield turns every riser into a slope one cell wide, on which feet slide off the edge. |
| `"pool"` | The same boxes, but each environment only gets the boxes near its own clip | Large maps with many clips, where testing every box in every environment would be slow. The table of boxes is built with `write_pool_file`. |

## Where episodes start

Where an episode starts follows from the terrain and is controlled by the argument `recenter_root_xy` of
the environment. With the default `None`, the reference trajectory is moved so that every episode starts at
the world origin, except on a terrain that is built around the clips of the dataset. There the trajectory
stays where it is in the world.

| Terrain | `recenter_root_xy=None` resolves to | Start of an episode |
|---|---|---|
| `FlatTerrain`, `RoughTerrain` | `True` | World origin |
| `HeightmapTerrain` | `False` | Position of the clip on the map |

Setting `recenter_root_xy=True` on a `HeightmapTerrain` raises an error, because the robot would no longer
stand on the part of the map that its motion belongs to.

## Height queries

All three terrains answer height queries through `get_height_at`, which returns the height of the surface
at given horizontal positions. This is what the
[`HeightScan` observation](observations.md#height-scan) reads: a grid of terrain heights around the root, in
the heading frame of the root.

```python
from loco_mujoco.core import ObservationType

observation_spec = [...,  # the other observations
                    ObservationType.HeightScan("height_scan", size=(1.6, 1.0), resolution=0.1)]
```

## Writing your own terrain

To add a terrain, derive from {class}`~loco_mujoco.core.terrain.base.Terrain`, or from
{class}`~loco_mujoco.core.terrain.static.StaticTerrain` if it does not change at run time, and call
`register()` on the class, as for every other module. The section on terrains in
[Writing custom modules](custom_modules.md#terrains) lists the methods and shows an example.

## See also

- [Observations](observations.md#height-scan) for the height scan.
- [Simulation backends](backends.md) for the differences between MJX and MuJoCo Warp.
- [Trajectories](trajectories.md) for how reference clips are placed in the world.
- API: {mod}`loco_mujoco.core.terrain.base`, {mod}`loco_mujoco.core.terrain.static`,
  {mod}`loco_mujoco.core.terrain.dynamic`, {mod}`loco_mujoco.core.terrain.rough`,
  {mod}`loco_mujoco.core.terrain.heightmap`.
