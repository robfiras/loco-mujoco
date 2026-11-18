Terrains
========

The terrain is a module of the environment like the observations or the reward: it is chosen with
``terrain_type`` and configured with ``terrain_params``.

.. code-block:: python

    env, traj = ImitationFactory.make("UnitreeG1", default_dataset_conf=dict(task="walk"),
                                      terrain_type="RoughTerrain",
                                      terrain_params=dict(random_min_height=-0.05, random_max_height=0.05))

.. image:: ../../../imgs/terrains.gif
   :align: center

*The three built-in terrains, with their base class in brackets: a* ``StaticTerrain`` *never changes after
construction, a* ``DynamicTerrain`` *writes a heightfield of its own for every env at run time.*

*The map of the HeightmapTerrain is an example; the terrain loads any height map.*

Which Terrain to Use
--------------------

.. list-table::
   :header-rows: 1
   :widths: 16 34 25 25

   * - terrain
     - what it is
     - where an episode starts
     - backends
   * - ``FlatTerrain`` (default)
     - The floor of the model, a plane at height 0. A ``StaticTerrain``; ``terrain_type="StaticTerrain"``
       selects the same terrain.
     - at the world origin
     - MuJoCo, MJX, MjWarp
   * - ``RoughTerrain``
     - A random uneven heightfield of 8 m x 8 m with a flat platform in the middle. **Every env has its own**,
       and it is generated anew on every reset. A ``DynamicTerrain``.
     - at the world origin, on the platform
     - MuJoCo, MJX. Not MjWarp, which keeps one heightfield for all worlds.
   * - ``HeightmapTerrain``
     - **One** static terrain for all envs, loaded from a height map file (``.npz`` with the heights ``h``, the
       cell size ``cell`` and optionally the ``origin``). It is aligned with the world frame, for datasets
       whose clips are already placed on the map, such as motions on stairs. A ``StaticTerrain``.
     - where its clip is on the map
     - MuJoCo, MJX, MjWarp

In short:

- Use ``RoughTerrain`` for robustness to uneven ground on MuJoCo and MJX. Each env sees a new random surface
  on every reset.
- Use ``HeightmapTerrain`` when the reference motions belong to a particular piece of terrain, and for any
  uneven terrain on MjWarp. The terrain comes from a file instead of being generated, it never changes after
  construction, so thousands of envs share it, and the robot starts where the clip starts instead of at the
  origin.

Where episodes start follows from the terrain and is controlled by ``recenter_root_xy`` of the environment:
by default the trajectory is moved to the origin, and on a ``HeightmapTerrain`` it stays where it is in the
world.

``HeightmapTerrain`` has three collision modes (``contact``): ``"hfield"`` collides with the heightfield
tiles, ``"boxes"`` covers the map with axis-aligned boxes, which keeps the edges of steps sharp, and ``"pool"``
gives each env only the boxes near its own clip.

All three terrains answer height queries (``get_height_at``), which is what the ``HeightScan`` observation
reads: a grid of terrain heights around the root, in the root's heading frame.

.. code-block:: python

    from loco_mujoco.core import ObservationType

    observation_spec = [...,  # the other observations
                        ObservationType.HeightScan("height_scan", size=(1.6, 1.0), resolution=0.1)]

To add a terrain of your own, derive from ``Terrain`` (or from ``StaticTerrain`` if it does not change at run
time) and call ``register()`` on the class, as for every other module (see
:doc:`../tutorials/customizing_environments`).

API
---

.. automodule:: loco_mujoco.core.terrain.base
    :members:
    :undoc-members:
    :private-members:
    :show-inheritance:


.. automodule:: loco_mujoco.core.terrain.static
    :members:
    :undoc-members:
    :private-members:
    :show-inheritance:


.. automodule:: loco_mujoco.core.terrain.dynamic
    :members:
    :undoc-members:
    :private-members:
    :show-inheritance:


.. automodule:: loco_mujoco.core.terrain.rough
    :members:
    :undoc-members:
    :private-members:
    :show-inheritance:


.. automodule:: loco_mujoco.core.terrain.heightmap
    :members:
    :undoc-members:
    :private-members:
    :show-inheritance:
