# LocoMuJoCo core

This package contains the simulation base classes (`Mujoco` for CPU MuJoCo, `Mjx` for MJX and MuJoCo Warp) and
the pluggable modules every environment is assembled from: observations, rewards, control functions, domain
randomization, terrain, initial and terminal state handlers, trajectories, wrappers and visuals.

The documentation has moved to the website:

- [Simulation backends](https://loco-mujoco.readthedocs.io/en/latest/user_guide/backends.html): the low-level
  `Mujoco` and `Mjx` classes, the functional MJX interface and the MuJoCo Warp backend
- [Modular architecture](https://loco-mujoco.readthedocs.io/en/latest/user_guide/modular_architecture.html):
  how the modules are selected and registered
- [Rendering and recording](https://loco-mujoco.readthedocs.io/en/latest/user_guide/rendering.html): the OpenGL
  viewer, browser rendering with viser and USD export
- [Writing custom modules](https://loco-mujoco.readthedocs.io/en/latest/user_guide/custom_modules.html)
- [API reference](https://loco-mujoco.readthedocs.io/en/latest/api/index.html)
