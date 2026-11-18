# Rendering and recording

LocoMuJoCo has three ways to look at a simulation. They all read the same simulation state, so you can
switch between them without changing the environment.

| Renderer | Methods | Output | Typical use |
|---|---|---|---|
| OpenGL viewer | `render`, `mjx_render` | A local window, or frames and an MP4 video | Interactive use on a workstation, video recording, also headless |
| Viser viewer | `render_viser`, `mjx_render_viser` | A 3D scene in the browser | Remote machines over SSH, machines without a display |
| USD export | `render_usd`, `mjx_render_usd` | An animated USD stage on disk | Offline rendering in Blender or another DCC tool |

```python
import numpy as np
from loco_mujoco import ImitationFactory

env, traj = ImitationFactory.make("UnitreeH1", default_dataset_conf=dict(task="walk"))

env.reset()
for _ in range(500):
    env.step(np.zeros(env.info.action_space.shape[0]))
    env.render()        # or env.render_viser()
env.stop()
```

To replay a dataset instead of running a policy, use
{meth}`~loco_mujoco.environments.base.LocoEnv.play_trajectory`, which accepts the same choice of renderer,
see [Replay a motion](../getting_started/replay_motion.md).

## The OpenGL viewer

{meth}`~loco_mujoco.core.mujoco_base.Mujoco.render` opens a window on its first call and draws the current
state on every further call. It plays the simulation in real time: the viewer draws as many frames per call
as it needs to keep pace. {meth}`~loco_mujoco.core.mujoco_base.Mujoco.stop` closes the window.

For parallel MJX environments, {meth}`~loco_mujoco.core.mujoco_mjx.Mjx.mjx_render` takes a batched
{class}`~loco_mujoco.core.mujoco_mjx.MjxState` and draws all environments side by side on a square grid:

```python
state = jax.vmap(env.mjx_reset)(keys)
env.mjx_render(state)
```

### Keyboard and mouse

| Input | Effect |
|---|---|
| `Space` | Pause and resume |
| `Tab` | Switch between the camera modes `static`, `follow` and `top_static` |
| `S` / `F` | Halve / double the playback speed |
| `H` | Hide or show the menu |
| `T` | Make the model transparent |
| `E` | Show or hide the reference frames |
| `C` | Show or hide contact forces |
| `M` | Show or hide the mimic sites, if the robot has them |
| `0` to `5` | Show or hide the geom group with this number |
| Left mouse button and drag | Rotate the camera |
| Right mouse button and drag | Move the camera |
| Scroll wheel, or middle button and drag | Zoom |

In the robot models, geom group 0 holds the geometry of the robot and group 2 the floor. These two are
visible by default. Some models keep their collision geometry in a separate group, for example group 1 in
UnitreeH1 and group 3 in UnitreeGo2. Pressing its number shows what the robot collides with.

### Viewer parameters

The viewer is configured through keyword arguments of the environment. Every argument that the environment
does not know is passed on to the viewer, so they can be given to a task factory directly:

```python
env, traj = ImitationFactory.make("UnitreeH1", default_dataset_conf=dict(task="walk"),
                                  viewer_size=(1920, 1080), default_camera_mode="static", start_paused=True)
```

| Parameter | Default | Meaning |
|---|---|---|
| `viewer_size` | `(1280, 720)` | Width and height of the window, and of recorded frames |
| `default_camera_mode` | `"follow"` | Camera at startup: `"follow"` tracks the robot, `"static"` and `"top_static"` are fixed |
| `camera_params` | `None` | Dictionary with one dictionary per camera mode (`distance`, `elevation`, `azimuth`, and `lookat` for the static modes). Missing entries keep their defaults. |
| `start_paused` | `False` | Start in pause mode |
| `hide_menu_on_startup` | `False` | Hide the text overlay. Always hidden in headless mode. |
| `geom_group_visualization_on_startup` | `[0, 2]` | Geom groups that are visible at startup |
| `mimic_site_visualization_on_startup` | `False` | Show the mimic sites at startup |
| `headless` | `False` | Render off-screen without a window, see below |
| `recorder_params` | `None` | Arguments for the video recorder, see below |
| `custom_render_callback` | `None` | Function that is called during rendering |

### Visualizing goals

Modules can draw into the scene. The goals do so when they are created with `visualize_goal=True`: the
velocity goals draw arrows at the root, `GoalTrajMimic` draws the target positions of the mimic sites, and
`GoalTrajMimicv2` draws the reference motion as a translucent copy of the robot.

```python
env, traj = ImitationFactory.make("UnitreeH1", default_dataset_conf=dict(task="walk"),
                                  goal_type="GoalTrajMimicv2", goal_params=dict(visualize_goal=True))
```

## Recording videos

Pass `record=True` to the render call. Every call then renders exactly one frame, adds it to a video and
returns it as an RGB array. `stop` finishes the video, and the property `video_file_path` gives its
location.

```python
env.reset()
for _ in range(500):
    env.step(action)
    frame = env.render(record=True)      # (height, width, 3) uint8
env.stop()
print(env.video_file_path)               # ./LocoMuJoCo_recordings/<timestamp>/recording.mp4
```

The video is written by {class}`~loco_mujoco.core.visuals.video_recorder.VideoRecorder`, which is configured
with the environment argument `recorder_params`:

| Key | Default | Meaning |
|---|---|---|
| `path` | `"./LocoMuJoCo_recordings"` | Directory for all recordings |
| `tag` | time stamp | Name of the subdirectory of this recording |
| `video_name` | `"recording"` | File name without extension |
| `fps` | `1 / env.dt` | Frame rate. Playback is in real time when it matches the control frequency. |
| `compress` | `True` | Re-encode the video with H.264 after recording. Needs `ffmpeg` on the path. |

Dataset replay and the `play_policy` methods of the algorithms take a `record` argument as well, for
example `env.play_trajectory(n_episodes=3, record=True)`.

### Headless rendering

On a machine without a display, create the environment with `headless=True`. The viewer then renders
off-screen and `render(record=True)` works as above. The OpenGL backend is taken from the environment
variable `MUJOCO_GL` (`glfw`, `egl` or `osmesa`). If the variable is not set, the three are tried in this
order.

```bash
MUJOCO_GL=egl python my_script.py
```

Use `egl` on a machine with a GPU and `osmesa` for software rendering. If you only want to look at the
simulation on a remote machine and do not need a video, the viser viewer below is usually more convenient.

## Rendering in the browser with viser

Next to the OpenGL viewer, every environment has a [viser](https://viser.studio)-based counterpart that
serves the scene to a browser instead of opening a local window. It needs the optional dependencies:

```bash
pip install -e ".[viser]"
```

The methods are drop-in replacements and use the same viewer parameters:

```python
env.render_viser()                               # instead of env.render()
env.mjx_render_viser(state)                      # instead of env.mjx_render(state)
env.play_trajectory(n_episodes=3, viser=True)
```

```{literalinclude} ../../examples/tutorials/12_viser_rendering.py
:language: python
```

The URL of the server (`http://localhost:8080` by default) is printed at startup. The environment arguments
`host`, `port` and `verbose` configure the server. If the port is taken, viser uses the next free one and
the viewer warns about it. On a remote machine, forward the port, for example with
`ssh -L 8080:localhost:8080 user@host`.

The panel in the browser has the controls of the OpenGL viewer (geom and site groups, the three camera
modes, pause and playback speed) and adds overlays for contacts, tendons, inertia and reference frames. The
markers that goals draw are shown as well.

### Parallel environments

{meth}`~loco_mujoco.core.mujoco_mjx.Mjx.mjx_render_viser` lays the parallel environments out on the same grid
as `mjx_render` and adds a selector to the panel for the environment that the overlays follow.

```{literalinclude} ../../examples/tutorials/13_viser_rendering_mjx.py
:language: python
```

### Appearance

The look is set with environment arguments. The ground plane and the sky can also be recolored live in the
*Visualization* tab of the browser panel.

| Argument | Default | Meaning |
|---|---|---|
| `floor_color` | `(28, 30, 34)` | RGB color of the ground plane. `"model"` takes the colors of the plane's material and texture, `None` keeps the default of the viewer. |
| `sky` | `"model"` | Background gradient: `"model"` reads it from the skybox of the model, a pair `(top_rgb, bottom_rgb)` sets it, `None` turns it off. |
| `logo` | LocoMuJoCo banner | Logo in the top left corner. A URL is fetched by the browser, a local path is embedded in the page, `None` removes it. |
| `logo_width` | `170` | Width of the logo in pixels |

### Caveats

- There is no local framebuffer. Frames are read back from a connected browser, so `record=True` only
  produces a video while a browser tab is open, and `render_viser(record=False)` returns an array of zeros
  instead of the displayed image.
- An environment has one viewer at a time. Call `stop()` before switching between `render` and
  `render_viser`.
- The sky is a flat backdrop and does not rotate with the camera.
- The default logo is loaded from GitHub by the browser. Pass a local path or `None` when working offline.

## Offline USD export

The simulation can be collected into an animated [USD](https://openusd.org) stage, a file format that
Blender, Omniverse and other tools read. The export needs the optional USD bindings:

```bash
pip install -e ".[usd]"
```

Each call of {meth}`~loco_mujoco.core.mujoco_base.Mujoco.render_usd` appends one frame, and
{meth}`~loco_mujoco.core.mujoco_base.Mujoco.stop_usd` writes the file and returns its path. Parameters of
the exporter can only be passed with the first frame.

```python
env.reset()
env.render_usd(output_directory="render", fps=30)
for _ in range(300):
    env.step(action)
    env.render_usd()
usd_path = env.stop_usd()
```

| Parameter | Default | Meaning |
|---|---|---|
| `output_directory` | `LocoMuJoCo_usd/<timestamp>` | Where the stage and its assets are written |
| `fps` | `1 / env.dt` | Frame rate of the animation. It cannot be higher than the control frequency. |
| `light_intensity` | `1000` | Intensity of the exported lights |
| `camera_body_name` | first moving body | Body that the animated camera target follows |
| `filetype` | `"usd"` | `"usd"`, `"usda"` or `"usdc"` |
| `verbose` | `True` | Print progress |

The same parameters can be set when the environment is created, with the prefix `usd_`, for example
`usd_output_directory="render"`. Dataset replay exports directly:

```python
env.play_trajectory(n_episodes=1, usd=True, usd_params={"output_directory": "render", "fps": 30})
```

For MJX, {meth}`~loco_mujoco.core.mujoco_mjx.Mjx.mjx_render_usd` exports one environment of a batch, chosen
with `env_idx`. The stage contains the geometry, materials, textures and lights of the model, the
animation, and an animated camera target. The exported geom groups are the ones that the viewer shows at
startup (`geom_group_visualization_on_startup`).

## Rendering with Blender

Scripts for photorealistic rendering with Blender are kept outside the package in
[`scripts/blender`](https://github.com/robfiras/loco-mujoco/tree/master/scripts/blender). They export a USD
stage as described above and render it with Cycles or Eevee, optionally with motion overlays and as a mosaic
of several robots. They need a Blender installation and the optional dependencies
`pip install -e ".[blender]"`.

```bash
python scripts/blender/render_locomujoco.py \
  --env UnitreeH1 --task walk --render --engine CYCLES \
  --blender-executable /path/to/blender
```

The [README of the scripts](https://github.com/robfiras/loco-mujoco/blob/master/scripts/blender/README.md)
lists the options.

## See also

- [Replay a motion](../getting_started/replay_motion.md) for rendering datasets.
- [Evaluation](../training/evaluation.md) for rendering and recording trained policies.
- [Observations](observations.md#goals) for the goals that can be visualized.
- API: {mod}`loco_mujoco.core.visuals.viewer`, {mod}`loco_mujoco.core.visuals.viser_viewer`,
  {mod}`loco_mujoco.core.visuals.video_recorder`, {mod}`loco_mujoco.core.visuals.usd_exporter`.
