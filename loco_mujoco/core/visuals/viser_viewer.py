import base64
import mimetypes
import time
import warnings
from pathlib import Path

import mujoco
import numpy as np

from loco_mujoco.core.visuals.video_recorder import VideoRecorder


# viewer parameters that only the viser viewer understands. They may be passed to the environment
# constructor together with the shared ones, so the OpenGL viewer filters them out.
VISER_ONLY_VIEWER_PARAMS = ("host", "port", "verbose", "num_envs", "logo", "logo_width",
                            "floor_color", "sky")

# Sentinel for `sky`: take the gradient from the model's own skybox texture, which reproduces
# the warm haze of the OpenGL viewer.
SKY_FROM_MODEL = "model"

# Sentinel for `floor_color`: take the ground plane's color from its material and texture, which
# reproduces the grey checkerboard of the OpenGL viewer. mjviser draws MuJoCo planes as a viser
# infinite grid, which cannot show a checker pattern, so the two checker tones become the plane
# color and the grid line color instead.
FLOOR_FROM_MODEL = "model"

# Default ground plane color. Pass FLOOR_FROM_MODEL instead to match the OpenGL viewer.
DEFAULT_FLOOR_COLOR = (28, 30, 34)

# The LocoMuJoCo banner shown in the README. It is loaded by the browser, not by us, so no asset
# has to be shipped -- at the cost of needing an internet connection on the viewing machine.
# Pass a local file path as `logo` to embed an image instead, or None to disable the overlay.
DEFAULT_LOGO = "https://github.com/robfiras/loco-mujoco/assets/69359729/bd2a219e-ddfd-4355-8024-d9af921fb92a"

VISER_IMPORT_ERROR = (
    "The viser viewer requires the optional dependencies `viser` and `mjviser`. "
    "Install them with:\n\n    pip install loco-mujoco[viser]\n"
)


def check_optional_imports():
    """
    Raises an informative error if the optional viser dependencies are missing.

    """
    try:
        import mjviser  # noqa: F401
        import trimesh  # noqa: F401
        import viser  # noqa: F401
    except ImportError as e:
        raise ImportError(VISER_IMPORT_ERROR) from e


# geom types that are never used as user markers. The default-initialized MjvGeom slots of
# stateful objects that do not visualize anything have type mjGEOM_PLANE, which would otherwise
# be drawn as a spurious marker at the origin.
_SKIPPED_MARKER_TYPES = (int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_HFIELD))

_MESH = int(mujoco.mjtGeom.mjGEOM_MESH)
_CYLINDER = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
_CAPSULE = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
_BOX = int(mujoco.mjtGeom.mjGEOM_BOX)
_ELLIPSOID = int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)
_SPHERE = int(mujoco.mjtGeom.mjGEOM_SPHERE)
_ARROW = int(mujoco.mjtGeom.mjGEOM_ARROW)
_ARROW1 = int(mujoco.mjtGeom.mjGEOM_ARROW1)
_ARROW2 = int(mujoco.mjtGeom.mjGEOM_ARROW2)
_ARROWS = (_ARROW, _ARROW1, _ARROW2)
_ARROW_HEAD = -1  # synthetic type for the cone at the tip of an arrow

_UNIT_MESHES = {}


def _get_unit_mesh(geom_type):
    """
    Returns a cached unit mesh for a given mjvGeom type. The conventions match the ones used by
    mjviser for decor geoms, so markers look consistent with the rest of the scene.

    Args:
        geom_type (int): The mjtGeom type (or the synthetic ``_ARROW_HEAD``).

    Returns:
        A trimesh.Trimesh of unit size.

    """
    import trimesh

    if geom_type not in _UNIT_MESHES:
        if geom_type in (_CYLINDER, _CAPSULE):
            mesh = trimesh.creation.cylinder(radius=1.0, height=1.0)
        elif geom_type == _BOX:
            mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
        elif geom_type in (_ELLIPSOID, _SPHERE):
            mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        elif geom_type in _ARROWS:
            # arrow shaft: cylinder with its base at the origin, pointing along +z
            mesh = trimesh.creation.cylinder(radius=1.0, height=1.0, sections=12)
            mesh.apply_translation([0.0, 0.0, 0.5])
        elif geom_type == _ARROW_HEAD:
            mesh = trimesh.creation.cone(radius=2.0, height=1.0, sections=12)
        else:
            mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
        _UNIT_MESHES[geom_type] = mesh
    return _UNIT_MESHES[geom_type]


def _mat_to_wxyz(mat):
    """
    Converts a batch of flat 9-element rotation matrices to wxyz quaternions.

    Args:
        mat (np.ndarray): Rotation matrices of shape (n, 9) or (n, 3, 3).

    Returns:
        np.ndarray of shape (n, 4).

    """
    import viser.transforms as vtf

    mat = np.asarray(mat, dtype=np.float64).reshape(-1, 3, 3)
    return vtf.SO3.from_matrix(mat).wxyz


class ViserViewer:
    """
    Web-based viewer for LocoMuJoCo built on `viser <https://viser.studio>`_ via
    `mjviser <https://github.com/mujocolab/mjviser>`_.

    It mirrors the feature set of :class:`~loco_mujoco.core.visuals.viewer.MujocoViewer`
    (geom/site group toggles, the three camera modes, pause, run-speed control, video
    recording and the ``carry.user_scene`` markers) but streams the scene to a browser
    instead of opening an OpenGL window. This makes it usable over SSH and on headless
    machines with a browser on the other end.

    Note:
        There is no local framebuffer to read back from: frames come from asking a connected
        browser to render one (``client.get_render``), which costs a network round-trip. Two
        consequences differ from :class:`MujocoViewer`:

        * frames are only read back when recording, so ``render(..., record=False)`` returns a
          zero-filled array instead of the displayed image;
        * with no browser connected there is nothing to read from at all, so recording produces
          empty frames and warns once.

    """

    def __init__(self, model, dt, viewer_size=(1280, 720), start_paused=False,
                 record=False, camera_params=None, default_camera_mode="static",
                 geom_group_visualization_on_startup=None,
                 mimic_site_visualization_on_startup=False,
                 headless=False, recorder_params=None, num_envs=1,
                 host="0.0.0.0", port=8080, verbose=True,
                 logo=DEFAULT_LOGO, logo_width=170, floor_color=DEFAULT_FLOOR_COLOR,
                 sky=SKY_FROM_MODEL, **unsupported_params):
        """
        Constructor.

        Args:
            model: Mujoco model.
            dt (float): Time between two rendered frames, used for real-time pacing and as the
                default recording fps.
            viewer_size (tuple): Width and height used when requesting frames for recording.
            start_paused (bool): If True, the viewer starts paused.
            record (bool): If True, a VideoRecorder is created and fed with the rendered frames.
            camera_params (dict): Dictionary of dictionaries with parameters for each camera mode.
            default_camera_mode (str): One of "static", "follow" or "top_static".
            geom_group_visualization_on_startup (int/list): Geom groups visible on startup.
                If None, all are visible.
            mimic_site_visualization_on_startup (bool): If True, site group 4 (the mimic sites)
                is visible on startup.
            headless (bool): Ignored. Accepted so that the same ``viewer_params`` work for both
                viewers. There is no local window to suppress; the scene is always served over
                the network and it is up to the user whether to open it in a browser.
            recorder_params (dict): Parameters passed to the VideoRecorder.
            num_envs (int): Number of parallel environments to visualize.
            host (str): Host the viser server binds to.
            port (int): Port the viser server binds to.
            verbose (bool): If True, viser logs client connections and disconnections. The
                server URL is printed by viser regardless.
            logo (str): Logo shown in the bottom right corner. Either a URL, which the browser
                fetches itself, or a path to a local image file, which is embedded in the page.
                Set to None to disable the overlay.
            logo_width (int): Width of the logo overlay in pixels.
            floor_color (str/tuple): Ground plane color. "model" reproduces the OpenGL viewer by
                reading the plane's material and texture, an RGB tuple sets an explicit base
                color, and None keeps mjviser's light default.
            sky (str/tuple): Background gradient. "model" reproduces the OpenGL viewer by
                reading the model's skybox texture, a (top_rgb, bottom_rgb) pair sets an
                explicit gradient, and None leaves viser's plain background.

        """
        check_optional_imports()

        import mjviser
        import viser

        if unsupported_params:
            warnings.warn(f"The viser viewer does not support the viewer parameters "
                          f"{sorted(unsupported_params.keys())}. They are ignored.", stacklevel=2)

        self._model = model
        self.dt = dt
        self._num_envs = num_envs
        self._width, self._height = viewer_size
        self._headless = headless

        self._camera_params = self._assert_camera_params({} if camera_params is None else camera_params)
        assert default_camera_mode in self._camera_params.keys(), \
            f"Unknown camera mode \"{default_camera_mode}\"."

        self._server = viser.ViserServer(host=host, port=port, verbose=verbose)

        # viser silently falls back to the next free port when the requested one is taken, which
        # is easy to miss when an older viewer is still running and serving a stale scene
        actual_port = self._server.get_port()
        if actual_port != port:
            warnings.warn(f"Port {port} is already in use, the viser viewer is served on port "
                          f"{actual_port} instead. Another viewer is likely still running on "
                          f"port {port}.", stacklevel=2)
        self._server.scene.set_up_direction("+z")
        self._scene = mjviser.ViserMujocoScene(self._server, model, num_envs=num_envs)

        # geom group visibility, mirroring MujocoViewer's startup behaviour
        if geom_group_visualization_on_startup is not None:
            if isinstance(geom_group_visualization_on_startup, int):
                geom_group_visualization_on_startup = [geom_group_visualization_on_startup]
            self._scene.geom_groups_visible = [i in geom_group_visualization_on_startup
                                               for i in range(len(self._scene.geom_groups_visible))]

        # mimic sites live in site group 4
        self._scene.site_groups_visible[4] = bool(mimic_site_visualization_on_startup)

        # the group flags above are only read when the handle visibilities are synchronized
        self._scene._sync_visibilities()

        self._paused = start_paused
        self._run_speed_factor = 1.0
        self._camera_mode = default_camera_mode
        self._apply_camera_mode(default_camera_mode)

        self.frames = 0
        self._last_render_time = None
        self._warned_no_client = False

        # reusable MjData objects for the parallel rendering path
        self._datas_for_parallel_render = None
        self._offsets_for_parallel_render = None
        self._visual_geom_offsets = None

        self._marker_handles = {}
        self._floor_rgb = None
        self._sky_rgb = None
        self._logo_handle = None
        # remembered so the GUI reset buttons restore what the viewer was created with
        self._default_floor_color = floor_color
        self._default_sky = sky

        self._apply_floor_color(floor_color)
        self._apply_sky(sky)

        # the overlay has to exist before the GUI, which adds a toggle bound to its handle
        self._add_logo_overlay(logo, logo_width)
        self._build_gui()

        # recorder, same defaults as MujocoViewer
        if record:
            recorder_params = {} if recorder_params is None else dict(recorder_params)
            fps = 1.0 / self.dt
            if "fps" in recorder_params.keys() and recorder_params["fps"] != fps:
                warnings.warn(f"Video recording fps {recorder_params['fps']} does not match the "
                              f"environment's fps {fps}. The video will not be in real-time.", stacklevel=2)
            elif "fps" not in recorder_params.keys():
                recorder_params["fps"] = fps
            self._recorder = VideoRecorder(**recorder_params)
        else:
            self._recorder = None

    # ------------------------------------------------------------------ GUI

    def _build_gui(self):
        """
        Builds the control panel.

        mjviser's ``create_visualization_gui`` would build the Scene/Visualization/Groups tabs in
        one go, but it returns only the tab group, so there would be no way to add anything to an
        existing tab. The tabs are therefore assembled here from its per-tab builders, which lets
        the floor and sky controls sit in the Visualization tab next to the other appearance
        settings. Tabs mjviser may add in future versions will not show up automatically.

        """
        import viser

        tabs = self._server.gui.add_tab_group()
        with tabs.add_tab("Scene", icon=viser.Icon.VIDEO):
            self._scene.create_scene_gui()
        with tabs.add_tab("Visualization", icon=viser.Icon.EYE):
            self._scene.create_overlay_gui()
            self._build_appearance_gui()
        with tabs.add_tab("Groups", icon=viser.Icon.LAYERS_INTERSECT):
            self._scene.create_groups_gui()

        with tabs.add_tab("LocoMuJoCo", icon=viser.Icon.RUN):
            pause_cb = self._server.gui.add_checkbox("Pause", initial_value=self._paused)
            speed_slider = self._server.gui.add_slider("Run speed", min=0.1, max=8.0, step=0.1,
                                                       initial_value=self._run_speed_factor,
                                                       hint="Playback speed relative to real time.")
            camera_dd = self._server.gui.add_dropdown("Camera mode",
                                                      tuple(self._camera_params.keys()),
                                                      initial_value=self._camera_mode)

            @pause_cb.on_update
            def _(_):
                self._paused = bool(pause_cb.value)

            @speed_slider.on_update
            def _(_):
                self._run_speed_factor = float(speed_slider.value)

            @camera_dd.on_update
            def _(_):
                self._apply_camera_mode(str(camera_dd.value))

    def _build_appearance_gui(self):
        """
        Adds live color pickers for the ground plane and the sky, so the look can be tuned in
        the browser without restarting the environment. Laid out as one folder per feature to
        match the rest of the Visualization tab.

        """
        if self._floor_rgb is not None:
            with self._server.gui.add_folder("Floor"):
                floor_rgb = self._server.gui.add_rgb("Color", initial_value=self._floor_rgb)
                reset_floor = self._server.gui.add_button("Reset")

                @floor_rgb.on_update
                def _(_):
                    self._apply_floor_color(tuple(floor_rgb.value))

                @reset_floor.on_click
                def _(_):
                    self._apply_floor_color(self._default_floor_color)
                    floor_rgb.value = self._floor_rgb

        if self._sky_rgb is not None:
            with self._server.gui.add_folder("Sky"):
                top_rgb = self._server.gui.add_rgb("Top", initial_value=self._sky_rgb[0])
                bottom_rgb = self._server.gui.add_rgb("Horizon", initial_value=self._sky_rgb[1])
                reset_sky = self._server.gui.add_button("Reset")

                def _update_sky(_):
                    self._apply_sky((tuple(top_rgb.value), tuple(bottom_rgb.value)))

                top_rgb.on_update(_update_sky)
                bottom_rgb.on_update(_update_sky)

                @reset_sky.on_click
                def _(_):
                    self._apply_sky(self._default_sky)
                    top_rgb.value, bottom_rgb.value = self._sky_rgb

        if self._logo_handle is not None:
            with self._server.gui.add_folder("Logo"):
                # a checkbox rather than a button, to match the other toggles in this tab and
                # to keep the current state visible
                hide_logo = self._server.gui.add_checkbox("Hide logo", initial_value=False)

                @hide_logo.on_update
                def _(_):
                    self._logo_handle.visible = not hide_logo.value

    def _skybox_gradient(self):
        """
        Reads the vertical gradient of the model's skybox texture.

        MuJoCo stores a skybox as a vertical strip of six ``width x width`` cube faces in the
        order right, left, up, down, front, back. The four side faces carry the same
        zenith-to-horizon gradient, so one column of a side face is all that is needed.

        Returns:
            An (n, 3) uint8 array running from the top of the sky to the horizon, or None if
            the model has no skybox.

        """
        for i in range(self._model.ntex):
            if self._model.tex_type[i] != mujoco.mjtTexture.mjTEXTURE_SKYBOX:
                continue
            width, height = int(self._model.tex_width[i]), int(self._model.tex_height[i])
            n_channels = int(self._model.tex_nchannel[i])
            adr = int(self._model.tex_adr[i])
            if n_channels < 3 or height < 5 * width:
                continue
            strip = self._model.tex_data[adr:adr + width * height * n_channels]
            strip = np.asarray(strip, dtype=np.uint8).reshape(height, width, n_channels)
            front = strip[4 * width:5 * width]  # a side face, top row = zenith
            return front[:, width // 2, :3]
        return None

    def _apply_sky(self, sky):
        """
        Sets a vertical gradient as the scene background, reproducing the warm sky of the
        OpenGL viewer.

        Args:
            sky: "model" to read the model's skybox, a (top_rgb, bottom_rgb) pair, or None.

        """
        if sky is None:
            return

        if isinstance(sky, str):
            if sky != SKY_FROM_MODEL:
                raise ValueError(f"Unknown sky \"{sky}\". Use \"{SKY_FROM_MODEL}\", a "
                                 f"(top_rgb, bottom_rgb) pair, or None.")
            column = self._skybox_gradient()
            if column is None:
                # nothing to reproduce; leave viser's default background alone
                return
        else:
            top, bottom = (np.asarray(c, dtype=np.float64) for c in sky)
            ramp = np.linspace(0.0, 1.0, 256)[:, None]
            column = np.clip(top * (1.0 - ramp) + bottom * ramp, 0, 255).astype(np.uint8)

        # remembered so the GUI pickers can start from the current gradient
        self._sky_rgb = (tuple(int(c) for c in column[0]), tuple(int(c) for c in column[-1]))

        # the background is stretched across the viewport, so a narrow strip is enough
        self._server.scene.set_background_image(np.repeat(column[:, None, :], 8, axis=1))

    def _apply_floor_color(self, floor_color):
        """
        Recolors the ground plane.

        mjviser draws every fixed plane geom as a viser infinite grid with hardcoded light
        colors and keeps no handle to it, so the grids are re-added under the same scene node
        names, which replaces them.

        Args:
            floor_color (tuple): RGB base color of the plane, or None to keep mjviser's default.

        """
        if floor_color is None:
            return
        if isinstance(floor_color, str) and floor_color != FLOOR_FROM_MODEL:
            raise ValueError(f"Unknown floor_color \"{floor_color}\". Use "
                             f"\"{FLOOR_FROM_MODEL}\", an RGB tuple, or None.")

        from mjviser.conversions import get_body_name, is_fixed_body

        for i in range(self._model.ngeom):
            if self._model.geom_type[i] != mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            body_id = self._model.geom_bodyid[i]
            if not is_fixed_body(self._model, body_id):
                continue

            if isinstance(floor_color, str):
                colors = self._model_floor_colors(i)
                if colors is None:
                    continue
            else:
                colors = self._derive_floor_colors(floor_color)
            plane, cell, section = colors
            self._floor_rgb = plane  # remembered so the GUI picker can start from it

            body_name = get_body_name(self._model, body_id)
            geom_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, i)
            self._server.scene.add_grid(
                f"/fixed_bodies/{body_name}/{geom_name}",
                infinite_grid=True, fade_distance=50.0, shadow_opacity=0.2,
                plane_color=plane, cell_color=cell, section_color=section,
                plane_opacity=1.0,
                position=self._model.geom_pos[i],
                wxyz=self._model.geom_quat[i])

    @staticmethod
    def _derive_floor_colors(base_rgb):
        """
        Turns a single RGB into a (plane, cell, section) triple. The grid lines are pushed away
        from the plane color -- towards white on a dark floor and towards black on a light one --
        so that they stay readable either way.

        Args:
            base_rgb (tuple): RGB color of the plane itself.

        Returns:
            Three RGB int tuples in (plane, cell, section) order.

        """
        base = np.clip(np.asarray(base_rgb, dtype=np.float64), 0, 255)
        target = 0.0 if base @ np.array([0.299, 0.587, 0.114]) > 127.0 else 255.0
        as_tuple = lambda c: tuple(int(v) for v in np.clip(c, 0, 255))  # noqa: E731
        return (as_tuple(base),
                as_tuple(base + (target - base) * 0.18),
                as_tuple(base + (target - base) * 0.32))

    def _model_floor_colors(self, geom_id):
        """
        Reads the colors the OpenGL viewer would draw a plane geom with.

        A textured plane (the usual grey checkerboard) contributes its two dominant tones, which
        become the grid's plane and line colors. An untextured plane falls back to the material
        or geom rgba. Everything is modulated by the material color, as MuJoCo does.

        Args:
            geom_id (int): Index of the plane geom.

        Returns:
            Three RGB int tuples in (plane, cell, section) order, or None if the plane is
            fully transparent.

        """
        matid = int(self._model.geom_matid[geom_id])
        if matid >= 0:
            rgba = np.asarray(self._model.mat_rgba[matid], dtype=np.float64)
        else:
            rgba = np.asarray(self._model.geom_rgba[geom_id], dtype=np.float64)
        if rgba[3] <= 0.0:
            return None

        tones = self._plane_texture_tones(matid)
        if tones is None:
            # untextured: a single flat color, so derive the grid lines from it
            return self._derive_floor_colors(rgba[:3] * 255.0)

        # MuJoCo multiplies the texture by the material color. The checkerboard cannot be
        # reproduced by a grid, so its brighter tone becomes the plane and its darker tone
        # the grid lines, which reads closest to the original.
        dark, light = (np.clip(t * rgba[:3], 0, 255) for t in tones)
        as_tuple = lambda c: tuple(int(v) for v in np.clip(c, 0, 255))  # noqa: E731
        return (as_tuple(light), as_tuple(dark), as_tuple(dark * 0.8))

    def _plane_texture_tones(self, matid):
        """
        Extracts the darkest and brightest tones of a material's 2D texture, which for the
        default MuJoCo floor are the two greys of the checkerboard.

        Args:
            matid (int): Material index, or -1 for no material.

        Returns:
            A (dark_rgb, light_rgb) pair of float arrays, or None if there is no 2D texture.

        """
        if matid < 0:
            return None
        for tex_id in np.atleast_1d(np.asarray(self._model.mat_texid[matid])).reshape(-1):
            tex_id = int(tex_id)
            if tex_id < 0 or self._model.tex_type[tex_id] != mujoco.mjtTexture.mjTEXTURE_2D:
                continue
            width, height = int(self._model.tex_width[tex_id]), int(self._model.tex_height[tex_id])
            n_channels = int(self._model.tex_nchannel[tex_id])
            adr = int(self._model.tex_adr[tex_id])
            if n_channels < 3:
                continue
            data = np.asarray(self._model.tex_data[adr:adr + width * height * n_channels],
                              dtype=np.float64).reshape(-1, n_channels)[:, :3]
            unique = np.unique(data, axis=0)
            luminance = unique @ np.array([0.299, 0.587, 0.114])
            return unique[luminance.argmin()], unique[luminance.argmax()]
        return None

    def _add_logo_overlay(self, logo, width_px):
        """
        Pins a small logo to the top left corner of the viewport, clear of the control panel.

        viser has no API for viewport overlays -- ``gui.add_image`` places images inside the
        control panel -- so this injects an absolutely positioned element through
        ``gui.add_html``. It is click-through so it never swallows camera drags.

        Args:
            logo (str): URL or path of the image, or None to skip the overlay.
            width_px (int): Width of the logo in pixels.

        """
        if logo is None:
            return

        logo = str(logo)
        if logo.startswith(("http://", "https://", "data:")):
            src = logo
        else:
            path = Path(logo)
            if not path.is_file():
                warnings.warn(f"Logo \"{path}\" does not exist, the logo overlay is disabled.",
                              stacklevel=3)
                return
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
            src = f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"

        self._logo_handle = self._server.gui.add_html(
            f'<div style="position:fixed; left:16px; top:14px; width:{int(width_px)}px;'
            f' z-index:100; pointer-events:none; opacity:0.85;">'
            f'<img src="{src}" style="width:100%; height:auto; display:block;"'
            f' alt="LocoMuJoCo"/></div>')

    # --------------------------------------------------------------- camera

    def _apply_camera_mode(self, mode):
        """
        Applies a camera mode to all connected clients. "follow" enables mjviser's scene
        tracking, which keeps the tracked body at the origin; the static modes disable it
        and place the camera from the mode's distance/elevation/azimuth/lookat.

        Args:
            mode (str): Camera mode. (either "follow", "static", or "top_static")

        """
        self._camera_mode = mode
        params = self._camera_params[mode]
        self._scene.camera_tracking_enabled = (mode == "follow")

        lookat = np.array(params.get("lookat", np.zeros(3)), dtype=np.float64)
        if mode == "follow":
            # the tracked body is held at the origin by the scene offset
            lookat = np.zeros(3)

        azimuth, elevation = np.radians(params["azimuth"]), np.radians(params["elevation"])
        # MuJoCo's camera looks along `forward`, so the eye sits `distance` behind the lookat point
        forward = np.array([np.cos(elevation) * np.cos(azimuth),
                            np.cos(elevation) * np.sin(azimuth),
                            np.sin(elevation)])
        position = lookat - params["distance"] * forward

        self._camera_position, self._camera_lookat = position, lookat
        for client in self._server.get_clients().values():
            client.camera.up_direction = np.array([0.0, 0.0, 1.0])
            client.camera.position = position
            client.camera.look_at = lookat

    def _assert_camera_params(self, camera_params):
        """
        Asserts if the provided camera parameters are valid or not. Also, if properties of some
        camera types are not specified, the default parameters are used.

        Args:
            camera_params (dict): Dictionary of dictionaries containing parameters for each camera type.

        Returns:
            Dictionary of dictionaries with parameters for each camera type.

        """
        default_camera_params = self.get_default_camera_params()

        for cam_type in camera_params.keys():
            assert cam_type in default_camera_params.keys(), \
                f"Camera type \"{cam_type}\" is unknown. Allowed camera types are " \
                f"{list(default_camera_params.keys())}."
            for param in camera_params[cam_type].keys():
                assert param in default_camera_params[cam_type].keys(), \
                    f"Parameter \"{param}\" of camera type \"{cam_type}\" is unknown. Allowed " \
                    f"parameters are {list(default_camera_params[cam_type].keys())}"

        for cam_type in default_camera_params.keys():
            if cam_type not in camera_params.keys():
                camera_params[cam_type] = default_camera_params[cam_type]
            else:
                for param in default_camera_params[cam_type].keys():
                    if param not in camera_params[cam_type].keys():
                        camera_params[cam_type][param] = default_camera_params[cam_type][param]

        return camera_params

    @staticmethod
    def get_default_camera_params():
        """
        Getter for default camera parameterization. Matches MujocoViewer's defaults.

        Returns:
            Dictionary of dictionaries with default parameters for each camera type.

        """
        return dict(static=dict(distance=15.0, elevation=-45.0, azimuth=90.0, lookat=np.array([0.0, 0.0, 0.0])),
                    follow=dict(distance=3.5, elevation=0.0, azimuth=90.0),
                    top_static=dict(distance=5.0, elevation=-90.0, azimuth=90.0, lookat=np.array([0.0, 0.0, 0.0])))

    # -------------------------------------------------------------- markers

    def _collect_markers(self, user_scene, env_offsets=None):
        """
        Converts the MjvGeoms carried in the environment state into flat per-type arrays that can
        be pushed to viser as batched meshes.

        Args:
            user_scene (MjvScene): The user scene from the environment's carry. Its fields are
                either of shape (n_geoms, ...) for a single environment or (n_envs, n_geoms, ...)
                for the parallel case.
            env_offsets (np.ndarray): XY offsets of shape (n_envs, 1, 2) applied to the marker
                positions in the parallel case, or None.

        Returns:
            A dict mapping a geom type to a tuple of (positions, wxyzs, scales, colors, opacities),
            plus mesh markers keyed by ``(_MESH, mesh_id)``.

        """
        types = np.asarray(user_scene.geoms.type).reshape(-1)
        sizes = np.asarray(user_scene.geoms.size, dtype=np.float64).reshape(-1, 3)
        positions = np.array(user_scene.geoms.pos, dtype=np.float64).reshape(-1, 3)
        mats = np.asarray(user_scene.geoms.mat, dtype=np.float64).reshape(-1, 9)
        rgbas = np.asarray(user_scene.geoms.rgba, dtype=np.float64).reshape(-1, 4)
        dataids = np.asarray(user_scene.geoms.dataid).reshape(-1)

        if env_offsets is not None:
            offsets = np.broadcast_to(env_offsets, (env_offsets.shape[0],
                                                    positions.shape[0] // env_offsets.shape[0], 2))
            positions[:, :2] += offsets.reshape(-1, 2)

        # the scene offset keeps the tracked body at the origin, so markers must follow it
        positions = positions + self._scene_offset()

        grouped = {}
        arrow_heads = []
        for i in range(types.shape[0]):
            geom_type = int(types[i])
            if geom_type in _SKIPPED_MARKER_TYPES or rgbas[i, 3] <= 0.0:
                continue

            mat = mats[i].reshape(3, 3)
            color = (np.clip(rgbas[i, :3], 0.0, 1.0) * 255).astype(np.uint8)
            opacity = float(rgbas[i, 3])
            size = sizes[i]
            pos = positions[i]

            if geom_type == _MESH:
                key = (_MESH, int(dataids[i]))
                scale = np.ones(3)
            elif geom_type in _ARROWS:
                # size = [shaft_radius, head_radius, total_length]; 80% shaft, 20% head
                key = (geom_type, None)
                shaft_len, head_len = size[2] * 0.8, size[2] * 0.2
                scale = np.array([size[0], size[0], shaft_len])
                arrow_heads.append((pos + mat[:, 2] * shaft_len, mat,
                                    np.array([size[0], size[0], head_len]), color, opacity))
            elif geom_type in (_CYLINDER, _CAPSULE):
                key = (geom_type, None)
                scale = np.array([size[0], size[0], max(size[2] * 2.0, size[0])])
            else:
                key = (geom_type, None)
                scale = size

            grouped.setdefault(key, []).append((pos, mat, scale, color, opacity))

        for pos, mat, scale, color, opacity in arrow_heads:
            grouped.setdefault((_ARROW_HEAD, None), []).append((pos, mat, scale, color, opacity))

        out = {}
        for key, entries in grouped.items():
            pos = np.array([e[0] for e in entries], dtype=np.float32)
            wxyz = _mat_to_wxyz(np.array([e[1] for e in entries])).astype(np.float32)
            scale = np.array([e[2] for e in entries], dtype=np.float32)
            color = np.array([e[3] for e in entries], dtype=np.uint8)
            opacity = np.array([e[4] for e in entries], dtype=np.float32)
            out[key] = (pos, wxyz, scale, color, opacity)
        return out

    def _scene_offset(self):
        """
        Returns the translation mjviser currently applies to the whole scene (non-zero while the
        follow camera tracks a body), so that the markers can be shifted along with it.

        """
        return np.asarray(getattr(self._scene, "_scene_offset", np.zeros(3)), dtype=np.float64)

    def _mesh_for_marker(self, mesh_id):
        """
        Builds a unit trimesh for a mesh-type marker geom from the model's mesh buffers. Used for
        the ghost-robot markers of the mimic goals.

        Args:
            mesh_id (int): Index into the model's mesh arrays.

        Returns:
            A trimesh.Trimesh.

        """
        import trimesh

        key = (_MESH, int(mesh_id))
        if key not in _UNIT_MESHES:
            vert_adr = self._model.mesh_vertadr[mesh_id]
            vert_num = self._model.mesh_vertnum[mesh_id]
            face_adr = self._model.mesh_faceadr[mesh_id]
            face_num = self._model.mesh_facenum[mesh_id]
            vertices = self._model.mesh_vert[vert_adr:vert_adr + vert_num].reshape(-1, 3)
            faces = self._model.mesh_face[face_adr:face_adr + face_num].reshape(-1, 3)
            _UNIT_MESHES[key] = trimesh.Trimesh(vertices=np.array(vertices, dtype=np.float64),
                                                faces=np.array(faces, dtype=np.int64), process=False)
        return _UNIT_MESHES[key]

    def _update_markers(self, user_scene, env_offsets=None):
        """
        Pushes the user-scene markers to viser as batched meshes, reusing the handles across frames.

        Args:
            user_scene (MjvScene): The user scene from the environment's carry.
            env_offsets (np.ndarray): XY offsets per environment, or None.

        """
        if user_scene is None:
            return

        markers = self._collect_markers(user_scene, env_offsets)

        for key, (pos, wxyz, scale, color, opacity) in markers.items():
            handle = self._marker_handles.get(key)
            if handle is not None and handle.batched_positions.shape[0] != pos.shape[0]:
                handle.remove()
                handle = None
                self._marker_handles.pop(key, None)

            if handle is None:
                mesh = self._mesh_for_marker(key[1]) if key[0] == _MESH else _get_unit_mesh(key[0])
                self._marker_handles[key] = self._server.scene.add_batched_meshes_simple(
                    f"/loco_mujoco_markers/{key[0]}_{key[1]}",
                    mesh.vertices, mesh.faces,
                    batched_wxyzs=wxyz, batched_positions=pos, batched_scales=scale,
                    batched_colors=color, batched_opacities=opacity,
                    lod="off", cast_shadow=False, receive_shadow=False)
            else:
                handle.batched_positions = pos
                handle.batched_wxyzs = wxyz
                handle.batched_scales = scale
                handle.batched_colors = color
                handle.batched_opacities = opacity
                handle.visible = True

        for key, handle in self._marker_handles.items():
            if key not in markers:
                handle.visible = False

    # --------------------------------------------------------------- render

    def render(self, data, carry, record):
        """
        Main rendering function for the single-environment (CPU) case.

        Args:
            data: Mujoco data structure.
            carry: Carry object holding the user scene with the markers.
            record (bool): If True, a frame is requested from a connected browser and fed to
                the recorder.

        Returns:
            The rendered image as a numpy array of shape (height, width, 3).

        """
        while self._paused:
            self._scene.update_from_mjdata(data)
            self._update_markers(getattr(carry, "user_scene", None))
            time.sleep(0.01)

        self._scene.update_from_mjdata(data)
        self._update_markers(getattr(carry, "user_scene", None))

        return self._finish_frame(record)

    def parallel_render(self, mjx_state, record, offset=2.0):
        """
        Main rendering function for the parallel (MJX) case. All environments are laid out on a
        square grid, matching MujocoViewer.parallel_render.

        Args:
            mjx_state: Batched Mjx state.
            record (bool): If True, a frame is requested from a connected browser and fed to
                the recorder.
            offset (float): Spacing between two environments on the grid.

        Returns:
            The rendered image as a numpy array of shape (height, width, 3).

        """
        n_envs = mjx_state.data.qpos.shape[0]
        assert n_envs <= self._num_envs, \
            f"The viser viewer was created for {self._num_envs} environments but got {n_envs}."

        if self._offsets_for_parallel_render is None or n_envs > len(self._offsets_for_parallel_render):
            self._offsets_for_parallel_render = self.generate_square_positions(0.0, 0.0, n_envs, offset)
            self._visual_geom_offsets = np.array(self._offsets_for_parallel_render)[:, np.newaxis, :]
        if self._datas_for_parallel_render is None or n_envs > len(self._datas_for_parallel_render):
            self._datas_for_parallel_render = [mujoco.MjData(self._model) for _ in range(n_envs)]

        qpos = np.array(mjx_state.data.qpos)
        qvel = np.array(mjx_state.data.qvel)
        mocap_pos = np.array(mjx_state.data.mocap_pos)
        mocap_quat = np.array(mjx_state.data.mocap_quat)

        body_xpos = np.empty((n_envs, self._model.nbody, 3))
        body_xmat = np.empty((n_envs, self._model.nbody, 3, 3))

        for i in range(n_envs):
            data = self._datas_for_parallel_render[i]
            env_offset = self._offsets_for_parallel_render[i]
            data.qpos, data.qvel = qpos[i], qvel[i]
            data.mocap_pos, data.mocap_quat = mocap_pos[i], mocap_quat[i]
            data.qpos[0] += env_offset[0]
            data.qpos[1] += env_offset[1]
            data.mocap_pos[:, 0] += env_offset[0]
            data.mocap_pos[:, 1] += env_offset[1]
            mujoco.mj_forward(self._model, data)
            body_xpos[i] = data.xpos
            body_xmat[i] = data.xmat.reshape(-1, 3, 3)
            # write the grid-offset state back so the contact/tendon overlays, which are
            # recomputed by mjviser from qpos, line up with the bodies
            qpos[i] = data.qpos
            mocap_pos[i] = data.mocap_pos
            mocap_quat[i] = data.mocap_quat

        while self._paused:
            time.sleep(0.01)

        self._scene.update_from_arrays(body_xpos, body_xmat, mocap_pos, mocap_quat,
                                       qpos=qpos, qvel=qvel)
        self._update_markers(mjx_state.additional_carry.user_scene, self._visual_geom_offsets[:n_envs])

        return self._finish_frame(record)

    @staticmethod
    def generate_square_positions(center_x, center_y, num_envs, offset):
        """
        Lays out `num_envs` environments on a square grid centered at (center_x, center_y). The
        environment shown in the middle of the first column is swapped to index 0, matching
        MujocoViewer.

        Args:
            center_x (float): Grid center along x.
            center_y (float): Grid center along y.
            num_envs (int): Number of environments.
            offset (float): Spacing between two environments.

        Returns:
            List of (x, y) tuples.

        """
        positions = []
        grid_size = int((num_envs - 1) ** 0.5) + 1
        half_grid = grid_size // 2

        done = False
        for i in range(grid_size):
            for j in range(grid_size):
                positions.append((center_x + (i - half_grid) * offset,
                                  center_y + (j - half_grid) * offset))
                if len(positions) == num_envs:
                    done = True
                    break
            if done:
                break

        col_length = min(grid_size, (num_envs + grid_size - 1) // grid_size)
        middle_index = (col_length // 2) * grid_size
        if middle_index < len(positions):
            positions[0], positions[middle_index] = positions[middle_index], positions[0]

        return positions

    def _finish_frame(self, record):
        """
        Paces the rendering to real time, reads back a frame and feeds the recorder.

        Args:
            record (bool): If True, a frame is requested from a connected browser.

        Returns:
            The rendered image as a numpy array of shape (height, width, 3).

        """
        self.frames += 1

        if not record:
            self._sync_to_real_time()

        im = self.read_pixels() if (record or self._recorder) else np.zeros((self._height, self._width, 3),
                                                                           dtype=np.uint8)
        if self._recorder:
            self._recorder(im)

        return im

    def _sync_to_real_time(self):
        """
        Sleeps so that consecutive frames are dt / run_speed_factor apart in wall-clock time.

        """
        now = time.time()
        if self._last_render_time is not None:
            remaining = (self.dt / self._run_speed_factor) - (now - self._last_render_time)
            if remaining > 0:
                time.sleep(remaining)
        self._last_render_time = time.time()

    def read_pixels(self, depth=False):
        """
        Requests a rendered frame from the first connected browser client. Unlike the OpenGL
        viewer there is no local framebuffer, so this returns zeros when no client is connected.

        Args:
            depth (bool): Unsupported; kept for API compatibility with MujocoViewer.

        Returns:
            The rendered image as a numpy array of shape (height, width, 3).

        """
        if depth:
            raise NotImplementedError("The viser viewer cannot read back depth images.")

        clients = self._server.get_clients()
        if not clients:
            if not self._warned_no_client:
                warnings.warn("No browser is connected to the viser server, so no frames can be "
                              "read back. Open the viser URL to enable recording. Returning "
                              "zero-filled frames until then.", stacklevel=2)
                self._warned_no_client = True
            return np.zeros((self._height, self._width, 3), dtype=np.uint8)

        client = next(iter(clients.values()))
        return np.asarray(client.get_render(height=self._height, width=self._width))[..., :3]

    def upload_hfield(self, model, hfield_id):
        """
        Rebuilds the scene geometry so that a modified height field becomes visible. Unlike the
        OpenGL viewer, which only re-uploads a texture, this rebuilds the height field mesh and
        is therefore expensive.

        Args:
            model: Mujoco model.
            hfield_id: Height field id.

        """
        self._model.hfield_data = model.hfield_data
        self._scene.rebuild_visual_handles()

    def stop(self):
        """
        Stops the viser server and finalizes the recording.

        Returns:
            The path to the recorded video, or None if nothing was recorded.

        """
        path = self._recorder.stop() if self._recorder else None
        self._server.stop()
        return path

    @property
    def server(self):
        """The underlying viser server, for building custom overlays."""
        return self._server

    @property
    def scene(self):
        """The underlying mjviser scene."""
        return self._scene

    @property
    def video_file_path(self):
        if self._recorder is not None:
            return self._recorder.file_path
        else:
            return None
