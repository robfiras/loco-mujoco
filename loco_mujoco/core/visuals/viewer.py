import os
import warnings

import glfw
import mujoco
import time
import collections
from itertools import cycle
import numpy as np

from loco_mujoco.core.visuals.video_recorder import VideoRecorder
from mujoco import mjx
import jax

def _import_egl(width, height):
    from mujoco.egl import GLContext

    return GLContext(width, height)


def _import_glfw(width, height):
    from mujoco.glfw import GLContext

    return GLContext(width, height)


def _import_osmesa(width, height):
    from mujoco.osmesa import GLContext

    return GLContext(width, height)


_ALL_RENDERERS = collections.OrderedDict(
    [
        ("glfw", _import_glfw),
        ("egl", _import_egl),
        ("osmesa", _import_osmesa),
    ]
)


class MujocoViewer:
    """
    Class that creates a viewer for mujoco environments.

    """

    def __init__(self, model, dt, viewer_size=(1280, 720), start_paused=False,
                 custom_render_callback=None, record=False, camera_params=None,
                 default_camera_mode="static", hide_menu_on_startup=None,
                 geom_group_visualization_on_startup=None,
                 mimic_site_visualization_on_startup=False,
                 headless=False, recorder_params=None):
        """
        Constructor.

        Args:
            model: Mujoco model.
            dt (float): Timestep of the environment, (not the simulation).
            viewer_size (tuple): Tuple of width and height of the viewer window.
            start_paused (bool): If True, the rendering is paused in the beginning of the simulation.
            custom_render_callback (func): Custom render callback function, which is supposed to be called
                during rendering.
            record (bool): If true, frames are returned during rendering.
            camera_params (dict): Dictionary of dictionaries including custom parameterization of the three cameras.
                Checkout the function get_default_camera_params() to know what parameters are expected. Is some camera
                type specification or parameter is missing, the default one is used.
            hide_menu_on_startup (bool): If True, the menu is hidden on startup.
            geom_group_visualization_on_startup (int/list): int or list defining which geom group_ids should be
                visualized on startup. If None, all are visualized.
            mimic_site_visualization_on_startup (bool): If True, mimic sites are visualized on startup.
            headless (bool): If True, render will be done in headless mode.
            recorder_params (dict): Dictionary of parameters for the video recorder.

        """

        if hide_menu_on_startup is None and headless:
            hide_menu_on_startup = True
        elif hide_menu_on_startup is None:
            hide_menu_on_startup = False

        self.button_left = False
        self.button_right = False
        self.button_middle = False
        self.last_x = 0
        self.last_y = 0
        self.dt = dt

        self.frames = 0
        self.start_time = time.time()

        self._headless = headless
        self._model = model
        self._font_scale = 100
        width, height = viewer_size

        if headless:
            # use the OpenGL render that is available on the machine
            self._opengl_context = self.setup_opengl_backend_headless(width, height)
            self._opengl_context.make_current()
            self._width, self._height = self.update_headless_size(width, height)
        else:
            # use glfw
            self._width, self._height = width, height
            glfw.init()
            glfw.window_hint(glfw.COCOA_RETINA_FRAMEBUFFER, 0)
            self._window = glfw.create_window(width=self._width, height=self._height,
                                              title="MuJoCo", monitor=None, share=None)
            glfw.make_context_current(self._window)
            glfw.set_mouse_button_callback(self._window, self.mouse_button)
            glfw.set_cursor_pos_callback(self._window, self.mouse_move)
            glfw.set_key_callback(self._window, self.keyboard)
            glfw.set_scroll_callback(self._window, self.scroll)

        self._set_mujoco_buffers()

        if record and not headless:
            # dont allow to change the window size to have equal frame size during recording
            glfw.window_hint(glfw.RESIZABLE, False)

        self._viewport = mujoco.MjrRect(0, 0, self._width, self._height)
        self._loop_count = 0
        self._time_per_render = 1 / 60.
        self._run_speed_factor = 1.0
        self._paused = start_paused

        # Disable v_sync, so swap_buffers does not block
        # glfw.swap_interval(0)

        self._scene = mujoco.MjvScene(self._model, 100000)
        self._user_scene = mujoco.MjvScene(self._model, 1000)
        self._scene_option = mujoco.MjvOption()
        if mimic_site_visualization_on_startup:
            self._scene_option.sitegroup[4] = True
        self._camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, self._camera)
        if camera_params is None:
            self._camera_params = self.get_default_camera_params()
        else:
            self._camera_params = self._assert_camera_params(camera_params)
        self._all_camera_modes = ("static", "follow", "top_static")
        self._camera_mode_iter = cycle(self._all_camera_modes)
        self._camera_mode = None
        self._camera_mode_target = next(self._camera_mode_iter)
        assert default_camera_mode in self._all_camera_modes
        while self._camera_mode_target != default_camera_mode:
            self._camera_mode_target = next(self._camera_mode_iter)
        self._set_camera()

        self.custom_render_callback = custom_render_callback

        self._overlay = {}
        self._hide_menu = hide_menu_on_startup

        if geom_group_visualization_on_startup is not None:
            assert type(geom_group_visualization_on_startup) == list or type(geom_group_visualization_on_startup) == int
            if type(geom_group_visualization_on_startup) is not list:
                geom_group_visualization_on_startup = [geom_group_visualization_on_startup]
            for group_id, _ in enumerate(self._scene_option.geomgroup):
                if group_id not in geom_group_visualization_on_startup:
                    self._scene_option.geomgroup[group_id] = False

        # things for parallel rendering
        self._offsets_for_parallel_render = None
        self._datas_for_parallel_render = None
        self._datas_for_parallel_render = None
        self._visual_geom_offsets = None

        if record:
            if recorder_params is None:
                recorder_params = dict(fps=1/dt)
            elif "fps" not in recorder_params.keys():
                recorder_params["fps"] = 1/dt
            elif recorder_params["fps"] != 1/dt:
                warnings.warn("The provided frame rate in recorder_params is different from the simulation frame rate.")
            self._recorder = VideoRecorder(**recorder_params)
        else:
            self._recorder = None

    def load_new_model(self, model):
        """
        Loads a new model to the viewer, and resets the scene and context.
        This is used in MultiMujoco environments.

        Args:
            model: Mujoco model.

        """

        self._model = model
        self._scene = mujoco.MjvScene(model, 1000)
        self._context = mujoco.MjrContext(model, mujoco.mjtFontScale(self._font_scale))

    def mouse_button(self, window, button, act, mods):
        """
        Mouse button callback for glfw.

        Args:
            window: glfw window.
            button: glfw button id.
            act: glfw action.
            mods: glfw mods.

        """

        self.button_left = glfw.get_mouse_button(self._window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self.button_right = glfw.get_mouse_button(self._window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        self.button_middle = glfw.get_mouse_button(self._window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS

        self.last_x, self.last_y = glfw.get_cursor_pos(self._window)

    def mouse_move(self, window, x_pos, y_pos):
        """
        Mouse mode callback for glfw.

        Args:
            window:  glfw window.
            x_pos: Current mouse x position.
            y_pos: Current mouse y position.

        """

        if not self.button_left and not self.button_right and not self.button_middle:
            return

        dx = x_pos - self.last_x
        dy = y_pos - self.last_y
        self.last_x = x_pos
        self.last_y = y_pos

        width, height = glfw.get_window_size(self._window)

        mod_shift = glfw.get_key(self._window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or glfw.get_key(self._window,
                                                                                                  glfw.KEY_RIGHT_SHIFT) == glfw.PRESS

        if self.button_right:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self.button_left:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        mujoco.mjv_moveCamera(self._model, action, dx / width, dy / height, self._scene, self._camera)

    def keyboard(self, window, key, scancode, act, mods):
        """
        Keyboard callback for glfw.

        Args:
            window: glfw window.
            key: glfw key event.
            scancode: glfw scancode.
            act: glfw action.
            mods: glfw mods.

        """

        if act != glfw.RELEASE:
            return

        if key == glfw.KEY_SPACE:
            self._paused = not self._paused

        if key == glfw.KEY_C:
            self._scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = not self._scene_option.flags[
                mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]
            self._scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = not self._scene_option.flags[
                mujoco.mjtVisFlag.mjVIS_CONSTRAINT]
            self._scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTSPLIT] = not self._scene_option.flags[
                mujoco.mjtVisFlag.mjVIS_CONTACTSPLIT]

        if key == glfw.KEY_T:
            self._scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = not self._scene_option.flags[
                mujoco.mjtVisFlag.mjVIS_TRANSPARENT]

        if key == glfw.KEY_0:
            self._scene_option.geomgroup[0] = not self._scene_option.geomgroup[0]

        if key == glfw.KEY_1:
            self._scene_option.geomgroup[1] = not self._scene_option.geomgroup[1]

        if key == glfw.KEY_2:
            self._scene_option.geomgroup[2] = not self._scene_option.geomgroup[2]

        if key == glfw.KEY_3:
            self._scene_option.geomgroup[3] = not self._scene_option.geomgroup[3]

        if key == glfw.KEY_4:
            self._scene_option.geomgroup[4] = not self._scene_option.geomgroup[4]

        if key == glfw.KEY_5:
            self._scene_option.geomgroup[5] = not self._scene_option.geomgroup[5]

        if key == glfw.KEY_M:
            # mimic sites
            self._scene_option.sitegroup[4] = not self._scene_option.sitegroup[4]

        if key == glfw.KEY_TAB:
            self._camera_mode_target = next(self._camera_mode_iter)

        if key == glfw.KEY_S:
            self._run_speed_factor /= 2.0

        if key == glfw.KEY_F:
            self._run_speed_factor *= 2.0

        if key == glfw.KEY_E:
            self._scene_option.frame = not self._scene_option.frame

        if key == glfw.KEY_H:
            if self._hide_menu:
                self._hide_menu = False
            else:
                self._hide_menu = True

    def scroll(self, window, x_offset, y_offset):
        """
        Scrolling callback for glfw.

        Args:
            window: glfw window.
            x_offset: x scrolling offset.
            y_offset: y scrolling offset.

        """

        mujoco.mjv_moveCamera(self._model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, 0.05 * y_offset, self._scene, self._camera)

    def _set_mujoco_buffers(self):
        self._context = mujoco.MjrContext(self._model, mujoco.mjtFontScale(self._font_scale))
        if self._headless:
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._context)
            if self._context.currentBuffer != mujoco.mjtFramebuffer.mjFB_OFFSCREEN:
                raise RuntimeError("Offscreen rendering not supported")
        else:
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self._context)
            if self._context.currentBuffer != mujoco.mjtFramebuffer.mjFB_WINDOW:
                raise RuntimeError("Window rendering not supported")

    def update_headless_size(self, width, height):
        _context = mujoco.MjrContext(self._model, mujoco.mjtFontScale(self._font_scale))
        if width > _context.offWidth or height > _context.offHeight:
            width = max(width, self._model.vis.global_.offwidth)
            height = max(height, self._model.vis.global_.offheight)

            if width != _context.offWidth or height != _context.offHeight:
                self._model.vis.global_.offwidth = width
                self._model.vis.global_.offheight = height

        return width, height

    def _add_user_scene_geoms(self):
        for i in range(self._user_scene.ngeom):
            j = self._scene.ngeom
            # Copy all attributes from obj1 to obj2
            mujoco.mjv_initGeom(self._scene.geoms[j],
                                self._user_scene.geoms[i].type,
                                self._user_scene.geoms[i].size,
                                self._user_scene.geoms[i].pos,
                                self._user_scene.geoms[i].mat.reshape(-1),
                                self._user_scene.geoms[i].rgba)
            self._scene.ngeom += 1
            if self._scene.ngeom >= self._scene.maxgeom:
                raise RuntimeError(
                    'Ran out of geoms. maxgeom: %d' %
                    self._scene.ngeom.maxgeom)

    def render(self, data, carry, record):
        """
        Main rendering function.

        Args:
            data: Mujoco data structure.
            carry: Carry object.
            record (bool): If true, frames are returned during rendering.

        Returns:
            If record is True, frames are returned during rendering, else None.

        """

        def render_inner_loop(self):

            if not self._headless:
                self._create_overlay()

            render_start = time.time()

            mujoco.mjv_updateScene(self._model, data, self._scene_option, None, self._camera,
                                   mujoco.mjtCatBit.mjCAT_ALL,
                                   self._scene)

            # add visual geoms
            carry_visual_start_idx = self._scene.ngeom
            for j in range(carry.user_scene.ngeoms):
                if self._scene.ngeom > self._scene.maxgeom:
                    raise RuntimeError(
                        'Ran out of geoms. maxgeom: %d' %
                        self._scene.ngeom.maxgeom)
                mujoco.mjv_initGeom(self._scene.geoms[carry_visual_start_idx + j],
                                    carry.user_scene.geoms.type[j],
                                    carry.user_scene.geoms.size[j],
                                    carry.user_scene.geoms.pos[j],
                                    carry.user_scene.geoms.mat[j],
                                    carry.user_scene.geoms.rgba[j])
                # set dataid to be able to identify the geom in the user scene
                self._scene.geoms[carry_visual_start_idx + j].dataid = int(carry.user_scene.geoms.dataid[j]*2)
                self._scene.geoms[carry_visual_start_idx + j].category =  mujoco.mjtCatBit.mjCAT_DECOR

                self._scene.ngeom += 1

            self._add_user_scene_geoms()

            if not self._headless:
                self._viewport.width, self._viewport.height = glfw.get_window_size(self._window)

            mujoco.mjr_render(self._viewport, self._scene, self._context)

            for gridpos, [t1, t2] in self._overlay.items():

                if self._hide_menu:
                    continue

                mujoco.mjr_overlay(
                    mujoco.mjtFont.mjFONT_SHADOW,
                    gridpos,
                    self._viewport,
                    t1,
                    t2,
                    self._context)

            if self.custom_render_callback is not None:
                self.custom_render_callback(self._viewport, self._context)

            if not self._headless:
                glfw.swap_buffers(self._window)
                glfw.poll_events()
                if glfw.window_should_close(self._window):
                    self.stop()
                    exit(0)

            self.frames += 1
            self._overlay.clear()
            self._time_per_render = 0.9 * self._time_per_render + 0.1 * (time.time() - render_start)

        if self._paused:
            while self._paused:
                render_inner_loop(self)

        if record:
            self._loop_count = 1
        else:
            self._loop_count += self.dt / (self._time_per_render * self._run_speed_factor)
        while self._loop_count > 0:
            render_inner_loop(self)
            self._set_camera()
            self._loop_count -= 1

        im = self.read_pixels()

        if self._recorder:
            self._recorder(im)

        return im

    def parallel_render(self, mjx_state, record, offset=2.0):
        """
        Main rendering function.

        Args:
            datas: List of Mjx state.
            record (bool): If true, frames are returned during rendering.

        Returns:
            If record is True, frames are returned during rendering, else None.

        """

        def generate_square_positions(center_x, center_y, num_envs, offset):
            positions = []
            # Determine the size of the square grid
            grid_size = int((num_envs - 1) ** 0.5) + 1

            # Calculate starting coordinates to center the grid around (center_x, center_y)
            half_grid = grid_size // 2

            done = False  # Flag to track when to stop
            for i in range(grid_size):
                for j in range(grid_size):
                    x = center_x + (i - half_grid) * offset
                    y = center_y + (j - half_grid) * offset
                    positions.append((x, y))

                    if len(positions) == num_envs:
                        done = True
                        break

                if done:
                    break

            # Identify the middle index of the first column
            col_length = min(grid_size,
                             (num_envs + grid_size - 1) // grid_size)  # Number of elements in the first column
            middle_index = (col_length // 2) * grid_size  # Index of middle element in the first column

            # Swap the middle element of the first column with the first element
            if middle_index < len(positions):
                positions[0], positions[middle_index] = positions[middle_index], positions[0]

            return positions


        n_envs = mjx_state.data.qpos.shape[0]
        if self._offsets_for_parallel_render is None or n_envs > len(self._offsets_for_parallel_render):
            self._offsets_for_parallel_render = generate_square_positions(0.0, 0.0, n_envs, offset)
            self._visual_geom_offsets = np.array(self._offsets_for_parallel_render)[:, np.newaxis, :]
        if self._datas_for_parallel_render is None or n_envs > len(self._datas_for_parallel_render):
            self._datas_for_parallel_render = [mujoco.MjData(self._model) for i in range(n_envs)]

        # get visual geoms
        visual_geoms_type = np.array(mjx_state.additional_carry.user_scene.geoms.type)
        visual_geoms_size = np.array(mjx_state.additional_carry.user_scene.geoms.size)
        visual_geoms_pos = np.array(mjx_state.additional_carry.user_scene.geoms.pos)
        # visual_geoms_pos[..., :2] += self._visual_geom_offsets
        visual_geoms_mat = np.array(mjx_state.additional_carry.user_scene.geoms.mat)
        visual_geoms_rgba = np.array(mjx_state.additional_carry.user_scene.geoms.rgba)
        visual_geoms_dataid = np.array(mjx_state.additional_carry.user_scene.geoms.dataid)
        n_visual_geoms = mjx_state.additional_carry.user_scene.ngeoms[0]

        def render_all_inner_loop(self):

            render_start = time.time()

            data_list = _unbatch_general(mjx_state.data)

            for i in range(n_envs):
                data = self._datas_for_parallel_render[i]
                # offset = self._offsets_for_parallel_render[i]
                # data.qpos, data.qvel = mjx_state.data.qpos[i, :], mjx_state.data.qvel[i, :]
                # data.mocap_pos, data.mocap_quat = mjx_state.data.mocap_pos[i, :], mjx_state.data.mocap_quat[i, :]

                mjx.get_data_into(data, self._model, data_list[i])

                # for j in np.where(self._model.body_parentid == 0)[0]:
                #     model.body_pos[j, 0] = self._model.body_pos[j, 0] + offset[0]
                #     model.body_pos[j, 1] = self._model.body_pos[j, 1] + offset[1]
                # for j in np.where(self._model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]:
                #     adr = model.jnt_qposadr[j]
                #     data.qpos[adr:adr + 2] += offset[:2]

                # data.mocap_pos[:, 0] += offset[0]
                # data.mocap_pos[:, 1] += offset[1]
                # mujoco.mj_forward(model, data)

                if i == 0 and not self._headless:
                    self._create_overlay()

                if i == 0:
                    mujoco.mj_forward(self._model, data)
                    mujoco.mjv_updateScene(self._model, data, self._scene_option, None, self._camera,
                                           mujoco.mjtCatBit.mjCAT_ALL,
                                           self._scene)
                else:
                    mujoco.mjv_addGeoms(self._model, data, self._scene_option, mujoco.MjvPerturb(),
                                        mujoco.mjtCatBit.mjCAT_DYNAMIC, self._scene)

                # add visual geoms
                carry_visual_start_idx = self._scene.ngeom
                for j in range(n_visual_geoms):
                    if self._scene.ngeom > self._scene.maxgeom:
                        raise RuntimeError(
                            'Ran out of geoms. maxgeom: %d' %
                            self._scene.ngeom.maxgeom)
                    mujoco.mjv_initGeom(self._scene.geoms[carry_visual_start_idx + j],
                                        visual_geoms_type[i, j],
                                        visual_geoms_size[i, j],
                                        visual_geoms_pos[i, j],
                                        visual_geoms_mat[i, j],
                                        visual_geoms_rgba[i, j])

                    # set dataid to be able to identify the geom in the user scene
                    self._scene.geoms[carry_visual_start_idx + j].dataid = int(visual_geoms_dataid[i, j] * 2)
                    self._scene.geoms[carry_visual_start_idx + j].category = mujoco.mjtCatBit.mjCAT_DECOR

                    self._scene.ngeom += 1

            self._add_user_scene_geoms()

            if not self._headless:
                self._viewport.width, self._viewport.height = glfw.get_window_size(self._window)

            mujoco.mjr_render(self._viewport, self._scene, self._context)

            for gridpos, [t1, t2] in self._overlay.items():

                if self._hide_menu:
                    continue

                mujoco.mjr_overlay(
                    mujoco.mjtFont.mjFONT_SHADOW,
                    gridpos,
                    self._viewport,
                    t1,
                    t2,
                    self._context)

            if self.custom_render_callback is not None:
                self.custom_render_callback(self._viewport, self._context)

            if not self._headless:
                glfw.swap_buffers(self._window)
                glfw.poll_events()
                if glfw.window_should_close(self._window):
                    self.stop()
                    exit(0)

            self.frames += 1
            self._overlay.clear()
            self._time_per_render = 0.9 * self._time_per_render + 0.1 * (time.time() - render_start)

        if self._paused:
            while self._paused:
                render_all_inner_loop(self)

        if record:
            self._loop_count = 1
        else:
            self._loop_count += self.dt / (self._time_per_render * self._run_speed_factor)
        while self._loop_count > 0:
            render_all_inner_loop(self)
            self._set_camera()
            self._loop_count -= 1

        im = self.read_pixels()

        if self._recorder:
            self._recorder(im)

        return im

    def read_pixels(self, depth=False):
        """
        Reads the pixels from the glfw viewer.

        Args:
            depth (bool): If True, depth map is also returned.

        Returns:
            If depth is True, tuple of np.arrays (rgb and depth), else just a single
            np.array for the rgb image.

        """

        if self._headless:
            shape = (self._width, self._height)
        else:
            shape = glfw.get_framebuffer_size(self._window)

        if depth:
            rgb_img = np.zeros((shape[1], shape[0], 3), dtype=np.uint8)
            depth_img = np.zeros((shape[1], shape[0], 1), dtype=np.float32)
            mujoco.mjr_readPixels(rgb_img, depth_img, self._viewport, self._context)
            return (np.flipud(rgb_img), np.flipud(depth_img))
        else:
            img = np.zeros((shape[1], shape[0], 3), dtype=np.uint8)
            mujoco.mjr_readPixels(img, None, self._viewport, self._context)
            return np.flipud(img)

    def stop(self):
        """
        Destroys the glfw image.

        Returns:
            If record was True, the video is saved and the path is returned.

        """
        if not self._headless:
            glfw.destroy_window(self._window)
        if self._recorder:
            return self._recorder.stop()
        else:
            return None

    def _create_overlay(self):
        """
        This function creates and adds all overlays used in the viewer.

        """

        topleft = mujoco.mjtGridPos.mjGRID_TOPLEFT
        topright = mujoco.mjtGridPos.mjGRID_TOPRIGHT
        bottomleft = mujoco.mjtGridPos.mjGRID_BOTTOMLEFT
        bottomright = mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT

        def add_overlay(gridpos, text1, text2="", make_new_line=True):
            if gridpos not in self._overlay:
                self._overlay[gridpos] = ["", ""]
            if make_new_line:
                self._overlay[gridpos][0] += text1 + "\n"
                self._overlay[gridpos][1] += text2 + "\n"
            else:
                self._overlay[gridpos][0] += text1
                self._overlay[gridpos][1] += text2

        add_overlay(
            bottomright,
            "Framerate:",
            str(int(1 / self._time_per_render * self._run_speed_factor)), make_new_line=False)

        add_overlay(
            topleft,
            "Press SPACE to pause.")

        add_overlay(
            topleft,
            "Press H to hide the menu.")

        add_overlay(
            topleft,
            "Press TAB to switch cameras.")

        add_overlay(
            topleft,
            "Press T to make the model transparent.")

        add_overlay(
            topleft,
            "Press E to toggle reference frames.")

        add_overlay(
            topleft,
            "Press M for mimic sites visualization (if available).")

        add_overlay(
            topleft,
            "Press 0-5 to disable/enable geom group visualization.")

        visualize_contact = "On" if self._scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] else "Off"
        add_overlay(
            topleft,
            "Contact force visualization (Press C):", visualize_contact)

        add_overlay(
            topleft,
            "Camera mode:",
            self._camera_mode)

        add_overlay(
            topleft,
            "Run speed = %.3f x real time" %
            self._run_speed_factor,
            "[S]lower, [F]aster", make_new_line=False)

    def _set_camera(self):
        """
        Sets the camera mode to the current camera mode target. Allowed camera
        modes are "follow" in which the model is tracked, "static" that is a static
        camera at the default camera positon, and "top_static" that is a static
        camera on top of the model.

        """

        if self._camera_mode_target == "follow":
            if self._camera_mode != "follow":
                self._camera.fixedcamid = -1
                self._camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self._camera.trackbodyid = 0
                self._set_camera_properties(self._camera_mode_target)
        elif self._camera_mode_target == "static":
            if self._camera_mode != "static":
                self._camera.fixedcamid = 0
                self._camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                self._camera.trackbodyid = -1
                self._set_camera_properties(self._camera_mode_target)
        elif self._camera_mode_target == "top_static":
            if self._camera_mode != "top_static":
                self._camera.fixedcamid = 0
                self._camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                self._camera.trackbodyid = -1
                self._set_camera_properties(self._camera_mode_target)

    def _set_camera_properties(self, mode):
        """
        Sets the camera properties "distance", "elevation", and "azimuth"
        as well as the camera mode based on the provided mode.

        Args:
            mode (str): Camera mode. (either "follow", "static", or "top_static")

        """

        cam_params = self._camera_params[mode]
        self._camera.distance = cam_params["distance"]
        self._camera.elevation = cam_params["elevation"]
        self._camera.azimuth = cam_params["azimuth"]
        if "lookat" in cam_params:
            self._camera.lookat = np.array(cam_params["lookat"])
        self._camera_mode = mode

    def _assert_camera_params(self, camera_params):
        """
        Asserts if the provided camera parameters are valid or not. Also, if
        properties of some camera types are not specified, the default parameters
        are used.

        Args:
            camera_params (dict): Dictionary of dictionaries containig parameters for each camera type.

        Returns:
            Dictionary of dictionaries with parameters for each camera type.

        """

        default_camera_params = self.get_default_camera_params()

        # check if the provided camera types and parameters are valid
        for cam_type in camera_params.keys():
            assert cam_type in default_camera_params.keys(), f"Camera type \"{cam_type}\" is unknown. Allowed " \
                                                             f"camera types are {list(default_camera_params.keys())}."
            for param in camera_params[cam_type].keys():
                assert param in default_camera_params[cam_type].keys(), f"Parameter \"{param}\" of camera type " \
                                                                        f"\"{cam_type}\" is unknown. Allowed " \
                                                                        f"parameters are" \
                                                                        f" {list(default_camera_params[cam_type].keys())}"

        # add default parameters if not specified
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
        Getter for default camera paramterization.

        Returns:
            Dictionary of dictionaries with default parameters for each camera type.

        """

        return dict(static=dict(distance=15.0, elevation=-45.0, azimuth=90.0, lookat=np.array([0.0, 0.0, 0.0])),
                    follow=dict(distance=3.5, elevation=0.0, azimuth=90.0),
                    top_static=dict(distance=5.0, elevation=-90.0, azimuth=90.0, lookat=np.array([0.0, 0.0, 0.0])))

    def setup_opengl_backend_headless(self, width, height):

        backend = os.environ.get("MUJOCO_GL")
        if backend is not None:
            try:
                opengl_context = _ALL_RENDERERS[backend](width, height)
            except KeyError:
                raise RuntimeError(
                    "Environment variable {} must be one of {!r}: got {!r}.".format(
                        "MUJOCO_GL", _ALL_RENDERERS.keys(), backend
                    )
                )

        else:
            # iterate through all OpenGL backends to see which one is available
            for name, _ in _ALL_RENDERERS.items():
                try:
                    opengl_context = _ALL_RENDERERS[name](width, height)
                    backend = name
                    break
                except:  # noqa:E722
                    pass
            if backend is None:
                raise RuntimeError(
                    "No OpenGL backend could be imported. Attempting to create a "
                    "rendering context will result in a RuntimeError."
                )

        return opengl_context

    def upload_hfield(self, model, hfield_id):
        """
        Uploads the height field to the GPU.

        Args:
            model: Mujoco model.
            hfield_id: Height field id.

        """
        mujoco.mjr_uploadHField(model, self._context, hfield_id)

    @property
    def user_scene(self):
        return self._user_scene

    @property
    def video_file_path(self):
        if self._recorder is not None:
            return self._recorder.file_path
        else:
            return None

@jax.jit
def _unbatch_general(batched_data):
    leaves, treedef = jax.tree_util.tree_flatten(batched_data)
    unbatched_leaves = zip(*[leaf for leaf in leaves])  # zip over batch axis
    return [jax.tree_util.tree_unflatten(treedef, elems) for elems in unbatched_leaves]