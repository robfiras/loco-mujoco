import numpy as np
import mujoco
import pytest

from loco_mujoco.core.visuals.scene import MjvScene


viser = pytest.importorskip("viser", reason="the viser viewer requires loco-mujoco[viser]")
mjviser = pytest.importorskip("mjviser", reason="the viser viewer requires loco-mujoco[viser]")

from loco_mujoco.core.visuals.viser_viewer import (  # noqa: E402
    DEFAULT_FLOOR_COLOR, DEFAULT_LOGO, FLOOR_FROM_MODEL, ViserViewer)
from loco_mujoco.task_factories import RLFactory  # noqa: E402


_BOX = int(mujoco.mjtGeom.mjGEOM_BOX)
_SPHERE = int(mujoco.mjtGeom.mjGEOM_SPHERE)
_ARROW = int(mujoco.mjtGeom.mjGEOM_ARROW)
_MESH = int(mujoco.mjtGeom.mjGEOM_MESH)
_PLANE = int(mujoco.mjtGeom.mjGEOM_PLANE)
_ARROW_HEAD = -1


@pytest.fixture
def env():
    e, _ = RLFactory.make("UnitreeH1", port=0, verbose=False)
    e.reset()
    yield e
    e.stop()


@pytest.fixture
def mjx_env():
    e, _ = RLFactory.make("MjxUnitreeH1", port=0, verbose=False, use_mjwarp=False)
    yield e
    e.stop()


def _user_scene(**geom_fields):
    """Builds a user scene with `n` geoms from the given per-geom arrays."""
    n = len(next(iter(geom_fields.values())))
    scene = MjvScene.init_n_geoms(n, np)
    return scene.replace(geoms=scene.geoms.replace(**{k: np.asarray(v) for k, v in geom_fields.items()}))


def test_render_returns_frame(env):
    frame = env.render_viser()
    assert isinstance(env.viewer, ViserViewer)
    assert frame.shape == (720, 1280, 3)
    assert frame.dtype == np.uint8


def test_viewer_params_are_forwarded():
    e, _ = RLFactory.make("UnitreeH1", port=0, verbose=False, viewer_size=(640, 480),
                          default_camera_mode="top_static",
                          geom_group_visualization_on_startup=[0, 2],
                          mimic_site_visualization_on_startup=True)
    try:
        e.reset()
        frame = e.render_viser()
        assert frame.shape == (480, 640, 3)
        assert e.viewer._camera_mode == "top_static"
        assert e.viewer.scene.geom_groups_visible == [True, False, True, False, False, False]
        assert e.viewer.scene.site_groups_visible[4] is True
    finally:
        e.stop()


@pytest.mark.parametrize("groups, hidden_group", [([0, 2], 1), ([1], 0)])
def test_geom_groups_are_actually_hidden(groups, hidden_group):
    """The startup groups must reach the handles, not just the flags on the scene."""
    e, _ = RLFactory.make("UnitreeH1", port=0, verbose=False,
                          geom_group_visualization_on_startup=groups)
    try:
        e.reset()
        e.render_viser()
        scene = e.viewer.scene
        visible_by_group = {}
        handles = ([(mg.group_id, mg.handle) for mg in scene._mesh_groups] +
                   [(gid, h) for (_, gid, _), h in scene._fixed_geom_handles.items()])
        for group_id, handle in handles:
            visible_by_group.setdefault(int(group_id), set()).add(bool(handle.visible))

        assert visible_by_group, "no geom handles were created"
        for group_id, states in visible_by_group.items():
            assert states == {group_id in groups}, f"geom group {group_id} has visibility {states}"
        # the collision geoms of the H1 are in a group that must be off in the first case
        if hidden_group in visible_by_group:
            assert visible_by_group[hidden_group] == {False}
    finally:
        e.stop()


def test_unsupported_viewer_params_warn(env):
    with pytest.warns(UserWarning, match="does not support the viewer parameters"):
        viewer = ViserViewer(env._model, env.dt, port=0, verbose=False, custom_render_callback=lambda *a: None)
    viewer.stop()


def test_camera_modes(env):
    env.render_viser()
    viewer = env.viewer

    viewer._apply_camera_mode("follow")
    assert viewer.scene.camera_tracking_enabled is True
    np.testing.assert_allclose(viewer._camera_lookat, np.zeros(3))

    viewer._apply_camera_mode("top_static")
    assert viewer.scene.camera_tracking_enabled is False
    # a top-down camera sits straight above its lookat point
    params = viewer.get_default_camera_params()["top_static"]
    np.testing.assert_allclose(viewer._camera_position, [0.0, 0.0, params["distance"]], atol=1e-6)


def test_marker_geometry(env):
    env.render_viser()
    viewer = env.viewer
    # the follow camera shifts the whole scene, so compare against the same offset
    offset = viewer._scene_offset()

    angle = np.radians(30.0)
    rot_z = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                      [np.sin(angle), np.cos(angle), 0.0],
                      [0.0, 0.0, 1.0]])
    scene = _user_scene(type=[[_BOX], [_ARROW]],
                        size=[[0.075, 0.05, 0.025], [0.025, 0.025, 1.0]],
                        pos=[[1.0, 2.0, 3.0], [0.0, 0.0, 0.5]],
                        mat=[rot_z.reshape(-1), np.eye(3).reshape(-1)],
                        rgba=[[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.75]])
    viewer._update_markers(scene)

    box = viewer._marker_handles[(_BOX, None)]
    np.testing.assert_allclose(box.batched_positions[0], np.array([1.0, 2.0, 3.0]) + offset, atol=1e-5)
    np.testing.assert_allclose(box.batched_scales[0], [0.075, 0.05, 0.025], atol=1e-6)
    np.testing.assert_array_equal(box.batched_colors[0], [0, 255, 0])
    # a 30 deg rotation about z
    np.testing.assert_allclose(box.batched_wxyzs[0], [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)], atol=1e-6)

    # an arrow is split into an 80% shaft and a 20% cone head at its tip
    arrow = viewer._marker_handles[(_ARROW, None)]
    head = viewer._marker_handles[(_ARROW_HEAD, None)]
    np.testing.assert_allclose(arrow.batched_positions[0], np.array([0.0, 0.0, 0.5]) + offset, atol=1e-5)
    np.testing.assert_allclose(arrow.batched_scales[0], [0.025, 0.025, 0.8], atol=1e-6)
    np.testing.assert_allclose(head.batched_positions[0], np.array([0.0, 0.0, 1.3]) + offset, atol=1e-5)
    np.testing.assert_allclose(head.batched_scales[0], [0.025, 0.025, 0.2], atol=1e-6)
    # opacity is not readable through the public handle property, so check the message props
    np.testing.assert_allclose(arrow._impl.props.batched_opacities, [0.75], atol=1e-6)


def test_invisible_and_placeholder_markers_are_skipped(env):
    env.render_viser()
    viewer = env.viewer

    # default-initialized geom slots are planes and must not show up as markers
    scene = _user_scene(type=[[_PLANE], [_SPHERE], [_SPHERE]],
                        rgba=[[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    viewer._update_markers(scene)

    assert (_PLANE, None) not in viewer._marker_handles
    # only the opaque sphere survives; the fully transparent one is dropped
    assert viewer._marker_handles[(_SPHERE, None)].batched_positions.shape[0] == 1


def test_mesh_markers_use_model_meshes(env):
    env.render_viser()
    viewer = env.viewer
    assert env._model.nmesh > 0

    mesh_id = 0
    scene = _user_scene(type=[[_MESH]], dataid=[[mesh_id]],
                        rgba=[[0.471, 0.38, 0.812, 0.5]])
    viewer._update_markers(scene)

    handle = viewer._marker_handles[(_MESH, mesh_id)]
    n_verts = env._model.mesh_vertnum[mesh_id]
    assert handle._impl.props.vertices.shape == (n_verts, 3)


def test_markers_hidden_when_gone(env):
    env.render_viser()
    viewer = env.viewer

    viewer._update_markers(_user_scene(type=[[_SPHERE]], rgba=[[1.0, 0.0, 0.0, 1.0]]))
    handle = viewer._marker_handles[(_SPHERE, None)]
    assert handle.visible is True

    viewer._update_markers(_user_scene(type=[[_BOX]], rgba=[[1.0, 0.0, 0.0, 1.0]]))
    assert handle.visible is False


def test_goal_markers_are_rendered():
    e, _ = RLFactory.make("UnitreeH1", port=0, verbose=False, goal_params=dict(visualize_goal=True))
    try:
        e.reset()
        e.render_viser()
        # the root velocity goal draws a velocity arrow (with its head) and a center sphere
        assert set(e.viewer._marker_handles.keys()) == {(_ARROW, None), (_ARROW_HEAD, None), (_SPHERE, None)}
    finally:
        e.stop()


def _grid_handles(env):
    nodes = env.viewer.server.scene._handle_from_node_name
    return {n: h for n, h in nodes.items() if "GridProps" in type(h._impl.props).__name__}


def test_default_floor_is_dark(env):
    env.render_viser()
    grids = _grid_handles(env)
    assert len(grids) == 1, "the ground plane should map to exactly one grid node"
    assert tuple(next(iter(grids.values()))._impl.props.plane_color) == DEFAULT_FLOOR_COLOR


def test_floor_from_model_matches_the_mujoco_checkerboard():
    """floor_color="model" must reproduce the two greys of the model's plane texture."""
    e, _ = RLFactory.make("UnitreeH1", port=0, verbose=False, floor_color=FLOOR_FROM_MODEL)
    try:
        e.reset()
        e.render_viser()
        props = next(iter(_grid_handles(e).values()))._impl.props
        # the two dominant tones of the MuJoCo checkerboard, straight from the texture
        dark, light = e.viewer._plane_texture_tones(int(e._model.geom_matid[0]))
        assert tuple(props.plane_color) == tuple(int(c) for c in light)
        assert tuple(props.cell_color) == tuple(int(c) for c in dark)
        # on a light floor the lines have to be darker than the plane to be visible
        assert all(c < p for c, p in zip(props.cell_color, props.plane_color))
        assert all(s < c for s, c in zip(props.section_color, props.cell_color))
        assert props.plane_opacity == 1.0
    finally:
        e.stop()


def test_colors_can_be_changed_live_without_duplicating_nodes(env):
    """The GUI pickers re-apply colors on a running viewer."""
    env.render_viser()
    viewer = env.viewer

    viewer._apply_floor_color((180, 40, 40))
    grids = _grid_handles(env)
    assert len(grids) == 1, "re-applying must replace the grid, not add another"
    assert tuple(next(iter(grids.values()))._impl.props.plane_color) == (180, 40, 40)
    assert viewer._floor_rgb == (180, 40, 40)

    launched_sky = viewer._sky_rgb
    viewer._apply_sky(((0, 0, 40), (255, 120, 0)))
    assert viewer._sky_rgb == ((0, 0, 40), (255, 120, 0))

    # the Reset buttons restore whatever the viewer was constructed with
    viewer._apply_floor_color(viewer._default_floor_color)
    viewer._apply_sky(viewer._default_sky)
    assert viewer._floor_rgb == DEFAULT_FLOOR_COLOR
    assert viewer._sky_rgb == launched_sky


def test_reset_restores_the_launch_colors_not_the_model():
    """A viewer started with model colors must reset back to those, not to the dark default."""
    e, _ = RLFactory.make("UnitreeH1", port=0, verbose=False, floor_color=FLOOR_FROM_MODEL)
    try:
        e.reset()
        e.render_viser()
        viewer = e.viewer
        launched = viewer._floor_rgb
        assert launched != DEFAULT_FLOOR_COLOR

        viewer._apply_floor_color((1, 2, 3))
        viewer._apply_floor_color(viewer._default_floor_color)
        assert viewer._floor_rgb == launched
    finally:
        e.stop()


@pytest.mark.parametrize("base", [(28, 30, 34), (230, 230, 235)])
def test_explicit_floor_color_lands_on_the_plane(base):
    """An explicit color is the plane itself; lines contrast away from it in either direction."""
    e, _ = RLFactory.make("UnitreeH1", port=0, verbose=False, floor_color=base)
    try:
        e.reset()
        e.render_viser()
        props = next(iter(_grid_handles(e).values()))._impl.props
        assert tuple(props.plane_color) == base
        lighten = sum(base) < 3 * 127
        for line in (props.cell_color, props.section_color):
            assert all((v > b) == lighten for v, b in zip(line, base))
    finally:
        e.stop()


def test_unknown_floor_color_is_rejected():
    e, _ = RLFactory.make("UnitreeH1", port=0, verbose=False, floor_color="chartreuse")
    try:
        e.reset()
        with pytest.raises(ValueError, match="Unknown floor_color"):
            e.render_viser()
    finally:
        e.stop()


def test_floor_color_can_be_left_to_mjviser():
    e, _ = RLFactory.make("UnitreeH1", port=0, verbose=False, floor_color=None)
    try:
        e.reset()
        e.render_viser()
        props = next(iter(_grid_handles(e).values()))._impl.props
        assert tuple(props.plane_color) == (255, 255, 255)
    finally:
        e.stop()


def test_sky_gradient_comes_from_the_model_skybox(env):
    env.render_viser()
    column = env.viewer._skybox_gradient()
    assert column is not None and column.ndim == 2 and column.shape[1] == 3
    # the MuJoCo skybox is a warm haze that darkens towards the horizon
    assert np.all(column[:, 0] >= column[:, 1]) and np.all(column[:, 1] >= column[:, 2])
    assert column[0].sum() > column[-1].sum()


def test_sky_accepts_an_explicit_gradient_and_can_be_disabled():
    for sky in (((10, 20, 60), (200, 90, 30)), None):
        e, _ = RLFactory.make("UnitreeH1", port=0, verbose=False, sky=sky)
        try:
            e.reset()
            e.render_viser()
        finally:
            e.stop()


def test_unknown_sky_is_rejected():
    e, _ = RLFactory.make("UnitreeH1", port=0, verbose=False, sky="sunset")
    try:
        e.reset()
        with pytest.raises(ValueError, match="Unknown sky"):
            e.render_viser()
    finally:
        e.stop()


def _logo_html(env):
    handle = getattr(env.viewer, "_logo_handle", None)
    return None if handle is None else handle._impl.props.content


def test_logo_overlay_is_pinned_top_left(env):
    env.render_viser()
    html = _logo_html(env)
    assert DEFAULT_LOGO in html
    for style in ("position:fixed", "left:16px", "top:14px", "pointer-events:none"):
        assert style in html, f"missing {style!r} in the logo overlay"
    assert "bottom:" not in html and "right:" not in html


def test_logo_can_be_hidden_from_the_gui(env):
    env.render_viser()
    handle = env.viewer._logo_handle
    assert handle.visible is True
    handle.visible = False
    assert handle.visible is False


def test_logo_from_local_file_is_embedded(tmp_path):
    logo = tmp_path / "logo.png"
    logo.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    e, _ = RLFactory.make("UnitreeH1", port=0, verbose=False, logo=str(logo), logo_width=64)
    try:
        e.reset()
        e.render_viser()
        html = _logo_html(e)
        assert "data:image/png;base64," in html
        assert "width:64px" in html
    finally:
        e.stop()


def test_logo_can_be_disabled():
    e, _ = RLFactory.make("UnitreeH1", port=0, verbose=False, logo=None)
    try:
        e.reset()
        e.render_viser()
        assert _logo_html(e) is None
    finally:
        e.stop()


def test_missing_logo_warns_instead_of_crashing():
    e, _ = RLFactory.make("UnitreeH1", port=0, verbose=False, logo="/does/not/exist.png")
    try:
        e.reset()
        with pytest.warns(UserWarning, match="does not exist"):
            e.render_viser()
        assert _logo_html(e) is None
    finally:
        e.stop()


def test_grid_positions_center_first_env():
    positions = ViserViewer.generate_square_positions(0.0, 0.0, 4, 2.0)
    assert len(positions) == 4
    assert len(set(positions)) == 4
    # env 0 is swapped into the middle of the first column
    assert positions[0] == (0.0, -2.0)


def test_parallel_render(mjx_env):
    import jax
    import jax.numpy as jnp

    n_envs = 4
    keys = jax.random.split(jax.random.key(0), n_envs)
    state = jax.jit(jax.vmap(mjx_env.mjx_reset))(keys)
    state = jax.jit(jax.vmap(mjx_env.mjx_step))(state, jnp.zeros((n_envs, mjx_env.info.action_space.shape[0])))

    frame = mjx_env.mjx_render_viser(state)
    assert frame.shape == (720, 1280, 3)
    assert isinstance(mjx_env.viewer, ViserViewer)
    assert len(mjx_env.viewer._offsets_for_parallel_render) == n_envs


def test_read_pixels_without_client_warns(env):
    env.render_viser()
    with pytest.warns(UserWarning, match="No browser is connected"):
        frame = env.viewer.read_pixels()
    assert frame.shape == (720, 1280, 3)
    assert not frame.any()


def test_mixing_backends_is_rejected(env):
    # stand in for a running OpenGL viewer; creating a real one needs a GL context
    env._viewer = object()
    try:
        with pytest.raises(AssertionError, match="OpenGL viewer is already running"):
            env.render_viser()
    finally:
        env._viewer = None


def test_viser_only_params_are_hidden_from_the_opengl_viewer(env):
    env._viewer_params.update(port=0, verbose=False, host="127.0.0.1", num_envs=1,
                              default_camera_mode="follow")
    filtered = env._glfw_viewer_params()
    assert not {"port", "verbose", "host", "num_envs"} & filtered.keys()
    assert filtered["default_camera_mode"] == "follow"


# ---------------------------------------------------------------- lifecycle


def test_port_collision_warns_and_falls_back():
    """A stale viewer holding the port must not silently serve the old scene."""
    blocker = viser.ViserServer(port=8931, verbose=False)
    try:
        e, _ = RLFactory.make("UnitreeH1", port=8931, verbose=False)
        e.reset()
        with pytest.warns(UserWarning, match="already in use"):
            e.render_viser()
        assert e.viewer.server.get_port() != 8931
        e.stop()
    finally:
        blocker.stop()


def test_stop_closes_the_server_and_returns_the_video_path(tmp_path):
    e, _ = RLFactory.make("UnitreeH1", port=0, verbose=False,
                          recorder_params=dict(path=str(tmp_path), compress=False))
    e.reset()
    with pytest.warns(UserWarning, match="No browser is connected"):
        e.render_viser(record=True)
    e.stop()

    assert e._viewer is None
    assert e.video_file_path is not None
    assert list(tmp_path.rglob("*.mp4")), "the recorder should have written a video"


def test_recording_feeds_the_recorder(tmp_path):
    """Frames are pushed to the VideoRecorder even though they are empty without a browser."""
    e, _ = RLFactory.make("UnitreeH1", port=0, verbose=False,
                          recorder_params=dict(path=str(tmp_path), compress=False))
    try:
        e.reset()
        with pytest.warns(UserWarning, match="No browser is connected"):
            for _ in range(3):
                e.step(np.zeros(e.info.action_space.shape))
                frame = e.render_viser(record=True)
        assert frame.shape == (720, 1280, 3)
        assert e.viewer._recorder is not None
    finally:
        e.stop()


def test_play_trajectory_with_viser(tmp_path):
    from loco_mujoco.task_factories import ImitationFactory

    e, _ = ImitationFactory.make("UnitreeH1", default_dataset_conf=dict(task="walk"),
                                 port=0, verbose=False)
    e.play_trajectory(n_episodes=1, n_steps_per_episode=3, viser=True, quiet=True)
    # play_trajectory calls stop() itself
    assert e._viewer is None


def test_read_pixels_rejects_depth(env):
    env.render_viser()
    with pytest.raises(NotImplementedError, match="depth"):
        env.viewer.read_pixels(depth=True)


# ------------------------------------------------------------ parallel path


def test_parallel_render_rejects_a_different_batch_size(mjx_env):
    import jax

    state = jax.jit(jax.vmap(mjx_env.mjx_reset))(jax.random.split(jax.random.key(0), 3))
    mjx_env.mjx_render_viser(state)

    bigger = jax.jit(jax.vmap(mjx_env.mjx_reset))(jax.random.split(jax.random.key(1), 5))
    with pytest.raises(AssertionError, match="was created for 3 environments"):
        mjx_env.mjx_render_viser(bigger)


def test_parallel_markers_follow_the_grid_offsets():
    """Each environment's markers must be shifted onto that environment's grid cell."""
    import jax

    n_envs = 4
    e, _ = RLFactory.make("MjxUnitreeH1", port=0, verbose=False, use_mjwarp=False,
                          goal_params=dict(visualize_goal=True))
    try:
        state = jax.jit(jax.vmap(e.mjx_reset))(jax.random.split(jax.random.key(0), n_envs))
        e.mjx_render_viser(state)
        viewer = e.viewer

        offsets = np.array(viewer._offsets_for_parallel_render)
        spheres = viewer._marker_handles[(_SPHERE, None)].batched_positions
        assert spheres.shape[0] == n_envs

        # the sphere marker sits above each robot's root, so its xy must track the grid
        raw_xy = np.array(state.additional_carry.user_scene.geoms.pos)[:, :, :2]
        sphere_idx = int(np.argmax(np.asarray(state.additional_carry.user_scene.geoms.type)[0] == _SPHERE))
        expected_xy = raw_xy[:, sphere_idx, :] + offsets
        np.testing.assert_allclose(spheres[:, :2], expected_xy + viewer._scene_offset()[:2],
                                   atol=1e-4)
    finally:
        e.stop()


# ------------------------------------------------------------- other models


def test_model_without_skybox_or_texture_is_handled():
    """A bare model has no skybox and an untextured plane; neither may raise."""
    from loco_mujoco.core.visuals.viser_viewer import ViserViewer as VV

    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><geom name='ground' type='plane' size='5 5 .1' rgba='.2 .4 .2 1'/>"
        "<body><joint type='free'/><geom type='sphere' size='.1'/></body></worldbody></mujoco>")
    # floor_color="model" so the untextured-plane fallback is the thing under test
    viewer = VV(model, dt=0.01, port=0, verbose=False, floor_color=FLOOR_FROM_MODEL)
    try:
        assert viewer._skybox_gradient() is None
        assert viewer._sky_rgb is None          # no skybox, so no background was set
        # no texture on the plane, so its geom rgba (.2 .4 .2) is used directly
        assert viewer._floor_rgb == (51, 102, 51)
        viewer.render(mujoco.MjData(model), None, False)
    finally:
        viewer.stop()


def test_dynamic_terrain_rebuilds_the_scene():
    e, _ = RLFactory.make("UnitreeH1", port=0, verbose=False,
                          terrain_type="RoughTerrain")
    try:
        e.reset()
        e.render_viser()
        assert e._terrain.is_dynamic
        # exercised through render_viser above; calling it directly must also be safe
        e.viewer.upload_hfield(e._model, hfield_id=e._terrain.hfield_id)
    finally:
        e.stop()
