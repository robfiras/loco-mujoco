"""Tests for portable animated USD trajectory export."""

from pathlib import Path
from types import SimpleNamespace

import jax
import mujoco
import numpy as np
import pytest

pytest.importorskip("pxr", reason="install the usd extra to test USD export")
from pxr import Sdf, Usd, UsdGeom  # noqa: E402

from loco_mujoco.core.visuals.usd_exporter import UsdTrajectoryExporter  # noqa: E402
from loco_mujoco.task_factories import ImitationFactory, RLFactory  # noqa: E402


def _tiny_model():
    return mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <visual><global offwidth="640" offheight="480"/></visual>
          <asset>
            <texture name="checker" type="2d" builtin="checker"
                     rgb1=".4 .4 .4" rgb2=".6 .6 .6" width="8" height="8"/>
            <material name="floor_mat" texture="checker"/>
          </asset>
          <worldbody>
            <geom name="floor" type="plane" size="5 5 .1" material="floor_mat"/>
            <light pos="0 -2 4"/>
            <body name="ball" pos="0 0 1">
              <joint type="free"/>
              <geom name="ball_geom" type="sphere" size=".2" rgba=".8 .2 .1 1"/>
              <geom name="hyphen-name" type="sphere" size=".05" pos="0 0 .3"/>
              <geom name="collision_geom" group="1" type="sphere" size=".25"
                    rgba="1 0 0 1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )


def _tiny_mocap_model():
    return mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <geom name="floor" type="plane" size="5 5 .1"/>
            <body name="marker" mocap="true"><geom type="sphere" size=".05"/></body>
            <body name="ball" pos="0 0 1">
              <joint type="free"/><geom type="sphere" size=".2"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )


def test_exports_an_animated_usd_with_correct_timing(tmp_path):
    model = _tiny_model()
    data = mujoco.MjData(model)
    renderer = UsdTrajectoryExporter(
        model, dt=0.02, output_directory=tmp_path, verbose=False
    )

    for x in (0.0, 0.25, 0.5):
        data.qpos[0] = x
        mujoco.mj_forward(model, data)
        renderer.render(data)

    path = Path(renderer.stop())
    assert path == tmp_path / "frames" / "frame_3.usd"
    assert path.is_file()
    assert (tmp_path / "assets").is_dir()

    stage = Usd.Stage.Open(str(path))
    assert stage.GetStartTimeCode() == 0
    assert stage.GetEndTimeCode() == 2
    assert stage.GetTimeCodesPerSecond() == 50
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert any("ball_geom" in prim.GetName() for prim in stage.Traverse())
    assert any("hyphen_name" in prim.GetName() for prim in stage.Traverse())
    assert not any("collision_geom" in prim.GetName() for prim in stage.Traverse())
    target = UsdGeom.Xform(stage.GetPrimAtPath("/World/LocoMuJoCoCameraTarget"))
    target_op = target.GetOrderedXformOps()[0]
    assert target_op.GetTimeSamples() == [0.0, 1.0, 2.0]
    assert target_op.Get(2.0)[0] == pytest.approx(0.5)
    varnames = [
        prim.GetAttribute("inputs:varname")
        for prim in stage.Traverse()
        if prim.GetAttribute("inputs:varname")
    ]
    assert varnames
    assert all(attr.GetTypeName() == Sdf.ValueTypeNames.String for attr in varnames)
    metallic = [
        prim.GetAttribute("inputs:metallic")
        for prim in stage.Traverse()
        if prim.GetAttribute("inputs:metallic")
    ]
    assert metallic
    assert all(attr.Get() == 0.0 for attr in metallic)


def test_stop_without_frames_warns_and_writes_nothing(tmp_path):
    renderer = UsdTrajectoryExporter(_tiny_model(), dt=0.01,
                               output_directory=tmp_path, verbose=False)
    with pytest.warns(UserWarning, match="No USD frames"):
        assert renderer.stop() is None
    assert not list((tmp_path / "frames").glob("*.usd"))


def test_lower_output_fps_decimates_without_changing_duration(tmp_path):
    model = _tiny_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = UsdTrajectoryExporter(model, dt=0.01, fps=30,
                               output_directory=tmp_path, verbose=False)

    # 101 source states cover exactly one second, including t=0 and t=1.
    for _ in range(101):
        renderer.render(data)
    path = Path(renderer.stop())

    assert renderer.source_frame_count == 101
    assert renderer.frame_count == 31
    stage = Usd.Stage.Open(str(path))
    assert stage.GetEndTimeCode() == 30
    assert stage.GetTimeCodesPerSecond() == 30
    assert stage.GetEndTimeCode() / stage.GetTimeCodesPerSecond() == 1.0


def test_output_fps_cannot_exceed_source_frequency(tmp_path):
    with pytest.raises(ValueError, match="cannot exceed the source frequency"):
        UsdTrajectoryExporter(_tiny_model(), dt=0.01, fps=120,
                        output_directory=tmp_path, verbose=False)


def test_unknown_camera_body_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown camera body"):
        UsdTrajectoryExporter(_tiny_model(), dt=0.01, camera_body_name="missing",
                        output_directory=tmp_path, verbose=False)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dt": 0}, "dt must be positive"),
        ({"dt": 0.01, "filetype": "obj"}, "filetype must be"),
        ({"dt": 0.01, "fps": 0}, "fps must be positive"),
        ({"dt": 0.01, "geom_groups": (6,)}, "Invalid geom group"),
    ],
)
def test_invalid_export_configuration_is_rejected(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        UsdTrajectoryExporter(
            _tiny_model(), output_directory=tmp_path, verbose=False, **kwargs
        )


def test_default_output_directory_and_idempotent_finalization(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = _tiny_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    exporter = UsdTrajectoryExporter(model, dt=0.01, verbose=False)
    exporter.render(data)

    path = exporter.save_usd()
    assert path == exporter.save_usd()
    assert Path(exporter.stop()) == path
    assert Path(exporter.stop()) == path
    with pytest.raises(RuntimeError, match="already been finalized"):
        exporter.render(data)


def test_save_without_frames_is_rejected(tmp_path):
    exporter = UsdTrajectoryExporter(
        _tiny_model(), dt=0.01, output_directory=tmp_path, verbose=False
    )
    with pytest.raises(RuntimeError, match="No frames were exported"):
        exporter.save_usd()


def test_missing_usd_dependency_has_actionable_error(tmp_path, monkeypatch):
    import builtins

    original_import = builtins.__import__

    def missing_pxr(name, *args, **kwargs):
        if name == "mujoco.usd.exporter":
            error = ModuleNotFoundError("No module named 'pxr'")
            error.name = "pxr"
            raise error
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_pxr)
    with pytest.raises(ImportError, match=r"loco-mujoco\[usd\]"):
        UsdTrajectoryExporter(_tiny_model(), dt=0.01, output_directory=tmp_path)


def test_non_pxr_import_error_is_not_masked(tmp_path, monkeypatch):
    import builtins

    original_import = builtins.__import__

    def missing_other(name, *args, **kwargs):
        if name == "mujoco.usd.exporter":
            error = ModuleNotFoundError("No module named 'other'")
            error.name = "other"
            raise error
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_other)
    with pytest.raises(ModuleNotFoundError, match="other"):
        UsdTrajectoryExporter(_tiny_model(), dt=0.01, output_directory=tmp_path)


def test_direct_mjx_index_validation_and_mocap_copy(tmp_path):
    model = _tiny_mocap_model()
    data = mujoco.MjData(model)
    data.mocap_pos[0] = (0.1, 0.2, 0.3)
    data.mocap_quat[0] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    state = SimpleNamespace(data=SimpleNamespace(
        qpos=data.qpos.copy(),
        qvel=data.qvel.copy(),
        mocap_pos=data.mocap_pos.copy(),
        mocap_quat=data.mocap_quat.copy(),
    ))
    exporter = UsdTrajectoryExporter(
        model, dt=0.01, output_directory=tmp_path, verbose=False
    )
    with pytest.raises(IndexError, match="unbatched"):
        exporter.render_mjx(state, env_idx=1)
    exporter.render_mjx(state)
    np.testing.assert_allclose(exporter._mjx_data.mocap_pos, data.mocap_pos)
    assert Path(exporter.stop()).is_file()

    batched = SimpleNamespace(data=SimpleNamespace(
        qpos=np.stack((data.qpos, data.qpos)),
        qvel=np.stack((data.qvel, data.qvel)),
        mocap_pos=np.stack((data.mocap_pos, data.mocap_pos)),
        mocap_quat=np.stack((data.mocap_quat, data.mocap_quat)),
    ))
    other = UsdTrajectoryExporter(
        model, dt=0.01, output_directory=tmp_path / "other", verbose=False
    )
    with pytest.raises(IndexError, match="outside a batch of size 2"):
        other.render_mjx(batched, env_idx=2)


def test_cpu_environment_api_exports_usd(tmp_path):
    env, _ = RLFactory.make("UnitreeH1", headless=True)
    try:
        env.reset()
        env.render_usd(output_directory=tmp_path, verbose=False)
        env.step(np.zeros(env.info.action_space.shape))
        env.render_usd()
        path = Path(env.stop_usd())

        assert path.is_file()
        assert env.usd_output_path == str(path)
        assert env._usd_exporter is None
    finally:
        env.stop()


def test_parameters_can_only_be_set_on_first_frame(tmp_path):
    env, _ = RLFactory.make("UnitreeH1", headless=True)
    try:
        env.reset()
        env.render_usd(output_directory=tmp_path, verbose=False)
        with pytest.raises(ValueError, match="only be set on the first frame"):
            env.render_usd(fps=25)
    finally:
        env.stop()


def test_play_trajectory_exports_usd(tmp_path):
    env, _ = ImitationFactory.make(
        "UnitreeH1", default_dataset_conf=dict(task="walk"), headless=True
    )
    env.play_trajectory(
        n_episodes=1,
        n_steps_per_episode=3,
        usd=True,
        usd_params=dict(output_directory=tmp_path, verbose=False),
        quiet=True,
    )
    path = Path(env.usd_output_path)
    assert path.is_file()
    assert Usd.Stage.Open(str(path)).GetEndTimeCode() == 2


def test_usd_only_environment_kwargs_do_not_leak_to_viewers(tmp_path):
    env, _ = RLFactory.make(
        "UnitreeH1",
        headless=True,
        usd_output_directory=tmp_path,
        usd_fps=30,
    )
    params = env._glfw_viewer_params()
    assert "usd_output_directory" not in params
    assert "usd_fps" not in params
    assert "usd_output_directory" not in env._viser_viewer_params()
    assert env._usd_exporter_params()["fps"] == 30


def test_mjx_exports_selected_environment(tmp_path):
    env, _ = RLFactory.make("MjxUnitreeH1", use_mjwarp=False, headless=True)
    try:
        state = jax.jit(jax.vmap(env.mjx_reset))(jax.random.split(jax.random.key(0), 3))
        env.mjx_render_usd(state, env_idx=2, output_directory=tmp_path, verbose=False)
        np.testing.assert_allclose(
            env._usd_exporter._mjx_data.qpos,
            np.asarray(state.data.qpos[2]),
            atol=1e-6,
        )
        assert Path(env.stop_usd()).is_file()
    finally:
        env.stop()


def test_mjx_rejects_bad_environment_index(tmp_path):
    env, _ = RLFactory.make("MjxUnitreeH1", use_mjwarp=False, headless=True)
    try:
        state = jax.jit(jax.vmap(env.mjx_reset))(jax.random.split(jax.random.key(0), 2))
        with pytest.raises(IndexError, match="outside a batch of size 2"):
            env.mjx_render_usd(state, env_idx=2,
                                   output_directory=tmp_path, verbose=False)
    finally:
        env.stop()
