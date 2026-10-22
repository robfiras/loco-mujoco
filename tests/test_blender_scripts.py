"""Tests for the standalone Blender orchestration helpers."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from PIL import Image
from pxr import Usd


SCRIPT = Path(__file__).parents[1] / "scripts" / "blender" / "render_locomujoco.py"
SPEC = spec_from_file_location("render_locomujoco", SCRIPT)
render_locomujoco = module_from_spec(SPEC)
SPEC.loader.exec_module(render_locomujoco)


def _rgba(path, color, size=(4, 3)):
    Image.new("RGBA", size, color).save(path)


def _render_args(**overrides):
    values = dict(
        render_frame="middle", overlay_interval=0.5, overlay_opacity=0.3,
        overlay_whiten=0.5, blender_executable=None, width=64, height=32,
        engine="CYCLES", samples=1, lighting="uniform", sun_energy=5.0,
        fill_ratio=0.55, robot_roughness=0.9, robot_specular=0.05,
        floor_style="viser", floor_cell_size=1.0, floor_extent=10000.0,
        floor_roughness=0.35, sky_style="daylight", sky_saturation=1.35,
        environment_strength=0.0, view_transform="Khronos PBR Neutral",
        exposure=0.7, quiet=False,
        format="png",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_find_blender_explicit_file_directory_path_and_missing(tmp_path, monkeypatch):
    executable = tmp_path / "blender"
    executable.touch()
    assert render_locomujoco._find_blender(executable) == executable.resolve()

    app = tmp_path / "Blender.app"
    app_executable = app / "Contents" / "MacOS" / "Blender"
    app_executable.parent.mkdir(parents=True)
    app_executable.touch()
    assert render_locomujoco._find_blender(app) == app_executable.resolve()

    monkeypatch.setattr(render_locomujoco.shutil, "which", lambda _: str(executable))
    assert render_locomujoco._find_blender(None) == executable
    assert render_locomujoco._find_blender("blender-from-path") == executable
    monkeypatch.setattr(render_locomujoco.shutil, "which", lambda _: None)
    monkeypatch.setattr(render_locomujoco, "DEFAULT_BLENDER_PATHS", (app_executable,))
    assert render_locomujoco._find_blender(None) == app_executable
    with pytest.raises(FileNotFoundError, match="not found"):
        render_locomujoco._find_blender(tmp_path / "missing")
    monkeypatch.setattr(render_locomujoco, "DEFAULT_BLENDER_PATHS", ())
    with pytest.raises(FileNotFoundError, match="was not found"):
        render_locomujoco._find_blender(None)


def test_camera_scales_from_visual_height():
    env = SimpleNamespace(
        reset=lambda: None,
        _model=SimpleNamespace(
            geom_group=np.array([0, 0, 2]),
            geom_type=np.array([
                mujoco.mjtGeom.mjGEOM_SPHERE,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                mujoco.mjtGeom.mjGEOM_PLANE,
            ]),
        ),
        _data=SimpleNamespace(geom_xpos=np.array([[0, 0, 0.1], [0, 0, 2.1], [0, 0, 0]])),
    )
    position, look_at = render_locomujoco._camera_for_robot(env, distance_scale=2)
    assert position[0] == pytest.approx(3.8 * 2 / 1.459 * 2)
    assert position[1] == -position[0]
    assert position[2] == pytest.approx(1.26)
    assert look_at[2] == pytest.approx(1.16)


def test_usd_info_reads_frame_count_and_rate(tmp_path):
    path = tmp_path / "stage.usda"
    stage = Usd.Stage.CreateNew(str(path))
    stage.SetStartTimeCode(4)
    stage.SetEndTimeCode(9)
    stage.SetTimeCodesPerSecond(30)
    stage.GetRootLayer().Save()
    assert render_locomujoco._usd_info(path) == (6, 30.0)


def test_overlay_composite_whitens_ghosts_and_keeps_current_opaque(tmp_path):
    _rgba(tmp_path / "_overlay_base.png", (0, 0, 0, 255))
    _rgba(tmp_path / "_overlay_ghost_0000.png", (255, 0, 0, 255))
    current = Image.new("RGBA", (4, 3), (0, 0, 0, 0))
    current.putpixel((0, 0), (0, 255, 0, 255))
    current.save(tmp_path / "_overlay_current.png")

    output = render_locomujoco._composite_overlay_layers(tmp_path, 7, 0.2, 0.5)
    with Image.open(output) as image:
        assert image.getpixel((0, 0)) == (0, 255, 0, 255)
        red, green, blue, alpha = image.getpixel((1, 1))
        assert (red, green, blue, alpha) == pytest.approx((51, 26, 26, 255), abs=1)
    assert not list(tmp_path.glob("_overlay_*.png"))


def test_overlay_composite_rejects_missing_layers(tmp_path):
    with pytest.raises(RuntimeError, match="expected trajectory-overlay layers"):
        render_locomujoco._composite_overlay_layers(tmp_path, 0, 0.3, 0)


def test_mosaic_preserves_native_resolution(tmp_path):
    paths = []
    for index in range(5):
        path = tmp_path / f"{index}.png"
        _rgba(path, (index, 0, 0, 255), size=(10, 6))
        paths.append(path)
    output = render_locomujoco._make_mosaic(paths, tmp_path / "mosaic.png", columns=4)
    with Image.open(output) as image:
        assert image.size == (40, 12)
        assert image.getpixel((0, 6)) == (4, 0, 0, 255)


def test_mosaic_rejects_mixed_dimensions(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _rgba(first, "white", size=(10, 6))
    _rgba(second, "white", size=(11, 6))
    with pytest.raises(ValueError, match="identical dimensions"):
        render_locomujoco._make_mosaic([first, second], tmp_path / "out.png")


def test_render_usd_builds_blender_command_and_composites(tmp_path, monkeypatch):
    usd_path = tmp_path / "stage.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(2)
    stage.SetTimeCodesPerSecond(30)
    stage.GetRootLayer().Save()
    executable = tmp_path / "blender"
    executable.touch()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        frames = tmp_path / "output" / "rendered_frames"
        _rgba(frames / "_overlay_base.png", (0, 0, 0, 255))
        _rgba(frames / "_overlay_ghost_0000.png", (255, 0, 0, 255))
        _rgba(frames / "_overlay_current.png", (0, 255, 0, 255))
        (frames / ".blender_render_complete").touch()

    monkeypatch.setattr(render_locomujoco.subprocess, "run", fake_run)
    args = _render_args(blender_executable=executable)
    output = render_locomujoco._render_usd(
        usd_path, tmp_path / "output", args, (4, -4, 1), (0, 0, 1), ["hidden"]
    )
    command, kwargs = calls[0]
    assert output == tmp_path / "output" / "rendered_frames" / "frame_0001.png"
    assert command[0] == str(executable.resolve())
    assert command[command.index("--render-frame") + 1] == "1"
    assert command[command.index("--overlay-interval-frames") + 1] == "15"
    assert command[command.index("--environment-strength") + 1] == "0.0"
    assert command[-2:] == ["--hide-geom", "hidden"]
    assert kwargs == {"check": True}


def test_render_usd_rejects_frame_outside_stage(tmp_path):
    usd_path = tmp_path / "stage.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(1)
    stage.SetTimeCodesPerSecond(30)
    stage.GetRootLayer().Save()
    args = SimpleNamespace(render_frame=2, overlay_interval=None)
    with pytest.raises(ValueError, match="outside"):
        render_locomujoco._render_usd(
            usd_path, tmp_path / "output", args, (0, 0, 0), (0, 0, 0), []
        )


def test_render_usd_detects_missing_sentinel(tmp_path, monkeypatch):
    usd_path = tmp_path / "stage.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    stage.SetEndTimeCode(0)
    stage.SetTimeCodesPerSecond(30)
    stage.GetRootLayer().Save()
    executable = tmp_path / "blender"
    executable.touch()
    monkeypatch.setattr(render_locomujoco.subprocess, "run", lambda *args, **kwargs: None)
    args = _render_args(
        render_frame=None, overlay_interval=None, blender_executable=executable, quiet=True
    )
    with pytest.raises(RuntimeError, match="did not finish"):
        render_locomujoco._render_usd(
            usd_path, tmp_path / "output", args, (0, 0, 0), (0, 0, 0), []
        )


def test_render_usd_encodes_mp4(tmp_path, monkeypatch):
    usd_path = tmp_path / "stage.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    stage.SetEndTimeCode(0)
    stage.SetTimeCodesPerSecond(30)
    stage.GetRootLayer().Save()
    executable = tmp_path / "blender"
    executable.touch()
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0] == str(executable.resolve()):
            frames = tmp_path / "output" / "rendered_frames"
            _rgba(frames / "frame_0000.png", "white")
            (frames / ".blender_render_complete").touch()
        else:
            Path(command[-1]).touch()

    monkeypatch.setattr(render_locomujoco.subprocess, "run", fake_run)
    monkeypatch.setattr(
        render_locomujoco.shutil, "which",
        lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None,
    )
    args = _render_args(
        render_frame=None, overlay_interval=None, blender_executable=executable, format="mp4"
    )
    output = render_locomujoco._render_usd(
        usd_path, tmp_path / "output", args, (0, 0, 0), (0, 0, 0), []
    )
    assert output == tmp_path / "output" / "render.mp4"
    assert calls[1][0] == "/usr/bin/ffmpeg"


def test_render_usd_requires_ffmpeg_for_mp4(tmp_path, monkeypatch):
    usd_path = tmp_path / "stage.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    stage.SetEndTimeCode(0)
    stage.SetTimeCodesPerSecond(30)
    stage.GetRootLayer().Save()
    executable = tmp_path / "blender"
    executable.touch()

    def fake_run(command, **kwargs):
        frames = tmp_path / "output" / "rendered_frames"
        (frames / ".blender_render_complete").touch()

    monkeypatch.setattr(render_locomujoco.subprocess, "run", fake_run)
    monkeypatch.setattr(render_locomujoco.shutil, "which", lambda _: None)
    args = _render_args(
        render_frame=None, overlay_interval=None, blender_executable=executable, format="mp4"
    )
    with pytest.raises(FileNotFoundError, match="FFmpeg"):
        render_locomujoco._render_usd(
            usd_path, tmp_path / "output", args, (0, 0, 0), (0, 0, 0), []
        )


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv", ["render_locomujoco.py"])
    args = render_locomujoco._parse_args()
    assert args.env == "UnitreeH1"
    assert args.dataset_source == "default"
    assert args.engine == "BLENDER_EEVEE_NEXT"


class _FakeEnv:
    def __init__(self, usd_path):
        self.usd_output_path = str(usd_path)
        self._model = SimpleNamespace(
            geom_group=np.array([0, 0]),
            geom_type=np.array([mujoco.mjtGeom.mjGEOM_SPHERE] * 2),
        )
        self._data = SimpleNamespace(geom_xpos=np.array([[0, 0, 0], [0, 0, 1]]))
        self.play_calls = []
        self.stopped = False

    def reset(self):
        return None

    def play_trajectory(self, **kwargs):
        self.play_calls.append(kwargs)
        self.usd_output_path = str(Path(kwargs["usd_params"]["output_directory"]) / "stage.usda")
        Path(self.usd_output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.usd_output_path).touch()

    def stop(self):
        self.stopped = True


def _main_args(tmp_path, **overrides):
    values = vars(_render_args()).copy()
    values.update(
        render=False, env="UnitreeH1", all_humanoids=False, dataset_source="default",
        task="walk", steps=2, output=tmp_path, fps=30, light_intensity=1000,
        camera_position=None, camera_look_at=None, camera_distance_scale=1.0,
        camera_body_name=None, filetype="usd", mosaic=False, hide_geom=[],
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_main_exports_single_default_dataset(tmp_path, monkeypatch):
    env = _FakeEnv(tmp_path / "unused.usda")
    calls = []
    monkeypatch.setattr(render_locomujoco, "_parse_args", lambda: _main_args(tmp_path))
    monkeypatch.setattr(
        render_locomujoco.ImitationFactory, "make",
        lambda *args, **kwargs: (calls.append((args, kwargs)) or env, None),
    )
    render_locomujoco.main()
    assert calls[0][1]["default_dataset_conf"] == {"task": "walk"}
    assert env.play_calls[0]["usd"] is True
    assert env.stopped


def test_main_renders_lafan_batch_and_mosaic(tmp_path, monkeypatch):
    envs = []
    render_calls = []

    def make(*args, **kwargs):
        env = _FakeEnv(tmp_path / "unused.usda")
        envs.append((args, kwargs, env))
        return env, None

    def render(usd_path, output_directory, args, camera_position, camera_look_at, hidden):
        output = output_directory / "still.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        _rgba(output, "white")
        render_calls.append((output_directory, hidden))
        return output

    mosaic_calls = []
    monkeypatch.setattr(
        render_locomujoco, "_parse_args",
        lambda: _main_args(
            tmp_path, render=True, all_humanoids=True, dataset_source="lafan1",
            task="dance2_subject4", format="png", mosaic=True,
        ),
    )
    monkeypatch.setattr(render_locomujoco.ImitationFactory, "make", make)
    monkeypatch.setattr(render_locomujoco, "_render_usd", render)
    monkeypatch.setattr(
        render_locomujoco, "_make_mosaic",
        lambda images, output: (mosaic_calls.append((images, output)) or output),
    )
    render_locomujoco.main()
    assert len(envs) == len(render_locomujoco.HUMANOID_ENVS)
    assert envs[0][1]["lafan1_dataset_conf"] == {"dataset_name": "dance2_subject4"}
    skeleton_hidden = dict(render_calls)[tmp_path / "SkeletonTorque"]
    assert skeleton_hidden[-2:] == ["foot_box_r", "foot_box_l"]
    assert len(mosaic_calls[0][0]) == len(render_locomujoco.HUMANOID_ENVS)


def test_main_rejects_invalid_mosaic_combination(tmp_path, monkeypatch):
    env = _FakeEnv(tmp_path / "unused.usda")
    monkeypatch.setattr(
        render_locomujoco, "_parse_args",
        lambda: _main_args(tmp_path, mosaic=True, render=False),
    )
    monkeypatch.setattr(render_locomujoco.ImitationFactory, "make", lambda *a, **k: (env, None))
    with pytest.raises(ValueError, match="--mosaic requires"):
        render_locomujoco.main()


def test_main_mosaic_requires_single_frame_outputs(tmp_path, monkeypatch):
    def make(*args, **kwargs):
        return _FakeEnv(tmp_path / "unused.usda"), None

    monkeypatch.setattr(
        render_locomujoco, "_parse_args",
        lambda: _main_args(
            tmp_path, render=True, all_humanoids=True, format="png", mosaic=True,
        ),
    )
    monkeypatch.setattr(render_locomujoco.ImitationFactory, "make", make)
    monkeypatch.setattr(
        render_locomujoco, "_render_usd",
        lambda *args, **kwargs: tmp_path / "not-a-file",
    )
    with pytest.raises(ValueError, match="single --render-frame"):
        render_locomujoco.main()
