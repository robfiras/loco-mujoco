"""Optional real-Blender smoke test for scripts/blender/render_usd.py."""

import shutil
import subprocess
from pathlib import Path

import mujoco
import pytest
from PIL import Image

pytest.importorskip("pxr", reason="install the blender extra for the Blender smoke test")

from loco_mujoco.core.visuals.usd_exporter import UsdTrajectoryExporter  # noqa: E402


def _blender_executable():
    found = shutil.which("blender")
    if found:
        return Path(found)
    candidate = Path("/Applications/Blender.app/Contents/MacOS/Blender")
    return candidate if candidate.is_file() else None


@pytest.mark.skipif(_blender_executable() is None, reason="Blender is not installed")
def test_blender_script_renders_base_ghost_and_current_layers(tmp_path):
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <geom name="floor" type="plane" size="5 5 .1"/>
            <light pos="0 -2 4"/>
            <body name="ball" pos="0 0 1">
              <joint type="free"/><geom name="ball" type="sphere" size=".2"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    exporter = UsdTrajectoryExporter(
        model, dt=0.02, output_directory=tmp_path / "usd", verbose=False
    )
    for x in (0.0, 0.2, 0.4):
        data.qpos[0] = x
        mujoco.mj_forward(model, data)
        exporter.render(data)
    usd_path = Path(exporter.stop())

    output = tmp_path / "rendered"
    script = Path(__file__).parents[1] / "scripts" / "blender" / "render_usd.py"
    subprocess.run([
        str(_blender_executable()), "--background", "--factory-startup", "--python",
        str(script), "--", "--usd", str(usd_path), "--output", str(output),
        "--width", "64", "--height", "64", "--fps", "50", "--frames", "3",
        "--engine", "CYCLES", "--samples", "1", "--lighting", "uniform",
        "--sun-energy", "2", "--fill-ratio", "0.5", "--robot-roughness", "0.9",
        "--robot-specular", "0.05", "--floor-style", "viser", "--floor-cell-size",
        "1", "--floor-extent", "100", "--floor-roughness", "0.5", "--sky-style",
        "daylight", "--sky-saturation", "1", "--view-transform", "Khronos PBR Neutral",
        "--exposure", "0", "--render-frame", "1", "--overlay-interval-frames", "1",
        "--overlay-opacity", "0.3", "--camera-position", "3", "-3", "1",
        "--camera-look-at", "0", "0", "1",
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    expected = {
        "_overlay_base.png", "_overlay_ghost_0000.png", "_overlay_ghost_0002.png",
        "_overlay_current.png", ".blender_render_complete",
    }
    assert expected <= {path.name for path in output.iterdir()}
    with Image.open(output / "_overlay_current.png") as image:
        assert image.size == (64, 64)
