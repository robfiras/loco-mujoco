"""Create a before/after figure for Stage 1 SMPL shape fitting.

The optimization itself is the production implementation in
``loco_mujoco.smpl.retargeting.fit_smpl_shape``.  This script only prepares
the two states used by a paper figure and delegates appearance to Blender.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import shutil
import subprocess
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import joblib
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as Rotation

from loco_mujoco.core.visuals import UsdTrajectoryExporter
from loco_mujoco.environments import LocoEnv
from loco_mujoco.smpl import SMPLH_BONE_ORDER_NAMES, SMPLH_Parser
from loco_mujoco.smpl.retargeting import (
    fit_smpl_shape,
    load_robot_conf_file,
    to_t_pose,
)


BLENDER_SCRIPT = Path(__file__).with_name("render_shape_fitting_blender.py")
DEFAULT_BLENDER = Path("/Applications/Blender.app/Contents/MacOS/Blender")
DEFAULT_OUTPUT = Path("LocoMuJoCo_blender/retargeting_shape_fit/UnitreeG1Full")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="UnitreeG1Full")
    parser.add_argument("--import-module", action="append", default=["lmj_real"])
    parser.add_argument("--smpl-model-path", type=Path, default=Path.home() / "SMPL")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--blender-executable", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=1400)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--engine", default="CYCLES")
    parser.add_argument("--camera-scale", type=float, default=2.35)
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _smpl_state(parser, pose, betas, scale, mimic_indices):
    import torch

    betas = torch.as_tensor(betas, dtype=torch.float32)
    if betas.ndim == 1:
        betas = betas[None]
    scale = float(np.asarray(scale).reshape(-1)[0])
    with torch.no_grad():
        vertices, _ = parser.get_joints_verts(pose, th_betas=betas, th_trans=None)
        transforms = parser.get_joint_transformations(pose, betas, None)
    root = transforms[:, :1, :3, 3]
    z_offset = torch.tensor([0.0, 0.0, 2.0], dtype=vertices.dtype)[None, None]
    vertices = (vertices - root + z_offset) * scale
    joints = (transforms[..., :3, 3] - root + z_offset) * scale
    return (
        vertices[0].detach().cpu().numpy(),
        joints[0, mimic_indices].detach().cpu().numpy(),
    )


def _prepare_assets(args):
    for module in args.import_module:
        importlib.import_module(module)

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_file = args.smpl_model_path.expanduser().resolve() / "SMPLH_NEUTRAL.pkl"
    if not model_file.is_file():
        raise FileNotFoundError(f"SMPL-H model not found: {model_file}")

    conf = load_robot_conf_file(args.env)
    shape_path = output / "shape_optimized.pkl"
    if args.refit or not shape_path.is_file():
        fit_smpl_shape(
            args.env,
            conf,
            str(args.smpl_model_path.expanduser().resolve()),
            str(shape_path),
            logging.getLogger("shape-fitting-figure"),
            visualize=False,
        )

    env_cls = LocoEnv.registered_envs[args.env]
    env = env_cls(
        **conf.env_params,
        th_params=dict(random_start=False, fixed_start_conf=(0, 0)),
    )
    to_t_pose(env, conf)
    qpos = env._data.qpos.copy()
    qpos[:3] = (0.0, 0.0, 2.0)
    env._data.qpos[:] = qpos
    mujoco.mj_forward(env._model, env._data)

    site_ids = np.asarray([
        mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_SITE, name)
        for name in env.sites_for_mimic
    ])
    target_joints = np.asarray(env._data.site_xpos[site_ids]).copy()
    mimic_indices = np.asarray([
        SMPLH_BONE_ORDER_NAMES.index(conf.site_joint_matches[name].smpl_joint)
        for name in env.sites_for_mimic
    ])

    parser = SMPLH_Parser(model_path=str(args.smpl_model_path), gender="neutral")
    pose = np.zeros((1, 156), dtype=np.float32)
    pose[0, :3] = Rotation.from_euler(
        "xyz", [np.pi / 2, 0.0, np.pi / 2]
    ).as_rotvec()
    import torch

    pose = torch.from_numpy(pose)
    before_vertices, before_joints = _smpl_state(
        parser, pose, np.zeros((1, 16), dtype=np.float32), np.ones(1), mimic_indices
    )
    shape, scale, *_ = joblib.load(shape_path)
    after_vertices, after_joints = _smpl_state(
        parser, pose, shape.detach().cpu(), scale.detach().cpu(), mimic_indices
    )

    data_path = output / "shape_fitting_states.npz"
    np.savez_compressed(
        data_path,
        faces=np.asarray(parser.faces, dtype=np.int32),
        target_joints=target_joints,
        before_vertices=before_vertices,
        before_joints=before_joints,
        after_vertices=after_vertices,
        after_joints=after_joints,
        before_error_mm=np.linalg.norm(before_joints - target_joints, axis=1).mean() * 1000,
        after_error_mm=np.linalg.norm(after_joints - target_joints, axis=1).mean() * 1000,
    )

    usd_exporter = UsdTrajectoryExporter(
        env._model,
        env.dt,
        output_directory=output / "robot_usd",
        geom_groups=(0,),
        verbose=False,
    )
    usd_exporter.render(env._data)
    robot_usd = Path(usd_exporter.stop())
    env.stop()
    return output, data_path, robot_usd


def _compose(output: Path, data_path: Path) -> Path:
    data = np.load(data_path)
    before = Image.open(output / "before.png").convert("RGB")
    after = Image.open(output / "after.png").convert("RGB")
    margin, gutter, header = 70, 34, 190
    canvas = Image.new(
        "RGB",
        (before.width + after.width + 2 * margin + gutter, before.height + header + margin),
        "white",
    )
    corner_radius = max(18, round(before.width * 0.025))
    panel_mask = Image.new("L", before.size, 0)
    ImageDraw.Draw(panel_mask).rounded_rectangle(
        (0, 0, before.width - 1, before.height - 1),
        radius=corner_radius,
        fill=255,
    )
    canvas.paste(before, (margin, header), panel_mask)
    canvas.paste(after, (margin + before.width + gutter, header), panel_mask)
    draw = ImageDraw.Draw(canvas)
    title_font = _font(48, bold=True)
    metric_font = _font(31)
    legend_font = _font(27)
    x_before = margin + before.width // 2
    x_after = margin + before.width + gutter + after.width // 2
    labels = (
        (x_before, "(a) Before shape fitting", float(data["before_error_mm"])),
        (x_after, "(b) After shape fitting", float(data["after_error_mm"])),
    )
    for x, label, error in labels:
        draw.text((x, 30), label, font=title_font, fill=(25, 29, 36), anchor="ma")
        draw.text(
            (x, 94),
            f"Mean correspondence error: {error:.1f} mm",
            font=metric_font,
            fill=(75, 80, 90),
            anchor="ma",
        )

    legend_y = 151
    legend_center = canvas.width // 2
    entries = (((34, 132, 220), "SMPL joints"), ((242, 126, 38), "Robot mimic sites"))
    widths = [draw.textlength(text, font=legend_font) + 52 for _, text in entries]
    x = legend_center - (sum(widths) + 55) / 2
    for color, text in entries:
        draw.ellipse((x, legend_y - 10, x + 20, legend_y + 10), fill=color)
        draw.text((x + 30, legend_y), text, font=legend_font, fill=(55, 60, 68), anchor="lm")
        x += draw.textlength(text, font=legend_font) + 107

    final = output / "shape_fitting_before_after.png"
    canvas.save(final, quality=100)
    return final


def main() -> None:
    args = _parser().parse_args()
    output, data_path, robot_usd = _prepare_assets(args)
    blender = args.blender_executable.expanduser().resolve()
    if not blender.is_file():
        found = shutil.which(str(args.blender_executable))
        if not found:
            raise FileNotFoundError(f"Blender executable not found: {args.blender_executable}")
        blender = Path(found)
    command = [
        str(blender),
        "--background",
        "--factory-startup",
        "--python",
        str(BLENDER_SCRIPT),
        "--",
        "--robot-usd",
        str(robot_usd),
        "--data",
        str(data_path),
        "--output",
        str(output),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--samples",
        str(args.samples),
        "--engine",
        args.engine,
        "--camera-scale",
        str(args.camera_scale),
    ]
    kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.STDOUT} if args.quiet else {}
    subprocess.run(command, check=True, **kwargs)
    final = _compose(output, data_path)
    print(final)


if __name__ == "__main__":
    main()
