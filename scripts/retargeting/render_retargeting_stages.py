"""Render a publication-style figure for retargeting Stages 2 and 3.

The figure is generated from a real robot-to-robot transfer:

    UnitreeG1 -> synthetic SMPL -> UnitreeG1Full

The synthetic SMPL motion is also visualized as the input to Stage 2, together
with the 15 target frames produced by the target robot's fitted shape.
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
from scipy.spatial.transform import Rotation

from loco_mujoco.core.trajectory import Trajectory, TrajectoryData
from loco_mujoco.core.visuals import UsdTrajectoryExporter
from loco_mujoco.environments import LocoEnv
from loco_mujoco.smpl import SMPLH_BONE_ORDER_NAMES, SMPLH_Parser
from loco_mujoco.smpl.retargeting import (
    OPTIMIZED_SHAPE_FILE_NAME,
    extend_motion,
    load_robot_conf_file,
    motion_transfer_robot_to_robot,
)


BLENDER_SCRIPT = Path(__file__).with_name("render_retargeting_stages_blender.py")
DEFAULT_BLENDER = Path("/Applications/Blender.app/Contents/MacOS/Blender")
DEFAULT_SOURCE = Path.home() / ".loco-mujoco-caches/LAFAN1/UnitreeG1/dance2_subject4.npz"
DEFAULT_OUTPUT = Path("LocoMuJoCo_blender/retargeting_pipeline_stages")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-env", default="UnitreeG1")
    parser.add_argument("--target-env", default="UnitreeG1Full")
    parser.add_argument("--import-module", action="append", default=["lmj_real"])
    parser.add_argument("--source-trajectory", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--segment-start", type=int, default=160)
    parser.add_argument("--segment-length", type=int, default=81)
    parser.add_argument("--smpl-model-path", type=Path, default=Path.home() / "SMPL")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--blender-executable", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--width", type=int, default=1050)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--exposure", type=float, default=0.0)
    parser.add_argument("--target-axes", action="store_true")
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _font(size, bold=False):
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _slice_trajectory(trajectory, start, length):
    data = TrajectoryData.dynamic_slice_in_dim(
        trajectory.data, 0, start, length, backend=np
    )
    return trajectory.replace(data=data)


def _prepare_retargeting(args, output):
    cache = output / "cache"
    source_dir = cache / args.source_env
    target_dir = cache / args.target_env
    source_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    source_segment_path = cache / "source_segment.npz"
    source_extended_path = cache / "source_extended.npz"
    synthetic_path = cache / "synthetic_smpl.npz"
    target_path = cache / "target_trajectory.npz"

    source = Trajectory.load(args.source_trajectory.expanduser(), backend=np)
    source_segment = _slice_trajectory(
        source, args.segment_start, args.segment_length
    )
    source_segment.save(source_segment_path)
    source_conf = load_robot_conf_file(args.source_env)
    target_conf = load_robot_conf_file(args.target_env)
    logger = logging.getLogger("retargeting-figure")

    if args.refit or not source_extended_path.is_file():
        source_extended = extend_motion(
            args.source_env, source_conf.env_params, source_segment, logger
        )
        source_extended.save(source_extended_path)
    else:
        source_extended = Trajectory.load(source_extended_path, backend=np)

    if args.refit or not (synthetic_path.is_file() and target_path.is_file()):
        target = motion_transfer_robot_to_robot(
            args.source_env,
            source_conf,
            source_segment,
            str(source_dir),
            args.target_env,
            target_conf,
            str(target_dir),
            str(args.smpl_model_path.expanduser().resolve()),
            logger,
            path_to_fitted_motion_source=str(synthetic_path),
            visualize=False,
        )
        target.save(target_path)
    else:
        target = Trajectory.load(target_path, backend=np)
    return source_extended, target, synthetic_path, source_dir, target_dir, source_conf, target_conf


def _smpl_sequence(parser, motion, shape_data, conf):
    import torch

    shape, scale, s2r_pos, s2r_rot, offset_z, height_scale = shape_data
    pose = torch.from_numpy(np.asarray(motion["pose_aa"], dtype=np.float32))
    trans = torch.from_numpy(np.asarray(motion["trans"], dtype=np.float32).copy())
    trans += torch.tensor(
        [0.0, 0.0, float(offset_z) + float(conf.optimization_params.z_offset_feet)]
    )
    trans[:, :2] *= float(height_scale)
    trans[:, 2] = (trans[:, 2] - trans[0, 2]) * float(height_scale) + trans[0, 2]
    shape = shape.detach().cpu().repeat(len(pose), 1)
    scale = float(scale.detach().cpu().numpy().reshape(-1)[0])
    with torch.no_grad():
        vertices, _ = parser.get_joints_verts(pose, shape, trans)
        transforms = parser.get_joint_transformations(pose, shape, trans)
    root = transforms[:, :1, :3, 3]
    vertices = (vertices - root) * scale + root
    joints = (transforms[..., :3, 3] - root) * scale + root
    rotations = transforms[..., :3, :3]
    return (
        vertices.detach().cpu().numpy(),
        joints.detach().cpu().numpy(),
        rotations.detach().cpu().numpy(),
        np.asarray(s2r_pos),
        np.asarray(s2r_rot),
    )


def _original_smpl_sequence(parser, motion):
    """Evaluate the motion with the untouched neutral SMPL-H body shape."""
    import torch

    pose = torch.from_numpy(np.asarray(motion["pose_aa"], dtype=np.float32))
    trans = torch.from_numpy(np.asarray(motion["trans"], dtype=np.float32))
    shape = torch.zeros((len(pose), 16), dtype=torch.float32)
    with torch.no_grad():
        vertices, _ = parser.get_joints_verts(pose, shape, trans)
        transforms = parser.get_joint_transformations(pose, shape, trans)
    return vertices.detach().cpu().numpy(), transforms[..., :3, 3].detach().cpu().numpy()


def _normalise_smpl(vertices, joints, current):
    offset = np.array([
        -joints[current, 0, 0],
        -joints[current, 0, 1],
        -vertices[current, :, 2].min(),
    ])
    return vertices + offset, joints + offset, offset


def _normalise_robot_qpos(qpos, current):
    qpos = np.asarray(qpos).copy()
    qpos[:, :2] -= qpos[current, :2]
    return qpos


def _export_robot_frames(env_name, conf, qpos, indices, output):
    env_cls = LocoEnv.registered_envs[env_name]
    env = env_cls(
        **conf.env_params,
        th_params=dict(random_start=False, fixed_start_conf=(0, 0)),
    )
    paths = []
    for label, index in zip(("past", "current", "future"), indices):
        env._data.qpos[:] = qpos[index]
        env._data.qvel[:] = 0
        mujoco.mj_forward(env._model, env._data)
        exporter = UsdTrajectoryExporter(
            env._model,
            env.dt,
            output_directory=output / label,
            geom_groups=(0,),
            verbose=False,
        )
        exporter.render(env._data)
        paths.append(Path(exporter.stop()))
    env.stop()
    return paths


def _prepare_assets(args):
    for module in args.import_module:
        importlib.import_module(module)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    source, target, synthetic_path, source_dir, target_dir, source_conf, target_conf = \
        _prepare_retargeting(args, output)

    motion = np.load(synthetic_path)
    parser = SMPLH_Parser(model_path=str(args.smpl_model_path.expanduser()), gender="neutral")
    source_shape = joblib.load(source_dir / OPTIMIZED_SHAPE_FILE_NAME)
    target_shape = joblib.load(target_dir / OPTIMIZED_SHAPE_FILE_NAME)
    source_vertices, source_joints, _, _, _ = _smpl_sequence(
        parser, motion, source_shape, source_conf
    )
    target_vertices, target_joints, target_rotations, target_s2r_pos, target_s2r_rot = \
        _smpl_sequence(parser, motion, target_shape, target_conf)
    original_vertices, original_joints = _original_smpl_sequence(parser, motion)

    n_motion = len(source_vertices)
    current = n_motion // 2
    interval = max(1, round(float(motion["fps"]) * 0.5))
    smpl_indices = np.asarray([current - interval, current, current + interval])
    target_indices = np.clip(smpl_indices - 1, 0, len(target.data.qpos) - 1)
    source_indices = np.clip(smpl_indices, 0, len(source.data.qpos) - 1)

    source_vertices, source_joints, _ = _normalise_smpl(
        source_vertices, source_joints, current
    )
    target_vertices, target_joints, target_offset = _normalise_smpl(
        target_vertices, target_joints, current
    )
    original_vertices, original_joints, _ = _normalise_smpl(
        original_vertices, original_joints, current
    )
    source_qpos = _normalise_robot_qpos(source.data.qpos, source_indices[1])
    target_qpos = _normalise_robot_qpos(target.data.qpos, target_indices[1])

    source_usd = _export_robot_frames(
        args.source_env, source_conf, source_qpos, source_indices,
        output / "source_robot_usd",
    )
    target_usd = _export_robot_frames(
        args.target_env, target_conf, target_qpos, target_indices,
        output / "target_robot_usd",
    )

    target_env = LocoEnv.registered_envs[args.target_env](
        **target_conf.env_params,
        th_params=dict(random_start=False, fixed_start_conf=(0, 0)),
    )
    site_names = list(target_env.sites_for_mimic)
    mimic_indices = np.asarray([
        SMPLH_BONE_ORDER_NAMES.index(target_conf.site_joint_matches[name].smpl_joint)
        for name in site_names
    ])
    target_env.stop()
    target_edge_names = (
        ("pelvis_mimic", "upper_body_mimic"),
        ("upper_body_mimic", "head_mimic"),
        ("upper_body_mimic", "left_shoulder_mimic"),
        ("left_shoulder_mimic", "left_elbow_mimic"),
        ("left_elbow_mimic", "left_hand_mimic"),
        ("upper_body_mimic", "right_shoulder_mimic"),
        ("right_shoulder_mimic", "right_elbow_mimic"),
        ("right_elbow_mimic", "right_hand_mimic"),
        ("pelvis_mimic", "left_hip_mimic"),
        ("left_hip_mimic", "left_knee_mimic"),
        ("left_knee_mimic", "left_foot_mimic"),
        ("pelvis_mimic", "right_hip_mimic"),
        ("right_hip_mimic", "right_knee_mimic"),
        ("right_knee_mimic", "right_foot_mimic"),
    )
    target_edges = np.asarray([
        (site_names.index(parent), site_names.index(child))
        for parent, child in target_edge_names
    ], dtype=np.int32)
    joint_pos = target_joints[current, mimic_indices]
    joint_rot = target_rotations[current, mimic_indices]
    target_frame_rot = np.einsum("bij,bjk->bik", joint_rot, target_s2r_rot)
    target_frame_pos = joint_pos - np.einsum(
        "bij,bj->bi", target_frame_rot, target_s2r_pos
    )
    data_path = output / "pipeline_states.npz"
    np.savez_compressed(
        data_path,
        faces=np.asarray(parser.faces, dtype=np.int32),
        original_smpl_vertices=original_vertices[smpl_indices],
        source_skeleton_joints=source_joints[smpl_indices, :22],
        target_skeleton_joints=target_joints[smpl_indices, :22],
        skeleton_parents=np.asarray(parser.parents[:22].detach().cpu(), dtype=np.int32),
        target_smpl_joints=target_joints[current, mimic_indices],
        target_frame_pos=target_frame_pos,
        target_frame_rot=target_frame_rot,
        target_frame_edges=target_edges,
        axis_indices=np.asarray([2, 5, 8, 11, 14], dtype=np.int32),
        source_usd=np.asarray([str(path) for path in source_usd]),
        target_usd=np.asarray([str(path) for path in target_usd]),
    )
    return output, data_path


def _rounded_paste(canvas, image, xy, radius):
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, image.width - 1, image.height - 1), radius=radius, fill=255
    )
    canvas.paste(image, xy, mask)


def _composite_layers(output, name, opacity=0.28, whiten=0.45):
    with Image.open(output / f"_{name}_base.png") as image:
        result = image.convert("RGBA")
    for suffix in ("past", "future"):
        with Image.open(output / f"_{name}_{suffix}.png") as image:
            ghost = image.convert("RGBA")
        alpha = ghost.getchannel("A")
        ghost = Image.blend(ghost, Image.new("RGBA", ghost.size, "white"), whiten)
        ghost.putalpha(alpha.point(lambda value: round(value * opacity)))
        result.alpha_composite(ghost)
    with Image.open(output / f"_{name}_current.png") as image:
        result.alpha_composite(image.convert("RGBA"))
    result.convert("RGB").save(output / f"{name}.png")


def _arrow(draw, start, end, color=(82, 91, 104), width=5):
    draw.line((start, end), fill=color, width=width)
    angle = np.arctan2(end[1] - start[1], end[0] - start[0])
    length = 18
    for delta in (2.55, -2.55):
        point = (
            end[0] + length * np.cos(angle + delta),
            end[1] + length * np.sin(angle + delta),
        )
        draw.line((end, point), fill=color, width=width)


def _compose(output, width, height, source_env, target_env):
    panel_names = (
        "stage2_smpl", "stage2_targets", "stage2_robot",
        "stage3_source", "stage3_smpl", "stage3_target",
    )
    panels = [Image.open(output / f"{name}.png").convert("RGB") for name in panel_names]
    margin, gutter, label_h, row_gap = 58, 105, 145, 155
    canvas_w = 2 * margin + 3 * width + 2 * gutter
    canvas_h = 2 * margin + 2 * (height + label_h) + row_gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(42, bold=True)
    label_font = _font(31, bold=True)
    arrow_font = _font(23)
    source_label = source_env.removeprefix("Unitree").replace("H1v2", "H1 v2")
    target_label = target_env.removeprefix("Unitree").replace("G1Full", "G1 Full")
    labels = (
        "Original SMPL motion", "Target-shaped SMPL skeleton", f"{target_label} trajectory",
        f"Source {source_label} trajectory", f"{source_label}-shaped SMPL skeleton",
        f"Target {target_label} trajectory",
    )
    row_titles = (
        "Stage 2 — SMPL → robot",
        "Stage 3 — Robot → SMPL → robot",
    )
    row_y = [margin + label_h, margin + label_h + height + row_gap + label_h]
    radius = max(22, round(width * 0.025))
    for row in range(2):
        title_y = row_y[row] - label_h + 4
        draw.text((margin, title_y), row_titles[row], font=title_font,
                  fill=(29, 35, 45), anchor="la")
        for col in range(3):
            index = row * 3 + col
            x = margin + col * (width + gutter)
            y = row_y[row]
            draw.text((x + width / 2, y - 43), labels[index], font=label_font,
                      fill=(56, 63, 74), anchor="ms")
            _rounded_paste(canvas, panels[index], (x, y), radius)
        arrow_y = row_y[row] - 45
        for col in range(2):
            x0 = margin + (col + 1) * width + col * gutter + 18
            x1 = x0 + gutter - 36
            _arrow(draw, (x0, arrow_y), (x1, arrow_y))
    stage2_arrow_labels = ("apply fitted shape", "soft weld constraints")
    stage3_arrow_labels = ("inverse site fitting", "reuse Stage 2")
    for row, arrow_labels in enumerate((stage2_arrow_labels, stage3_arrow_labels)):
        y = row_y[row] - 76
        for col, text in enumerate(arrow_labels):
            x = margin + (col + 1) * width + col * gutter + gutter / 2
            draw.text((x, y), text, font=arrow_font, fill=(82, 91, 104), anchor="ms")
    final = output / "retargeting_stages_2_3.png"
    canvas.save(final)
    for panel in panels:
        panel.close()
    return final


def main():
    args = _parse_args()
    output, data_path = _prepare_assets(args)
    blender = args.blender_executable.expanduser().resolve()
    if not blender.is_file():
        found = shutil.which(str(args.blender_executable))
        if not found:
            raise FileNotFoundError(f"Blender executable not found: {args.blender_executable}")
        blender = Path(found)
    command = [
        str(blender), "--background", "--factory-startup", "--python",
        str(BLENDER_SCRIPT), "--", "--data", str(data_path), "--output", str(output),
        "--width", str(args.width), "--height", str(args.height),
        "--samples", str(args.samples), "--exposure", str(args.exposure),
    ]
    if args.target_axes:
        command.append("--target-axes")
    kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.STDOUT} if args.quiet else {}
    subprocess.run(command, check=True, **kwargs)
    for name in (
        "stage2_smpl", "stage2_targets", "stage2_robot",
        "stage3_source", "stage3_smpl", "stage3_target"
    ):
        _composite_layers(output, name)
    final = _compose(output, args.width, args.height, args.source_env, args.target_env)
    print(final)


if __name__ == "__main__":
    main()
