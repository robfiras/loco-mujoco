"""Export LocoMuJoCo trajectories to USD and optionally render them in Blender.

This is intentionally outside the ``loco_mujoco`` package: core owns portable
USD export, while this script owns Blender, FFmpeg, appearance, and mosaics.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from loco_mujoco import ImitationFactory


HUMANOID_ENVS = (
    "Atlas", "Talos", "UnitreeH1", "UnitreeH1v2", "UnitreeG1", "Apollo",
    "BoosterT1", "ToddlerBot", "FourierGR1T2", "SkeletonTorque",
    "SkeletonMuscle", "MyoSkeleton",
)
BLENDER_SCRIPT = Path(__file__).with_name("render_usd.py")
DEFAULT_BLENDER_PATHS = (
    Path("/Applications/Blender.app/Contents/MacOS/Blender"),
    Path.home() / "Applications/Blender.app/Contents/MacOS/Blender",
)


def _camera_for_robot(env, distance_scale=1.0):
    env.reset()
    visual = np.asarray(env._model.geom_group) == 0
    visual &= np.asarray(env._model.geom_type) != int(mujoco.mjtGeom.mjGEOM_PLANE)
    z = np.asarray(env._data.geom_xpos)[visual, 2]
    height = float(np.max(z) - np.min(z))
    scale = height / 1.459
    floor_z = float(np.min(z))
    return (
        (3.8 * scale * distance_scale, -3.8 * scale * distance_scale,
         floor_z + 0.58 * height),
        (0.0, 0.0, floor_z + 0.53 * height),
    )


def _find_blender(explicit):
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.is_dir():
            candidate = candidate / "Contents" / "MacOS" / "Blender"
        if candidate.is_file():
            return candidate.resolve()
        found = shutil.which(str(explicit))
        if found:
            return Path(found)
        raise FileNotFoundError(f"Blender executable not found: {explicit}")
    found = shutil.which("blender")
    if found:
        return Path(found)
    for candidate in DEFAULT_BLENDER_PATHS:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Blender was not found; pass --blender-executable.")


def _usd_info(path):
    from pxr import Usd

    stage = Usd.Stage.Open(str(path))
    frames = int(stage.GetEndTimeCode() - stage.GetStartTimeCode()) + 1
    return frames, float(stage.GetTimeCodesPerSecond())


def _composite_overlay_layers(frames_path, render_frame, opacity, whiten):
    from PIL import Image

    base_path = frames_path / "_overlay_base.png"
    current_path = frames_path / "_overlay_current.png"
    ghost_paths = sorted(frames_path.glob("_overlay_ghost_*.png"))
    if not base_path.is_file() or not current_path.is_file() or not ghost_paths:
        raise RuntimeError("Blender did not produce the expected trajectory-overlay layers.")

    with Image.open(base_path) as image:
        result = image.convert("RGBA")
    for path in ghost_paths:
        with Image.open(path) as image:
            ghost = image.convert("RGBA")
        alpha = ghost.getchannel("A")
        if whiten:
            ghost = Image.blend(ghost, Image.new("RGBA", ghost.size, "white"), whiten)
        ghost.putalpha(alpha.point(lambda value: round(value * opacity)))
        result.alpha_composite(ghost)
    with Image.open(current_path) as image:
        result.alpha_composite(image.convert("RGBA"))

    output = frames_path / f"frame_{render_frame:04d}.png"
    result.save(output)
    for path in (base_path, current_path, *ghost_paths):
        path.unlink()
    return output


def _render_usd(usd_path, output_directory, args, camera_position, camera_look_at,
                hidden_geom_names):
    frames, fps = _usd_info(usd_path)
    if args.render_frame == "middle":
        render_frame = frames // 2
    elif args.render_frame is None:
        render_frame = -1
    else:
        render_frame = int(args.render_frame)
    if render_frame >= frames:
        raise ValueError(f"render frame {render_frame} is outside a {frames}-frame stage")
    interval_frames = (
        0 if args.overlay_interval is None
        else max(1, round(args.overlay_interval * fps))
    )

    frames_path = output_directory / "rendered_frames"
    frames_path.mkdir(parents=True, exist_ok=True)
    sentinel = frames_path / ".blender_render_complete"
    sentinel.unlink(missing_ok=True)
    command = [
        str(_find_blender(args.blender_executable)), "--background", "--factory-startup",
        "--python", str(BLENDER_SCRIPT), "--", "--usd", str(usd_path),
        "--output", str(frames_path), "--width", str(args.width), "--height",
        str(args.height), "--fps", str(fps), "--frames", str(frames), "--engine",
        args.engine, "--samples", str(args.samples), "--lighting", args.lighting,
        "--sun-energy", str(args.sun_energy), "--fill-ratio", str(args.fill_ratio),
        "--robot-roughness", str(args.robot_roughness), "--robot-specular",
        str(args.robot_specular), "--floor-style", args.floor_style,
        "--floor-cell-size", str(args.floor_cell_size), "--floor-extent",
        str(args.floor_extent), "--floor-roughness", str(args.floor_roughness),
        "--sky-style", args.sky_style, "--sky-saturation", str(args.sky_saturation),
        "--environment-strength", str(args.environment_strength),
        "--view-transform", args.view_transform, "--exposure", str(args.exposure),
        "--render-frame", str(render_frame), "--overlay-interval-frames",
        str(interval_frames), "--overlay-opacity", str(args.overlay_opacity),
        "--camera-position", *map(str, camera_position), "--camera-look-at",
        *map(str, camera_look_at),
    ]
    for name in hidden_geom_names:
        command.extend(("--hide-geom", name))
    kwargs = ({"stdout": subprocess.DEVNULL, "stderr": subprocess.STDOUT}
              if args.quiet else {})
    subprocess.run(command, check=True, **kwargs)
    if not sentinel.is_file():
        raise RuntimeError("Blender did not finish rendering; inspect its output above.")

    if render_frame >= 0 and interval_frames > 0:
        still = _composite_overlay_layers(
            frames_path, render_frame, args.overlay_opacity, args.overlay_whiten
        )
    else:
        still = frames_path / f"frame_{max(0, render_frame):04d}.png"

    if args.format == "png":
        return still if render_frame >= 0 else frames_path

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("FFmpeg is required for --format mp4.")
    output = output_directory / "render.mp4"
    subprocess.run([
        ffmpeg, "-framerate", str(fps), "-start_number", str(max(0, render_frame)),
        "-i", str(frames_path / "frame_%04d.png"), "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-y", str(output),
    ], check=True, **kwargs)
    return output


def _make_mosaic(images, output, columns=4):
    from PIL import Image

    opened = [Image.open(path).convert("RGBA") for path in images]
    try:
        width, height = opened[0].size
        if any(image.size != (width, height) for image in opened):
            raise ValueError("Mosaic inputs must have identical dimensions.")
        rows = (len(opened) + columns - 1) // columns
        mosaic = Image.new("RGBA", (columns * width, rows * height))
        for index, image in enumerate(opened):
            mosaic.paste(image, ((index % columns) * width, (index // columns) * height))
        mosaic.save(output)
    finally:
        for image in opened:
            image.close()
    return output


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--env", choices=HUMANOID_ENVS, default="UnitreeH1")
    parser.add_argument("--all-humanoids", action="store_true")
    parser.add_argument("--dataset-source", choices=("default", "lafan1"), default="default")
    parser.add_argument("--task", default="walk")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--light-intensity", type=int, default=1000)
    parser.add_argument("--lighting", choices=("uniform", "studio", "model"), default="uniform")
    parser.add_argument("--sun-energy", type=float, default=5.0)
    parser.add_argument("--fill-ratio", type=float, default=0.55)
    parser.add_argument("--robot-roughness", type=float, default=0.9)
    parser.add_argument("--robot-specular", type=float, default=0.05)
    parser.add_argument("--floor-style", choices=("viser", "model"), default="viser")
    parser.add_argument("--floor-cell-size", type=float, default=1.0)
    parser.add_argument("--floor-extent", type=float, default=10000.0)
    parser.add_argument("--floor-roughness", type=float, default=0.35)
    parser.add_argument("--sky-style", choices=("daylight", "model"), default="daylight")
    parser.add_argument("--sky-saturation", type=float, default=1.35)
    parser.add_argument("--environment-strength", type=float, default=0.0)
    parser.add_argument("--view-transform", default="Khronos PBR Neutral")
    parser.add_argument("--exposure", type=float, default=0.7)
    parser.add_argument("--render-frame", default=None)
    parser.add_argument("--overlay-interval", type=float)
    parser.add_argument("--overlay-opacity", type=float, default=0.3)
    parser.add_argument("--overlay-whiten", type=float, default=0.55)
    parser.add_argument("--hide-geom", action="append", default=[])
    parser.add_argument("--camera-position", nargs=3, type=float)
    parser.add_argument("--camera-look-at", nargs=3, type=float)
    parser.add_argument("--camera-distance-scale", type=float, default=1.0)
    parser.add_argument("--camera-body-name")
    parser.add_argument("--format", choices=("mp4", "png"), default="mp4")
    parser.add_argument("--engine", choices=("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"),
                        default="BLENDER_EEVEE_NEXT")
    parser.add_argument("--blender-executable")
    parser.add_argument("--filetype", choices=("usd", "usda", "usdc"), default="usd")
    parser.add_argument("--mosaic", action="store_true",
                        help="Combine all stills without downsampling (requires --all-humanoids).")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    env_names = HUMANOID_ENVS if args.all_humanoids else (args.env,)
    output_root = args.output or Path(
        f"LocoMuJoCo_blender/{'all_humanoids' if args.all_humanoids else args.env}_{args.task}"
    )
    output_root = output_root.resolve()
    outputs = []

    for index, env_name in enumerate(env_names, start=1):
        print(f"[{index}/{len(env_names)}] Preparing {env_name}...")
        dataset_conf = (
            {"default_dataset_conf": {"task": args.task}}
            if args.dataset_source == "default"
            else {"lafan1_dataset_conf": {"dataset_name": args.task}}
        )
        env, _ = ImitationFactory.make(env_name, headless=True, **dataset_conf)
        output_directory = output_root / env_name if args.all_humanoids else output_root
        try:
            automatic_position, automatic_look_at = _camera_for_robot(
                env, args.camera_distance_scale
            )
            camera_position = args.camera_position or automatic_position
            camera_look_at = args.camera_look_at or automatic_look_at
            env.play_trajectory(
                n_episodes=1,
                n_steps_per_episode=args.steps,
                usd=True,
                usd_params={
                    "output_directory": output_directory,
                    "fps": args.fps,
                    "light_intensity": args.light_intensity,
                    "camera_body_name": args.camera_body_name,
                    "filetype": args.filetype,
                    "verbose": not args.quiet,
                },
                quiet=args.quiet,
            )
            usd_path = Path(env.usd_output_path)
            if args.render:
                hidden = list(args.hide_geom)
                if env_name in ("SkeletonTorque", "SkeletonMuscle"):
                    hidden.extend(("foot_box_r", "foot_box_l"))
                result = _render_usd(
                    usd_path, output_directory, args, camera_position, camera_look_at, hidden
                )
            else:
                result = usd_path
            outputs.append((env_name, result))
        finally:
            env.stop()

    if args.mosaic:
        if not args.all_humanoids or not args.render or args.format != "png":
            raise ValueError("--mosaic requires --all-humanoids --render --format png.")
        if not all(Path(path).is_file() for _, path in outputs):
            raise ValueError("--mosaic requires a single --render-frame.")
        mosaic = _make_mosaic(
            [Path(path) for _, path in outputs], output_root / "all_humanoids.png"
        )
        print("Mosaic:", mosaic)

    print("Outputs:")
    for env_name, path in outputs:
        print(f"  {env_name}: {path}")


if __name__ == "__main__":
    main()
