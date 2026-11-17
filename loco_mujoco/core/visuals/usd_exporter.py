"""Testable offline trajectory export to an animated USD stage."""

from __future__ import annotations

import datetime
import re
import warnings
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import mujoco
import numpy as np


USD_EXPORTER_PARAMS = {
    "usd_output_directory",
    "usd_fps",
    "usd_light_intensity",
    "usd_camera_body_name",
    "usd_verbose",
    "usd_filetype",
}


class UsdTrajectoryExporter:
    """Accumulate MuJoCo states into a portable animated USD stage."""

    def __init__(
        self,
        model: mujoco.MjModel,
        dt: float,
        *,
        output_directory: str | Path | None = None,
        fps: float | None = None,
        light_intensity: int = 1000,
        camera_body_name: str | None = None,
        geom_groups: Optional[Sequence[int]] = (0,),
        filetype: Literal["usd", "usda", "usdc"] = "usd",
        verbose: bool = True,
    ):
        try:
            from mujoco.usd.exporter import USDExporter
        except ModuleNotFoundError as exc:
            if exc.name == "pxr":
                raise ImportError(
                    "USD export requires the optional USD bindings. Install them with "
                    "`pip install loco-mujoco[usd]`."
                ) from exc
            raise

        if dt <= 0:
            raise ValueError("dt must be positive.")
        if filetype not in ("usd", "usda", "usdc"):
            raise ValueError("filetype must be 'usd', 'usda', or 'usdc'.")

        if output_directory is None:
            timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            output_directory = Path("LocoMuJoCo_usd") / timestamp
        self.output_directory = Path(output_directory).expanduser().resolve()
        self.output_directory.mkdir(parents=True, exist_ok=True)

        self.model = model
        self.dt = float(dt)
        self.source_fps = 1.0 / self.dt
        self.fps = self.source_fps if fps is None else float(fps)
        if self.fps <= 0:
            raise ValueError("fps must be positive.")
        if self.fps > self.source_fps + 1e-9:
            raise ValueError(
                f"fps ({self.fps}) cannot exceed the source frequency ({self.source_fps}); "
                "frame interpolation is not implemented."
            )

        self.filetype = filetype
        self.verbose = bool(verbose)
        self.frame_count = 0
        self.source_frame_count = 0
        self._next_frame_time = 0.0
        self.usd_path: Optional[Path] = None
        self._stopped = False
        self._usd_finalized = False
        self._camera_target_op = None
        self._mjx_data = mujoco.MjData(model)

        if camera_body_name is None:
            moving_bodies = np.flatnonzero(np.asarray(model.body_jntnum) > 0)
            self.camera_body_id = int(moving_bodies[0]) if len(moving_bodies) else 0
        else:
            self.camera_body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, camera_body_name
            )
            if self.camera_body_id < 0:
                raise ValueError(f"Unknown camera body {camera_body_name!r}.")

        self._scene_option = mujoco.MjvOption()
        if geom_groups is not None:
            self._scene_option.geomgroup[:] = 0
            for group in geom_groups:
                group = int(group)
                if not 0 <= group < len(self._scene_option.geomgroup):
                    raise ValueError(f"Invalid geom group {group}; expected a value in [0, 5].")
                self._scene_option.geomgroup[group] = 1

        class _SafeUSDExporter(USDExporter):
            def _get_geom_name(self, geom):
                name = super()._get_geom_name(geom)
                name = re.sub(r"[^A-Za-z0-9_]", "_", name)
                return name if name[0].isalpha() or name[0] == "_" else f"_{name}"

        self._exporter = _SafeUSDExporter(
            model,
            width=int(model.vis.global_.offwidth),
            height=int(model.vis.global_.offheight),
            output_directory=self.output_directory.name,
            output_directory_root=str(self.output_directory.parent),
            light_intensity=int(light_intensity),
            verbose=self.verbose,
        )

    def render(self, data: mujoco.MjData, *_: Any) -> None:
        """Append one MuJoCo state to the animated USD stage."""
        if self._stopped or self._usd_finalized:
            raise RuntimeError("This UsdTrajectoryExporter has already been finalized.")
        source_time = self.source_frame_count * self.dt
        self.source_frame_count += 1
        if source_time + 1e-12 < self._next_frame_time:
            return

        self._exporter.update_scene(data, scene_option=self._scene_option)
        if self.frame_count == 0:
            self._exporter.stage.SetTimeCodesPerSecond(self.fps)
            self._exporter.stage.SetFramesPerSecond(self.fps)
            from pxr import UsdGeom

            UsdGeom.SetStageMetersPerUnit(self._exporter.stage, 1.0)
            target = UsdGeom.Xform.Define(
                self._exporter.stage, "/World/LocoMuJoCoCameraTarget"
            )
            self._camera_target_op = target.AddTranslateOp()

        from pxr import Gf

        target_position = np.asarray(data.xpos[self.camera_body_id], dtype=float)
        self._camera_target_op.Set(Gf.Vec3d(*target_position.tolist()), self.frame_count)
        self.frame_count += 1
        self._next_frame_time = self.frame_count / self.fps

    def render_mjx(self, state: Any, env_idx: int = 0) -> None:
        """Append one selected environment from a batched MJX state."""
        qpos = np.asarray(state.data.qpos)
        if qpos.ndim == 1:
            if env_idx != 0:
                raise IndexError("env_idx must be 0 for an unbatched MJX state.")
            source = state.data
        else:
            if not 0 <= env_idx < qpos.shape[0]:
                raise IndexError(f"env_idx {env_idx} is outside a batch of size {qpos.shape[0]}.")
            source = _IndexedData(state.data, env_idx)

        self._mjx_data.qpos[:] = np.asarray(source.qpos)
        self._mjx_data.qvel[:] = np.asarray(source.qvel)
        if self.model.nu and hasattr(source, "ctrl"):
            self._mjx_data.ctrl[:] = np.asarray(source.ctrl)
        if self.model.nmocap:
            self._mjx_data.mocap_pos[:] = np.asarray(source.mocap_pos)
            self._mjx_data.mocap_quat[:] = np.asarray(source.mocap_quat)
        mujoco.mj_forward(self.model, self._mjx_data)
        self.render(self._mjx_data)

    def save_usd(self) -> Path:
        """Save and return the USD stage path."""
        if self.usd_path is not None:
            return self.usd_path
        if self.frame_count == 0:
            raise RuntimeError("No frames were exported. Call render() before save_usd().")

        self._exporter.save_scene(filetype=self.filetype)
        self.usd_path = (
            self.output_directory / "frames" / f"frame_{self.frame_count}.{self.filetype}"
        )

        from pxr import Sdf, Usd

        stage = Usd.Stage.Open(str(self.usd_path))
        stage.SetEndTimeCode(max(0, self.frame_count - 1))
        for prim in stage.Traverse():
            varname = prim.GetAttribute("inputs:varname")
            if varname and varname.GetTypeName() == Sdf.ValueTypeNames.Token:
                value = varname.Get()
                prim.RemoveProperty("inputs:varname")
                prim.CreateAttribute("inputs:varname", Sdf.ValueTypeNames.String).Set(value)
            metallic = prim.GetAttribute("inputs:metallic")
            if metallic:
                metallic.Set(0.0)
        stage.GetRootLayer().Save()
        self._usd_finalized = True
        return self.usd_path

    def stop(self) -> Optional[str]:
        """Finalize the USD stage and return its path."""
        if self._stopped:
            return str(self.usd_path) if self.usd_path is not None else None
        if self.frame_count == 0:
            warnings.warn("No USD frames were collected; no USD was written.", stacklevel=2)
            self._stopped = True
            return None
        path = self.save_usd()
        self._stopped = True
        return str(path)


class _IndexedData:
    def __init__(self, data: Any, index: int):
        self._data = data
        self._index = index

    def __getattr__(self, name: str) -> Any:
        return getattr(self._data, name)[self._index]
