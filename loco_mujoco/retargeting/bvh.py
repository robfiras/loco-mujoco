from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class BVHSkeleton:
    """Joint hierarchy of a BVH file.

    Attributes:
        names: Joint names, parents always before their children.
        parents: Index of each joint's parent, -1 for the root.
        offsets: Rest translation of each joint relative to its parent in meters, shape (J, 3).

    """
    names: Tuple[str, ...]
    parents: Tuple[int, ...]
    offsets: np.ndarray

    def index(self, name: str) -> int:
        return self.names.index(name)


@dataclass(frozen=True)
class BVHMotion:
    """Local joint transforms of a BVH clip.

    Attributes:
        skeleton: The joint hierarchy.
        local_pos: Local joint translations in meters, shape (T, J, 3).
        local_rot: Local joint rotation matrices, shape (T, J, 3, 3).
        fps: Frames per second.

    """
    skeleton: BVHSkeleton
    local_pos: np.ndarray
    local_rot: np.ndarray
    fps: float


def _axis_rotation(axis: str, angle: np.ndarray) -> np.ndarray:
    """Rotation matrices about a principal axis for an array of angles in radians."""
    c, s = np.cos(angle), np.sin(angle)
    r = np.zeros(angle.shape + (3, 3))
    i = "XYZ".index(axis)
    j, k = (i + 1) % 3, (i + 2) % 3
    r[..., i, i] = 1.0
    r[..., j, j] = c
    r[..., k, k] = c
    r[..., j, k] = -s
    r[..., k, j] = s
    return r


def load_bvh(path: str, unit_scale: float = 0.01) -> BVHMotion:
    """Load a BVH file.

    Joints with position channels use them as their local translation, all others keep their
    rest offset. Rotation channels are composed in the order they are listed in the file.

    Args:
        path: Path to the BVH file.
        unit_scale: Factor converting the file's length unit to meters.

    Returns:
        The parsed motion.

    Raises:
        ValueError: If the file has no motion data or contains non-finite values.

    """
    with open(path, "r") as f:
        tokens = f.read().split()

    names: List[str] = []
    parents: List[int] = []
    offsets: List[List[float]] = []
    channels: List[List[str]] = []
    stack: List[int] = []
    pending_end_site = False

    i = 0
    while tokens[i] != "MOTION":
        token = tokens[i]
        if token in ("ROOT", "JOINT"):
            names.append(tokens[i + 1])
            parents.append(stack[-1] if stack else -1)
            offsets.append([0.0, 0.0, 0.0])
            channels.append([])
            i += 2
        elif token == "End":
            pending_end_site = True
            i += 2
        elif token == "{":
            stack.append(-2 if pending_end_site else len(names) - 1)
            pending_end_site = False
            i += 1
        elif token == "}":
            stack.pop()
            i += 1
        elif token == "OFFSET":
            if stack[-1] != -2:
                offsets[stack[-1]] = [float(v) for v in tokens[i + 1:i + 4]]
            i += 4
        elif token == "CHANNELS":
            n = int(tokens[i + 1])
            channels[stack[-1]] = tokens[i + 2:i + 2 + n]
            i += 2 + n
        else:
            i += 1

    n_frames = int(tokens[tokens.index("Frames:") + 1])
    frame_time_idx = tokens.index("Time:", i)
    frame_time = float(tokens[frame_time_idx + 1])
    n_channels = sum(len(c) for c in channels)
    values = np.array(tokens[frame_time_idx + 2:], dtype=np.float64)
    if n_frames == 0 or values.size != n_frames * n_channels:
        raise ValueError(f"Expected {n_frames} frames of {n_channels} channels in {path}, "
                         f"found {values.size} values.")
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite motion values in {path}.")
    values = values.reshape(n_frames, n_channels)

    skeleton = BVHSkeleton(tuple(names), tuple(parents), np.array(offsets) * unit_scale)
    n_joints = len(names)
    local_pos = np.tile(skeleton.offsets, (n_frames, 1, 1))
    local_rot = np.tile(np.eye(3), (n_frames, n_joints, 1, 1))

    col = 0
    for j, joint_channels in enumerate(channels):
        for channel in joint_channels:
            axis, kind = channel[0].upper(), channel[1:].lower()
            if kind == "position":
                local_pos[:, j, "XYZ".index(axis)] = values[:, col] * unit_scale
            else:
                local_rot[:, j] = local_rot[:, j] @ _axis_rotation(axis, np.deg2rad(values[:, col]))
            col += 1

    return BVHMotion(skeleton, local_pos, local_rot, 1.0 / frame_time)
