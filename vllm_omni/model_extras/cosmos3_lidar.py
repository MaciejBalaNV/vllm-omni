# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numeric LiDAR admission. No image/video decoding is used by this contract."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def required_lidar_sweeps(num_frames: int, camera_fps: float, lidar_fps: float) -> int:
    if num_frames <= 0 or any(not math.isfinite(rate) or rate <= 0 for rate in (camera_fps, lidar_fps)):
        raise ValueError("Camera frame count and camera/LiDAR FPS must be positive and finite.")
    sweeps = round(num_frames * lidar_fps / camera_fps)
    if sweeps < 1:
        raise ValueError("The camera duration must cover at least one LiDAR sweep.")
    return sweeps


@contextmanager
def _open_lidar_frames(path: str | Path) -> Iterator[Any]:
    """Validate the safetensors structure without reading tensor values."""
    from safetensors import SafetensorError, safe_open

    if Path(path).suffix.lower() != ".safetensors":
        raise ValueError("LiDAR control must be a .safetensors file.")
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            if list(handle.keys()) != ["frames"]:
                raise ValueError("LiDAR control must contain only a tensor named 'frames'.")
            tensor = handle.get_slice("frames")
            shape = tensor.get_shape()
            if len(shape) != 4 or shape[0] != 3 or shape[1] < 1 or shape[2:] != [128, 1800]:
                raise ValueError(f"LiDAR frames must have shape [3,T,128,1800], got {shape}.")
            if tensor.get_dtype() != "F32":
                raise ValueError("LiDAR frames must be float32.")
            yield tensor
    except (OSError, RuntimeError, SafetensorError) as exc:
        raise ValueError(f"Invalid LiDAR safetensors input: {exc}") from exc


def validate_lidar_header(path: str | Path) -> tuple[int, ...]:
    """Check upload structure at admission; values are checked after sweep selection."""
    with _open_lidar_frames(path) as tensor:
        return tuple(tensor.get_shape())


def load_lidar_frames(path: str | Path, *, num_sweeps: int | None = None) -> torch.Tensor:
    import torch

    if num_sweeps is not None and (type(num_sweeps) is not int or num_sweeps < 1):
        raise ValueError("Required LiDAR sweep count must be a positive integer.")
    with _open_lidar_frames(path) as tensor:
        available_sweeps = tensor.get_shape()[1]
        if num_sweeps is not None and available_sweeps < num_sweeps:
            raise ValueError(f"LiDAR control requires {num_sweeps} sweeps, but contains {available_sweeps}.")
        frames = tensor[:, :num_sweeps].contiguous()
    if not torch.isfinite(frames).all():
        raise ValueError("LiDAR frames must contain finite values.")
    if (frames[0] < 0).any() or (frames[1:] < 0).any() or (frames[1:] > 1).any():
        raise ValueError("LiDAR range must be non-negative; intensity and validity must be in [0,1].")
    return frames
