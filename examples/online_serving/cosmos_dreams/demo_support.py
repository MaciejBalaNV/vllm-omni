# SPDX-License-Identifier: Apache-2.0
"""Startup-only scene/replay loading shared by Cosmos-Dreams demos."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from vllm_omni.diffusion.models.cosmos_dreams.controller import (
    AgiBotControllerLimits,
    AgiBotSceneState,
)


def load_scene_bundle(path: Path) -> AgiBotSceneState:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    seed_path = Path(payload.pop("seed_rgb"))
    if not seed_path.is_absolute():
        seed_path = path.parent / seed_path
    seed_rgb = Image.open(seed_path).convert("RGB")
    limits_payload = payload.pop("limits", {})
    limits = AgiBotControllerLimits(**limits_payload)
    return AgiBotSceneState(seed_rgb=seed_rgb, limits=limits, **payload)


def load_replay_actions(path: Path) -> list[torch.Tensor]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    actions = torch.as_tensor(value, dtype=torch.float32)
    if actions.ndim == 2:
        actions = actions.unsqueeze(0)
    if actions.ndim != 3 or tuple(actions.shape[1:]) != (16, 29):
        raise ValueError(f"Replay actions must have shape [chunks,16,29], got {tuple(actions.shape)}.")
    return [chunk.contiguous() for chunk in actions]
