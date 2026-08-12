# SPDX-License-Identifier: Apache-2.0
"""Typed AR-Diffusion control adapter for Cosmos-Dreams AgiBot ticks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vllm_omni.diffusion.models.cosmos_dreams.controller import (
    ACTION_COORDINATE_VERSION,
    ACTION_STEPS_PER_TICK,
    AGIBOT_ACTION_DIM,
    AGIBOT_DOMAIN_ID,
    AGIBOT_EMBODIMENT,
)
from vllm_omni.experimental.ar_diffusion.tick_protocol import (
    ARDiffusionControlInput,
    ARDiffusionTickRequest,
)

COSMOS_DREAMS_ACTION_TRACK = "robot_action"
COSMOS_DREAMS_ACTION_SCHEMA = "robot_action.v1"
COSMOS_DREAMS_LATENT_FRAMES_PER_TICK = 4


@dataclass(frozen=True, slots=True)
class CosmosDreamsTickInputs:
    """Validated model inputs reconstructed from one typed control payload."""

    action: torch.Tensor
    frame_idx: int
    num_latent_frames: int
    domain_name: str
    domain_id: int
    measure_tick_latency: bool


def build_cosmos_dreams_action_control(
    action: torch.Tensor,
    *,
    frame_idx: int,
    measure_tick_latency: bool,
) -> ARDiffusionControlInput:
    """Serialize a raw AgiBot chunk into the model-neutral tick contract."""

    tensor = torch.as_tensor(action, dtype=torch.float32).detach().cpu().contiguous()
    expected = (ACTION_STEPS_PER_TICK, AGIBOT_ACTION_DIM)
    if tuple(tensor.shape) != expected:
        raise ValueError(f"Cosmos-Dreams ticks require raw action shape {expected}, got {tuple(tensor.shape)}.")
    if not torch.isfinite(tensor).all():
        raise ValueError("Cosmos-Dreams raw action contains NaN or Inf values.")
    if isinstance(frame_idx, bool) or not isinstance(frame_idx, int) or frame_idx < 0:
        raise ValueError("Cosmos-Dreams frame_idx must be a non-negative integer.")
    return ARDiffusionControlInput(
        track=COSMOS_DREAMS_ACTION_TRACK,
        schema=COSMOS_DREAMS_ACTION_SCHEMA,
        data={
            "values": tensor.tolist(),
            "frame_idx": frame_idx,
            "num_latent_frames": COSMOS_DREAMS_LATENT_FRAMES_PER_TICK,
            "domain_name": AGIBOT_EMBODIMENT,
            "domain_id": AGIBOT_DOMAIN_ID,
            "coordinate_version": ACTION_COORDINATE_VERSION,
            "measure_tick_latency": bool(measure_tick_latency),
        },
    )


def _int_field(data: Any, name: str) -> int:
    value = data.get(name) if hasattr(data, "get") else None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{COSMOS_DREAMS_ACTION_SCHEMA}.{name} must be an integer.")
    return value


def parse_cosmos_dreams_tick(tick: ARDiffusionTickRequest) -> CosmosDreamsTickInputs:
    """Validate the schema-tagged action and reconstruct its float32 tensor."""

    controls = [control for control in tick.controls if control.track == COSMOS_DREAMS_ACTION_TRACK]
    if len(controls) != 1:
        raise ValueError("Cosmos-Dreams typed ticks require exactly one robot_action control.")
    control = controls[0]
    if control.schema != COSMOS_DREAMS_ACTION_SCHEMA:
        raise ValueError(
            f"Cosmos-Dreams robot_action schema must be {COSMOS_DREAMS_ACTION_SCHEMA!r}, got {control.schema!r}."
        )
    data = control.data
    frame_idx = _int_field(data, "frame_idx")
    expected_frame_idx = 0 if tick.chunk_index == 0 else 1 + tick.chunk_index * COSMOS_DREAMS_LATENT_FRAMES_PER_TICK
    if frame_idx != expected_frame_idx:
        raise ValueError(
            "Cosmos-Dreams robot_action frame_idx does not match chunk_index: "
            f"expected {expected_frame_idx}, got {frame_idx}."
        )
    num_latent_frames = _int_field(data, "num_latent_frames")
    if num_latent_frames != COSMOS_DREAMS_LATENT_FRAMES_PER_TICK:
        raise ValueError(
            f"Cosmos-Dreams typed ticks require exactly {COSMOS_DREAMS_LATENT_FRAMES_PER_TICK} generated latent frames."
        )
    domain_name = data.get("domain_name")
    domain_id = _int_field(data, "domain_id")
    if domain_name != AGIBOT_EMBODIMENT or domain_id != AGIBOT_DOMAIN_ID:
        raise ValueError(f"Cosmos-Dreams typed ticks are pinned to {AGIBOT_EMBODIMENT}/domain {AGIBOT_DOMAIN_ID}.")
    if data.get("coordinate_version") != ACTION_COORDINATE_VERSION:
        raise ValueError("Cosmos-Dreams robot_action coordinate_version does not match the controller contract.")
    measure_tick_latency = data.get("measure_tick_latency", False)
    if not isinstance(measure_tick_latency, bool):
        raise ValueError(f"{COSMOS_DREAMS_ACTION_SCHEMA}.measure_tick_latency must be a boolean.")
    try:
        action = torch.tensor(data["values"], dtype=torch.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{COSMOS_DREAMS_ACTION_SCHEMA}.values must contain numeric action rows.") from exc
    expected_shape = (ACTION_STEPS_PER_TICK, AGIBOT_ACTION_DIM)
    if tuple(action.shape) != expected_shape:
        raise ValueError(
            f"{COSMOS_DREAMS_ACTION_SCHEMA}.values must have shape {expected_shape}, got {tuple(action.shape)}."
        )
    if not torch.isfinite(action).all():
        raise ValueError(f"{COSMOS_DREAMS_ACTION_SCHEMA}.values contains NaN or Inf values.")
    return CosmosDreamsTickInputs(
        action=action.contiguous(),
        frame_idx=frame_idx,
        num_latent_frames=num_latent_frames,
        domain_name=domain_name,
        domain_id=domain_id,
        measure_tick_latency=measure_tick_latency,
    )


__all__ = [
    "COSMOS_DREAMS_ACTION_SCHEMA",
    "COSMOS_DREAMS_ACTION_TRACK",
    "COSMOS_DREAMS_LATENT_FRAMES_PER_TICK",
    "CosmosDreamsTickInputs",
    "build_cosmos_dreams_action_control",
    "parse_cosmos_dreams_tick",
]
