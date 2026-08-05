# SPDX-License-Identifier: Apache-2.0
"""Pure packing, mRoPE, cache-accounting, and hashing helpers."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import torch

from vllm_omni.diffusion.models.cosmos_dreams.config import CosmosDreamsManifest


def iter_ar_chunk_ranges(start_frame: int, num_frames: int, chunk_size: int) -> Iterator[tuple[int, int]]:
    """Yield the training-aligned partition ``[1, C, C, ...]``."""
    if start_frame < 0 or num_frames < 0 or start_frame > num_frames:
        raise ValueError(f"Invalid Cosmos-Dreams frame range [{start_frame}, {num_frames})")
    if chunk_size <= 0:
        raise ValueError(f"Cosmos-Dreams chunk_size must be positive, got {chunk_size}")
    frame = start_frame
    while frame < num_frames:
        if frame == 0:
            chunk_end = 1
        else:
            chunk_end = 1 + ((frame - 1) // chunk_size + 1) * chunk_size
        chunk_end = min(chunk_end, num_frames)
        yield frame, chunk_end
        frame = chunk_end


def iter_clean_commit_frames(
    chunk_start: int,
    chunk_end: int,
    *,
    target_frame: int,
    terminal_request: bool,
) -> Iterator[tuple[int, int]]:
    """Yield ``(local, absolute)`` clean-refresh frames in commit order.

    The globally final frame has no downstream reader and is omitted only for
    a terminal request. Every other frame is refreshed individually so later
    frames in the same denoised chunk see clean, committed history.
    """
    if chunk_start < 0 or chunk_end <= chunk_start or target_frame < chunk_end:
        raise ValueError(
            "Invalid Cosmos-Dreams clean-commit range: "
            f"chunk=[{chunk_start}, {chunk_end}), target={target_frame}"
        )
    for local_idx, frame_idx in enumerate(range(chunk_start, chunk_end)):
        if terminal_request and frame_idx == target_frame - 1:
            continue
        yield local_idx, frame_idx


def interleave_action_vision_tokens(
    action_tokens: torch.Tensor,
    vision_tokens: torch.Tensor,
) -> torch.Tensor:
    """Pack per-frame hidden states as ``[action, vision]`` supertokens.

    Args:
        action_tokens: ``[B, T, A, D]``.
        vision_tokens: ``[B, T, P, D]``.
    """
    if action_tokens.ndim != 4 or vision_tokens.ndim != 4:
        raise ValueError(
            "Cosmos-Dreams interleaving expects action [B,T,A,D] and vision [B,T,P,D], "
            f"got {tuple(action_tokens.shape)} and {tuple(vision_tokens.shape)}"
        )
    if action_tokens.shape[:2] != vision_tokens.shape[:2] or action_tokens.shape[-1] != vision_tokens.shape[-1]:
        raise ValueError(
            "Cosmos-Dreams action/vision batch, frame, and hidden dimensions must match; "
            f"got {tuple(action_tokens.shape)} and {tuple(vision_tokens.shape)}"
        )
    return torch.cat([action_tokens, vision_tokens], dim=2).flatten(1, 2)


def split_interleaved_action_vision_tokens(
    tokens: torch.Tensor,
    *,
    num_frames: int,
    action_tokens_per_frame: int,
    vision_tokens_per_frame: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`interleave_action_vision_tokens`."""
    if tokens.ndim != 3:
        raise ValueError(f"Cosmos-Dreams packed tokens must have shape [B,S,D], got {tuple(tokens.shape)}")
    tokens_per_frame = action_tokens_per_frame + vision_tokens_per_frame
    expected = num_frames * tokens_per_frame
    if tokens.shape[1] != expected:
        raise ValueError(f"Cosmos-Dreams packed length must be {expected}, got {tokens.shape[1]}")
    framed = tokens.view(tokens.shape[0], num_frames, tokens_per_frame, tokens.shape[-1])
    return framed[:, :, :action_tokens_per_frame], framed[:, :, action_tokens_per_frame:]


def build_interleaved_mrope_position_ids(
    *,
    frame_start: int,
    num_frames: int,
    grid_h: int,
    grid_w: int,
    text_temporal_offset: int,
    temporal_modality_margin: int,
    fps: float,
    base_fps: float = 24.0,
    action_tokens_per_frame: int = 4,
    null_action_frames: Iterable[int] = (),
) -> torch.Tensor:
    """Build reference-compatible mRoPE IDs in interleaved supertoken order.

    Real action sub-tokens span the interval ending at their associated latent
    frame. The first null-action supertoken in an AR unit is co-located with
    its vision frame. Later all-null supertokens retain the architectural real-
    action IDs used by the reference packer, even though their values are zero.
    """
    if frame_start < 0 or num_frames <= 0 or grid_h <= 0 or grid_w <= 0:
        raise ValueError(
            "Cosmos-Dreams mRoPE dimensions must be positive and frame_start non-negative; "
            f"got start={frame_start}, frames={num_frames}, grid={grid_h}x{grid_w}"
        )
    if fps <= 0 or base_fps <= 0:
        raise ValueError(f"Cosmos-Dreams FPS values must be positive, got fps={fps}, base_fps={base_fps}")
    if action_tokens_per_frame <= 0:
        raise ValueError(
            f"Cosmos-Dreams action_tokens_per_frame must be positive, got {action_tokens_per_frame}"
        )

    null_frames = {int(frame) for frame in null_action_frames}
    patch_count = grid_h * grid_w
    base_offset = float(text_temporal_offset + temporal_modality_margin)
    frame_stride = float(base_fps) / float(fps)
    position_parts: list[torch.Tensor] = []
    h_ids = torch.arange(grid_h, dtype=torch.float32).view(-1, 1).expand(-1, grid_w).flatten()
    w_ids = torch.arange(grid_w, dtype=torch.float32).view(1, -1).expand(grid_h, -1).flatten()

    for local_frame in range(num_frames):
        absolute_frame = frame_start + local_frame
        vision_t = base_offset + absolute_frame * frame_stride
        if absolute_frame in null_frames and local_frame == 0:
            action_t = torch.full((action_tokens_per_frame,), vision_t, dtype=torch.float32)
        else:
            substep = frame_stride / action_tokens_per_frame
            action_t = vision_t - frame_stride + substep * torch.arange(
                1,
                action_tokens_per_frame + 1,
                dtype=torch.float32,
            )
        zeros = torch.zeros(action_tokens_per_frame, dtype=torch.float32)
        action_ids = torch.stack([action_t, zeros, zeros], dim=0)
        vision_ids = torch.stack(
            [torch.full((patch_count,), vision_t, dtype=torch.float32), h_ids, w_ids],
            dim=0,
        )
        position_parts.extend((action_ids, vision_ids))

    return torch.cat(position_parts, dim=1)


def zero_null_action_values(
    value: torch.Tensor,
    *,
    num_frames: int,
    tokens_per_frame: int,
    action_tokens_per_frame: int,
    null_frame_indexes: Sequence[int],
) -> torch.Tensor:
    """Zero V (not K) for null action slots before persistent storage."""
    if value.ndim != 4:
        raise ValueError(f"Cosmos-Dreams K/V must have shape [B,S,H,D], got {tuple(value.shape)}")
    if value.shape[1] != num_frames * tokens_per_frame:
        raise ValueError(
            "Cosmos-Dreams K/V length does not match frame geometry: "
            f"length={value.shape[1]}, frames={num_frames}, tokens_per_frame={tokens_per_frame}"
        )
    if not null_frame_indexes:
        return value
    result = value.clone()
    positions: list[int] = []
    for frame in null_frame_indexes:
        if frame < 0 or frame >= num_frames:
            raise ValueError(f"Cosmos-Dreams null action frame {frame} is outside [0, {num_frames})")
        start = frame * tokens_per_frame
        positions.extend(range(start, start + action_tokens_per_frame))
    result[:, positions] = 0
    return result


def prompt_token_hash(token_ids: Sequence[int] | torch.Tensor) -> str:
    """Stable SHA-256 over prompt token IDs, independent of tensor dtype."""
    if isinstance(token_ids, torch.Tensor):
        values = [int(value) for value in token_ids.detach().cpu().reshape(-1).tolist()]
    else:
        values = [int(value) for value in token_ids]
    payload = json.dumps(values, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CosmosDreamsKVMemoryEstimate:
    page_bytes: int
    managed_blocks: int
    scratch_blocks: int
    self_attention_bytes: int
    scratch_bytes: int
    cross_attention_bytes: int
    total_bytes: int


def estimate_kv_memory_bytes(
    manifest: CosmosDreamsManifest,
    *,
    num_layers: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    num_local_kv_branches: int = 1,
    num_logical_kv_branches: int = 1,
    session_capacity: int = 1,
    frames_per_block: int = 1,
    max_scratch_tokens_per_branch: int = 0,
) -> CosmosDreamsKVMemoryEstimate:
    """Estimate the manager floor, scratch reservation, and text pools."""
    positive = {
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_size": head_size,
        "num_local_kv_branches": num_local_kv_branches,
        "num_logical_kv_branches": num_logical_kv_branches,
        "session_capacity": session_capacity,
        "frames_per_block": frames_per_block,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"Cosmos-Dreams KV estimate {name} must be positive, got {value}")
    if max_scratch_tokens_per_branch < 0:
        raise ValueError("Cosmos-Dreams max_scratch_tokens_per_branch must be non-negative")
    page_bytes = int(
        2
        * manifest.tokens_per_frame
        * num_kv_heads
        * head_size
        * dtype.itemsize
        * num_layers
    )
    managed_blocks = num_local_kv_branches * (
        manifest.sink_frames + manifest.window_frames + frames_per_block
    ) + 2
    scratch_per_branch = frames_per_block + math.ceil(
        max_scratch_tokens_per_branch / manifest.tokens_per_frame
    )
    scratch_blocks = num_local_kv_branches * scratch_per_branch
    self_attention_bytes = managed_blocks * page_bytes
    scratch_bytes = scratch_blocks * page_bytes
    cross_attention_bytes = int(
        2
        * session_capacity
        * num_logical_kv_branches
        * manifest.text_cache_max_len
        * num_kv_heads
        * head_size
        * dtype.itemsize
        * num_layers
    )
    return CosmosDreamsKVMemoryEstimate(
        page_bytes=page_bytes,
        managed_blocks=managed_blocks,
        scratch_blocks=scratch_blocks,
        self_attention_bytes=self_attention_bytes,
        scratch_bytes=scratch_bytes,
        cross_attention_bytes=cross_attention_bytes,
        total_bytes=self_attention_bytes + scratch_bytes + cross_attention_bytes,
    )
