# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Strict, request-owned bridge between vLLM paged K/V and Cosmos3 GEN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    from .transformer_cosmos3 import Cosmos3VFMTransformer


@dataclass
class ExternalBranch:
    expected_length: int
    kv: list[tuple[torch.Tensor, torch.Tensor]]
    freqs: tuple[torch.Tensor, torch.Tensor] | None = None

    def text_tensors(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        # IDs are unused by GEN. Length and mask include the two appended tokens.
        mask = torch.ones((1, self.expected_length), dtype=torch.long, device=device)
        return torch.zeros_like(mask), mask


def validate_branch(
    cache: object,
    metadata: dict,
    expected_length: int,
    *,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    checkpoint: dict[str, str] | None = None,
) -> ExternalBranch:
    if checkpoint is not None and metadata.get("cosmos3_checkpoint") != checkpoint:
        raise ValueError("Cosmos3 UND and GEN must use the same model and checkpoint revision")
    if expected_length <= 0 or metadata.get("seq_len") != expected_length:
        raise ValueError(f"Cosmos3 UND export length {metadata.get('seq_len')} != expected {expected_length}")
    if metadata.get("dtype") not in ("auto", str(dtype), str(dtype).removeprefix("torch.")):
        raise ValueError("Cosmos3 UND export dtype metadata mismatch")
    if metadata.get("cosmos3_conditioning_length") != expected_length:
        raise ValueError("Cosmos3 UND export canonical length metadata mismatch")
    if metadata.get("confirmed_computed", -1) < expected_length:
        raise ValueError("Cosmos3 UND export contains uncomputed conditioning tokens")
    if metadata.get("num_layers") != num_layers:
        raise ValueError("Cosmos3 UND export layer count mismatch")
    if metadata.get("layer_order") != list(range(num_layers)):
        raise ValueError("Cosmos3 UND export layer order is missing or incorrect")
    keys = getattr(cache, "key_cache", ())
    values = getattr(cache, "value_cache", ())
    if len(keys) != num_layers or len(values) != num_layers:
        raise ValueError("Cosmos3 UND export is missing layers")
    kv = []
    shape = (expected_length, num_heads, head_dim)
    for index, (key, value) in enumerate(zip(keys, values, strict=True)):
        for tensor in (key, value):
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise ValueError(f"Cosmos3 UND layer {index} must have shape {shape} and dtype {dtype}")
        kv.append((key.unsqueeze(0), value.unsqueeze(0)))
    return ExternalBranch(expected_length, kv)


def load_external_branches(
    sp: OmniDiffusionSamplingParams,
    transformer: Cosmos3VFMTransformer,
    do_cfg: bool,
    dtype: torch.dtype,
    *,
    checkpoint: dict[str, str],
) -> dict[bool, ExternalBranch]:
    lengths = (sp.extra_args or {}).get("cosmos3_expected_lengths", {})
    branches = {}
    for negative in (False, True) if do_cfg else (False,):
        role = "negative" if negative else "positive"
        prefix = "cfg_text_" if negative else ""
        expected = lengths.get(role)
        if not isinstance(expected, int):
            raise ValueError(f"Missing canonical Cosmos3 {role} token count")
        branches[negative] = validate_branch(
            getattr(sp, f"{prefix}past_key_values", None),
            getattr(sp, f"{prefix}kv_metadata", None) or {},
            expected,
            num_layers=transformer.num_hidden_layers,
            num_heads=transformer.num_key_value_heads,
            head_dim=transformer.head_dim,
            dtype=dtype,
            checkpoint=checkpoint,
        )
    return branches


def prepare_branch_frequencies(
    transformer: Cosmos3VFMTransformer,
    branches: dict[bool, ExternalBranch],
    latents: torch.Tensor,
    fps: float | None,
    *,
    dtype: torch.dtype,
) -> None:
    _, _, frames, height, width = latents.shape
    hp, wp, _, _ = transformer._pad_to_patch_size(height, width)
    for branch in branches.values():
        _, mask = branch.text_tensors(latents.device)
        # Sampling state may be FP32 while transformer execution is BF16.
        _, branch.freqs = transformer._compute_rope_freqs(mask, frames, hp, wp, fps, latents.device, dtype)
