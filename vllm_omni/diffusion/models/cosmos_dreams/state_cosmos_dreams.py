# SPDX-License-Identifier: Apache-2.0
"""Non-KV per-session state and fail-closed request fingerprinting."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class CosmosDreamsSessionFingerprint:
    prompt_hash: str
    real_text_kv_lengths: tuple[tuple[str, int], ...]
    height: int
    width: int
    fps: float
    domain_id: int
    normalizer_id: str
    checkpoint_id: str
    manifest_id: str
    sampler_id: str

    def __post_init__(self) -> None:
        if not self.prompt_hash:
            raise ValueError("Cosmos-Dreams session fingerprint requires prompt_hash")
        if self.height <= 0 or self.width <= 0 or self.fps <= 0:
            raise ValueError(
                "Cosmos-Dreams fingerprint resolution/FPS must be positive, "
                f"got {self.height}x{self.width}@{self.fps}"
            )
        if self.domain_id < 0:
            raise ValueError(f"Cosmos-Dreams domain_id must be non-negative, got {self.domain_id}")
        if not self.real_text_kv_lengths:
            raise ValueError("Cosmos-Dreams fingerprint requires at least one text KV branch")
        if any(length <= 0 for _, length in self.real_text_kv_lengths):
            raise ValueError(
                f"Cosmos-Dreams real text KV lengths must be positive, got {self.real_text_kv_lengths}"
            )

    def text_length(self, branch: str) -> int:
        try:
            return dict(self.real_text_kv_lengths)[branch]
        except KeyError as exc:
            raise KeyError(
                f"Cosmos-Dreams fingerprint has no text length for branch {branch!r}"
            ) from exc


@dataclass
class CosmosDreamsSessionState:
    session_id: str
    fingerprint: CosmosDreamsSessionFingerprint | None = None
    next_frame_idx: int = 0
    terminal: bool = False
    prompt_ids_by_branch: dict[str, torch.Tensor] = field(default_factory=dict)
    prompt_masks_by_branch: dict[str, torch.Tensor] = field(default_factory=dict)
    text_kv_by_branch: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = field(default_factory=dict)
    dense_kv_by_branch: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = field(default_factory=dict)
    latents: list[torch.Tensor] = field(default_factory=list)
    vae_decoder_feat_cache: list[torch.Tensor | None] | None = None

    def initialize(
        self,
        fingerprint: CosmosDreamsSessionFingerprint,
        *,
        next_frame_idx: int = 0,
    ) -> None:
        if self.fingerprint is not None:
            raise RuntimeError(
                f"Cosmos-Dreams session {self.session_id!r} is already initialized; session reset required"
            )
        if next_frame_idx < 0:
            raise ValueError(f"Cosmos-Dreams next_frame_idx must be non-negative, got {next_frame_idx}")
        self.fingerprint = fingerprint
        self.next_frame_idx = int(next_frame_idx)

    def validate_request(
        self,
        fingerprint: CosmosDreamsSessionFingerprint,
        *,
        frame_idx: int,
    ) -> None:
        if self.fingerprint is None:
            raise RuntimeError(
                f"Cosmos-Dreams session {self.session_id!r} is not initialized; session reset required"
            )
        if self.terminal:
            raise ValueError(
                f"Cosmos-Dreams session {self.session_id!r} already completed a full rollout; "
                "session reset required"
            )
        if fingerprint != self.fingerprint:
            changed = [
                field_name
                for field_name in self.fingerprint.__dataclass_fields__
                if getattr(self.fingerprint, field_name) != getattr(fingerprint, field_name)
            ]
            raise ValueError(
                "Cosmos-Dreams session conditioning changed "
                f"({', '.join(changed)}); session reset required"
            )
        if frame_idx != self.next_frame_idx:
            raise ValueError(
                "Cosmos-Dreams request is out of order: "
                f"expected latent frame {self.next_frame_idx}, got {frame_idx}; session reset required"
            )

    def append_chunk(self, chunk: torch.Tensor, *, frame_start: int) -> None:
        if chunk.ndim != 5 or chunk.shape[0] != 1:
            raise ValueError(
                "Cosmos-Dreams session chunks must have shape [1,C,T,H,W], "
                f"got {tuple(chunk.shape)}"
            )
        if frame_start != self.next_frame_idx:
            raise ValueError(
                "Cosmos-Dreams cannot append an out-of-order chunk: "
                f"expected {self.next_frame_idx}, got {frame_start}; session reset required"
            )
        self.latents.append(chunk.detach())
        self.next_frame_idx += int(chunk.shape[2])

    def reset(self) -> None:
        self.fingerprint = None
        self.next_frame_idx = 0
        self.terminal = False
        self.prompt_ids_by_branch.clear()
        self.prompt_masks_by_branch.clear()
        self.text_kv_by_branch.clear()
        self.dense_kv_by_branch.clear()
        self.latents.clear()
        self.vae_decoder_feat_cache = None

    @property
    def accumulated_latents(self) -> torch.Tensor | None:
        if not self.latents:
            return None
        return torch.cat(self.latents, dim=2)
