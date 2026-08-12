# SPDX-License-Identifier: Apache-2.0
"""Session-owned incremental decode for the causal Wan video VAE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def snapshot_feature_cache(cache: list[Any]) -> list[Any]:
    """Snapshot a Wan cache list without losing non-tensor sentinels.

    Wan's temporal upsamplers use the string sentinel ``"Rep"`` alongside
    tensors and ``None``. Decoder layers replace list entries rather than
    mutating cached tensors in place, so a detached list copy is sufficient to
    isolate the committed session state from the next decode transaction.
    """

    return [entry.detach() if isinstance(entry, torch.Tensor) else entry for entry in cache]


def _unpatchify(video: torch.Tensor, patch_size: int) -> torch.Tensor:
    if patch_size <= 1:
        return video
    batch, channels, frames, height, width = video.shape
    patch_area = patch_size * patch_size
    if channels % patch_area != 0:
        raise ValueError(f"Wan decoder output channels {channels} are not divisible by patch_size^2={patch_area}.")
    channels //= patch_area
    video = video.reshape(batch, channels, patch_size, patch_size, frames, height, width)
    video = video.permute(0, 1, 4, 5, 2, 6, 3).contiguous()
    return video.reshape(batch, channels, frames, height * patch_size, width * patch_size)


@dataclass(frozen=True, slots=True)
class WanStreamingDecodeResult:
    video: torch.Tensor
    feature_cache: list[Any]


def decode_wan_causal_chunk(
    vae,
    denormalized_latents: torch.Tensor,
    *,
    feature_cache: list[Any] | None,
    initialized: bool,
) -> WanStreamingDecodeResult:
    """Decode only new latent frames while preserving causal decoder state.

    This is the streaming form of diffusers ``AutoencoderKLWan._decode``: the
    post-quant projection is applied to the incoming latent block, each latent
    frame advances the decoder feature cache once, and only the very first
    frame in a session uses ``first_chunk=True``.
    """

    if denormalized_latents.ndim != 5 or denormalized_latents.shape[0] != 1:
        raise ValueError(
            f"Incremental Wan decode expects latents shaped [1,C,T,H,W], got {tuple(denormalized_latents.shape)}."
        )
    if denormalized_latents.shape[2] <= 0:
        raise ValueError("Incremental Wan decode requires at least one new latent frame.")
    if feature_cache is None:
        vae.clear_cache()
        raw_cache = getattr(vae, "_feat_map", None)
        if not isinstance(raw_cache, list):
            raise TypeError("Wan VAE clear_cache() did not expose the expected decoder _feat_map list.")
        feature_cache = snapshot_feature_cache(raw_cache)

    vae._feat_map = snapshot_feature_cache(feature_cache)
    try:
        projected = vae.post_quant_conv(denormalized_latents)
        decoded_parts: list[torch.Tensor] = []
        for latent_idx in range(projected.shape[2]):
            vae._conv_idx = [0]
            decoded = vae.decoder(
                projected[:, :, latent_idx : latent_idx + 1],
                feat_cache=vae._feat_map,
                feat_idx=vae._conv_idx,
                first_chunk=not initialized and latent_idx == 0,
            )
            decoded_parts.append(decoded)

        video = torch.cat(decoded_parts, dim=2)
        patch_size = getattr(getattr(vae, "config", None), "patch_size", None)
        if patch_size is not None:
            video = _unpatchify(video, int(patch_size))
        committed_cache = snapshot_feature_cache(vae._feat_map)
    finally:
        # The state owns the committed cache; the shared VAE must not retain a
        # second session reference after success, reset, or decoder failure.
        vae.clear_cache()
    return WanStreamingDecodeResult(
        video=video,
        feature_cache=committed_cache,
    )


__all__ = [
    "WanStreamingDecodeResult",
    "decode_wan_causal_chunk",
    "snapshot_feature_cache",
]
