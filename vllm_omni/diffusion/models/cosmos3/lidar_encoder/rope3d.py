# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""3D rotary positional embedding for ``(T, H, W)`` token grids.

The head dimension is split into three chunks ``(dim_t, dim_h, dim_w)``.
Each axis contributes its own rotary frequency band, and the resulting angles are duplicated
across the two rotation halves (GPT-NeoX style) so that
:func:`apply_rotary_emb` can rotate Q/K with a single tensor.

For a query at position ``p_q = (t_q, h_q, w_q)`` and a key at
``p_k = (t_k, h_k, w_k)``, the dot product after rotation depends only on
``p_q - p_k``. If the query and key share the same value along some axis
(e.g. a factored *spatial* attention block where both are on the same
frame), the rotation along that axis cancels out -- so this single 3D
RoPE specializes correctly to 2D RoPE for spatial-only blocks and 1D RoPE
for temporal-only blocks.
"""

from __future__ import annotations

import torch
from torch import nn

from vllm_omni.diffusion.layers.rope import apply_rotary_emb_torch


class VideoRopePosition3DEmb(nn.Module):
    """3D RoPE producing angles for a ``(T, H, W)`` token grid.

    Args:
        head_dim: Per-head channel count. Must satisfy
            ``head_dim - 2 * (head_dim // 6 * 2) > 0``.
        len_t: Maximum supported temporal grid size.
        len_h: Maximum supported grid height.
        len_w: Maximum supported grid width. Frequencies are cached for these
            maxima; ``forward`` accepts any smaller ``(T, H, W)``.
        base_theta: RoPE base (``10000.0`` matches LLaMA/Cosmos).
    """

    def __init__(
        self,
        *,
        head_dim: int,
        len_t: int,
        len_h: int,
        len_w: int,
        base_theta: float = 10000.0,
    ) -> None:
        super().__init__()
        dim = head_dim
        dim_h = (dim // 6) * 2
        dim_w = dim_h
        dim_t = dim - dim_h - dim_w
        assert dim_t >= 2, f"head_dim={dim} too small for 3D RoPE (dim_t={dim_t}, dim_h={dim_h}, dim_w={dim_w})"
        self.head_dim = dim
        self.dim_t = dim_t
        self.dim_h = dim_h
        self.dim_w = dim_w
        self.max_t = len_t
        self.max_h = len_h
        self.max_w = len_w

        idx_h = torch.arange(0, dim_h, 2, dtype=torch.float32) / dim_h
        idx_w = torch.arange(0, dim_w, 2, dtype=torch.float32) / dim_w
        idx_t = torch.arange(0, dim_t, 2, dtype=torch.float32) / dim_t
        self.register_buffer("freqs_h", 1.0 / (base_theta**idx_h), persistent=False)
        self.register_buffer("freqs_w", 1.0 / (base_theta**idx_w), persistent=False)
        self.register_buffer("freqs_t", 1.0 / (base_theta**idx_t), persistent=False)
        self.register_buffer("seq_h", torch.arange(len_h, dtype=torch.float32), persistent=False)
        self.register_buffer("seq_w", torch.arange(len_w, dtype=torch.float32), persistent=False)
        self.register_buffer("seq_t", torch.arange(len_t, dtype=torch.float32), persistent=False)

    def forward(self, num_frames: int, height: int, width: int) -> torch.Tensor:  # returns [T*H*W,D]
        """Returns rope angles of shape ``(T*H*W, head_dim)``.

        Each half of the last dim duplicates the per-axis angles
        ``[t_half | h_half | w_half | t_half | h_half | w_half]`` so that
        :func:`apply_rotary_emb` (GPT-NeoX style) can rotate Q/K via
        ``[x_lo, x_hi] @ rot``.
        """
        assert num_frames <= self.max_t and height <= self.max_h and width <= self.max_w, (
            f"Requested ({num_frames},{height},{width}) exceeds cached ({self.max_t},{self.max_h},{self.max_w})"
        )
        ang_t = torch.outer(self.seq_t[:num_frames], self.freqs_t)  # [T,Dt/2]
        ang_h = torch.outer(self.seq_h[:height], self.freqs_h)  # [H,Dh/2]
        ang_w = torch.outer(self.seq_w[:width], self.freqs_w)  # [W,Dw/2]
        ang_t_b = ang_t.view(num_frames, 1, 1, -1).expand(num_frames, height, width, -1)  # [T,H,W,Dt/2]
        ang_h_b = ang_h.view(1, height, 1, -1).expand(num_frames, height, width, -1)  # [T,H,W,Dh/2]
        ang_w_b = ang_w.view(1, 1, width, -1).expand(num_frames, height, width, -1)  # [T,H,W,Dw/2]
        half = torch.cat([ang_t_b, ang_h_b, ang_w_b], dim=-1)  # [T,H,W,D/2]
        full = torch.cat([half, half], dim=-1)  # [T,H,W,D]
        return full.reshape(num_frames * height * width, -1)  # [T*H*W,D]


def apply_rotary_emb(
    x: torch.Tensor, rope_emb: torch.Tensor
) -> torch.Tensor:  # x: [...,S,N,D], rope_emb: [S,D], returns [...,S,N,D]
    """Apply half-split rotation using the shared helper and duplicated 3D angles."""
    angles = rope_emb[:, : x.shape[-1] // 2].to(x.dtype)
    return apply_rotary_emb_torch(x, angles.cos(), angles.sin(), interleaved=False)
