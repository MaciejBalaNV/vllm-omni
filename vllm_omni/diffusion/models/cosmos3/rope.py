# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fused Qwen3-style RoPE for Cosmos3 Q/K tensors."""

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

if HAS_TRITON:

    @triton.jit
    def _rope_rows(
        x_ptr,
        out_ptr,
        cos,
        sin,
        x_stride_b,
        x_stride_s,
        x_stride_h,
        b,
        s,
        seq_len,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        block_h: tl.constexpr,
    ):
        heads = tl.arange(0, block_h)[:, None]
        columns = tl.arange(0, head_dim)[None, :]
        rotated = (columns + head_dim // 2) % head_dim
        mask = heads < num_heads
        row = x_ptr + b * x_stride_b + s * x_stride_s + heads * x_stride_h
        x = tl.load(row + columns, mask=mask, other=0.0)
        x_rot = tl.load(row + rotated, mask=mask, other=0.0)
        x_rot = tl.where(columns < head_dim // 2, -x_rot, x_rot)
        # Eager rounds each product to the input dtype before the add.
        a = (x.to(tl.float32) * cos).to(x.dtype).to(tl.float32)
        r = (x_rot.to(tl.float32) * sin).to(x.dtype).to(tl.float32)
        out_row = out_ptr + ((b * seq_len + s) * num_heads + heads) * head_dim
        tl.store(out_row + columns, (a + r).to(x.dtype), mask=mask)

    @triton.jit
    def _rope_qk_kernel(
        q_ptr,
        k_ptr,
        cos_ptr,
        sin_ptr,
        q_out_ptr,
        k_out_ptr,
        seq_len,
        q_stride_b,
        q_stride_s,
        q_stride_h,
        k_stride_b,
        k_stride_s,
        k_stride_h,
        cs_stride_b,
        cs_stride_s,
        sn_stride_b,
        sn_stride_s,
        num_q_heads: tl.constexpr,
        num_k_heads: tl.constexpr,
        head_dim: tl.constexpr,
        block_qh: tl.constexpr,
        block_kh: tl.constexpr,
    ):
        token = tl.program_id(0).to(tl.int64)
        b = token // seq_len
        s = token % seq_len
        columns = tl.arange(0, head_dim)[None, :]
        cos = tl.load(cos_ptr + b * cs_stride_b + s * cs_stride_s + columns).to(tl.float32)
        sin = tl.load(sin_ptr + b * sn_stride_b + s * sn_stride_s + columns).to(tl.float32)
        _rope_rows(
            q_ptr,
            q_out_ptr,
            cos,
            sin,
            q_stride_b,
            q_stride_s,
            q_stride_h,
            b,
            s,
            seq_len,
            num_q_heads,
            head_dim,
            block_qh,
        )
        _rope_rows(
            k_ptr,
            k_out_ptr,
            cos,
            sin,
            k_stride_b,
            k_stride_s,
            k_stride_h,
            b,
            s,
            seq_len,
            num_k_heads,
            head_dim,
            block_kh,
        )


def _broadcast_table(table: torch.Tensor, batch: int, seq_len: int, head_dim: int) -> torch.Tensor | None:
    if table.ndim != 4 or table.shape[2] != 1:
        return None
    try:
        table = table.expand(batch, seq_len, 1, head_dim)
    except RuntimeError:
        return None
    return table if table.stride(3) == 1 else None


def fused_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``(x * cos) + (rotate_half(x) * sin)`` for Q and K in one kernel, bit-exact with eager.

    Returns ``None`` when the inputs are outside the fused kernel's contract.
    """
    if (
        not HAS_TRITON
        or not q.is_cuda
        or torch.compiler.is_compiling()
        or (torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, cos, sin)))
        or q.ndim != 4
        or k.ndim != 4
        or q.dtype not in (torch.bfloat16, torch.float16)
        or any(t.dtype != q.dtype or t.device != q.device for t in (k, cos, sin))
        or q.shape[:2] != k.shape[:2]
        or q.shape[3] != k.shape[3]
        or q.stride(3) != 1
        or k.stride(3) != 1
        or q.numel() == 0
        or k.numel() == 0
    ):
        return None
    batch, seq_len, num_q_heads, head_dim = q.shape
    num_k_heads = k.shape[2]
    if head_dim < 2 or head_dim & (head_dim - 1):
        return None
    cos = _broadcast_table(cos, batch, seq_len, head_dim)
    sin = _broadcast_table(sin, batch, seq_len, head_dim)
    if cos is None or sin is None:
        return None
    q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    _rope_qk_kernel[(batch * seq_len,)](
        q,
        k,
        cos,
        sin,
        q_out,
        k_out,
        seq_len,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        num_q_heads,
        num_k_heads,
        head_dim,
        triton.next_power_of_2(num_q_heads),
        triton.next_power_of_2(num_k_heads),
        enable_fp_fusion=False,
    )
    return q_out, k_out
