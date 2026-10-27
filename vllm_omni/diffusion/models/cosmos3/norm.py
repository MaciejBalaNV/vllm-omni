# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

from vllm_omni.diffusion.layers.norm import RMSNorm as _VllmRMSNorm

if HAS_TRITON:

    @triton.jit
    def _square_f32_kernel(x_ptr, out_ptr, hidden_size: tl.constexpr, block_n: tl.constexpr):
        # ``x.float().pow(2)``: eager pow special-cases exponent 2 as ``x * x``.
        row = tl.program_id(0).to(tl.int64)
        columns = tl.arange(0, block_n)
        mask = columns < hidden_size
        x = tl.load(x_ptr + row * hidden_size + columns, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + row * hidden_size + columns, x * x, mask=mask)

    @triton.jit
    def _scale_rows_kernel(
        x_ptr, inv_rms_ptr, weight_ptr, out_ptr, hidden_size: tl.constexpr, block_n: tl.constexpr
    ):
        # ``(weight.float() * (x.float() * inv_rms)).to(x.dtype)`` with the eager rounding order.
        row = tl.program_id(0).to(tl.int64)
        columns = tl.arange(0, block_n)
        mask = columns < hidden_size
        x = tl.load(x_ptr + row * hidden_size + columns, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
        inv_rms = tl.load(inv_rms_ptr + row)
        out = weight * (x * inv_rms)
        tl.store(out_ptr + row * hidden_size + columns, out.to(out_ptr.dtype.element_ty), mask=mask)


class RMSNorm(_VllmRMSNorm):
    """Cosmos3 RMSNorm that uses the FP32 native implementation."""

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        if (
            not HAS_TRITON
            or torch.compiler.is_compiling()
            or not x.is_contiguous()
            or x.numel() == 0
            or x.dtype not in (torch.bfloat16, torch.float16)
            or self.weight.dtype not in (torch.bfloat16, torch.float16, torch.float32)
            or x.shape[-1] != self.hidden_size
            or self.weight.device != x.device
        ):
            return self.forward_native(x)
        # Same arithmetic as ``forward_native``: the elementwise steps run as two
        # Triton kernels (no FP32 copies of ``x``), while the variance keeps
        # the eager ``mean`` reduction so its summation order is unchanged.
        rows = x.numel() // self.hidden_size
        block_n = triton.next_power_of_2(self.hidden_size)
        squared = torch.empty(x.shape, dtype=torch.float32, device=x.device)
        _square_f32_kernel[(rows,)](x, squared, self.hidden_size, block_n)
        inv_rms = torch.rsqrt(squared.mean(-1, keepdim=True) + self.variance_epsilon)
        del squared
        out = torch.empty_like(x)
        _scale_rows_kernel[(rows,)](x, inv_rms, self.weight, out, self.hidden_size, block_n)
        return out

    def forward_hip(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)

    def forward_npu(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)
