# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803
"""Fused scatter-merge for the maskless Cosmos3 multiview LSE accumulator.

:func:`~.multiview_maskless_attention.merge_step` folds each attention pass
into an FP32 ``[tokens, heads, head_dim]`` accumulator.  As tensor ops that
costs three traversals of the accumulator per chunk -- ``index_select`` gathers
the rows, the compiled recurrence rewrites them, ``index_copy_`` scatters them
back -- and the accumulator is the largest tensor in the denoise step, so the
round trips dominate the merge.

The kernel below does the indirection in registers and updates the accumulator
in place: one read and one write per merge.  Its expressions mirror the Triton
that Inductor generates for ``merge_step`` term for term, so the accumulator is
bit-identical; the tensor path stays the reference on CPU and without Triton.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import HAS_TRITON, tl, tldevice, triton

#: ``(row, head)`` slots per program.  The merge is memory bound, so this only
#: has to keep enough of the accumulator in flight to saturate HBM.
_SLOTS_PER_PROGRAM = 16


if HAS_TRITON:

    @triton.jit
    def _fused_merge_kernel(
        acc_ptr,
        acc_lse_ptr,
        out_ptr,
        lse_ptr,
        index_ptr,
        num_slots,
        heads,
        head_dim,
        IDENTITY: tl.constexpr,
        SEED: tl.constexpr,
        BLOCK_SLOTS: tl.constexpr,
        BLOCK_DIM: tl.constexpr,
    ):
        # A slot is one (row, head) pair: the recurrence shares its weight
        # across head_dim, which one program holds as a single tile.  All
        # offsets are int64 -- the accumulator approaches 2**31 elements at the
        # released geometry, so int32 would wrap on a larger request.
        slot = tl.program_id(0).to(tl.int64) * BLOCK_SLOTS + tl.arange(0, BLOCK_SLOTS)
        slot_mask = slot < num_slots
        row = slot // heads
        head = slot - row * heads
        # An identity pass covers every GEN row in order, so its q_index is
        # exactly arange and the load can be skipped.
        destination = row if IDENTITY else tl.load(index_ptr + row, slot_mask, other=0)
        acc_slot = destination * heads + head

        dim = tl.arange(0, BLOCK_DIM)
        tile_mask = slot_mask[:, None] & (dim < head_dim)[None, :]
        acc_row = acc_ptr + acc_slot * head_dim
        out_row = out_ptr + slot * head_dim

        out = tl.load(out_row[:, None] + dim[None, :], tile_mask).to(tl.float32)
        lse = tl.load(lse_ptr + slot, slot_mask)
        if SEED:
            # The first pass covers every row and seeds the accumulators.
            tl.store(acc_row[:, None] + dim[None, :], out, tile_mask)
            tl.store(acc_lse_ptr + acc_slot, lse, slot_mask)
        else:
            acc_lse = tl.load(acc_lse_ptr + acc_slot, slot_mask)
            weight = tl.sigmoid(lse - acc_lse)
            acc = tl.load(acc_row[:, None] + dim[None, :], tile_mask)
            tl.store(acc_row[:, None] + dim[None, :], acc - weight[:, None] * (acc - out), tile_mask)
            # Inductor lowers logsigmoid(x) to minimum(0, x) - log1p(exp(-|x|))
            # over libdevice, and its minimum is where(0 < x, 0, x), which keeps
            # a NaN x where tl.minimum would discard it in favour of the 0.
            shift = acc_lse - lse
            log_sigmoid = tl.where(0.0 < shift, 0.0, shift) - tldevice.log1p(tldevice.exp(-tl.abs(shift)))
            tl.store(acc_lse_ptr + acc_slot, acc_lse - log_sigmoid, slot_mask)


def fused_merge_available(device: torch.device) -> bool:
    """Whether :func:`fused_merge` can serve ``device``."""
    return HAS_TRITON and device.type == "cuda"


def fused_merge(
    acc: torch.Tensor,
    acc_lse: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    index: torch.Tensor,
    *,
    identity: bool,
    seed: bool,
) -> None:
    """Fold ``out``/``lse`` into ``acc``/``acc_lse`` in place.

    ``index`` maps pass rows to accumulator rows; ``identity`` says it is the
    arange the kernel can skip loading.  ``seed`` writes the accumulators
    instead of folding into them, for the first pass.  Updating in place is
    safe because the planner proves a pass never attends a query row twice, so
    the rows one launch writes are distinct.
    """
    rows, heads, head_dim = out.shape
    if index.numel() != rows:
        raise ValueError(f"Fused merge index covers {index.numel()} rows for {rows} pass rows.")
    num_slots = rows * heads
    _fused_merge_kernel[(triton.cdiv(num_slots, _SLOTS_PER_PROGRAM),)](
        acc,
        acc_lse,
        out,
        lse,
        index,
        num_slots,
        heads,
        head_dim,
        IDENTITY=identity,
        SEED=seed,
        BLOCK_SLOTS=_SLOTS_PER_PROGRAM,
        BLOCK_DIM=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
