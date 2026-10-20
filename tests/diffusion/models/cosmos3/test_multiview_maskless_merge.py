# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The fused CUDA scatter-merge must be bit-identical to the tensor merge it replaces."""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.models.cosmos3 import multiview_maskless_attention as m
from vllm_omni.diffusion.models.cosmos3.multiview_attention import multiview_attention
from vllm_omni.diffusion.models.cosmos3.multiview_maskless_merge import fused_merge
from vllm_omni.platforms import current_omni_platform

from .test_multiview_maskless_attention import layout, prepared_context, tensors

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.gpu]

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused merge is CUDA only")


@pytest.fixture(autouse=True)
def keep_the_reference_compiled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bit-exactness is defined against the *compiled* merge_step.

    Eager and compiled disagree -- Inductor contracts the accumulator update
    into an FMA -- so a reference that fell back to eager past Dynamo's
    recompile limit would compare the kernel against the wrong arithmetic.
    """
    monkeypatch.setattr(torch._dynamo.config, "cache_size_limit", 256)


def tensor_merge(acc, acc_lse, out, lse, index, *, identity: bool, seed: bool) -> None:
    """The chunked gather / compiled merge_step / scatter the kernel replaces."""
    merge = m._merge_step_for(acc.device)
    for start in range(0, out.shape[0], m.MERGE_CHUNK_SIZE):
        chunk = slice(start, start + m.MERGE_CHUNK_SIZE)
        if seed and identity:
            acc[chunk].copy_(out[chunk])
            acc_lse[chunk].copy_(lse[chunk])
        elif seed:
            acc.index_copy_(0, index[chunk], out[chunk].float())
            acc_lse.index_copy_(0, index[chunk], lse[chunk].float())
        elif identity:
            merged, merged_lse = merge(acc[chunk], acc_lse[chunk], out[chunk], lse[chunk])
            acc[chunk].copy_(merged)
            acc_lse[chunk].copy_(merged_lse)
        else:
            gather = index[chunk]
            merged, merged_lse = merge(
                acc.index_select(0, gather), acc_lse.index_select(0, gather), out[chunk], lse[chunk]
            )
            acc.index_copy_(0, gather, merged)
            acc_lse.index_copy_(0, gather, merged_lse)


def identical(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)


def inputs(rows, heads, head_dim, *, dtype=torch.bfloat16, identity=True):
    """Accumulators, one pass output and the row map, on CUDA.

    An identity pass covers every accumulator row, exactly as the planner
    builds it, so its arange spans the whole accumulator.
    """
    tokens = rows if identity else rows * 2
    generator = torch.Generator(device="cuda").manual_seed(0)

    def randn(*shape, dtype=torch.float32):
        return torch.randn(*shape, generator=generator, device="cuda", dtype=torch.float32).to(dtype)

    acc, acc_lse = randn(tokens, heads, head_dim), randn(tokens, heads) * 3
    out, lse = randn(rows, heads, head_dim, dtype=dtype), randn(rows, heads) * 3
    index = (
        torch.arange(rows, device="cuda")
        if identity
        else torch.randperm(tokens, generator=generator, device="cuda")[:rows].contiguous()
    )
    return acc, acc_lse, out, lse, index


def run_both(acc, acc_lse, out, lse, index, *, identity, seed) -> None:
    expected = (acc.clone(), acc_lse.clone())
    tensor_merge(*expected, out, lse, index, identity=identity, seed=seed)
    actual = (acc.clone(), acc_lse.clone())
    fused_merge(*actual, out, lse, index, identity=identity, seed=seed)
    identical(actual[0], expected[0])
    identical(actual[1], expected[1])


@cuda_only
@pytest.mark.parametrize(
    ("rows", "heads", "head_dim", "dtype"),
    [
        (8193, 32, 128, torch.bfloat16),  # released geometry; also crosses MERGE_CHUNK_SIZE
        (37, 3, 80, torch.float16),  # ragged: neither heads nor head_dim fills a tile
        (1, 1, 64, torch.float32),  # one slot, and a no-op dtype conversion
    ],
)
def test_fused_merge_matches_tensor_merge(rows: int, heads: int, head_dim: int, dtype: torch.dtype) -> None:
    for identity in (True, False):
        for seed in (True, False):
            run_both(*inputs(rows, heads, head_dim, dtype=dtype, identity=identity), identity=identity, seed=seed)


@cuda_only
def test_fused_merge_matches_tensor_merge_on_extreme_lse() -> None:
    """NaN is what pins the kernel to Inductor's codegen.

    The recurrence's logsigmoid is lowered through ``minimum(0, x)``, which
    returns a NaN ``x`` where ``tl.minimum`` would discard it for the 0.
    """
    extremes = [float("-inf"), float("inf"), float("nan"), torch.finfo(torch.float32).min, -0.0]
    acc, acc_lse, out, lse, index = inputs(len(extremes) * 2 + 1, 2, 64, identity=True)
    for row, value in enumerate(extremes):
        lse[row] = value
        acc_lse[row + len(extremes)] = value
    lse[-1] = acc_lse[-1] = float("-inf")  # both accumulators saturated at once
    run_both(acc, acc_lse, out, lse, index, identity=True, seed=False)


@cuda_only
def test_fused_merge_addresses_past_the_int32_offset_limit() -> None:
    """An accumulator over 2**31 elements must not wrap its offsets.

    The released geometry already sits near that limit, so with 32-bit
    addressing a larger request would scatter into low rows instead.
    """
    heads, head_dim = 1, 128
    tokens = 2**31 // head_dim + 1024
    required = tokens * heads * (head_dim * 6 + 16) + 2 * 1024**3
    free = current_omni_platform.get_free_memory()
    if free < required:
        pytest.skip(f"needs {required / 1024**3:.1f} GiB free, has {free / 1024**3:.1f} GiB")

    acc = torch.zeros(tokens, heads, head_dim, dtype=torch.float32, device="cuda")
    acc_lse = torch.zeros(tokens, heads, dtype=torch.float32, device="cuda")
    try:
        assert acc.numel() > 2**31
        # An identity pass addresses the whole accumulator by row, so its last
        # rows sit past the limit; a scattered pass reaches them through index.
        out = torch.full((tokens, heads, head_dim), 2.0, dtype=torch.bfloat16, device="cuda")
        lse = torch.zeros(tokens, heads, dtype=torch.float32, device="cuda")
        fused_merge(acc, acc_lse, out, lse, torch.arange(tokens, device="cuda"), identity=True, seed=True)
        del out, lse
        current_omni_platform.empty_cache()
        identical(acc[-8:], torch.full((8, heads, head_dim), 2.0, device="cuda"))

        out = torch.full((8, heads, head_dim), 5.0, dtype=torch.bfloat16, device="cuda")
        lse = torch.full((8, heads), float("inf"), dtype=torch.float32, device="cuda")  # pass takes all the weight
        fused_merge(acc, acc_lse, out, lse, torch.arange(tokens - 8, tokens, device="cuda"), identity=False, seed=False)
        identical(acc[-8:], torch.full((8, heads, head_dim), 5.0, device="cuda"))
    finally:
        del acc, acc_lse
        current_omni_platform.empty_cache()


@cuda_only
@pytest.mark.parametrize(("views", "heads", "kv_heads", "head_dim"), [(1, 4, 4, 64), (3, 8, 2, 128), (6, 16, 2, 128)])
def test_attention_is_bit_identical_with_and_without_the_fused_merge(
    monkeypatch: pytest.MonkeyPatch, views: int, heads: int, kv_heads: int, head_dim: int
) -> None:
    """End to end over camera counts and head geometries, through the real plan."""
    spec = layout(views=views)
    num_und = sum(spec.caption_lengths)
    qkv = tensors(spec, num_und, dtype=torch.bfloat16, heads=heads, kv_heads=kv_heads, head_dim=head_dim, device="cuda")
    context = prepared_context(spec, num_und, kernel=m.resolve_maskless_kernel(), device="cuda")
    with torch.inference_mode():
        fused = multiview_attention(*qkv, context)
        monkeypatch.setattr(m, "fused_merge_available", lambda device: False)
        identical(fused, multiview_attention(*qkv, context))
