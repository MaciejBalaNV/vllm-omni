# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end check of the Cosmos3 multiview FlashAttention-4 backend.

Requires a datacenter-Blackwell GPU and the optional ``vllm-omni[fa4]`` extra;
everything that can be verified without one lives in
``test_multiview_flex_attention.py``.
"""

from __future__ import annotations

import math

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.B200]

# Fixture UND capacity; production uses DEFAULT_MAX_UND_TOKENS (4098). 128 is
# the smallest legal value here because real_und_len is parametrized up to 128,
# and being exactly one FA4 KV block it reproduces the padded length the old
# prompt-dependent rule gave for both 96 and 128 -- so the dense oracle below
# compares against the same geometry it did before capacity padding. The 96 case
# leaves a pad region and the 128 case fills the block exactly, covering both.
_MAX_UND = 128


def _fa4_unavailable() -> str | None:
    if not torch.cuda.is_available():
        return "CUDA is not available"
    if torch.cuda.get_device_capability()[0] != 10:
        return "FlashAttention-4 multiview backend targets datacenter Blackwell (SM100)"
    try:
        import flash_attn.cute  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return f"flash-attn-4 is not importable: {exc}"
    return None


_SKIP_REASON = _fa4_unavailable()
pytestmark.append(pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or ""))


def _dense_oracle(q, k_und, k, v_und, v, metadata, multiview_pair_predicate):
    """Masked dense attention over the unpadded sequence, in float32."""
    heads = q.shape[2]
    kv_heads = k.shape[2]
    repeat = heads // kv_heads
    dense_k = torch.cat([k_und, k], dim=1).repeat_interleave(repeat, dim=2).float()
    dense_v = torch.cat([v_und, v], dim=1).repeat_interleave(repeat, dim=2).float()
    allowed = multiview_pair_predicate(
        metadata,
        torch.arange(q.shape[1], device=q.device)[:, None],
        torch.arange(dense_k.shape[1], device=q.device)[None, :],
    )
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), dense_k) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~allowed[None, None], float("-inf"))
    return torch.einsum("bhqk,bkhd->bqhd", scores.softmax(dim=-1), dense_v)


# Cosmos3 defaults to 32 query heads over 8 KV heads (transformer_cosmos3.py:
# num_attention_heads / num_key_value_heads), so the production path is 4:1 GQA,
# not MHA.  FA4 packs those query heads by default; upstream covers that exact
# shape against a head-broadcast block map in tests/cute/test_mask_mod.py, and
# these cases pin it down for the multiview mask.
HEAD_GEOMETRIES = [
    pytest.param(4, 4, id="mha"),
    pytest.param(32, 8, id="gqa-production-4to1"),
    pytest.param(8, 1, id="mqa"),
]


@pytest.mark.parametrize("num_heads,num_kv_heads", HEAD_GEOMETRIES)
@pytest.mark.parametrize("real_und_len", [128, 96])
def test_fa4_multiview_attention_matches_dense_oracle(real_und_len: int, num_heads: int, num_kv_heads: int) -> None:
    from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import (
        MultiviewAttentionContext,
        MultiviewLayout,
        build_multiview_flex_metadata,
        multiview_pair_predicate,
        padded_multiview_flex_attention,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    head_dim = 128

    # Two views x four latent frames x 8x8 patches -> 256 tokens per item,
    # 512 packed GEN tokens, which is exactly two 256-row FA4 query blocks.
    layout = MultiviewLayout(2, 4, 8, 8, backend="fa4", max_und_tokens=_MAX_UND)
    gen = layout.gen_tokens
    context = MultiviewAttentionContext(layout, {})

    q = torch.randn(1, gen, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, gen, num_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    k_und = torch.randn(1, real_und_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_und = torch.randn_like(k_und)

    actual = padded_multiview_flex_attention(q, k, v, k_und, v_und, context)
    assert actual.shape == (1, gen, num_heads, head_dim)

    metadata = build_multiview_flex_metadata(
        seq_len=real_und_len + gen,
        full_q_offsets=(real_und_len, real_und_len + layout.items[0].num_tokens, real_und_len + gen),
        items_per_sample=layout.items,
        device=device,
        num_und=real_und_len,
        attention_scope=layout.attention_scope,
    )
    expected = _dense_oracle(q, k_und, k, v_und, v, metadata, multiview_pair_predicate)

    torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("num_heads,num_kv_heads", HEAD_GEOMETRIES)
@pytest.mark.parametrize(
    "patch_hw",
    [
        (8, 8),
        (20, 20),
        (17, 23),
        (23, 17),
        (15, 26),
        (26, 15),
        (30, 30),
        (26, 35),
        (35, 26),
        (23, 40),
        (40, 23),
    ],
)
def test_fa4_and_triton_backends_agree(num_heads: int, num_kv_heads: int, patch_hw) -> None:
    """Both backends project the same predicate, so their outputs must match."""
    from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import (
        MultiviewAttentionContext,
        MultiviewLayout,
        padded_multiview_flex_attention,
    )

    torch.manual_seed(1)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    head_dim = 128

    tensors = None
    outputs = {}
    for backend in ("triton", "fa4"):
        layout = MultiviewLayout(2, 4, *patch_hw, backend=backend, max_und_tokens=_MAX_UND)
        gen = layout.gen_tokens
        if tensors is None:
            tensors = (
                torch.randn(1, gen, num_heads, head_dim, device=device, dtype=dtype),
                torch.randn(1, gen, num_kv_heads, head_dim, device=device, dtype=dtype),
                torch.randn(1, gen, num_kv_heads, head_dim, device=device, dtype=dtype),
                torch.randn(1, 128, num_kv_heads, head_dim, device=device, dtype=dtype),
                torch.randn(1, 128, num_kv_heads, head_dim, device=device, dtype=dtype),
            )
        context = MultiviewAttentionContext(layout, {})
        outputs[backend] = padded_multiview_flex_attention(*tensors, context)

    torch.testing.assert_close(outputs["fa4"].float(), outputs["triton"].float(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    "num_heads,num_kv_heads",
    [
        pytest.param(32, 8, id="production-gqa"),
        pytest.param(4, 1, id="cp8-local-gqa"),
    ],
)
def test_fa4_vector_masks_match_scalar_bitwise(num_heads: int, num_kv_heads: int) -> None:
    """Changing Boolean mask evaluation must preserve the attention result's bits."""
    import cutlass
    import cutlass.cute as cute
    from flash_attn.cute import utils as fa_utils

    from vllm_omni.diffusion.models.cosmos3.multiview_fa4 import (
        _build_mask_mod,
        _build_run_mask_mod,
        _load_fa4,
        multiview_fa4_attention,
    )
    from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import (
        MultiviewAttentionContext,
        MultiviewBlockSparsity,
        MultiviewLayout,
        _pack_padded_bshd,
        get_multiview_attention_plan,
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16
    head_dim = 128
    entry = _load_fa4()
    masks = {
        f"vec_size={vec_size}": _build_mask_mod(cutlass, cute, fa_utils, vec_size=vec_size) for vec_size in (1, 8, 32)
    }
    masks["run_chunks"] = _build_run_mask_mod(cutlass, cute, fa_utils)
    cache = {}
    for caption_lengths in ((35, 61), (7, 9)):
        real_und_len = sum(caption_lengths)
        # Seventeen frames per view produce more than 32 key runs. The
        # 35-token runs cross vector/tile boundaries; short CFG captions put
        # two caption runs and UND padding into the first 32-key chunk.
        layout = MultiviewLayout(
            2,
            34,
            5,
            7,
            backend="fa4",
            max_und_tokens=_MAX_UND,
            control_attends_sensor=True,
            caption_lengths=caption_lengths,
        )
        plan, geometry = get_multiview_attention_plan(
            MultiviewAttentionContext(layout, cache),
            real_und_len=real_und_len,
            real_q_len=layout.gen_tokens,
            device=device,
        )
        assert isinstance(plan, MultiviewBlockSparsity)
        assert geometry.real_q_len < geometry.padded_q_len
        assert geometry.real_und_len < geometry.padded_und_len
        assert plan.k_group_ids.max().item() > 31
        assert (plan.allowed_words < 0).any().item(), "Exercise bit 31 in signed int32 packed words"
        assert plan.partial_counts.sum().item() > 0
        assert plan.full_counts.sum().item() > 0
        first, last = plan.k_run_chunks[:, 0], plan.k_run_chunks[:, 1]
        assert (first == last).any().item(), "Exercise homogeneous chunks"
        assert ((last >= 0) & (first != last)).any().item(), "Exercise two-run chunks"
        if real_und_len < 32:
            assert last[0].item() == -1, "Short captions must exercise the three-run fallback"

        torch.manual_seed(23)
        q = torch.randn(1, layout.gen_tokens, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(1, layout.gen_tokens, num_kv_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn_like(k)
        k_und = torch.randn(1, real_und_len, num_kv_heads, head_dim, device=device, dtype=dtype)
        v_und = torch.randn_like(k_und)
        q_padded = _pack_padded_bshd((q, geometry.padded_q_len))
        k_padded = _pack_padded_bshd((k_und, geometry.padded_und_len), (k, geometry.padded_q_len))
        v_padded = _pack_padded_bshd((v_und, geometry.padded_und_len), (v, geometry.padded_q_len))

        block_sparse = entry.block_sparse_cls(
            mask_block_cnt=plan.partial_counts[None, None],
            mask_block_idx=plan.partial_indices[None, None],
            full_block_cnt=plan.full_counts[None, None],
            full_block_idx=plan.full_indices[None, None],
            block_size=(plan.q_block_size, plan.kv_block_size),
        )
        reference_bits = None
        # Four mask specializations per head geometry, reused across CFG
        # lengths. The wrapper should reuse the optimized specialization.
        for name, mask_mod in masks.items():
            output = entry.flash_attn_func(
                q_padded,
                k_padded,
                v_padded,
                mask_mod=mask_mod,
                aux_tensors=plan.aux_tensors(),
                block_sparse_tensors=block_sparse,
            )
            if isinstance(output, tuple):
                output = output[0]
            assert torch.isfinite(output).all().item()
            assert output[:, geometry.real_q_len :].count_nonzero().item() == 0
            bits = output.contiguous().view(torch.int16)
            if reference_bits is None:
                reference_bits = bits.clone()
            else:
                assert torch.equal(bits, reference_bits), f"{name}, captions={caption_lengths} changed output bits"
        actual = multiview_fa4_attention(q_padded, k_padded, v_padded, plan)
        assert torch.equal(actual.contiguous().view(torch.int16), reference_bits)


@pytest.mark.parametrize("seqlen_k", [33, 63, None], ids=["tail-33", "tail-63", "all-boundaries"])
def test_fa4_run_mask_chunks_match_vector_bitwise(seqlen_k: int | None) -> None:
    """Check run compression against the original per-key mask on arbitrary tables.

    Run memory checking on a supported GPU with:
        compute-sanitizer --tool memcheck --error-exitcode=1 python -m pytest \
            tests/diffusion/models/cosmos3/test_multiview_fa4.py -k run_mask_chunks
    """
    import cutlass
    import cutlass.cute as cute
    from flash_attn.cute import utils as fa_utils

    from vllm_omni.diffusion.models.cosmos3.multiview_fa4 import (
        _build_mask_mod,
        _build_run_mask_mod,
        _load_fa4,
    )
    from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import (
        _build_key_run_chunks,
        _pack_allowed_bits,
    )

    if seqlen_k is None:
        chunks = [[group] * 32 for group in (31, 32, 63, 64)]
        for first, last in ((31, 32), (63, 64)):
            chunks.extend([first] * boundary + [last] * (32 - boundary) for boundary in range(1, 32))
        chunks.extend(
            (
                [31, 64] * 16,  # Two IDs need not form contiguous runs.
                [31] * 10 + [32] * 11 + [64] * 11,
                [31] * 15 + [63] + [31] * 16,  # Equal endpoints cannot prove homogeneity.
            )
        )
        group_ids = [group for chunk in chunks for group in chunk] + [64]
    else:
        group_ids = [31] * 32 + [32] * (seqlen_k - 32)

    device = torch.device("cuda")
    k_group_ids = torch.tensor(group_ids, dtype=torch.int32, device=device)
    k_run_chunks = _build_key_run_chunks(k_group_ids)
    seqlen_k = len(group_ids)
    # Exercise every visibility combination of four IDs spanning packed-word
    # boundaries, including bits 31 and 63 and completely masked query rows.
    query_patterns = torch.arange(16, dtype=torch.int64, device=device)
    bit_offsets = torch.arange(4, dtype=torch.int64, device=device)
    group_allowed = torch.zeros((16, 65), dtype=torch.bool, device=device)
    group_allowed[:, [31, 32, 63, 64]] = ((query_patterns[:, None] >> bit_offsets) & 1).bool()
    allowed_words, words_per_row = _pack_allowed_bits(group_allowed)
    q_len = 65
    q_groups = torch.arange(q_len, dtype=torch.int64, device=device) % 16
    q_word_base = (q_groups * words_per_row).to(torch.int32)
    allowed = group_allowed[q_groups][:, k_group_ids.long()]
    all_masked = ~allowed.any(dim=-1)
    assert all_masked.any().item()
    assert (allowed_words < 0).any().item()
    if seqlen_k > 128:
        assert (k_run_chunks[:, 1] < 0).sum().item() == 2
    # Both Q and K have unaligned lengths. FA4 wraps auxiliary reads for the
    # padded lanes; the optimized mask must fall back for wrapped vectors.
    assert q_len % 32 and seqlen_k % 32

    torch.manual_seed(41)
    q = torch.randn(1, q_len, 4, 128, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, seqlen_k, 1, 128, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    entry = _load_fa4()
    kv_blocks = (seqlen_k + 127) // 128
    # Metadata counts logical query rows: Q=65 needs one 256-row entry even
    # though GQA packs 260 rows into two physical work tiles.
    counts = torch.full((1, 1, 1), kv_blocks, dtype=torch.int32, device=device)
    indices = torch.arange(kv_blocks, dtype=torch.int32, device=device)[None, None, None, :]
    # Mark every visited block partial, so full-block shortcuts cannot hide a
    # bad mask. The map is broadcast across all four packed query heads.
    block_sparse = entry.block_sparse_cls(
        mask_block_cnt=counts,
        mask_block_idx=indices,
        full_block_cnt=torch.zeros_like(counts),
        full_block_idx=torch.zeros_like(indices),
        block_size=(256, 128),
    )
    outputs = []
    for mask_mod in (
        _build_mask_mod(cutlass, cute, fa_utils, vec_size=32),
        _build_run_mask_mod(cutlass, cute, fa_utils),
    ):
        output = entry.flash_attn_func(
            q,
            k,
            v,
            mask_mod=mask_mod,
            aux_tensors=[q_word_base, k_group_ids, allowed_words, k_run_chunks],
            block_sparse_tensors=block_sparse,
        )
        if isinstance(output, tuple):
            output = output[0]
        assert torch.isfinite(output).all().item()
        assert output[:, all_masked].count_nonzero().item() == 0
        outputs.append(output)
    assert torch.equal(outputs[0].contiguous().view(torch.int16), outputs[1].contiguous().view(torch.int16))

    scores = torch.matmul(q.float().transpose(1, 2), k.float().permute(0, 2, 3, 1)) / math.sqrt(128)
    probabilities = scores.masked_fill(~allowed[None, None], float("-inf")).softmax(dim=-1).nan_to_num(0.0)
    expected = torch.matmul(probabilities, v.float().transpose(1, 2)).transpose(1, 2)
    torch.testing.assert_close(outputs[1].float(), expected, atol=2e-2, rtol=2e-2)
