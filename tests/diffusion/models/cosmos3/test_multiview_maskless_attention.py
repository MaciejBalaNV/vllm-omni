# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU mathematical oracles; CUDA tests exercise real FA and NATTEN separately."""

from __future__ import annotations

import math
import sys
import types
from dataclasses import replace

import pytest
import torch

from vllm_omni.diffusion.models.cosmos3 import multiview_maskless_attention as m
from vllm_omni.diffusion.models.cosmos3.multiview_attention import multiview_attention
from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import (
    MaskItem,
    MultiviewAttentionContext,
    MultiviewLayout,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def merge_oracle(outputs, lse_tensors, **kwargs):
    """Independent mathematical merge, used only in tests."""
    weights = torch.stack(lse_tensors).double().softmax(0)
    return (torch.stack(outputs).double() * weights[..., None]).sum(0).to(outputs[0].dtype), None


def varlen_oracle(q, k, v, *, cu_seqlens_q, cu_seqlens_k, **kwargs):
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[1], q.shape[0], dtype=torch.float32, device=q.device)
    k, v = [x.repeat_interleave(q.shape[1] // k.shape[1], 1) for x in (k, v)]
    for qa, qb, ka, kb in zip(cu_seqlens_q[:-1], cu_seqlens_q[1:], cu_seqlens_k[:-1], cu_seqlens_k[1:]):
        scores = torch.einsum("qhd,khd->hqk", q[qa:qb].double(), k[ka:kb].double()) / math.sqrt(q.shape[-1])
        out[qa:qb] = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v[ka:kb].double()).to(q.dtype)
        lse[:, qa:qb] = scores.logsumexp(-1)
    return out, lse


@pytest.fixture
def cpu_kernels(monkeypatch):
    fa = types.ModuleType("vllm_omni.diffusion.attention.backends.utils.fa")
    fa.vllm_flash_attn_varlen_with_lse = varlen_oracle
    monkeypatch.setitem(sys.modules, fa.__name__, fa)
    natten = types.ModuleType("natten.functional")
    natten.merge_attentions = merge_oracle
    monkeypatch.setitem(sys.modules, natten.__name__, natten)


def layout(joint=True, views=2, controls=True, **kwargs):
    items = [
        MaskItem((views * 2, 1, 2), views, is_control=c, seconds_per_frame=0.2)
        for c in ((True, False) if controls else (False,))
    ]
    if joint:
        items += [
            MaskItem((3, 1, 1), 1, view_offset=views, is_control=c, seconds_per_frame=0.1, is_lidar=True)
            for c in ((True, False) if controls else (False,))
        ]
    return MultiviewLayout(tuple(items), backend="maskless", control_attends_sensor=True, **kwargs)


def tensors(spec, num_und=5, device="cpu", dtype=torch.float64):
    torch.manual_seed(4)
    return [
        torch.randn(1, n, heads, 4, device=device, dtype=dtype)
        for n, heads in ((spec.gen_tokens, 4), (spec.gen_tokens, 2), (spec.gen_tokens, 2), (num_und, 2), (num_und, 2))
    ]


def context(spec, qkv):
    q, k, _, ku, _ = qkv
    plan = m.build_maskless_plan(spec, ku.shape[1], q.device, q.shape[2], k.shape[2], q.shape[3])
    scratch = m.make_merge_scratch(q.shape[2], q.shape[3], q.dtype, q.device)
    return MultiviewAttentionContext(spec, {}, {}, (plan, scratch), 2)


def attention_oracle(spec, qkv):
    """Enumerate token semantics independently, concatenate branch keys WITH duplicates."""
    q, k, v, ku, vu = [x[0].double() for x in qkv]
    k, v, ku, vu = [x.repeat_interleave(2, 1) for x in (k, v, ku, vu)]
    anchor = next(x.seconds_per_frame for x in spec.items if not x.is_lidar)
    tokens = []
    for item in spec.items:
        for view in range(item.num_views):
            for frame in range(item.token_shape[0] // item.num_views):
                for _ in range(item.token_shape[1] * item.token_shape[2]):
                    tokens.append(
                        (
                            (item.is_lidar, item.view_offset + view),
                            item.is_control,
                            math.floor((frame + 0.5) * item.seconds_per_frame / anchor + 1e-6),
                        )
                    )
    group_count = len({t[0] for t in tokens})
    result = torch.empty_like(q)
    for index, (group, control, instant) in enumerate(tokens):
        keys = [j for j, t in enumerate(tokens) if t[0] == group]
        if spec.attention_scope == "decomposed" and group_count > 1 and not control:
            keys += [j for j, t in enumerate(tokens) if not t[1] and t[2] == instant]
        caption = list(range(ku.shape[0]))
        if group[0] and not spec.lidar_attends_captions:
            caption = []
        elif not group[0] and spec.caption_lengths:
            start = sum(spec.caption_lengths[: group[1]])
            caption = list(range(start, start + spec.caption_lengths[group[1]]))
        key, value = torch.cat((k[keys], ku[caption])), torch.cat((v[keys], vu[caption]))
        scores = torch.einsum("hd,khd->hk", q[index], key) / math.sqrt(q.shape[-1])
        result[index] = torch.einsum("hk,khd->hd", scores.softmax(-1), value)
    return result[None]


@pytest.mark.parametrize("joint,views", [(False, 1), (False, 2), (True, 1), (True, 2)])
@pytest.mark.parametrize("controls", [True, False])
@pytest.mark.parametrize("scope", ["same_view", "decomposed"])
@pytest.mark.parametrize("captions", [(), (2, 3)])
@pytest.mark.parametrize("lidar_captions", [True, False])
def test_attention_matches_concatenated_key_oracle(
    cpu_kernels, joint, views, controls, scope, captions, lidar_captions
):
    if views == 1 and captions:
        captions = (5,)
    spec = layout(
        joint, views, controls, attention_scope=scope, caption_lengths=captions, lidar_attends_captions=lidar_captions
    )
    qkv = tensors(spec)
    actual = multiview_attention(*qkv, context(spec, qkv))
    torch.testing.assert_close(actual, attention_oracle(spec, qkv), atol=1e-6, rtol=1e-6)


def test_backend_geometry_and_semantics():
    spec = layout()
    assert len(spec.items) == 4
    with pytest.raises(RuntimeError, match="no sparse block geometry"):
        _ = spec.block_sizes
    with pytest.raises(ValueError, match="mixed view offsets"):
        replace(spec, backend="triton")
    for kwargs in (
        {"attention_scope": "all_views"},
        {"control_attends_sensor": False},
        {"decomposed_temporal_window_seconds": 0.1},
    ):
        with pytest.raises(ValueError, match="Maskless attention requires"):
            replace(spec, **kwargs)


def test_batch_two_rejected(cpu_kernels):
    spec = layout()
    qkv = tensors(spec)
    with pytest.raises(ValueError, match="B == 1"):
        multiview_attention(*(x.expand(2, -1, -1, -1) for x in qkv), context(spec, qkv))


@pytest.mark.parametrize("lengths,heads,dim", [([2**31], 1, 1), ([2**30, 2**30], 1, 1), ([2**20], 16, 128)])
def test_int32_boundaries_without_allocating(lengths, heads, dim):
    with pytest.raises(ValueError, match="int32 indexing"):
        m.validate_indexing(lengths, heads, dim)
    m.validate_indexing([2**31 - 1], 1, 1)


def test_explicit_lse_axis_normalization():
    square = torch.arange(16).reshape(4, 4)
    assert torch.equal(m.normalize_varlen_lse(square, 4, 4)[0], square.T)
    with pytest.raises(ValueError, match="Expected FlashAttention LSE"):
        m.normalize_varlen_lse(torch.zeros(1, 4, 4), 4, 4)


def test_zero_caption_groups_are_removed(cpu_kernels):
    spec = layout(caption_lengths=(0, 5), lidar_attends_captions=False)
    qkv = tensors(spec)
    torch.testing.assert_close(
        multiview_attention(*qkv, context(spec, qkv)), attention_oracle(spec, qkv), atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize(
    "joint,scope,chunk_size",
    [
        (False, "same_view", 3),
        (False, "same_view", 4),
        (True, "decomposed", 3),
        (True, "decomposed", 11),
    ],
)
def test_chunk_shapes_and_tail(cpu_kernels, monkeypatch, joint, scope, chunk_size):
    spec = layout(joint=joint, controls=joint, attention_scope=scope)
    qkv = tensors(spec)
    # Force multiple chunks cheaply; production constant is checked separately.
    assert m.MERGE_CHUNK_SIZE == 8192
    monkeypatch.setattr(m, "MERGE_CHUNK_SIZE", chunk_size)
    ctx = context(spec, qkv)
    for buffer in ctx.maskless_plan[1]:
        buffer.fill_(float("nan"))
    seen = []
    chunks_per_call = math.ceil(spec.gen_tokens / chunk_size)

    def merge(outputs, lse_tensors, **kwargs):
        chunk_index = len(seen) % chunks_per_call
        count = min(chunk_size, spec.gen_tokens - chunk_index * chunk_size)
        seen.append(outputs[0].shape[1])
        assert torch.is_inference_mode_enabled()
        assert kwargs == {"use_autograd_fix": True, "torch_compile": True}
        assert len(outputs) == (3 if joint else 2)
        for branch, (out, lse) in enumerate(zip(outputs, lse_tensors, strict=True)):
            assert torch.isfinite(out).all() and torch.isfinite(lse).all()
            assert torch.count_nonzero(out[:, count:]) == 0
            sentinel = 0 if branch == 0 else torch.finfo(lse.dtype).min
            assert (lse[:, count:] == sentinel).all()
        result = merge_oracle(outputs, lse_tensors)
        # A later chunk/call must overwrite every row it consumes, including
        # absent contributions and dummy tail rows, regardless of stale data.
        for buffer in (*outputs, *lse_tensors):
            buffer.fill_(float("nan"))
        return result

    monkeypatch.setattr(sys.modules["natten.functional"], "merge_attentions", merge)
    for _ in range(2):
        actual = multiview_attention(*qkv, ctx)
        torch.testing.assert_close(actual, attention_oracle(spec, qkv), atol=1e-6, rtol=1e-6)
    assert seen == [chunk_size] * (2 * chunks_per_call)


@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_output_preserves_callers_tensor_mode(cpu_kernels, inference, compiled):
    spec = layout()
    qkv = tensors(spec)
    ctx = context(spec, qkv)
    attention = (
        torch.compile(multiview_attention, backend="inductor", fullgraph=True) if compiled else multiview_attention
    )
    with torch.inference_mode() if inference else torch.no_grad():
        result = attention(*qkv, ctx)
        assert result.is_inference() == inference
        if not inference:
            # HSDP/offload callers require outputs with accessible version counters.
            assert isinstance(result._version, int)
        torch.testing.assert_close(result, attention_oracle(spec, qkv), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("compiler_backend", ["eager", "inductor"])
def test_prompt_lengths_do_not_recompile(cpu_kernels, compiler_backend):
    from torch._dynamo.testing import CompileCounterWithBackend

    torch._dynamo.reset()
    counter = CompileCounterWithBackend(compiler_backend)

    # Exercise the actual dispatch boundary, with a small projection as a GEN region.
    def region(q, k, v, ku, vu, ctx):
        return multiview_attention(q + 0.0, k, v, ku, vu, ctx) * 1.0

    compiled = torch.compile(region, backend=counter, fullgraph=True)
    with torch.inference_mode():
        for length in list(range(2, 13)) + [4, 8, 2]:
            for branch_length in (length, length + 3):
                spec = layout(caption_lengths=(1, branch_length - 1))
                qkv = tensors(spec, branch_length)
                for t in qkv[3:]:
                    torch._dynamo.mark_dynamic(t, 1)
                ctx = context(spec, qkv)  # Request cache reset, unequal CFG lengths.
                torch.testing.assert_close(compiled(*qkv, ctx), attention_oracle(spec, qkv), atol=1e-6, rtol=1e-6)
    assert counter.frame_count == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA, real FlashAttention and NATTEN")
def test_gpu_attention():
    spec = layout(caption_lengths=(2, 3))
    qkv = tensors(spec, device="cuda", dtype=torch.bfloat16)
    ctx = replace(context(spec, qkv), fa_version=m.load_maskless_runtime())
    with torch.inference_mode():
        torch.testing.assert_close(
            multiview_attention(*qkv, ctx).double(), attention_oracle(spec, qkv), atol=1e-2, rtol=1e-2
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and real NATTEN")
@pytest.mark.parametrize("branches", [2, 3])
@pytest.mark.parametrize("absent", [float("-inf"), torch.finfo(torch.float32).min])
def test_real_natten_merge(branches, absent):
    m.load_maskless_runtime()
    from natten.functional import merge_attentions

    outputs = [torch.full((1, 8192, 2, 4), float(i + 1), device="cuda") for i in range(branches)]
    lses = [torch.full((1, 8192, 2), math.log(i + 1), device="cuda") for i in range(branches)]
    lses[1][:, :7] = absent
    lses[1][:, 7:14] = -10000
    with torch.inference_mode():
        result, _ = merge_attentions(outputs=outputs, lse_tensors=lses, use_autograd_fix=True, torch_compile=True)
        torch.testing.assert_close(result, merge_oracle(outputs, lses)[0], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("failure", ["missing", "version", "api"])
def test_missing_or_incompatible_natten(monkeypatch, failure):
    m.load_maskless_runtime.cache_clear()
    natten = types.ModuleType("natten")
    functional = types.ModuleType("natten.functional")
    functional.merge_attentions = lambda outputs, lse_tensors, torch_compile, use_autograd_fix: None
    monkeypatch.setitem(sys.modules, "natten", natten)
    monkeypatch.setitem(sys.modules, "natten.functional", functional)
    monkeypatch.setattr(m, "version", lambda _: "0.21.6" if failure != "version" else "0.20.0")
    if failure == "missing":
        monkeypatch.setitem(sys.modules, "natten.functional", None)
    if failure == "api":
        functional.merge_attentions = lambda: None
    with pytest.raises(RuntimeError, match="cosmos3-maskless extra"):
        m.load_maskless_runtime()
    m.load_maskless_runtime.cache_clear()


def test_sparse_honors_lidar_caption_disable():
    from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import (
        PaddedAttentionGeometry,
        build_multiview_flex_metadata,
        multiview_pair_predicate,
    )

    spec = replace(layout(), backend="triton", decomposed_temporal_window_seconds=0.2, lidar_attends_captions=False)
    metadata = build_multiview_flex_metadata(
        spec, PaddedAttentionGeometry(spec.gen_tokens, spec.gen_tokens, 5, 5), "cpu"
    )
    # Camera controls/targets read shared text; LiDAR controls/targets do not.
    assert multiview_pair_predicate(metadata, torch.tensor(0), torch.tensor(0))
    assert not multiview_pair_predicate(metadata, torch.tensor(16), torch.tensor(0))


def test_ulysses_maskless_trims_padding_and_preserves_batch_one(cpu_kernels, monkeypatch):
    from vllm_omni.diffusion.models.cosmos3 import multiview_parallel as parallel

    spec = layout(views=1)
    qkv = tensors(spec)
    q, k, v, ku, vu = qkv
    cp = 2
    length = (spec.gen_tokens + cp - 1) // cp
    padded = [torch.nn.functional.pad(x, (0, 0, 0, 0, 0, length * cp - x.shape[1])) for x in (q, k, v)]
    for rank in range(cp):
        full = [x[:, :, rank * (x.shape[2] // cp) : (rank + 1) * (x.shape[2] // cp)] for x in padded]
        calls = iter(full)

        def exchange(x, group, scatter, gather):
            assert x.shape[0] == 1
            if scatter == 2:
                return next(calls)
            assert x.shape[1] == length * cp
            return x[:, rank * length : (rank + 1) * length]

        monkeypatch.setattr(parallel, "_all_to_all", exchange)
        local_qkv = [x[:, rank * length : (rank + 1) * length] for x in padded] + [ku, vu]
        head_qkv = [x[:, : spec.gen_tokens] for x in full] + [ku[:, :, rank : rank + 1], vu[:, :, rank : rank + 1]]
        ctx = context(spec, head_qkv)
        result = parallel.multiview_ulysses_attention(*local_qkv, ctx, group=object(), rank=rank, world_size=cp)
        expected = multiview_attention(*head_qkv, ctx)
        expected = torch.nn.functional.pad(expected, (0, 0, 0, 0, 0, length * cp - spec.gen_tokens))
        torch.testing.assert_close(result, expected[:, rank * length : (rank + 1) * length])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and real NATTEN")
def test_natten_internal_compile_is_stable_and_matches_unchunked():
    from natten.functional import merge_attentions
    from torch._dynamo.utils import counters

    m.load_maskless_runtime()
    torch._dynamo.reset()
    before = counters["stats"]["unique_graphs"]
    kwargs = dict(use_autograd_fix=True, torch_compile=True)
    with torch.inference_mode():
        warm_out = [torch.randn(1, 8192, 2, 8, device="cuda") for _ in range(3)]
        warm_lse = [torch.randn(1, 8192, 2, device="cuda") for _ in range(3)]
        merge_attentions(outputs=warm_out, lse_tensors=warm_lse, **kwargs)
        warmed = counters["stats"]["unique_graphs"]
        assert warmed > before
        for length in [3, 17, 29, 55, 89, 144, 233, 377, 610, 8193, 16401]:
            outputs = [torch.randn(1, length, 2, 8, device="cuda") for _ in range(3)]
            lses = [torch.randn(1, length, 2, device="cuda") for _ in range(3)]
            expected, _ = merge_attentions(
                outputs=outputs, lse_tensors=lses, use_autograd_fix=True, torch_compile=False
            )
            chunks = []
            for start in range(0, length, 8192):
                count = min(length - start, 8192)
                for branch in range(3):
                    warm_out[branch].zero_()
                    warm_lse[branch].fill_(torch.finfo(torch.float32).min)
                    warm_out[branch][:, :count] = outputs[branch][:, start : start + count]
                    warm_lse[branch][:, :count] = lses[branch][:, start : start + count]
                warm_lse[0][:, count:] = 0
                merged, _ = merge_attentions(outputs=warm_out, lse_tensors=warm_lse, **kwargs)
                chunks.append(merged[:, :count].clone())
            torch.testing.assert_close(torch.cat(chunks, 1), expected)
        assert counters["stats"]["unique_graphs"] == warmed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires reference CUDA environment")
def test_natten_matches_reference_wrapper():
    reference = pytest.importorskip("imaginaire.attention")
    from natten.functional import merge_attentions

    m.load_maskless_runtime()
    outputs = [torch.randn(1, 8192, 2, 16, dtype=torch.bfloat16, device="cuda") for _ in range(3)]
    lses = [torch.randn(1, 8192, 2, device="cuda") for _ in range(3)]
    with torch.inference_mode():
        actual, _ = merge_attentions(outputs=outputs, lse_tensors=lses, torch_compile=True, use_autograd_fix=True)
        expected, _ = reference.merge_attentions(outputs=outputs, lse_tensors=lses, torch_compile=True)
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


def test_float64_midpoint_boundary():
    spec = MultiviewLayout(
        (
            MaskItem((2, 1, 1), 1, seconds_per_frame=0.3),
            MaskItem((3, 1, 1), 1, view_offset=1, is_lidar=True, seconds_per_frame=0.2),
        ),
        backend="maskless",
        control_attends_sensor=True,
    )
    plan = m.build_maskless_plan(spec, 1, torch.device("cpu"), 2, 1, 4)
    # Camera frames map to 0/1; LiDAR midpoints to 0/1/1, including the exact tie.
    assert plan[6].tolist() == [0, 2, 1, 3, 4]
    assert plan[8].tolist() == [0, 2, 5]


def test_worker_pins_fa_selection(monkeypatch):
    m.load_maskless_runtime.cache_clear()
    natten = types.ModuleType("natten")
    functional = types.ModuleType("natten.functional")
    functional.merge_attentions = lambda outputs, lse_tensors, torch_compile, use_autograd_fix: None
    fa = types.ModuleType("vllm_omni.diffusion.attention.backends.utils.fa")
    calls = []

    def resolve():
        calls.append(1)
        return 3

    fa.resolve_vllm_flash_attn_version = resolve
    monkeypatch.setitem(sys.modules, "natten", natten)
    monkeypatch.setitem(sys.modules, "natten.functional", functional)
    monkeypatch.setitem(sys.modules, fa.__name__, fa)
    monkeypatch.setattr(m, "version", lambda _: "0.21.6")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "test GPU")
    try:
        assert m.load_maskless_runtime() == m.load_maskless_runtime() == 3
        assert calls == [1]
    finally:
        m.load_maskless_runtime.cache_clear()


def test_plan_maxima_keep_static_shape():
    for caption_lengths in ((2, 3), (4, 5)):
        spec = layout(caption_lengths=caption_lengths)
        plan = m.build_maskless_plan(spec, sum(caption_lengths), torch.device("cpu"), 4, 2, 4)
        for maxima in plan[5::6]:
            assert maxima.shape == (2,)
            assert maxima.device.type == "cpu"
            assert not getattr(maxima, "_dynamo_dynamic_indices", set())
        # Caption gather lengths must still accommodate different prompts.
        assert getattr(plan[13], "_dynamo_dynamic_indices", set()) == {0}


@pytest.mark.parametrize("planned,actual", [(1, 2), (2, 1), (2, 3)])
@pytest.mark.parametrize("compiled", [False, True])
def test_mismatched_gen_length_rejected_before_attention(cpu_kernels, monkeypatch, planned, actual, compiled):
    spec = MultiviewLayout((MaskItem((planned, 1, 1), 1),), backend="maskless", control_attends_sensor=True)
    ctx = context(spec, tensors(spec))
    qkv = tensors(replace(spec, items=(MaskItem((actual, 1, 1), 1),)))

    def unexpected_kernel(*args, **kwargs):
        pytest.fail("GEN length mismatch must be rejected before launching attention")

    monkeypatch.setattr(
        sys.modules["vllm_omni.diffusion.attention.backends.utils.fa"],
        "vllm_flash_attn_varlen_with_lse",
        unexpected_kernel,
    )
    attention = (
        torch.compile(multiview_attention, backend="inductor", fullgraph=True) if compiled else multiview_attention
    )
    with torch.no_grad(), pytest.raises(ValueError, match="GEN length does not match the request plan"):
        attention(*qkv, ctx)


@pytest.mark.parametrize("branch", [1, 2])
def test_excluded_rows_zero_nonfinite_gather_placeholders(cpu_kernels, monkeypatch, branch):
    spec = layout(lidar_attends_captions=False)
    qkv = tensors(spec)
    ctx = context(spec, qkv)
    excluded = torch.cat(
        [
            torch.full((item.num_tokens,), item.is_control if branch == 1 else item.is_lidar, dtype=torch.bool)
            for item in spec.items
        ]
    )
    calls = 0

    def poison_placeholder(*args, **kwargs):
        nonlocal calls
        out, lse = varlen_oracle(*args, **kwargs)
        if calls == branch:
            out[0] = float("nan")
        calls += 1
        return out, lse

    def check_merge(outputs, lse_tensors, **kwargs):
        assert torch.count_nonzero(outputs[branch][0, : spec.gen_tokens][excluded]) == 0
        assert (lse_tensors[branch][0, : spec.gen_tokens][excluded] == torch.finfo(torch.float32).min).all()
        return merge_oracle(outputs, lse_tensors)

    monkeypatch.setattr(
        sys.modules["vllm_omni.diffusion.attention.backends.utils.fa"],
        "vllm_flash_attn_varlen_with_lse",
        poison_placeholder,
    )
    monkeypatch.setattr(sys.modules["natten.functional"], "merge_attentions", check_merge)
    actual = multiview_attention(*qkv, ctx)
    torch.testing.assert_close(actual[:, excluded], attention_oracle(spec, qkv)[:, excluded], atol=1e-6, rtol=1e-6)
