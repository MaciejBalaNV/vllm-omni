# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Frame visibility, bounded upstream allocation, and installed FA4 numerics."""

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.frame_pages import FramePageState, paged_write_attn
from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionKVBranchSpec
from vllm_omni.experimental.ar_diffusion.kv_cache.config import ARDiffusionKVConfig
from vllm_omni.experimental.ar_diffusion.kv_cache.manager import ARDiffusionKVCache
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState

pytestmark = pytest.mark.core_model


def make_state(page=5, device="cpu", dtype=torch.float32, heads=1, dim=4):
    branches = (ARDiffusionKVBranchSpec("main", 0),)
    page_bytes = 2 * page * heads * dim * torch.empty((), dtype=dtype).element_size() * 2
    cache = ARDiffusionKVCache(
        ARDiffusionKVConfig(enable=True, chunk_size=page, window_chunks=2, sink_chunks=1, gpu_memory_fraction=1.0),
        num_layers=2,
        num_kv_heads=heads,
        head_size=dim,
        dtype=dtype,
        block_size=page,
        available_bytes=page_bytes * 64,
        kv_branches=branches,
        max_model_len=1 << 20,
        session_capacity=1,
        cross_attention_lengths={"text": page + 3},
        frames_per_block=2,
        max_scratch_tokens_per_branch=page + 3,
        device=torch.device(device),
    )
    state = ARDiffusionKVState(cache, "test", {"main": cache.begin_request("test-main")}, num_layers=2)
    return FramePageState(state)


def publish(state, values):
    ctx = state.get_kv_caches("main", len(values) * state.kv_cache.block_size, commit_current=True, frame_causal=True)[
        0
    ].forward_ctx
    for page, value in zip(ctx.current_video_block_ids, values):
        for layer in range(state.num_layers):
            state.kv_cache.key_cache(layer)[page].fill_(value)
            state.kv_cache.value_cache(layer)[page].fill_(-value)
    state.commit_paged_context("main")


@pytest.mark.cpu
def test_bounded_frame_reclamation_and_exact_windows():
    state = make_state()
    cache = state.kv_cache
    free = cache.manager.block_pool.get_num_free_blocks()
    try:
        publish(state, [0])
        for first in range(1, 257, 2):
            ctx = state.get_kv_caches("main", 10, commit_current=True, frame_causal=True)[0].forward_ctx
            for offset in range(2):
                pages, count = ctx.video_block_table(seq_len=(offset + 1) * 5)
                history = [int(cache.key_cache(0)[p, 0, 0, 0]) for p in pages if p not in ctx.current_video_block_ids]
                assert history == [0] + list(range(max(1, first + offset - 2), first))
                assert count == len(pages) * 5
            publish(state, [first, first + 1])
            assert len(cache.window_block_ids(state.adapter("main"))) <= 3
            assert len(cache.block_table(state.adapter("main"))) <= 3
        assert state.adapter("main").absolute_num_computed_tokens == 257 * 5
        assert state.adapter("main").compacted_tokens > 0
    finally:
        state.close()
    assert cache.manager.block_pool.get_num_free_blocks() == free


@pytest.mark.cpu
def test_separate_text_pages_real_length_and_causal_rows():
    state = make_state()
    try:
        publish(state, [0, 1])
        publish(state, [2])
        ctx = state.get_kv_caches("main", 10, commit_current=True, frame_causal=True)[0].forward_ctx
        ctx.prepare("cpu", action_len=8, query_len=10)
        assert ctx.seq_lens.tolist() == [28, 28]
        assert ctx.query_start_loc.tolist() == [0, 5, 10]
        assert ctx.action_slots[0] % 5 == 0
        for row in range(2):
            pages, _ = ctx.video_block_table(seq_len=(row + 1) * 5)
            assert ctx.block_table[row, : len(pages)].tolist() == pages
            assert ctx.block_table[row, len(pages) : len(pages) + 2].tolist() == ctx.action_scratch_block_ids
        assert not set(ctx.current_video_block_ids) & set(ctx.action_scratch_block_ids)
    finally:
        state.close()


@hardware_test(res={"cuda": "B200"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires supported CUDA GPU and FA4")
def test_installed_fa4_matches_exact_dense_attention():
    import torch.nn.functional as F

    state = make_state(page=924, device="cuda", dtype=torch.bfloat16, heads=2, dim=64)
    cache = state.kv_cache
    torch.manual_seed(2026)
    try:
        for _ in range(4):
            ctx = state.get_kv_caches("main", 924, commit_current=True)[0].forward_ctx
            for layer in range(2):
                cache.key_cache(layer)[ctx.current_video_block_ids[0]].normal_()
                cache.value_cache(layer)[ctx.current_video_block_ids[0]].normal_()
            state.commit_paged_context("main")
        ctx = state.get_kv_caches("main", 1848, commit_current=True, frame_causal=True)[0].forward_ctx
        ctx.prepare("cuda", action_len=301, query_len=1848)
        q = torch.randn(1848, 4, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1848, 2, 64, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        tk = torch.randn(301, 2, 64, device="cuda", dtype=torch.bfloat16)
        tv = torch.randn_like(tk)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as trace:
            output = paged_write_attn(ctx.layer_inputs(0), q, k, v, tk, tv, 64**-0.5)
            torch.accelerator.synchronize()
        kernels = [event.name for event in trace.events() if event.device_type == torch.autograd.DeviceType.CUDA]
        assert any("FlashAttentionForwardSm100" in name and "vllm" not in name for name in kernels)
        print("installed_fa4_kernels=" + repr(kernels))
        reference = []
        for row in range(2):
            pages, _ = ctx.video_block_table(seq_len=(row + 1) * 924)
            rk = torch.cat([cache.key_cache(0)[p] for p in pages] + [tk]).repeat_interleave(2, dim=1)
            rv = torch.cat([cache.value_cache(0)[p] for p in pages] + [tv]).repeat_interleave(2, dim=1)
            ref = F.scaled_dot_product_attention(
                q[row * 924 : (row + 1) * 924].float().transpose(0, 1),
                rk.float().transpose(0, 1),
                rv.float().transpose(0, 1),
            )
            reference.append(ref.transpose(0, 1))
        torch.testing.assert_close(output.float(), torch.cat(reference), atol=0.012, rtol=0.025)
        state.commit_paged_context("main")
        assert len(cache.window_block_ids(state.adapter("main"))) <= 3
    finally:
        state.close()


@pytest.mark.cpu
def test_pipeline_binds_actual_upstream_cache_geometry():
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.pipeline_cosmos3_nano_sim_bimanual import (
        Cosmos3NanoSimBimanualPipeline,
    )

    pipeline = object.__new__(Cosmos3NanoSimBimanualPipeline)
    pipeline.transformer = SimpleNamespace(num_hidden_layers=2, num_kv_heads_local=1, head_dim=4)
    pipeline.manifest = SimpleNamespace(
        conditioning_tokens_per_frame=4, chunk_size=2, window_frames=2, sink_frames=1, text_cache_max_len=8
    )
    geometry = SimpleNamespace(tokens_per_frame=lambda _: 5)
    pipeline.resolution_policy = SimpleNamespace(default_resolution=(1, 1), resolve=lambda *args: geometry)
    state = make_state()
    try:
        pipeline._validate_bound_kv_geometry(state, geometry)
        with pipeline.bind_ar_diffusion_state("test", state.state):
            assert isinstance(pipeline._ar_diffusion_kv_state, FramePageState)
        assert pipeline._ar_diffusion_kv_state is None
    finally:
        state.close()


@pytest.mark.cpu
def test_attention_rejects_missing_frame_pages_before_computing():
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.transformer_cosmos3_nano_sim_bimanual import (
        Cosmos3NanoSimBimanualJointAttention,
    )

    with pytest.raises(RuntimeError, match="requires FA4 frame-page inputs"):
        Cosmos3NanoSimBimanualJointAttention.forward(
            SimpleNamespace(),
            torch.zeros(1, 5, 8),
            text_k=torch.zeros(1, 1, 1, 4),
            text_v=torch.zeros(1, 1, 1, 4),
            real_text_kv_len=1,
            freqs_cos=torch.ones(1, 5, 1, 4),
            freqs_sin=torch.zeros(1, 5, 1, 4),
            num_frames=1,
            tokens_per_frame=5,
            action_tokens_per_frame=1,
        )
