# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare real clean-refresh layer math and managed caches across window rolls."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.diffusion.models.cosmos3_nano_sim_bimanual.reference_attention import (
    ReferenceJointAttention,
    append_dense_kv_history,
)
from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.frame_causal_attention import FrameCausalAttentionState
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.transformer_cosmos3_nano_sim_bimanual import (
    Cosmos3NanoSimBimanualGenDecoderLayer,
    Cosmos3NanoSimBimanualTransformer,
)
from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionKVBranchSpec
from vllm_omni.experimental.ar_diffusion.kv_cache import ARDiffusionKVCache, ARDiffusionKVConfig
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]
BLOCK, WIDTH, HEAD, LAYERS = 16, 64, 64, 3


@pytest.fixture(autouse=True)
def cpu_attention_oracle(monkeypatch, request):
    if request.node.get_closest_marker("cpu") is None:
        return
    from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual import transformer_cosmos3_nano_sim_bimanual as module

    def oracle(inputs, q, k, v, tk, tv, scale, **kwargs):
        assert q.device.type == "cpu"
        size = inputs.block_size
        for pool, values, text in [(inputs.key_pool, k, tk), (inputs.value_pool, v, tv)]:
            flat = pool
            flat.index_copy_(0, inputs.video_slots, values)
            flat.index_copy_(0, inputs.action_slots, text)
        result = []
        for row, length in enumerate(inputs.seq_lens.tolist()):
            pages = inputs.block_table[row, : (length + size - 1) // size].long()
            key = inputs.key_pool.unflatten(0, (-1, size))[pages].flatten(0, 1)[:length]
            value = inputs.value_pool.unflatten(0, (-1, size))[pages].flatten(0, 1)[:length]
            start, end = inputs.query_start_loc[row : row + 2].tolist()
            result.append(
                torch.nn.functional.scaled_dot_product_attention(
                    q[start:end].transpose(0, 1),
                    key.transpose(0, 1),
                    value.transpose(0, 1),
                    scale=scale,
                    enable_gqa=True,
                ).transpose(0, 1)
            )
        return torch.cat(result)

    monkeypatch.setattr(module, "paged_write_attn", oracle)


class DenseAttention(nn.Module):
    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), enable_gqa=True
        ).transpose(1, 2)


def layers(device, dtype):
    """Use production forward methods with small weights; no distributed init."""
    result = nn.ModuleList()
    for _ in range(LAYERS):
        attn = ReferenceJointAttention.__new__(ReferenceJointAttention)
        nn.Module.__init__(attn)
        attn.num_heads_local, attn.num_kv_heads_local, attn.head_dim = 4, 2, HEAD
        attn.qk_norm = True
        attn.to_q, attn.to_k, attn.to_v = [nn.Linear(WIDTH, n * HEAD, bias=False) for n in (4, 2, 2)]
        attn.to_out = nn.Linear(4 * HEAD, WIDTH, bias=False)
        attn.norm_q, attn.norm_k = nn.RMSNorm(HEAD), nn.RMSNorm(HEAD)
        attn.norm_q.variance_epsilon = attn.norm_k.variance_epsilon = 1e-6
        attn.attn = DenseAttention()
        layer = Cosmos3NanoSimBimanualGenDecoderLayer.__new__(Cosmos3NanoSimBimanualGenDecoderLayer)
        nn.Module.__init__(layer)
        layer.cross_attention = attn
        layer.input_layernorm = nn.RMSNorm(WIDTH, eps=1e-6)
        layer.post_attention_layernorm = nn.RMSNorm(WIDTH, eps=1e-6)
        layer.mlp = nn.Sequential(nn.Linear(WIDTH, 2 * WIDTH), nn.SiLU(), nn.Linear(2 * WIDTH, WIDTH))
        result.append(layer)
    return result.to(device=device, dtype=dtype).eval()


def state(device, dtype, window, sink, page_size=None):
    cache = ARDiffusionKVCache(
        ARDiffusionKVConfig(enable=True, chunk_size=BLOCK, window_chunks=window, sink_chunks=sink),
        num_layers=LAYERS,
        num_kv_heads=2,
        head_size=HEAD,
        dtype=dtype,
        block_size=BLOCK if page_size is None else page_size,
        max_model_len=4096,
        available_bytes=1 << 25,
        kv_branches=(ARDiffusionKVBranchSpec("main", 0),),
        session_capacity=1,
        frames_per_block=4,
        max_scratch_tokens_per_branch=BLOCK,
        device=device,
    )
    return FrameCausalAttentionState(
        ARDiffusionKVState(cache, "test", {"main": cache.begin_request("main")}, num_layers=LAYERS)
    )


@torch.no_grad()
def run_forward(
    net,
    h,
    text,
    start,
    *,
    paged=None,
    dense=None,
    batched=False,
    window=5,
    sink=0,
    commit=True,
):
    frames = h.shape[1] // BLOCK
    nulls = tuple(i for i in range(frames) if (start + i) % 3 == 0)
    positions = torch.arange(start * BLOCK, (start + frames) * BLOCK, device=h.device).float()
    phases = positions.view(1, -1, 1, 1) * torch.arange(1, HEAD + 1, device=h.device).view(1, 1, 1, -1) * 0.001
    cos, sin = phases.cos().to(h.dtype), phases.sin().to(h.dtype)
    contexts = None
    if paged is not None:
        contexts = paged.get_kv_caches(
            "main",
            seq_len=frames * BLOCK,
            commit_current=commit,
            extra_visible_tokens=BLOCK if batched else frames * BLOCK,
            frame_causal=batched,
        )
        contexts[0].forward_ctx.prepare(device=h.device, action_len=text[0][0].shape[1], query_len=h.shape[1])
    current = []
    for index, layer in enumerate(net):
        h, k, v = layer(
            h,
            text_k=text[index][0],
            text_v=text[index][1],
            real_text_kv_len=text[index][0].shape[1],
            freqs_cos=cos,
            freqs_sin=sin,
            dense_history=None if dense is None else dense[index],
            paged_context=None if contexts is None else contexts[index].to_layer_inputs(),
            num_frames=frames,
            tokens_per_frame=BLOCK,
            action_tokens_per_frame=2,
            null_action_frame_indexes=nulls,
            clean_history_window=(sink, window) if batched else None,
        )
        current.append((k, v))
    if paged is not None:
        paged.commit_paged_context("main")
    elif commit:
        for offset in range(frames):
            dense = append_dense_kv_history(
                dense,
                [
                    (k[:, offset * BLOCK : (offset + 1) * BLOCK], v[:, offset * BLOCK : (offset + 1) * BLOCK])
                    for k, v in current
                ],
                tokens_per_frame=BLOCK,
                sink_frames=sink,
                window_frames=window,
            )
    return h, dense


def compare(*, device, dtype, history, window, sink, frames, paged, page_size=None):
    torch.manual_seed(123)
    net = layers(device, dtype)
    text = [
        (torch.randn(1, 5, 2, HEAD, device=device, dtype=dtype), torch.randn(1, 5, 2, HEAD, device=device, dtype=dtype))
        for _ in net
    ]
    inputs = torch.randn(1, (history + frames + 1) * BLOCK, WIDTH, device=device, dtype=dtype)
    a, b = (
        (state(device, dtype, window, sink, page_size), state(device, dtype, window, sink, page_size))
        if paged
        else (None, None)
    )
    da = db = None
    tolerance = 5e-2 if dtype == torch.bfloat16 else 2e-5
    try:
        for i in range(history):
            x = inputs[:, i * BLOCK : (i + 1) * BLOCK]
            _, da = run_forward(net, x, text, i, paged=a, dense=da, window=window, sink=sink)
            _, db = run_forward(net, x, text, i, paged=b, dense=db, window=window, sink=sink)
        sequential = []
        for i in range(history, history + frames):
            output, da = run_forward(
                net, inputs[:, i * BLOCK : (i + 1) * BLOCK], text, i, paged=a, dense=da, window=window, sink=sink
            )
            sequential.append(output)
        actual, db = run_forward(
            net,
            inputs[:, history * BLOCK : (history + frames) * BLOCK],
            text,
            history,
            paged=b,
            dense=db,
            batched=True,
            window=window,
            sink=sink,
        )
        torch.testing.assert_close(actual, torch.cat(sequential, dim=1), atol=tolerance, rtol=tolerance)
        if paged:
            assert a is not None and b is not None
            assert a.adapter("main").completed_chunks == b.adapter("main").completed_chunks == history + frames
            for layer in range(LAYERS):
                ia = a.kv_cache.window_block_ids(a.adapter("main"))
                ib = b.kv_cache.window_block_ids(b.adapter("main"))
                assert len(ia) == len(ib)
                for accessor in ("key_cache", "value_cache"):
                    ka = getattr(a.kv_cache, accessor)(layer)[ia].flatten(0, 1)
                    kb = getattr(b.kv_cache, accessor)(layer)[ib].flatten(0, 1)
                    tail = a.adapter("main").num_computed_tokens % a.kv_cache.block_size
                    if tail:
                        padding = a.kv_cache.block_size - tail
                        ka, kb = ka[:-padding], kb[:-padding]
                    torch.testing.assert_close(ka, kb, atol=tolerance, rtol=tolerance)
        else:
            assert da is not None and db is not None
            for pair_a, pair_b in zip(da, db, strict=True):
                for ka, kb in zip(pair_a, pair_b, strict=True):
                    torch.testing.assert_close(ka, kb, atol=tolerance, rtol=tolerance)
        # The next denoising call consumes the newly committed history.
        x = inputs[:, -BLOCK:]
        oa, _ = run_forward(net, x, text, history + frames, paged=a, dense=da, window=window, sink=sink, commit=False)
        ob, _ = run_forward(net, x, text, history + frames, paged=b, dense=db, window=window, sink=sink, commit=False)
        torch.testing.assert_close(oa, ob, atol=tolerance, rtol=tolerance)
    finally:
        if a is not None:
            a.close()
        if b is not None:
            b.close()


@pytest.mark.cpu
@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("frames", [1, 3, 4])
@pytest.mark.parametrize(
    "history,window,sink", [(0, 5, 0), (2, 5, 0), (4, 5, 0), (8, 5, 0), (8, 2, 0), (8, 2, 1), (0, 1, 2)]
)
def test_batched_clean_cache_matches_sequential_cpu(history, window, sink, frames, paged):
    compare(
        device=torch.device("cpu"),
        dtype=torch.float32,
        history=history,
        window=window,
        sink=sink,
        frames=frames,
        paged=paged,
    )


@hardware_test(res={"cuda": "B200"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA paged FlashAttention")
@pytest.mark.parametrize("history,window,sink,frames", [(0, 5, 0, 4), (4, 5, 0, 4), (8, 2, 0, 3), (8, 2, 1, 4)])
def test_batched_clean_cache_matches_sequential_cuda(history, window, sink, frames):
    compare(
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        history=history,
        window=window,
        sink=sink,
        frames=frames,
        paged=True,
    )


@pytest.mark.cpu
@pytest.mark.parametrize("nulls", [(), (0,), (1, 3), (0, 1, 2, 3)])
@pytest.mark.parametrize("fps", [15.0, 29.97])
def test_clean_rope_matches_individual_frames(nulls, fps):
    # Return the position IDs themselves to inspect the real packing code.
    def rotary(hidden, *, position_ids):
        ids = position_ids[:, 0].T.unsqueeze(0)
        return ids, torch.zeros_like(ids)

    model = SimpleNamespace(
        manifest=SimpleNamespace(action_tokens_per_frame=4),
        temporal_modality_margin=2,
        base_fps=24.0,
        temporal_compression_factor=4,
        enable_fps_modulation=True,
        language_model=SimpleNamespace(rotary_emb=rotary),
    )
    common = dict(grid_h=2, grid_w=2, real_text_kv_len=5, fps=fps)
    hidden = torch.empty(1, 32, WIDTH)
    batched, _ = Cosmos3NanoSimBimanualTransformer._current_rope(
        model,
        hidden,
        frame_start=7,
        num_frames=4,
        null_action_frame_indexes=nulls,
        frame_causal=True,
        **common,
    )
    sequential = [
        Cosmos3NanoSimBimanualTransformer._current_rope(
            model,
            hidden[:, :8],
            frame_start=7 + i,
            num_frames=1,
            null_action_frame_indexes=(0,) if i in nulls else (),
            **common,
        )[0]
        for i in range(4)
    ]
    torch.testing.assert_close(batched, torch.cat(sequential, dim=1), rtol=0, atol=0)


@pytest.mark.cpu
def test_non_multiple_of_16_frame_geometry_cpu(monkeypatch):
    monkeypatch.setitem(globals(), "BLOCK", 394)
    compare(
        device=torch.device("cpu"),
        dtype=torch.float32,
        history=1,
        window=5,
        sink=0,
        frames=2,
        paged=True,
        page_size=394,
    )
