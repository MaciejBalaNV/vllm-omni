# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Frame-page FA4 attention using upstream AR-Diffusion allocation/lifetime.

Only current frames are staged. Once every layer finishes a clean forward,
publish those pages through the upstream allocator, which reclaims whole old
frames and compacts metadata. History is never gathered or copied for attention.
"""

from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import version
from typing import NamedTuple

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


@lru_cache(maxsize=1)
def fa4_paged_func():
    try:
        from flash_attn.cute import flash_attn_varlen_func
    except ImportError as exc:
        raise RuntimeError("Cosmos3 Sim requires flash-attn-4 with the CuTe paged-attention API") from exc
    logger.info(
        "Cosmos3 Sim frame pages: flash-attn-4=%s; flash_attn.cute.flash_attn_varlen_func; no fallback",
        version("flash-attn-4"),
    )
    return flash_attn_varlen_func


class ARDiffusionPagedLayerInputs(NamedTuple):
    key_pool: torch.Tensor
    value_pool: torch.Tensor
    video_slots: torch.Tensor
    action_slots: torch.Tensor
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    max_query_len: int
    max_seq_len: int


@torch.library.custom_op("vllm_omni::cosmos_sim_frame_attention", mutates_args=("key_pool", "value_pool"))
def _frame_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    text_key: torch.Tensor,
    text_value: torch.Tensor,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    video_slots: torch.Tensor,
    action_slots: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    max_query_len: int,
    max_seq_len: int,
    scale: float,
) -> torch.Tensor:
    if not query.is_cuda:
        raise RuntimeError("Cosmos3 Sim frame attention requires CUDA and installed FA4; no CPU/dense fallback")
    flat_key = key_pool.view(-1, *key_pool.shape[2:])
    flat_value = value_pool.view(-1, *value_pool.shape[2:])
    flat_key.index_copy_(0, video_slots, key)
    flat_value.index_copy_(0, video_slots, value)
    flat_key.index_copy_(0, action_slots, text_key)
    flat_value.index_copy_(0, action_slots, text_value)
    output, _ = fa4_paged_func()(
        query,
        key_pool,
        value_pool,
        cu_seqlens_q=query_start_loc,
        seqused_k=seq_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_seq_len,
        page_table=block_table,
        softmax_scale=scale,
        causal=False,
    )
    return output


@_frame_attention.register_fake
def _fake_frame_attention(
    query,
    key,
    value,
    text_key,
    text_value,
    key_pool,
    value_pool,
    video_slots,
    action_slots,
    block_table,
    query_start_loc,
    seq_lens,
    max_query_len,
    max_seq_len,
    scale,
):
    return torch.empty_like(query)


def paged_write_attn(inputs, query, key, value, text_key, text_value, scale, *, framewise_attention=False):
    return _frame_attention(query, key, value, text_key, text_value, *inputs, scale)


def _device(value, device, dtype=torch.int32):
    tensor = torch.tensor(value, dtype=dtype)
    if torch.device(device).type == "cuda":
        return tensor.pin_memory().to(device, non_blocking=True)
    return tensor.to(device)


class FramePageForwardContext:
    def __init__(self, state, branch, seq_len, commit_current, frame_causal):
        self.state = state
        self.kv_cache = state.kv_cache
        self.kv_branch = branch
        self.adapter = state.adapter(branch)
        self.seq_len = seq_len
        self.commit_current = commit_current
        self.frame_causal = frame_causal
        self.block_size = self.kv_cache.block_size
        self.chunk_size = self.kv_cache.spec.chunk_size
        if self.block_size != self.chunk_size or seq_len <= 0 or seq_len % self.block_size:
            raise ValueError("Cosmos3 Sim requires exactly one complete latent frame per page")
        self.num_frames = seq_len // self.block_size
        if self.num_frames > self.kv_cache.frames_per_block:
            raise ValueError("Current frame span exceeds the declared scratch frame capacity")
        self.current_video_block_ids = self.kv_cache.scratch_block_ids(branch, 0, self.num_frames)
        self._history_tokens = self.adapter.num_computed_tokens
        self._prepared = False

    def video_block_table(self, device=None, seq_len=None):
        """Return physical pages in sink/recent/current order, entirely on host."""
        history_frames = self._history_tokens // self.block_size
        current_frames = self.num_frames if seq_len is None else seq_len // self.block_size
        end = history_frames + current_frames
        sink = min(self.kv_cache.spec.sink_chunks, end)
        recent = self.kv_cache.spec.window_chunks
        start = max(sink, end - recent - (1 if self.frame_causal else current_frames))
        indices = [*range(sink), *range(start, end)]
        old = [index for index in indices if index < history_frames]
        pages = dict(zip(old, self.kv_cache.block_ids_at(self.adapter, old)))
        blocks = [
            pages[index] if index < history_frames else self.current_video_block_ids[index - history_frames]
            for index in indices
        ]
        if self.kv_cache.null_block_id in blocks:
            raise RuntimeError("An evicted frame was selected for Cosmos3 Sim attention")
        return blocks, len(blocks) * self.block_size

    def prepare(self, device, action_len, query_len):
        if self._prepared:
            return
        if query_len != self.seq_len or action_len <= 0:
            raise ValueError("Invalid frame-page query or text length")
        size = self.block_size
        self.action_scratch_block_ids = self.kv_cache.scratch_block_ids(
            self.kv_branch,
            self.kv_cache.frames_per_block,
            (action_len + size - 1) // size,
        )
        self.video_slots = _device(
            [page * size + i for page in self.current_video_block_ids for i in range(size)], device, torch.int64
        )
        self.action_slots = _device(
            [self.action_scratch_block_ids[i // size] * size + i % size for i in range(action_len)], device, torch.int64
        )
        views = (
            [self.video_block_table(device, end) for end in range(size, self.seq_len + 1, size)]
            if self.frame_causal
            else [self.video_block_table(device)]
        )
        capacity = (
            self.kv_cache.spec.sink_chunks
            + self.kv_cache.spec.window_chunks
            + self.kv_cache.frames_per_block
            + (action_len + size - 1) // size
        )
        rows = [pages + self.action_scratch_block_ids for pages, _ in views]
        self.block_table = _device(
            [row + [self.kv_cache.null_block_id] * (capacity - len(row)) for row in rows], device
        )
        self.seq_lens = _device([length + action_len for _, length in views], device)
        self.max_query_len = size if self.frame_causal else self.seq_len
        self.query_start_loc = _device(list(range(0, self.seq_len + 1, self.max_query_len)), device)
        self.max_seq_len = capacity * size
        self._prepared = True

    def layer_inputs(self, layer_idx):
        if not self._prepared:
            raise RuntimeError("Frame-page metadata must be prepared before compiled blocks execute")
        return ARDiffusionPagedLayerInputs(
            self.kv_cache.key_cache(layer_idx),
            self.kv_cache.value_cache(layer_idx),
            self.video_slots,
            self.action_slots,
            self.block_table,
            self.query_start_loc,
            self.seq_lens,
            self.max_query_len,
            self.max_seq_len,
        )


@dataclass
class FramePageLayerContext:
    layer_idx: int
    forward_ctx: FramePageForwardContext

    def to_layer_inputs(self):
        return self.forward_ctx.layer_inputs(self.layer_idx)


class FramePageState:
    """Model-local attention adapter; upstream retains ownership of all pages."""

    def __init__(self, state):
        self.state = state
        self.kv_cache = state.kv_cache
        self.pending = {}
        if self.kv_cache.block_size != self.kv_cache.spec.chunk_size:
            raise ValueError("Cosmos3 Sim only supports one page per latent frame")
        if self.kv_cache.history_staging:
            raise ValueError("Cosmos3 Sim FA4 does not support gathered history staging")

    def __getattr__(self, name):
        return getattr(self.state, name)

    def get_kv_caches(self, branch, seq_len, commit_current=False, *, extra_visible_tokens=None, frame_causal=False):
        context = FramePageForwardContext(self, branch, seq_len, commit_current, frame_causal)
        self.pending[branch] = context
        return [FramePageLayerContext(index, context) for index in range(self.num_layers)]

    def commit_paged_context(self, branch):
        context = self.pending.pop(branch)
        if not context.commit_current:
            return
        # Every layer has finished reading old history before upstream may
        # evict it. Copy only newly computed frames; never compact/gather K/V.
        for source in context.current_video_block_ids:
            table = self.kv_cache.allocate_chunk(context.adapter)
            destination = table[context.adapter.num_computed_tokens // context.block_size]
            for layer in range(self.num_layers):
                self.kv_cache.key_cache(layer)[destination].copy_(self.kv_cache.key_cache(layer)[source])
                self.kv_cache.value_cache(layer)[destination].copy_(self.kv_cache.value_cache(layer)[source])
            self.kv_cache.commit_chunk(context.adapter)
            self.state._committed[branch] += context.block_size
