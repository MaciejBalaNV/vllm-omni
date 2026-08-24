# SPDX-License-Identifier: Apache-2.0
"""Paged self-attention helpers for AR-Diffusion KV reuse."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, NamedTuple

import torch

from vllm_omni.experimental.ar_diffusion.kv_cache.paged import compute_slot_mapping

_LAYER_IDX_TENSORS: dict[int, torch.Tensor] = {}


def _layer_idx_tensor(layer_idx: int) -> torch.Tensor:
    t = _LAYER_IDX_TENSORS.get(layer_idx)
    if t is None:
        t = torch.tensor(layer_idx, dtype=torch.int64)
        _LAYER_IDX_TENSORS[layer_idx] = t
    return t


class ARDiffusionPagedLayerInputs(NamedTuple):
    """Compiled-region payload for one layer's paged self-attention.

    A NamedTuple of plain tensors + ints so ``torch.compile`` treats every field
    as a pytree graph input (no object-attribute guards, no recompiles when only
    tensor *values* change). All layers of one KV branch forward share the same
    metadata tensor objects, built once by ``prepare()``.

    ``layer_idx`` is a 0-dim CPU tensor, NOT a python int: all 40 DiT blocks
    share one compiled code object, and an int here becomes a per-layer dynamo
    value guard (``layer_idx == k``) — 40 cache variants that blow the
    recompile limit. A tensor input guards on shape/dtype only, so one graph
    serves every layer.
    """

    layer_idx: torch.Tensor
    key_pool: torch.Tensor
    value_pool: torch.Tensor
    block_size: int
    seq_len: int
    video_slots: torch.Tensor
    action_slots: torch.Tensor
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    max_query_len: int
    max_seq_len: int


@dataclass
class ARDiffusionPagedForwardContext:
    """Mutable KV-branch state shared by all layer contexts in one forward."""

    kv_cache: Any
    adapter: Any
    kv_branch: str
    history_block_ids: list[int]
    seq_len: int
    commit_current: bool
    max_video_tokens: int
    current_video_block_ids: list[int] = field(default_factory=list)
    current_video_slot_mapping: torch.Tensor | None = None
    action_scratch_block_ids: list[int] = field(default_factory=list)
    action_slot_mapping: torch.Tensor | None = None
    query_len: int = 0
    kv_len: int = 0
    _allocated_video: bool = False
    _committed: bool = False
    _action_len: int = 0
    # Set once by prepare(); shared by all layers of the KV branch forward.
    block_table: torch.Tensor | None = None
    query_start_loc: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None
    max_query_len: int = 0
    max_seq_len: int = 0
    _prepared: bool = False

    @property
    def block_size(self) -> int:
        return int(self.kv_cache.block_size)

    @property
    def num_current_video_blocks(self) -> int:
        if self.seq_len % self.block_size != 0:
            raise AssertionError(
                "AR-Diffusion paged attention expects frame-aligned seq_len "
                f"(multiple of block_size={self.block_size}), got {self.seq_len}"
            )
        return self.seq_len // self.block_size

    def ensure_video_slots(self, device: torch.device) -> None:
        """Allocate/write targets for the current video tokens, once per KV branch."""
        if self._allocated_video:
            return

        n_blocks = self.num_current_video_blocks
        if self.commit_current:
            start = int(self.adapter.num_computed_tokens)
            self.kv_cache.allocate_token_slots(self.adapter, self.seq_len)
            table = self.kv_cache.block_table(self.adapter)
            start_block = start // self.block_size
            self.current_video_block_ids = [int(b) for b in table[start_block : start_block + n_blocks]]
            positions = torch.arange(start, start + self.seq_len, dtype=torch.long)
            self.current_video_slot_mapping = compute_slot_mapping(table, positions, self.block_size).to(device=device)
        else:
            self.current_video_block_ids = self.kv_cache.scratch_block_ids(self.kv_branch, 0, n_blocks)
            positions = torch.arange(self.seq_len, dtype=torch.long)
            self.current_video_slot_mapping = compute_slot_mapping(
                self.current_video_block_ids,
                positions,
                self.block_size,
            ).to(device=device)
        self._allocated_video = True

    @property
    def num_layers(self) -> int:
        return int(self.kv_cache.num_layers)

    def _validate_layer_idx(self, layer_idx: int) -> int:
        layer_idx = int(layer_idx)
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(f"AR-Diffusion layer index {layer_idx} is outside [0, {self.num_layers})")
        return layer_idx

    def visible_history_block_ids(self) -> list[int]:
        """History snapshot trimmed to the sink + sliding-window capacity.

        The manager evicts lazily — out-of-window blocks are only released by
        the *next* allocation — so between a commit and the following forward
        the snapshot can hold up to one chunk beyond the window. Attention must
        never see that overhang: trim to the sink prefix plus the window tail,
        the shape the dense oracle maintains eagerly and the pool converges to.
        """

        history = self.history_block_ids
        max_history_blocks = self.max_video_tokens // self.block_size
        if len(history) <= max_history_blocks:
            return list(history)
        sink_blocks = int(self.kv_cache.spec.sink_chunks)
        tail_blocks = max_history_blocks - sink_blocks
        visible = list(history[:sink_blocks])
        if tail_blocks > 0:
            visible += history[-tail_blocks:]
        return visible

    def gather_history(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather one layer's immutable history snapshot into dense K/V.

        ``history_block_ids`` is captured before any current-page allocation.
        Reading only that snapshot is load-bearing: the manager's live block
        table also contains in-flight allocations after the first layer writes.
        The snapshot is additionally trimmed to the visible sink + window span
        (see :meth:`visible_history_block_ids`).
        """

        layer_idx = self._validate_layer_idx(layer_idx)
        key_cache = self.kv_cache.key_cache(layer_idx)
        value_cache = self.kv_cache.value_cache(layer_idx)
        visible_block_ids = self.visible_history_block_ids()
        if not visible_block_ids:
            shape = (1, 0, int(self.kv_cache.num_kv_heads), int(self.kv_cache.head_size))
            return key_cache.new_empty(shape), value_cache.new_empty(shape)

        block_ids = torch.as_tensor(
            visible_block_ids,
            dtype=torch.long,
            device=key_cache.device,
        )
        keys = key_cache.index_select(0, block_ids).reshape(
            1,
            -1,
            int(self.kv_cache.num_kv_heads),
            int(self.kv_cache.head_size),
        )
        values = value_cache.index_select(0, block_ids).reshape_as(keys)
        return keys, values

    def _validate_layer_write(self, layer_idx: int) -> int:
        if not self.commit_current:
            raise RuntimeError(
                "AR-Diffusion write-only KV updates require commit_current=True; "
                "non-committing denoise contexts are read-only"
            )
        layer_idx = self._validate_layer_idx(layer_idx)
        if layer_idx in self._written_layers:
            raise RuntimeError(f"AR-Diffusion current-page K/V for layer {layer_idx} was written more than once")
        return layer_idx

    def write_only(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Write one layer's current frame K/V without running attention.

        This is the gathered-dense counterpart of the fused paged op.  It is
        intentionally unavailable on scratch/non-committing contexts so noisy
        denoise steps cannot accidentally enter the persistent pool.
        """

        if not self.commit_current:
            raise RuntimeError(
                "AR-Diffusion write_only() requires commit_current=True; non-committing denoise contexts are read-only"
            )
        layer_idx = self._validate_layer_write(layer_idx)
        if key.device != value.device:
            raise ValueError(f"AR-Diffusion write_only() K/V devices differ: {key.device} != {value.device}")
        expected = (
            self.seq_len,
            int(self.kv_cache.num_kv_heads),
            int(self.kv_cache.head_size),
        )
        if key.ndim == 4 and key.shape[0] == 1:
            key = key[0]
        if value.ndim == 4 and value.shape[0] == 1:
            value = value[0]
        if tuple(key.shape) != expected or tuple(value.shape) != expected:
            raise ValueError(
                "AR-Diffusion write_only() expected K/V shape "
                f"{expected} (or batch-prefixed (1, ...)), got "
                f"{tuple(key.shape)} and {tuple(value.shape)}"
            )

        self.ensure_video_slots(key.device)
        assert self.current_video_slot_mapping is not None
        self.kv_cache.write_token_kv(
            layer_idx,
            self.current_video_slot_mapping,
            key,
            value,
        )
        self._written_layers.add(layer_idx)

    def validate_commit_complete(self) -> None:
        """Fail closed unless every configured layer wrote the current page."""

        if not self.commit_current:
            return
        expected = set(range(self.num_layers))
        if self._written_layers != expected:
            missing = sorted(expected - self._written_layers)
            extra = sorted(self._written_layers - expected)
            detail = f"missing layers {missing}"
            if extra:
                detail += f", unexpected layers {extra}"
            raise RuntimeError("AR-Diffusion cannot commit an incomplete current page: " + detail)

    def ensure_action_slots(self, action_len: int, device: torch.device) -> None:
        """Reserve scratch slots for action/state K/V, if present."""
        if action_len <= 0:
            self.action_scratch_block_ids = []
            self.action_slot_mapping = torch.empty(0, dtype=torch.long, device=device)
            self._action_len = 0
            return

        self.ensure_video_slots(device)
        if self.action_slot_mapping is not None and self._action_len == action_len:
            return

        action_blocks = (action_len + self.block_size - 1) // self.block_size
        scratch_offset = 0 if self.commit_current else len(self.current_video_block_ids)
        self.action_scratch_block_ids = self.kv_cache.scratch_block_ids(
            self.kv_branch,
            scratch_offset,
            action_blocks,
        )
        positions = torch.arange(action_len, dtype=torch.long)
        self.action_slot_mapping = compute_slot_mapping(
            self.action_scratch_block_ids,
            positions,
            self.block_size,
        ).to(device=device)
        self._action_len = action_len

    def video_block_table(self, device: torch.device) -> tuple[list[int], int]:
        self.ensure_video_slots(device)
        if self.max_video_tokens % self.block_size != 0:
            raise AssertionError(
                "AR-Diffusion paged attention requires max_video_tokens to be block-aligned, "
                f"got max_video_tokens={self.max_video_tokens}, block_size={self.block_size}"
            )
        all_video_blocks = self.history_block_ids + self.current_video_block_ids
        max_video_blocks = self.max_video_tokens // self.block_size
        sink_blocks = int(self.kv_cache.spec.sink_chunks)
        if len(all_video_blocks) <= max_video_blocks:
            visible_video_blocks = all_video_blocks
        else:
            tail_blocks = max_video_blocks - sink_blocks
            visible_video_blocks = all_video_blocks[:sink_blocks]
            if tail_blocks:
                visible_video_blocks += all_video_blocks[-tail_blocks:]
        video_len = len(visible_video_blocks) * self.block_size
        return visible_video_blocks, video_len

    def build_block_table(
        self,
        *,
        action_len: int,
        query_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """Build FlashAttention block-table metadata for one self-attn call.

        The block table is tail-padded to a fixed width and ``max_seq_len`` is a
        constant upper bound, so across window growth only tensor *values* change
        — tensor shapes and the int consts stay stable for ``torch.compile``
        (and, later, CUDA-graph capture). The kernel only dereferences the first
        ``ceil(seq_lens/block_size)`` entries, so padding is never read.
        """
        video_blocks, video_len = self.video_block_table(device)
        self.ensure_action_slots(action_len, device)
        action_blocks = self.action_scratch_block_ids if action_len > 0 else []
        block_ids = video_blocks + action_blocks
        if not block_ids:
            raise RuntimeError("AR-Diffusion paged attention needs at least current video KV blocks")

        # Fixed capacity: full visible video window + one action-capacity block.
        action_capacity_blocks = max(1, (action_len + self.block_size - 1) // self.block_size)
        width = max(self.max_video_tokens // self.block_size + action_capacity_blocks, len(block_ids))
        padded = block_ids + [0] * (width - len(block_ids))

        self.query_len = int(query_len)
        self.kv_len = int(video_len + action_len)
        max_seq_len = int(self.max_video_tokens + action_capacity_blocks * self.block_size)
        block_table = torch.tensor([padded], dtype=torch.int32, device=device)
        query_start_loc = torch.tensor([0, self.query_len], dtype=torch.int32, device=device)
        seq_lens = torch.tensor([self.kv_len], dtype=torch.int32, device=device)
        return block_table, query_start_loc, seq_lens, self.query_len, max_seq_len

    def prepare(self, device: torch.device, action_len: int, query_len: int) -> None:
        """Host-side, once-per-KV-branch setup (called OUTSIDE torch.compile).

        Allocates the current video/action slots (still lazy: only the KV branch a
        CFG-parallel rank actually runs reaches its ``_forward_blocks``), builds
        the padded block-table metadata ONCE for all layers, and publishes the
        pool registry for the fused custom op. The compiled per-layer code then
        only consumes prebuilt tensors via ``ARDiffusionPagedLayerInputs``.
        """
        if getattr(self, "_prepared", False):
            return
        self.ensure_video_slots(device)
        (
            self.block_table,
            self.query_start_loc,
            self.seq_lens,
            self.max_query_len,
            self.max_seq_len,
        ) = self.build_block_table(action_len=action_len, query_len=query_len, device=device)
        if self.action_slot_mapping is None:
            self.action_slot_mapping = torch.empty(0, dtype=torch.long, device=device)
        self._prepared = True

    def layer_inputs(self, layer_idx: int) -> ARDiffusionPagedLayerInputs:
        if not getattr(self, "_prepared", False):
            raise RuntimeError("ARDiffusionPagedForwardContext.layer_inputs() before prepare()")
        return ARDiffusionPagedLayerInputs(
            layer_idx=_layer_idx_tensor(layer_idx),
            key_pool=self.kv_cache._k_pools[layer_idx],
            value_pool=self.kv_cache._v_pools[layer_idx],
            block_size=int(self.kv_cache.block_size),
            seq_len=int(self.seq_len),
            video_slots=self.current_video_slot_mapping,
            action_slots=self.action_slot_mapping,
            block_table=self.block_table,
            query_start_loc=self.query_start_loc,
            seq_lens=self.seq_lens,
            max_query_len=int(self.max_query_len),
            max_seq_len=int(self.max_seq_len),
        )

    def mark_committed(self) -> None:
        self._committed = True


@dataclass
class ARDiffusionPagedLayerContext:
    """Layer-specific handle passed through a model's ``kv_cache`` slot."""

    is_ar_diffusion_paged_context: ClassVar[bool] = True
    layer_idx: int
    forward_ctx: ARDiffusionPagedForwardContext

    @property
    def kv_cache(self):
        return self.forward_ctx.kv_cache

    @property
    def adapter(self):
        return self.forward_ctx.adapter

    @property
    def kv_branch(self) -> str:
        return self.forward_ctx.kv_branch

    @property
    def history_block_ids(self) -> list[int]:
        return self.forward_ctx.history_block_ids

    @property
    def current_video_block_ids(self) -> list[int]:
        return self.forward_ctx.current_video_block_ids

    @property
    def current_video_slot_mapping(self) -> torch.Tensor | None:
        return self.forward_ctx.current_video_slot_mapping

    @property
    def action_scratch_block_ids(self) -> list[int]:
        return self.forward_ctx.action_scratch_block_ids

    @property
    def action_slot_mapping(self) -> torch.Tensor | None:
        return self.forward_ctx.action_slot_mapping

    @property
    def seq_len(self) -> int:
        return self.forward_ctx.seq_len

    @property
    def query_len(self) -> int:
        return self.forward_ctx.query_len

    @property
    def kv_len(self) -> int:
        return self.forward_ctx.kv_len

    @property
    def commit_current(self) -> bool:
        return self.forward_ctx.commit_current

    def to_layer_inputs(self) -> ARDiffusionPagedLayerInputs:
        """Compiled-region payload; requires ``forward_ctx.prepare()`` first."""
        return self.forward_ctx.layer_inputs(self.layer_idx)

    def gather_history(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather this layer's dense history from the captured block snapshot."""

        return self.forward_ctx.gather_history(self.layer_idx)

    def write_only(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Persist this layer's current-frame K/V without running attention."""

        self.forward_ctx.write_only(self.layer_idx, key, value)


def is_ar_diffusion_paged_context(value: object) -> bool:
    return isinstance(value, ARDiffusionPagedLayerContext)


def _reference_paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    softmax_scale: float,
    *,
    causal: bool,
) -> torch.Tensor:
    if causal:
        raise NotImplementedError("AR-Diffusion paged self-attention uses causal=False")
    outs: list[torch.Tensor] = []
    block_size = key_cache.shape[1]
    for i in range(seq_lens.shape[0]):
        q_start = int(query_start_loc[i].item())
        q_end = int(query_start_loc[i + 1].item())
        kv_len = int(seq_lens[i].item())
        q = query[q_start:q_end]
        positions = torch.arange(kv_len, device=query.device)
        logical_blocks = torch.div(positions, block_size, rounding_mode="floor")
        offsets = positions % block_size
        physical_blocks = block_table[i, logical_blocks].long()
        k = key_cache[physical_blocks, offsets]
        v = value_cache[physical_blocks, offsets]
        scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * float(softmax_scale)
        probs = torch.softmax(scores, dim=-1).to(v.dtype)
        outs.append(torch.einsum("hqk,khd->qhd", probs, v))
    return torch.cat(outs, dim=0)


_FA_VERSION_BY_HEAD_SIZE: dict[int, int] = {}


def _resolve_fa_version(head_size: int) -> int:
    # get_flash_attn_version -> current_platform.get_device_capability() is not
    # dynamo-traceable, and the answer is fixed per head size for the process.
    version = _FA_VERSION_BY_HEAD_SIZE.get(head_size)
    if version is None:
        try:
            from vllm.v1.attention.backends.fa_utils import get_flash_attn_version

            version = int(get_flash_attn_version(requires_alibi=False, head_size=head_size) or 2)
        except Exception:
            version = 2
        _FA_VERSION_BY_HEAD_SIZE[head_size] = version
    return version


def _rocm_flash_attn_varlen_func():
    """Resolve a ROCm varlen kernel, preferring AITER when available.

    The caller gathers paged KV into packed tensors before invoking this
    function, avoiding AITER releases whose ``block_table`` kernel is broken.
    """
    try:
        from aiter import flash_attn_varlen_func

        return flash_attn_varlen_func
    except ImportError:
        from flash_attn import flash_attn_varlen_func

        return flash_attn_varlen_func


def ar_diffusion_paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    max_query_len: int,
    max_seq_len: int,
    softmax_scale: float,
    causal: bool = False,
) -> torch.Tensor:
    """Run non-causal paged attention over a vLLM block table.

    ``query`` may be ``(B, L, H, D)`` or already flattened as ``(T, H, D)``.
    ``key_cache`` / ``value_cache`` are ``(num_blocks, block_size, H, D)``.
    """
    batched = query.dim() == 4
    if batched:
        batch, q_len = query.shape[:2]
        query_flat = query.reshape(batch * q_len, *query.shape[2:])
    else:
        query_flat = query

    if not query_flat.is_cuda:
        out = _reference_paged_attention(
            query_flat,
            key_cache,
            value_cache,
            block_table,
            query_start_loc,
            seq_lens,
            softmax_scale,
            causal=causal,
        )
    elif torch.version.hip is not None:
        # vllm.vllm_flash_attn contains CUDA-only extensions. ROCm's AITER and
        # upstream flash-attn expose the standard cu_seqlens_k API instead of
        # vLLM's seqused_k/fa_version API. The ROCm flash-attn paged kernel also
        # requires 128-token blocks, while AR-Diffusion uses frame-aligned
        # 16-token blocks, so gather the visible blocks on-device first.
        flash_attn_varlen_func = _rocm_flash_attn_varlen_func()
        cu_seqlens_k = torch.cat([seq_lens.new_zeros(1), torch.cumsum(seq_lens, dim=0, dtype=torch.int32)])
        positions = torch.arange(int(max_seq_len), device=query_flat.device)
        logical_blocks = torch.div(positions, key_cache.shape[1], rounding_mode="floor")
        offsets = positions % key_cache.shape[1]
        physical_blocks = block_table[:, logical_blocks].long()
        gathered_k = key_cache[physical_blocks, offsets]
        gathered_v = value_cache[physical_blocks, offsets]
        valid = positions.unsqueeze(0) < seq_lens.unsqueeze(1)
        packed_k = gathered_k[valid]
        packed_v = gathered_v[valid]
        out = flash_attn_varlen_func(
            q=query_flat,
            k=packed_k,
            v=packed_v,
            cu_seqlens_q=query_start_loc,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=int(max_query_len),
            max_seqlen_k=int(max_seq_len),
            softmax_scale=float(softmax_scale),
            causal=causal,
        )
    else:
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        fa_version = _resolve_fa_version(query_flat.shape[-1])

        out = torch.empty_like(query_flat)
        flash_attn_varlen_func(
            q=query_flat,
            k=key_cache,
            v=value_cache,
            out=out,
            cu_seqlens_q=query_start_loc,
            max_seqlen_q=int(max_query_len),
            seqused_k=seq_lens,
            max_seqlen_k=int(max_seq_len),
            softmax_scale=float(softmax_scale),
            causal=causal,
            block_table=block_table,
            fa_version=fa_version,
        )

    if batched:
        return out.reshape(query.shape)
    return out


# ── Fused write+attend custom op (torch.compile-safe) ──────────────────────
#
# One opaque op per layer keeps the compiled DiT block fullgraph: dynamo treats
# it as a single graph node (no eager island, no graph breaks), and the K/V slot
# writes happen inside the op so write→read ordering with the block-table kernel
# is internal. The flat pools are explicit mutable inputs: Inductor/CUDA Graph
# must track their storage lifetime instead of observing an undeclared mutation
# through a process-global registry.
def _paged_write_attn_impl(
    query: torch.Tensor,
    k_curr: torch.Tensor,
    v_curr: torch.Tensor,
    k_act: torch.Tensor | None,
    v_act: torch.Tensor | None,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    block_size: int,
    video_slots: torch.Tensor,
    action_slots: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    max_query_len: int,
    max_seq_len: int,
    softmax_scale: float,
) -> torch.Tensor:
    key_pool[video_slots] = k_curr.to(key_pool.dtype)
    value_pool[video_slots] = v_curr.to(value_pool.dtype)
    if k_act is not None and v_act is not None and k_act.shape[0] > 0:
        key_pool[action_slots] = k_act.to(key_pool.dtype)
        value_pool[action_slots] = v_act.to(value_pool.dtype)
    key_cache = key_pool.unflatten(0, (-1, block_size))
    value_cache = value_pool.unflatten(0, (-1, block_size))
    return ar_diffusion_paged_attention(
        query,
        key_cache,
        value_cache,
        block_table=block_table,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        softmax_scale=softmax_scale,
        causal=False,
    )


# hasattr guard keeps registration idempotent across test re-imports that pop
# the module from sys.modules (same as sage_attn3.py).
if not hasattr(torch.ops.vllm_omni, "ar_diffusion_paged_write_attn"):

    @torch.library.custom_op(
        "vllm_omni::ar_diffusion_paged_write_attn",
        mutates_args=("key_pool", "value_pool"),
    )
    def _paged_write_attn_op(
        query: torch.Tensor,
        k_curr: torch.Tensor,
        v_curr: torch.Tensor,
        k_act: torch.Tensor | None,
        v_act: torch.Tensor | None,
        key_pool: torch.Tensor,
        value_pool: torch.Tensor,
        block_size: int,
        video_slots: torch.Tensor,
        action_slots: torch.Tensor,
        block_table: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        max_query_len: int,
        max_seq_len: int,
        softmax_scale: float,
    ) -> torch.Tensor:
        return _paged_write_attn_impl(
            query,
            k_curr,
            v_curr,
            k_act,
            v_act,
            key_pool,
            value_pool,
            block_size,
            video_slots,
            action_slots,
            block_table,
            query_start_loc,
            seq_lens,
            max_query_len,
            max_seq_len,
            softmax_scale,
        )

    @_paged_write_attn_op.register_fake
    def _(
        query,
        k_curr,
        v_curr,
        k_act,
        v_act,
        key_pool,
        value_pool,
        block_size,
        video_slots,
        action_slots,
        block_table,
        query_start_loc,
        seq_lens,
        max_query_len,
        max_seq_len,
        softmax_scale,
    ):
        return torch.empty_like(query)


def paged_write_attn(
    inputs: ARDiffusionPagedLayerInputs, query, k_curr, v_curr, k_act, v_act, softmax_scale: float
) -> torch.Tensor:
    """Model-facing entry: routes through the custom op (traceable in fullgraph)."""
    return torch.ops.vllm_omni.ar_diffusion_paged_write_attn(
        query,
        k_curr,
        v_curr,
        k_act,
        v_act,
        inputs.key_pool,
        inputs.value_pool,
        inputs.block_size,
        inputs.video_slots,
        inputs.action_slots,
        inputs.block_table,
        inputs.query_start_loc,
        inputs.seq_lens,
        inputs.max_query_len,
        inputs.max_seq_len,
        softmax_scale,
    )
