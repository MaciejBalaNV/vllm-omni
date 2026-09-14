# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cosmos3 transformer variant with Multiview-AV FlexAttention."""

from __future__ import annotations

from typing import Any

import torch
from vllm.distributed import tensor_model_parallel_all_reduce

from .multiview_flex_attention import (
    MultiviewAttentionContext,
    MultiviewLayout,
    padded_multiview_flex_attention,
)
from .multiview_parallel import multiview_ulysses_attention
from .transformer_cosmos3 import (
    COSMOS3_MULTIVIEW_BACKBONE_TYPE,
    Cosmos3CrossAttention,
    Cosmos3GenDecoderLayer,
    Cosmos3VFMTransformer,
    _get_ulysses_state,
    _is_sp_active,
    _tf_config_get,
)


class Cosmos3MultiviewCrossAttention(Cosmos3CrossAttention):
    """Use sparse rectangular attention when a multiview context is present."""

    def _forward_multiview(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_und: torch.Tensor,
        v_und: torch.Tensor,
        multiview_layout: Any,
    ) -> torch.Tensor:
        if not isinstance(multiview_layout, MultiviewAttentionContext):
            raise TypeError(
                "Cosmos3 multiview cross-attention expected MultiviewAttentionContext, "
                f"got {type(multiview_layout).__name__}."
            )
        if _is_sp_active():
            size, rank, group = _get_ulysses_state()
            if group is None:
                raise RuntimeError("Cosmos3 multiview CP is active without an initialized Ulysses group.")
            output = multiview_ulysses_attention(
                q, k, v, k_und, v_und, multiview_layout, group=group, rank=rank, world_size=size
            )
        else:
            output = padded_multiview_flex_attention(q, k, v, k_und, v_und, multiview_layout)
        return output.reshape(q.shape[0], q.shape[1], -1)


class Cosmos3MultiviewGenDecoderLayer(Cosmos3GenDecoderLayer):
    """Bound post-attention norm/MLP activations for long multiview sequences."""

    # Bound the total token rows across the batch in each norm/MLP call.
    _mlp_chunk_size = 65536

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Assemble local chunk outputs before doing a single TP reduction.
        self.mlp.down_proj.reduce_results = False

    def _forward_mlp_chunk(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.post_attention_layernorm(hidden_states))

    @torch.compiler.disable(recursive=False)
    def _forward_mlp_chunked(self, hidden_states: torch.Tensor, chunk_size: int) -> torch.Tensor:
        # Keep the loop out of the graph so compilation cannot unroll it and
        # retain intermediates across chunks. Child calls may still compile.
        batch, sequence_length, hidden_size = hidden_states.shape
        # Flatten before the compiled call so its strides do not depend on the
        # full sequence length. Any copy for a batched slice is chunk-sized.
        chunk = self._forward_mlp_chunk(hidden_states[:, :chunk_size].reshape(-1, hidden_size))
        output = chunk.new_empty(batch, sequence_length, hidden_size)
        output[:, :chunk_size] = chunk.view(batch, chunk_size, hidden_size)
        del chunk
        for start in range(chunk_size, sequence_length, chunk_size):
            end = min(start + chunk_size, sequence_length)
            output[:, start:end] = self._forward_mlp_chunk(hidden_states[:, start:end].reshape(-1, hidden_size)).view(
                batch, end - start, hidden_size
            )
        return output

    def _forward_mlp(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, sequence_length, _ = hidden_states.shape
        chunk_size = max(1, self._mlp_chunk_size // batch)
        if sequence_length <= chunk_size:
            output = self._forward_mlp_chunk(hidden_states)
        else:
            output = self._forward_mlp_chunked(hidden_states, chunk_size)
        if self.mlp.down_proj.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output)
        return output


class Cosmos3MultiviewVFMTransformer(Cosmos3VFMTransformer):
    """Cosmos3 Nano weights with request-local multiview block-mask caching."""

    _gen_layer_cls = Cosmos3MultiviewGenDecoderLayer
    _repeated_blocks = ["Cosmos3MultiviewGenDecoderLayer"]
    _cross_attention_cls = Cosmos3MultiviewCrossAttention

    @staticmethod
    def _validate_supported_config(model_config: Any) -> None:
        Cosmos3VFMTransformer._validate_supported_config(model_config)
        backbone_type = _tf_config_get(model_config, "backbone_type", None)
        if backbone_type != COSMOS3_MULTIVIEW_BACKBONE_TYPE:
            raise ValueError(
                "Cosmos3MultiviewVFMTransformer requires transformer/config.json "
                f"backbone_type={COSMOS3_MULTIVIEW_BACKBONE_TYPE!r}, got {backbone_type!r}."
            )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._multiview_mask_cache: dict[tuple[Any, ...], Any] = {}
        # Padded q/k/v packing buffers, keyed by shape/dtype/device. Held on the
        # transformer rather than the per-forward context so the ~2.5 GiB of
        # packed tensors are zeroed once per request instead of once per layer.
        self._multiview_buffer_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    def reset_cache(self) -> None:
        super().reset_cache()
        self._multiview_mask_cache.clear()
        self._multiview_buffer_cache.clear()

    def forward(
        self,
        *args,
        multiview_layout: MultiviewLayout | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if multiview_layout is None:
            return super().forward(*args, **kwargs)
        control_latents = kwargs.get("control_latents")
        if isinstance(control_latents, torch.Tensor):
            control_count = 1
        elif control_latents is None:
            control_count = 0
        else:
            control_count = len(control_latents)
        if control_count != 1:
            raise ValueError(f"Cosmos3 multiview v1 requires exactly one packed WSM control item, got {control_count}.")
        if kwargs.get("action_latents") is not None or kwargs.get("sound_latents") is not None:
            raise ValueError("Cosmos3 multiview v1 cannot be combined with action or sound streams.")
        context = MultiviewAttentionContext(
            multiview_layout,
            self._multiview_mask_cache,
            self._multiview_buffer_cache,
        )
        return super().forward(*args, multiview_layout=context, **kwargs)
