# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Test-only dense numerical oracle. Never imported by the model."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import _apply_rotary_pos_emb
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.transformer_cosmos3_nano_sim_bimanual import (
    Cosmos3NanoSimBimanualJointAttention,
)
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.utils import zero_null_action_values
from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import ARDiffusionPagedLayerInputs


def _dense_clean_kv(text, history, current, start, end, sink_tokens, tail_tokens):
    """Assemble text + retained prefix + current frame with one allocation.

    Select views before concatenating, preserving the exact sequential clean
    attention order even when a chunk spans a sink/window eviction boundary.
    """
    if tail_tokens is None or history.shape[1] + start <= sink_tokens + tail_tokens:
        return torch.cat([text, history, current[:, :end]], dim=1)
    old_sink = min(sink_tokens, history.shape[1])
    new_sink = sink_tokens - old_sink
    new_tail = min(start, tail_tokens)
    old_tail = tail_tokens - new_tail
    parts = [text, history[:, :old_sink], current[:, :new_sink]]
    if old_tail:
        parts.append(history[:, -old_tail:])
    parts.append(current[:, start - new_tail : end])
    return torch.cat(parts, dim=1)


def append_dense_kv_history(
    history: list[tuple[torch.Tensor, torch.Tensor]] | None,
    current_kv: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    tokens_per_frame: int,
    sink_frames: int,
    window_frames: int | None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Append a dense oracle block while retaining only one rebuilt layer."""

    if isinstance(tokens_per_frame, bool) or not isinstance(tokens_per_frame, int) or tokens_per_frame <= 0:
        raise ValueError("tokens_per_frame must be positive.")
    if (
        isinstance(sink_frames, bool)
        or not isinstance(sink_frames, int)
        or sink_frames < 0
        or (
            window_frames is not None
            and (isinstance(window_frames, bool) or not isinstance(window_frames, int) or window_frames <= 0)
        )
        or (window_frames is None and sink_frames != 0)
    ):
        raise ValueError("sink_frames must be non-negative and window_frames must be positive.")
    sink_tokens = sink_frames * tokens_per_frame
    tail_tokens = window_frames * tokens_per_frame if window_frames is not None else 0
    max_tokens = sink_tokens + tail_tokens if window_frames is not None else None
    if not current_kv:
        raise ValueError("Cosmos3-Nano-Sim-Bimanual dense K/V update must contain at least one layer.")

    def validate_pair(key: torch.Tensor, value: torch.Tensor, *, label: str) -> None:
        if key.ndim < 2 or key.shape != value.shape:
            raise ValueError(
                f"Cosmos3-Nano-Sim-Bimanual {label} K/V must have matching rank-2+ shapes, "
                f"got {tuple(key.shape)} and {tuple(value.shape)}."
            )
        if key.device != value.device or key.dtype != value.dtype:
            raise ValueError(f"Cosmos3-Nano-Sim-Bimanual {label} K/V must have matching dtypes and devices.")
        if key.shape[1] <= 0 or key.shape[1] % tokens_per_frame:
            raise ValueError(
                f"Cosmos3-Nano-Sim-Bimanual {label} K/V token length must be a positive multiple "
                f"of tokens_per_frame={tokens_per_frame}, got {key.shape[1]}."
            )

    for layer_idx, (new_k, new_v) in enumerate(current_kv):
        validate_pair(new_k, new_v, label=f"current layer {layer_idx}")

    if history is None:
        # The transformer output already owns exactly the first committed
        # block. Retain its detached storage directly instead of copying it
        # through a concatenation with zero-length views.
        def retain_initial(tensor):
            if max_tokens is None or tensor.shape[1] <= max_tokens:
                return tensor.detach()
            return torch.cat([tensor[:, :sink_tokens], tensor[:, -tail_tokens:]], dim=1).detach()

        return [(retain_initial(key), retain_initial(value)) for key, value in current_kv]
    if len(history) != len(current_kv):
        raise ValueError(
            "Cosmos3-Nano-Sim-Bimanual dense K/V layer count changed within a session: "
            f"history={len(history)}, current={len(current_kv)}."
        )

    # Validate every layer before mutating the list. Once this succeeds,
    # replacing each entry immediately releases that layer's previous history
    # instead of retaining complete old and next-history lists.
    for layer_idx, ((old_k, old_v), (new_k, new_v)) in enumerate(zip(history, current_kv, strict=True)):
        validate_pair(old_k, old_v, label=f"history layer {layer_idx}")
        if (
            old_k.device != new_k.device
            or old_k.dtype != new_k.dtype
            or old_k.shape[:1] + old_k.shape[2:] != new_k.shape[:1] + new_k.shape[2:]
        ):
            raise ValueError(
                "Cosmos3-Nano-Sim-Bimanual dense K/V layer "
                f"{layer_idx} geometry, dtype, or device changed within a session."
            )

    def combined_prefix(old: torch.Tensor, new: torch.Tensor, count: int) -> list[torch.Tensor]:
        old_count = min(count, old.shape[1])
        parts = [old[:, :old_count]] if old_count else []
        remaining = count - old_count
        if remaining:
            parts.append(new[:, :remaining])
        return parts

    def combined_suffix(old: torch.Tensor, new: torch.Tensor, count: int) -> list[torch.Tensor]:
        new_count = min(count, new.shape[1])
        old_count = count - new_count
        parts = [old[:, -old_count:]] if old_count else []
        if new_count:
            parts.append(new[:, -new_count:])
        return parts

    def append_bounded(old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        if max_tokens is None or old.shape[1] + new.shape[1] <= max_tokens:
            return torch.cat([old, new], dim=1)
        # Build the final sink+tail tensor directly. Appending the full history
        # and trimming it afterward creates another layer-sized transient at
        # the window boundary.
        parts = combined_prefix(old, new, sink_tokens)
        parts.extend(combined_suffix(old, new, tail_tokens))
        return parts[0].clone() if len(parts) == 1 else torch.cat(parts, dim=1)

    for layer_idx, ((old_k, old_v), (new_k, new_v)) in enumerate(zip(history, current_kv, strict=True)):
        key = append_bounded(old_k, new_k)
        value = append_bounded(old_v, new_v)
        history[layer_idx] = (key.detach(), value.detach())
    return history


class ReferenceJointAttention(Cosmos3NanoSimBimanualJointAttention):
    """One softmax over text, committed history, and the current chunk."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        text_k: torch.Tensor,
        text_v: torch.Tensor,
        real_text_kv_len: int,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        dense_history: tuple[torch.Tensor, torch.Tensor] | None = None,
        paged_context: ARDiffusionPagedLayerInputs | None = None,
        num_frames: int,
        tokens_per_frame: int,
        action_tokens_per_frame: int,
        null_action_frame_indexes: tuple[int, ...] = (),
        clean_history_window: tuple[int, int | None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if paged_context is not None:
            return super().forward(
                hidden_states,
                text_k=text_k,
                text_v=text_v,
                real_text_kv_len=real_text_kv_len,
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
                paged_context=paged_context,
                num_frames=num_frames,
                tokens_per_frame=tokens_per_frame,
                action_tokens_per_frame=action_tokens_per_frame,
                null_action_frame_indexes=null_action_frame_indexes,
                clean_history_window=clean_history_window,
            )
        if hidden_states.shape[0] != 1:
            raise ValueError(
                f"Cosmos3-Nano-Sim-Bimanual causal attention supports batch_size=1, got {hidden_states.shape[0]}"
            )
        if real_text_kv_len <= 0 or real_text_kv_len > text_k.shape[1]:
            raise ValueError(
                "Cosmos3-Nano-Sim-Bimanual real text KV length must be in the stored range, "
                f"got real={real_text_kv_len}, stored={text_k.shape[1]}"
            )
        if text_k.shape != text_v.shape:
            raise ValueError(
                f"Cosmos3-Nano-Sim-Bimanual text K/V shapes differ: {tuple(text_k.shape)} != {tuple(text_v.shape)}"
            )
        if dense_history is not None and paged_context is not None:
            raise ValueError(
                "Cosmos3-Nano-Sim-Bimanual attention accepts either dense_history or paged_context, not both"
            )

        batch, seq_len, _ = hidden_states.shape
        q = self.to_q(hidden_states).view(batch, seq_len, self.num_heads_local, self.head_dim)
        k = self.to_k(hidden_states).view(batch, seq_len, self.num_kv_heads_local, self.head_dim)
        v = self.to_v(hidden_states).view(batch, seq_len, self.num_kv_heads_local, self.head_dim)
        if self.qk_norm:
            q = F.rms_norm(q, (self.head_dim,), self.norm_q.weight, eps=self.norm_q.variance_epsilon)
            k = F.rms_norm(k, (self.head_dim,), self.norm_k.weight, eps=self.norm_k.variance_epsilon)
        q, k = _apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin)
        v = zero_null_action_values(
            v,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            action_tokens_per_frame=action_tokens_per_frame,
            null_frame_indexes=null_action_frame_indexes,
        )

        if clean_history_window is not None:
            # Dense oracle for batched clean refresh. Projections and MLPs
            # remain batched, while each attention sees its sequential prefix.
            sink_frames, window_frames = clean_history_window
            sink_tokens = sink_frames * tokens_per_frame
            tail_tokens = window_frames * tokens_per_frame if window_frames is not None else None
            history_k, history_v = dense_history if dense_history is not None else (k[:, :0], v[:, :0])
            outputs = []
            for frame in range(num_frames):
                start, end = frame * tokens_per_frame, (frame + 1) * tokens_per_frame
                outputs.append(
                    self.attn(
                        q[:, start:end],
                        _dense_clean_kv(
                            text_k[:, :real_text_kv_len], history_k, k, start, end, sink_tokens, tail_tokens
                        ),
                        _dense_clean_kv(
                            text_v[:, :real_text_kv_len], history_v, v, start, end, sink_tokens, tail_tokens
                        ),
                    )
                )
            output = torch.cat(outputs, dim=1)
        else:
            key_parts = [text_k[:, :real_text_kv_len]]
            value_parts = [text_v[:, :real_text_kv_len]]
            if dense_history is not None:
                history_k, history_v = dense_history
                if history_k.shape[1]:
                    key_parts.append(history_k)
                    value_parts.append(history_v)
            key_parts.append(k)
            value_parts.append(v)
            output = self.attn(q, torch.cat(key_parts, dim=1), torch.cat(value_parts, dim=1))

        return self.to_out(output.reshape(batch, seq_len, -1)), k, v
