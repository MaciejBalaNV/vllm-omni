# SPDX-License-Identifier: Apache-2.0
"""Causal autoregressive pipeline for Cosmos-Dreams checkpoints."""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any, ClassVar

import torch

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.cosmos3.action import (
    load_action_tensor,
    pad_action_to_dim,
)
from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
    Cosmos3OmniDiffusersPipeline,
    get_cosmos3_ir_op_priority_func,
    get_cosmos3_post_process_func,
    get_cosmos3_pre_process_func,
)
from vllm_omni.diffusion.models.cosmos_dreams.config import CosmosDreamsManifest
from vllm_omni.diffusion.models.cosmos_dreams.normalizer import QuantileRotAffineNormalizer
from vllm_omni.diffusion.models.cosmos_dreams.sampler import CosmosDreamsDistilledSampler
from vllm_omni.diffusion.models.cosmos_dreams.state_cosmos_dreams import (
    CosmosDreamsSessionFingerprint,
    CosmosDreamsSessionState,
)
from vllm_omni.diffusion.models.cosmos_dreams.transformer_cosmos_dreams import (
    CosmosDreamsTransformer,
    CosmosDreamsTransformerOutput,
)
from vllm_omni.diffusion.models.cosmos_dreams.utils import (
    estimate_kv_memory_bytes,
    iter_ar_chunk_ranges,
    iter_clean_commit_frames,
    prompt_token_hash,
)
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.experimental.ar_diffusion.capability import (
    ARDiffusionCrossAttentionKVSpec,
    ARDiffusionKVBranchSpec,
    ARDiffusionKVCacheSpec,
    ARDiffusionRequestRejectedError,
)

logger = logging.getLogger(__name__)


def _nested_config_value(config: Any, key: str, default: Any = None) -> Any:
    """Read one deploy/checkpoint option from the known config envelopes."""

    roots = [
        getattr(config, "custom_pipeline_args", None),
        getattr(config, "model_config", None),
        getattr(config, "tf_model_config", None),
    ]
    nested_names = (
        "cosmos_dreams",
        "causal_manifest",
        "interactive_config",
        "diffusion_expert_config",
    )
    for root in roots:
        if not isinstance(root, dict):
            to_dict = getattr(root, "to_dict", None)
            root = to_dict() if callable(to_dict) else getattr(root, "params", None)
        if not isinstance(root, dict):
            continue
        for nested_name in nested_names:
            nested = root.get(nested_name)
            if isinstance(nested, dict) and nested.get(key) is not None:
                return nested[key]
        if root.get(key) is not None:
            return root[key]
    return default


def get_cosmos_dreams_pre_process_func(od_config: OmniDiffusionConfig):
    """Use Cosmos3 media preprocessing with the deployment-fixed resolution."""

    manifest = CosmosDreamsManifest.from_od_config(od_config, require_explicit=True)
    cosmos3_pre_process = get_cosmos3_pre_process_func(od_config)

    def pre_process_func(request):
        sp = request.sampling_params
        if sp.height is not None and int(sp.height) != manifest.height:
            raise ValueError(
                "Cosmos-Dreams resolution is fixed per deployment: "
                f"requested height={sp.height}, configured height={manifest.height}."
            )
        if sp.width is not None and int(sp.width) != manifest.width:
            raise ValueError(
                "Cosmos-Dreams resolution is fixed per deployment: "
                f"requested width={sp.width}, configured width={manifest.width}."
            )
        sp.height = manifest.height
        sp.width = manifest.width
        return cosmos3_pre_process(request)

    return pre_process_func


def get_cosmos_dreams_post_process_func(od_config: OmniDiffusionConfig):
    return get_cosmos3_post_process_func(od_config)


def get_cosmos_dreams_ir_op_priority_func(od_config: OmniDiffusionConfig):
    return get_cosmos3_ir_op_priority_func(od_config)


class CosmosDreamsPipeline(Cosmos3OmniDiffusersPipeline):
    """Cosmos3-Interactive inference with dense-or-paged persistent GEN K/V.

    The default diffusion engine exercises the dense numerical-oracle path.
    When the AR-Diffusion runner binds a state, the exact same attention uses
    paged storage and gathers one layer of immutable history at a time.
    """

    # The engine's generic warmup request is 512x512 with a one-step sampler,
    # while Cosmos-Dreams has artifact-fixed geometry and a four-step sampler.
    # Skip that incompatible request; AR-Diffusion owns any model-valid rollout
    # warmup when CUDA graphs are enabled.
    dummy_run_num_frames: ClassVar[int] = 0
    _transformer_cls_override: ClassVar[type[CosmosDreamsTransformer]] = CosmosDreamsTransformer
    _MAIN_BRANCH = "main"
    _SESSION_CAPACITY = 1
    _ar_diffusion_kv_state = None
    _bound_session_id: str | None = None

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        self.manifest = CosmosDreamsManifest.from_od_config(od_config, require_explicit=True)
        self.manifest.require_exported_artifact()
        if not isinstance(self.transformer, CosmosDreamsTransformer):
            raise TypeError(
                f"Cosmos-Dreams pipeline resolved the wrong transformer type: {type(self.transformer).__name__}."
            )
        if not self.is_distilled_model:
            raise ValueError("Cosmos-Dreams requires a distilled fixed-step checkpoint.")
        if len(self.manifest.t_list) != 4:
            raise ValueError(
                f"Cosmos-Dreams requires exactly four distilled denoise steps, got {len(self.manifest.t_list)}."
            )
        if od_config.parallel_config.sequence_parallel_size > 1:
            raise ValueError(
                "Cosmos-Dreams supports tensor parallelism but not sequence parallelism; "
                f"got sequence_parallel_size={od_config.parallel_config.sequence_parallel_size}."
            )
        self.distilled_sampler = CosmosDreamsDistilledSampler(
            self.manifest.t_list,
            sample_type=self.manifest.sample_type,
            num_train_timesteps=self.manifest.num_train_timesteps,
        )
        action_schema = self.manifest.action_schema
        if action_schema is None:
            raise ValueError("Cosmos-Dreams requires a validated v2 action_schema.")
        self.action_normalizers = {
            embodiment: QuantileRotAffineNormalizer.from_contract(contract)
            for embodiment, contract in action_schema.normalizers.items()
        }
        self.default_domain_id = int(_nested_config_value(od_config, "default_domain_id", 0))
        self.default_embodiment = action_schema.resolve_embodiment(
            action_schema.default_embodiment,
            self.default_domain_id,
        )
        self.default_fps = float(_nested_config_value(od_config, "default_fps", 15.0))
        if not math.isfinite(self.default_fps) or self.default_fps <= 0:
            raise ValueError(f"Cosmos-Dreams default_fps must be positive, got {self.default_fps}.")
        self.checkpoint_id = (
            self.manifest.checkpoint_id if self.manifest.checkpoint_id != "unknown" else str(od_config.model)
        )
        self._states: OrderedDict[str, CosmosDreamsSessionState] = OrderedDict()

        estimate = estimate_kv_memory_bytes(
            self.manifest,
            num_layers=self.transformer.num_hidden_layers,
            num_kv_heads=self.transformer.num_kv_heads_local,
            head_size=self.transformer.head_dim,
            dtype=self.dtype,
            session_capacity=self._SESSION_CAPACITY,
        )
        logger.info(
            "Cosmos-Dreams KV floor estimate: %.2f GiB (managed %.2f, scratch %.2f, text %.2f)",
            estimate.total_bytes / 2**30,
            estimate.self_attention_bytes / 2**30,
            estimate.scratch_bytes / 2**30,
            estimate.cross_attention_bytes / 2**30,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Reject artifact tensors that do not map into this exact model."""

        allowed = set(self.state_dict())
        tp_aware = {name for name, parameter in self.named_parameters() if hasattr(parameter, "weight_loader")}
        unexpected: list[str] = []

        def is_export_only_tensor(name: str) -> bool:
            key = name.removeprefix("transformer.").removeprefix("model.")
            return key.startswith(("lm_head.", "action_pos_embed."))

        def checked_weights():
            for name, tensor in weights:
                remapped = self._remap_ckpt_key(name)
                if (
                    not is_export_only_tensor(name)
                    and name not in allowed
                    and name not in tp_aware
                    and (remapped is None or (remapped not in allowed and remapped not in tp_aware))
                ):
                    unexpected.append(name)
                yield name, tensor

        loaded = super().load_weights(checked_weights())
        if unexpected:
            preview = ", ".join(sorted(unexpected)[:12])
            suffix = "" if len(unexpected) <= 12 else f" (and {len(unexpected) - 12} more)"
            raise ValueError(f"Cosmos-Dreams checkpoint contains unexpected transformer tensors: {preview}{suffix}")
        return loaded

    # -- AR-Diffusion pipeline capability ---------------------------------

    def ar_diffusion_kv_cache_spec(self) -> ARDiffusionKVCacheSpec:
        return ARDiffusionKVCacheSpec(
            num_layers=self.transformer.num_hidden_layers,
            num_kv_heads=self.transformer.num_kv_heads_local,
            head_size=self.transformer.head_dim,
            tokens_per_frame=self.manifest.tokens_per_frame,
            frames_per_block=1,
            window_frames=self.manifest.window_frames,
            sink_frames=self.manifest.sink_frames,
            kv_branches=(ARDiffusionKVBranchSpec(self._MAIN_BRANCH, 0),),
            session_capacity=self._SESSION_CAPACITY,
            cross_attention=(ARDiffusionCrossAttentionKVSpec("text", self.manifest.text_cache_max_len),),
            max_model_len=(self.manifest.sink_frames + self.manifest.window_frames + 1)
            * self.manifest.tokens_per_frame,
            max_scratch_tokens_per_branch=0,
        )

    @contextmanager
    def bind_ar_diffusion_state(self, session_id, state):
        if self._ar_diffusion_kv_state is not None:
            raise RuntimeError("Cosmos-Dreams AR-Diffusion state is already bound.")
        if state.session_id != session_id:
            raise ValueError(f"Cosmos-Dreams bound session mismatch: {state.session_id!r} != {session_id!r}.")
        self._ar_diffusion_kv_state = state
        self._bound_session_id = str(session_id)
        try:
            yield
        finally:
            self._ar_diffusion_kv_state = None
            self._bound_session_id = None

    def reset_ar_diffusion_session(self, session_id: str) -> None:
        self._drop_session(session_id)

    def close_ar_diffusion_session(self, session_id: str) -> None:
        self._drop_session(session_id)

    def _drop_session(self, session_id: str) -> None:
        state = self._states.pop(str(session_id or "default"), None)
        if state is not None:
            state.reset()

    # -- Session and conditioning -----------------------------------------

    def _get_or_create_state(self, session_id: str) -> CosmosDreamsSessionState:
        state = self._states.get(session_id)
        if state is None:
            while len(self._states) >= self._SESSION_CAPACITY:
                _, evicted = self._states.popitem(last=False)
                evicted.reset()
            state = CosmosDreamsSessionState(session_id=session_id)
            self._states[session_id] = state
        self._states.move_to_end(session_id)
        return state

    def _ensure_text_kv(
        self,
        state: CosmosDreamsSessionState,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        cached = state.text_kv_by_branch.get(self._MAIN_BRANCH)
        if cached is not None:
            return cached

        paged_state = self._ar_diffusion_kv_state
        if paged_state is not None and paged_state.is_cross_attention_populated(self._MAIN_BRANCH, "text"):
            pooled = paged_state.get_cross_attention_kv(self._MAIN_BRANCH, "text")
            cached = [(entry["k"], entry["v"]) for entry in pooled]
        else:
            raw_kv, real_len = self.transformer.encode_text_kv(text_ids, text_mask)
            if real_len > self.manifest.text_cache_max_len:
                raise ValueError(
                    f"Cosmos-Dreams prompt exceeds text_cache_max_len: {real_len} > {self.manifest.text_cache_max_len}."
                )
            if paged_state is None:
                cached = raw_kv
            else:
                padded = self.transformer.pad_text_kv(
                    raw_kv,
                    max_len=self.manifest.text_cache_max_len,
                )
                paged_state.populate_cross_attention(self._MAIN_BRANCH, "text", padded)
                pooled = paged_state.get_cross_attention_kv(self._MAIN_BRANCH, "text")
                cached = [(entry["k"], entry["v"]) for entry in pooled]
        state.text_kv_by_branch[self._MAIN_BRANCH] = cached
        state.prompt_ids_by_branch[self._MAIN_BRANCH] = text_ids.detach().cpu()
        state.prompt_masks_by_branch[self._MAIN_BRANCH] = text_mask.detach().cpu()
        return cached

    def _fingerprint(
        self,
        text_ids: torch.Tensor,
        *,
        real_text_kv_len: int,
        height: int,
        width: int,
        fps: float,
        domain_id: int,
        embodiment: str,
    ) -> CosmosDreamsSessionFingerprint:
        return CosmosDreamsSessionFingerprint(
            prompt_hash=prompt_token_hash(text_ids),
            real_text_kv_lengths=((self._MAIN_BRANCH, real_text_kv_len),),
            height=height,
            width=width,
            fps=fps,
            domain_id=domain_id,
            embodiment=embodiment,
            action_contract_sha256=self.manifest.action_contract_sha256,
            checkpoint_id=self.checkpoint_id,
            manifest_id=self.manifest.digest,
            sampler_id=self.manifest.sampler_id,
        )

    # -- Dense/paged transformer bridge -----------------------------------

    def _append_dense_kv(
        self,
        state: CosmosDreamsSessionState,
        current_kv: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        history = state.dense_kv_by_branch.get(self._MAIN_BRANCH)
        if history is None:
            history = [(key[:, :0], value[:, :0]) for key, value in current_kv]
        next_history: list[tuple[torch.Tensor, torch.Tensor]] = []
        tokens_per_frame = self.manifest.tokens_per_frame
        for (old_k, old_v), (new_k, new_v) in zip(history, current_kv, strict=True):
            key = torch.cat([old_k, new_k], dim=1)
            value = torch.cat([old_v, new_v], dim=1)
            resident_frames = key.shape[1] // tokens_per_frame
            max_frames = self.manifest.sink_frames + self.manifest.window_frames
            if resident_frames > max_frames:
                sink_tokens = self.manifest.sink_frames * tokens_per_frame
                tail_tokens = self.manifest.window_frames * tokens_per_frame
                key = torch.cat([key[:, :sink_tokens], key[:, -tail_tokens:]], dim=1)
                value = torch.cat([value[:, :sink_tokens], value[:, -tail_tokens:]], dim=1)
            next_history.append((key, value))
        state.dense_kv_by_branch[self._MAIN_BRANCH] = next_history

    def _transformer_forward(
        self,
        state: CosmosDreamsSessionState,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        *,
        text_kv: list[tuple[torch.Tensor, torch.Tensor]],
        real_text_kv_len: int,
        frame_start: int,
        fps: float,
        action_latents: torch.Tensor,
        action_domain_ids: torch.Tensor,
        condition_vision: bool,
        null_action_frame_indexes: tuple[int, ...],
        commit_current: bool,
    ) -> CosmosDreamsTransformerOutput:
        paged_state = self._ar_diffusion_kv_state
        seq_len = hidden_states.shape[2] * self.manifest.tokens_per_frame
        paged_kv = None
        dense_history = None
        if paged_state is not None:
            paged_kv = paged_state.get_kv_caches(
                self._MAIN_BRANCH,
                seq_len=seq_len,
                commit_current=commit_current,
            )
        else:
            dense_history = state.dense_kv_by_branch.get(self._MAIN_BRANCH)

        output = self.transformer(
            hidden_states,
            timestep,
            text_kv=text_kv,
            real_text_kv_len=real_text_kv_len,
            frame_start=frame_start,
            fps=fps,
            action_latents=action_latents,
            action_domain_ids=action_domain_ids,
            paged_kv=paged_kv,
            dense_history=dense_history,
            condition_vision=condition_vision,
            null_action_frame_indexes=null_action_frame_indexes,
        )
        if paged_state is not None:
            paged_state.commit_paged_context(self._MAIN_BRANCH)
        elif commit_current:
            self._append_dense_kv(state, output.current_kv)
        return output

    # -- Action and latent preparation ------------------------------------

    def _prepare_raw_action(self, sp, *, embodiment: str) -> torch.Tensor | None:
        action_value = self._get_sp_param(sp, "action", None)
        if action_value is None:
            return None
        action = self.action_normalizers[embodiment].normalize(load_action_tensor(action_value))
        action = pad_action_to_dim(action, self.manifest.max_action_dim)
        return action.to(device=self.device, dtype=self.dtype)

    def _resolve_action_layout(
        self,
        raw_action: torch.Tensor | None,
        *,
        start_frame: int,
        target_frame: int,
    ) -> str | None:
        """Decide once per request how raw action rows are indexed.

        ``global``: row block ``[(f-1)*A, f*A)`` conditions latent frame ``f``
        ``local``:
        rows cover exactly this request's non-prefix frames in order (the
        chunk-per-request tick layout). Resolving once keeps the
        interpretation stable across every chunk of the request and turns
        insufficient coverage into an admission rejection instead of a
        mid-rollout failure after frames were already committed.
        """
        if raw_action is None:
            return None
        action_count = self.manifest.action_tokens_per_frame
        global_rows = max((target_frame - 1) * action_count, 0)
        local_rows = sum(action_count for frame in range(start_frame, target_frame) if frame > 0)
        rows = raw_action.shape[0]
        if rows >= global_rows:
            return "global"
        if rows == local_rows:
            return "local"
        raise ARDiffusionRequestRejectedError(
            "Cosmos-Dreams action length cannot cover the requested latent frames: "
            f"rows={rows}, frame_range=[{start_frame}, {target_frame}), "
            f"expected local rows={local_rows} or at least global rows={global_rows}."
        )

    def _actions_for_frames(
        self,
        raw_action: torch.Tensor | None,
        *,
        layout: str | None,
        request_start_frame: int,
        frame_start: int,
        frame_end: int,
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        action_count = self.manifest.action_tokens_per_frame
        frame_count = frame_end - frame_start
        if raw_action is None or layout is None:
            zeros = torch.zeros(
                1,
                frame_count * action_count,
                self.manifest.max_action_dim,
                device=self.device,
                dtype=self.dtype,
            )
            return zeros, tuple(range(frame_count))

        # First raw row conditions the first non-prefix frame of the request.
        local_base_frame = max(request_start_frame, 1)
        rows: list[torch.Tensor] = []
        null_indexes: list[int] = []
        for local_idx, frame_idx in enumerate(range(frame_start, frame_end)):
            if frame_idx == 0:
                rows.append(raw_action.new_zeros(action_count, self.manifest.max_action_dim))
                null_indexes.append(local_idx)
                continue
            if layout == "global":
                start = (frame_idx - 1) * action_count
            else:
                start = (frame_idx - local_base_frame) * action_count
            rows.append(raw_action[start : start + action_count])
        return torch.cat(rows, dim=0).unsqueeze(0), tuple(null_indexes)

    def _initial_condition_latent(self, prompt_data: Any, sp) -> torch.Tensor | None:
        explicit = self._get_sp_param(sp, "initial_latent", None)
        if explicit is not None:
            latent = explicit if isinstance(explicit, torch.Tensor) else torch.as_tensor(explicit)
            if latent.ndim == 4:
                latent = latent.unsqueeze(2)
            expected = (
                1,
                self.transformer.latent_channel_size,
                1,
                self.manifest.latent_height,
                self.manifest.latent_width,
            )
            if tuple(latent.shape) != expected:
                raise ValueError(f"Cosmos-Dreams initial_latent must have shape {expected}, got {tuple(latent.shape)}.")
            return latent.to(device=self.device, dtype=self.dtype)

        if isinstance(prompt_data, str):
            return None
        additional = prompt_data.get("additional_information", {}) or {}
        image = additional.get("preprocessed_image")
        video = additional.get("preprocessed_video")
        if image is None and isinstance(video, torch.Tensor):
            if video.ndim != 5:
                raise ValueError(
                    f"Cosmos-Dreams preprocessed video must have shape [1,3,T,H,W], got {tuple(video.shape)}."
                )
            image = video[:, :, 0]
        if image is None:
            return None
        if not isinstance(image, torch.Tensor):
            raise TypeError("Cosmos-Dreams preprocessed image must be a torch.Tensor.")
        return self._encode_conditioning_image_latent(image)

    def _commit_clean_frame(
        self,
        state: CosmosDreamsSessionState,
        latent: torch.Tensor,
        *,
        frame_idx: int,
        text_kv: list[tuple[torch.Tensor, torch.Tensor]],
        real_text_kv_len: int,
        fps: float,
        action: torch.Tensor,
        domain_ids: torch.Tensor,
        null_action: bool,
    ) -> None:
        self._transformer_forward(
            state,
            latent.to(self.dtype),
            torch.zeros(1, device=self.device, dtype=torch.float32),
            text_kv=text_kv,
            real_text_kv_len=real_text_kv_len,
            frame_start=frame_idx,
            fps=fps,
            action_latents=action,
            action_domain_ids=domain_ids,
            condition_vision=True,
            null_action_frame_indexes=(0,) if null_action else (),
            commit_current=True,
        )

    # -- Generation --------------------------------------------------------

    @torch.no_grad()
    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        try:
            return self._forward_impl(req)
        except ARDiffusionRequestRejectedError:
            # Admission rejection: guaranteed to be raised before any session
            # or KV side effect, so the session (and its paid-for history)
            # survives for a corrected retry or an explicit reset.
            raise
        except Exception:
            extra = req.sampling_params.extra_args or {}
            session_id = str(extra.get("session_id") or self._bound_session_id or "default")
            self._drop_session(session_id)
            raise

    def _forward_impl(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        # ---- Admission (pure) ------------------------------------------------
        # Everything in this section only reads request and session state and
        # raises ARDiffusionRequestRejectedError on invalid input. No session may be
        # created, initialized, evicted, or written before it completes: the
        # rejection contract promises the client an unchanged session.
        if len(req.prompts) != 1:
            raise ARDiffusionRequestRejectedError("CosmosDreamsPipeline supports exactly one prompt per request.")
        prompt_data = req.prompts[0]
        prompt = prompt_data if isinstance(prompt_data, str) else str(prompt_data.get("prompt", ""))
        sp = req.sampling_params
        extra = sp.extra_args or {}
        session_id = str(extra.get("session_id") or self._bound_session_id or "default")
        if self._bound_session_id is not None and session_id != self._bound_session_id:
            raise ARDiffusionRequestRejectedError(
                f"Cosmos-Dreams request session {session_id!r} does not match bound session {self._bound_session_id!r}."
            )

        reset = bool(extra.get("reset", False))
        close_session = bool(extra.get("close_session", False))
        tick = bool(extra.get("ar_diffusion_tick", False) or extra.get("chunk_only", False))
        existing_state = self._states.get(session_id)
        if reset or existing_state is None or existing_state.fingerprint is None:
            existing_state = None
        state_was_new = existing_state is None
        if not tick and state_was_new and session_id == "default" and not reset and not close_session:
            raise ARDiffusionRequestRejectedError(
                "Cosmos-Dreams full rollouts on the default session require reset=True at start "
                "or close_session=True at end."
            )

        height = int(sp.height or self.manifest.height)
        width = int(sp.width or self.manifest.width)
        if (height, width) != (self.manifest.height, self.manifest.width):
            raise ARDiffusionRequestRejectedError(
                "Cosmos-Dreams resolution is fixed per deployment: "
                f"requested {height}x{width}, configured {self.manifest.height}x{self.manifest.width}."
            )
        fps = float(
            self._get_sp_param(sp, "resolved_frame_rate", None)
            or self._get_sp_param(sp, "frame_rate", None)
            or self._get_sp_param(sp, "fps", None)
            or self.default_fps
        )
        if not math.isfinite(fps) or fps <= 0:
            raise ARDiffusionRequestRejectedError(f"Cosmos-Dreams FPS must be positive, got {fps}.")
        domain_name = self._get_sp_param(sp, "domain_name", None)
        domain_value = self._get_sp_param(sp, "domain_id", None)
        if domain_value is None and domain_name is None:
            domain_value = self.default_domain_id
        if domain_value is not None:
            domain_id = int(domain_value)
            if domain_id < 0:
                raise ARDiffusionRequestRejectedError(f"Cosmos-Dreams domain_id must be non-negative, got {domain_id}.")
        else:
            try:
                domain_id = self.manifest.resolve_domain_name(str(domain_name))
            except ValueError as exc:
                raise ARDiffusionRequestRejectedError(str(exc)) from exc
        if domain_id >= self.manifest.num_embodiment_domains:
            raise ARDiffusionRequestRejectedError(
                "Cosmos-Dreams domain_id is outside the exported embodiment table: "
                f"{domain_id} not in [0, {self.manifest.num_embodiment_domains})."
            )
        try:
            embodiment = self.manifest.resolve_embodiment(domain_name, domain_id)
        except ValueError as exc:
            raise ARDiffusionRequestRejectedError(str(exc)) from exc

        guidance_scale = float(self._get_sp_param(sp, "guidance_scale", 1.0) or 1.0)
        if guidance_scale != 1.0:
            raise ARDiffusionRequestRejectedError(
                f"Cosmos-Dreams distilled inference requires guidance_scale=1.0, got {guidance_scale}."
            )
        if sp.num_inference_steps not in (None, len(self.manifest.t_list)):
            raise ARDiffusionRequestRejectedError(
                "Cosmos-Dreams distilled inference uses the checkpoint-defined four-step schedule; "
                f"got num_inference_steps={sp.num_inference_steps}."
            )

        text_ids, text_mask = self._tokenize_prompt(
            prompt,
            max_sequence_length=1 << 30,
            use_system_prompt=False,
        )
        real_text_kv_len = int(text_mask[0].sum().item())
        if real_text_kv_len > self.manifest.text_cache_max_len:
            raise ARDiffusionRequestRejectedError(
                "Cosmos-Dreams prompt exceeds text_cache_max_len: "
                f"{real_text_kv_len} > {self.manifest.text_cache_max_len}."
            )
        fingerprint = self._fingerprint(
            text_ids,
            real_text_kv_len=real_text_kv_len,
            height=height,
            width=width,
            fps=fps,
            domain_id=domain_id,
            embodiment=embodiment,
        )
        start_frame = 0 if state_was_new else existing_state.next_frame_idx
        requested_frame_idx = int(extra.get("frame_idx", start_frame))
        if state_was_new:
            if requested_frame_idx != 0:
                raise ARDiffusionRequestRejectedError(
                    f"Cosmos-Dreams new sessions must start at latent frame 0; got {requested_frame_idx}."
                )
        else:
            try:
                existing_state.validate_request(fingerprint, frame_idx=requested_frame_idx)
            except (ValueError, RuntimeError) as exc:
                raise ARDiffusionRequestRejectedError(str(exc)) from exc

        try:
            raw_action = self._prepare_raw_action(sp, embodiment=embodiment)
        except (TypeError, ValueError) as exc:
            raise ARDiffusionRequestRejectedError(str(exc)) from exc
        try:
            initial_latent = self._initial_condition_latent(prompt_data, sp)
        except (TypeError, ValueError) as exc:
            raise ARDiffusionRequestRejectedError(str(exc)) from exc
        if start_frame > 0 and initial_latent is not None:
            raise ARDiffusionRequestRejectedError(
                "Cosmos-Dreams initial media may only be supplied at frame 0; session reset required."
            )

        if tick:
            tick_frames = int(extra.get("num_latent_frames", self.manifest.chunk_size))
            if tick_frames <= 0:
                raise ARDiffusionRequestRejectedError(
                    f"Cosmos-Dreams num_latent_frames must be positive, got {tick_frames}."
                )
            target_frame = start_frame + tick_frames
            # Frame zero is the singleton causal prefix. A normal first tick
            # therefore advances through [0, 1) and then one [1, 5) chunk,
            # regardless of whether frame zero is supplied or generated.
            if start_frame == 0:
                target_frame += 1
            if not close_session and (target_frame - 1) % self.manifest.chunk_size != 0:
                raise ARDiffusionRequestRejectedError(
                    "Cosmos-Dreams non-terminal ticks must end on a canonical [1,4,4,...] "
                    f"chunk boundary, got target latent frame {target_frame}."
                )
        else:
            requested_pixel_frames = int(sp.num_frames or 1)
            if requested_pixel_frames <= 0:
                raise ARDiffusionRequestRejectedError(
                    f"Cosmos-Dreams num_frames must be positive, got {requested_pixel_frames}."
                )
            target_frame = (requested_pixel_frames - 1) // self.manifest.temporal_compression_factor + 1
            if target_frame < start_frame:
                raise ARDiffusionRequestRejectedError(
                    "Cosmos-Dreams full rollout target precedes existing session state; session reset required."
                )
        action_layout = self._resolve_action_layout(
            raw_action,
            start_frame=start_frame,
            target_frame=target_frame,
        )

        # ---- Side effects begin ---------------------------------------------
        if reset:
            self._drop_session(session_id)
        state = self._get_or_create_state(session_id)
        if state.fingerprint is None:
            state.initialize(fingerprint)
        domain_ids = torch.tensor([domain_id], device=self.device, dtype=torch.long)
        text_kv = self._ensure_text_kv(state, text_ids, text_mask)

        terminal_request = close_session or not tick
        seed = self._resolve_seed(sp, sp.generator if isinstance(sp.generator, torch.Generator) else None)

        if initial_latent is not None and state.next_frame_idx == 0:
            initial_action, initial_null = self._actions_for_frames(
                raw_action,
                layout=action_layout,
                request_start_frame=start_frame,
                frame_start=0,
                frame_end=1,
            )
            if target_frame > 1 or not terminal_request:
                self._commit_clean_frame(
                    state,
                    initial_latent,
                    frame_idx=0,
                    text_kv=text_kv,
                    real_text_kv_len=real_text_kv_len,
                    fps=fps,
                    action=initial_action,
                    domain_ids=domain_ids,
                    null_action=bool(initial_null),
                )
            state.append_chunk(initial_latent, frame_start=0)

        generation_start = state.next_frame_idx
        for chunk_start, chunk_end in iter_ar_chunk_ranges(
            generation_start,
            target_frame,
            self.manifest.chunk_size,
        ):
            chunk_frames = chunk_end - chunk_start
            action_chunk, null_action_indexes = self._actions_for_frames(
                raw_action,
                layout=action_layout,
                request_start_frame=start_frame,
                frame_start=chunk_start,
                frame_end=chunk_end,
            )
            noise_generator = torch.Generator(device=self.device).manual_seed(seed + chunk_start)
            initial_noise = torch.randn(
                1,
                self.transformer.latent_channel_size,
                chunk_frames,
                self.manifest.latent_height,
                self.manifest.latent_width,
                generator=noise_generator,
                device=self.device,
                # The reference draws checkpoint-dtype noise, then promotes it
                # to fp32 inside the distilled sampler.
                dtype=self.dtype,
            )

            def velocity_fn(x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
                output = self._transformer_forward(
                    state,
                    x.to(self.dtype),
                    timestep,
                    text_kv=text_kv,
                    real_text_kv_len=real_text_kv_len,
                    frame_start=chunk_start,
                    fps=fps,
                    action_latents=action_chunk,
                    action_domain_ids=domain_ids,
                    condition_vision=False,
                    null_action_frame_indexes=null_action_indexes,
                    commit_current=False,
                )
                return output.video.float()

            clean_chunk = self.distilled_sampler.sample(
                velocity_fn,
                initial_noise,
                seed=seed,
                frame_idx=chunk_start,
            ).to(self.dtype)

            action_count = self.manifest.action_tokens_per_frame
            for local_idx, frame_idx in iter_clean_commit_frames(
                chunk_start,
                chunk_end,
                target_frame=target_frame,
                terminal_request=terminal_request,
            ):
                action_start = local_idx * action_count
                action_frame = action_chunk[:, action_start : action_start + action_count]
                self._commit_clean_frame(
                    state,
                    clean_chunk[:, :, local_idx : local_idx + 1],
                    frame_idx=frame_idx,
                    text_kv=text_kv,
                    real_text_kv_len=real_text_kv_len,
                    fps=fps,
                    action=action_frame,
                    domain_ids=domain_ids,
                    null_action=local_idx in null_action_indexes,
                )
            state.append_chunk(clean_chunk, frame_start=chunk_start)

        accumulated = state.accumulated_latents
        if accumulated is None:
            raise RuntimeError("Cosmos-Dreams rollout produced no latent frames.")
        if tick:
            previous_pixel_frames = (
                0 if start_frame == 0 else (start_frame - 1) * self.manifest.temporal_compression_factor + 1
            )
        else:
            previous_pixel_frames = 0
        if sp.output_type == "latent":
            output_value = accumulated[:, :, start_frame:] if tick else accumulated
        else:
            output_value = self._decode_latents(accumulated).clamp(-1, 1)
            if tick and previous_pixel_frames:
                output_value = output_value[:, :, previous_pixel_frames:]
            elif not tick:
                output_value = output_value[:, :, : int(sp.num_frames or 1)]

        if not tick:
            state.terminal = True
        result = DiffusionOutput(output={"video": output_value})
        if close_session and self._ar_diffusion_kv_state is None:
            self._drop_session(session_id)
        return result


# Export-manifest aliases used while the upstream name settles.
CosmosDreamsOmniPipeline = CosmosDreamsPipeline
Cosmos3InteractivePipeline = CosmosDreamsPipeline
