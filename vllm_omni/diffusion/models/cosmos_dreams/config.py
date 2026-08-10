# SPDX-License-Identifier: Apache-2.0
"""Validated deployment manifest for Cosmos-Dreams checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields
from typing import Any

from vllm_omni.diffusion.models.cosmos_dreams.action_contract import (
    ACTION_TOKENS_PER_FRAME,
    MODEL_ACTION_DIM,
    NUM_EMBODIMENT_DOMAINS,
    RAW_ACTION_DIM,
    CosmosDreamsActionSchema,
)

_EXPORTED_ARTIFACT_FIELDS = {
    "schema_version",
    "checkpoint_id",
    "checkpoint_iteration",
    "checkpoint_hash",
    "chunk_size",
    "window_frames",
    "sink_frames",
    "text_cache_max_len",
    "deploy_resolution",
    "attention_mode",
    "video_temporal_causal",
    "latent_patch_size",
    "vae_spatial_compression_factor",
    "temporal_compression_factor",
    "fixed_step_sampler_config",
    "action_schema",
    "temporal_modality_margin",
    "unified_3d_mrope_reset_spatial_ids",
    "base_fps",
    "enable_fps_modulation",
}
_LEGACY_NORMALIZER_FIELDS = {"action_normalizer", "normalizer_id", "normalizer_source"}


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        return converted if isinstance(converted, dict) else {}
    params = getattr(value, "params", None)
    return params if isinstance(params, dict) else {}


def _first(sources: list[dict[str, Any]], *keys: str, default: Any = None) -> Any:
    for source in sources:
        for key in keys:
            if key in source and source[key] is not None:
                return source[key]
    return default


def _manifest_sources(config: Any) -> list[dict[str, Any]]:
    roots: list[dict[str, Any]] = []
    for attr in ("custom_pipeline_args", "model_config", "tf_model_config"):
        value = _mapping(getattr(config, attr, None))
        if value:
            roots.append(value)

    sources: list[dict[str, Any]] = []
    for root in roots:
        for key in (
            "cosmos_dreams",
            "causal_manifest",
            "interactive_config",
            "diffusion_expert_config",
        ):
            nested = _mapping(root.get(key))
            if nested:
                sources.append(nested)
        sources.append(root)
    return sources


def _exported_artifact_source(config: Any) -> dict[str, Any]:
    """Return the causal payload embedded in ``transformer/config.json``."""

    transformer_config = _mapping(getattr(config, "tf_model_config", None))
    return _mapping(transformer_config.get("cosmos_dreams"))


def _validate_deploy_overrides(config: Any, artifact: dict[str, Any]) -> None:
    """Reject deploy values that contradict the signed artifact contract."""

    for attr in ("custom_pipeline_args", "model_config"):
        root = _mapping(getattr(config, attr, None))
        legacy_root_fields = sorted(_LEGACY_NORMALIZER_FIELDS & set(root))
        if legacy_root_fields:
            raise ValueError(
                f"Cosmos-Dreams deployment contains legacy normalizer fields in {attr}: {legacy_root_fields}."
            )
        if "action_schema" in root:
            raise ValueError(
                f"Cosmos-Dreams action_schema may only come from transformer/config.json, not {attr}.action_schema."
            )
        for key in ("cosmos_dreams", "causal_manifest", "interactive_config"):
            override = _mapping(root.get(key))
            legacy_override_fields = sorted(_LEGACY_NORMALIZER_FIELDS & set(override))
            if legacy_override_fields:
                raise ValueError(
                    "Cosmos-Dreams deployment contains legacy normalizer fields in "
                    f"{attr}.{key}: {legacy_override_fields}."
                )
            if "action_schema" in override:
                raise ValueError(
                    "Cosmos-Dreams action_schema may only come from transformer/config.json, "
                    f"not {attr}.{key}.action_schema."
                )
            for field in artifact.keys() & override.keys():
                if artifact[field] != override[field]:
                    raise ValueError(
                        "Cosmos-Dreams deployment override contradicts the exported artifact: "
                        f"{attr}.{key}.{field}={override[field]!r}, artifact={artifact[field]!r}."
                    )


@dataclass(frozen=True)
class CosmosDreamsManifest:
    """Causal settings stripped by the native Cosmos3 base-model export.

    Resolution is fixed at model load because ``tokens_per_frame`` is the AR
    paged-cache block size.  Pixel dimensions need not be multiples of 32; the
    latent patch grid uses ceiling division, matching Cosmos3 patchification.
    """

    schema_version: int = 2
    chunk_size: int = 4
    window_frames: int = 96
    sink_frames: int = 0
    text_cache_max_len: int = 512
    height: int = 720
    width: int = 1280
    latent_patch_size: int = 2
    vae_spatial_compression_factor: int = 16
    temporal_compression_factor: int = 4
    temporal_modality_margin: int = 15_000
    base_fps: float = 24.0
    enable_fps_modulation: bool = True
    attention_mode: str = "three_way"
    video_temporal_causal: bool = True
    sample_type: str = "sde"
    t_list: tuple[float, ...] = (1.0, 15 / 16, 5 / 6, 5 / 8)
    num_train_timesteps: int = 1000
    checkpoint_id: str = "unknown"
    checkpoint_iteration: int = 0
    checkpoint_hash: str = "unknown"
    action_schema: CosmosDreamsActionSchema | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError(f"Cosmos-Dreams manifest schema_version must be 2, got {self.schema_version}")
        positive = {
            "schema_version": self.schema_version,
            "chunk_size": self.chunk_size,
            "window_frames": self.window_frames,
            "text_cache_max_len": self.text_cache_max_len,
            "height": self.height,
            "width": self.width,
            "latent_patch_size": self.latent_patch_size,
            "vae_spatial_compression_factor": self.vae_spatial_compression_factor,
            "temporal_compression_factor": self.temporal_compression_factor,
            "action_tokens_per_frame": self.action_tokens_per_frame,
            "model_action_dim": self.max_action_dim,
            "raw_action_dim": self.raw_action_dim,
            "num_embodiment_domains": self.num_embodiment_domains,
            "num_train_timesteps": self.num_train_timesteps,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"Cosmos-Dreams manifest {name} must be positive, got {value}")
        if self.sink_frames < 0:
            raise ValueError(f"Cosmos-Dreams manifest sink_frames must be non-negative, got {self.sink_frames}")
        if self.checkpoint_iteration < 0:
            raise ValueError(
                f"Cosmos-Dreams checkpoint_iteration must be non-negative, got {self.checkpoint_iteration}"
            )
        if self.attention_mode != "three_way":
            raise ValueError(f"Cosmos-Dreams requires attention_mode='three_way', got {self.attention_mode!r}")
        if not self.video_temporal_causal:
            raise ValueError("Cosmos-Dreams requires video_temporal_causal=True")
        if not self.enable_fps_modulation:
            raise ValueError("Cosmos-Dreams AR inference requires FPS modulation")
        if not math.isfinite(self.base_fps) or self.base_fps <= 0:
            raise ValueError(f"Cosmos-Dreams base_fps must be positive, got {self.base_fps}")
        if self.sample_type not in {"ode", "sde"}:
            raise ValueError(f"Cosmos-Dreams sample_type must be 'ode' or 'sde', got {self.sample_type!r}")
        if not self.t_list:
            raise ValueError("Cosmos-Dreams t_list must not be empty")
        if abs(self.t_list[0] - 1.0) > 1e-6:
            raise ValueError(f"Cosmos-Dreams t_list must start at 1.0, got {self.t_list[0]}")
        if any(not math.isfinite(sigma) or sigma <= 0.0 or sigma > 1.0 for sigma in self.t_list):
            raise ValueError(f"Cosmos-Dreams t_list entries must be in (0, 1], got {self.t_list}")
        if any(left <= right for left, right in zip(self.t_list, self.t_list[1:])):
            raise ValueError(f"Cosmos-Dreams t_list must be strictly descending, got {self.t_list}")
        if self.action_tokens_per_frame != self.temporal_compression_factor:
            raise ValueError(
                "Cosmos-Dreams action_tokens_per_frame must equal temporal_compression_factor; "
                f"got {self.action_tokens_per_frame} and {self.temporal_compression_factor}"
            )
        if self.checkpoint_hash != "unknown" and (
            len(self.checkpoint_hash) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in self.checkpoint_hash)
        ):
            raise ValueError(
                f"Cosmos-Dreams checkpoint_hash must be a 64-character SHA-256 digest, got {self.checkpoint_hash!r}"
            )
        if self.checkpoint_hash != "unknown" and set(self.checkpoint_hash) == {"0"}:
            raise ValueError("Cosmos-Dreams checkpoint_hash cannot be the all-zero template value")

    @property
    def action_tokens_per_frame(self) -> int:
        if self.action_schema is None:
            return ACTION_TOKENS_PER_FRAME
        return self.action_schema.action_tokens_per_frame

    @property
    def raw_action_dim(self) -> int:
        if self.action_schema is None:
            return RAW_ACTION_DIM
        return self.action_schema.raw_action_dim

    @property
    def max_action_dim(self) -> int:
        if self.action_schema is None:
            return MODEL_ACTION_DIM
        return self.action_schema.model_action_dim

    @property
    def num_embodiment_domains(self) -> int:
        if self.action_schema is None:
            return NUM_EMBODIMENT_DOMAINS
        return self.action_schema.num_embodiment_domains

    @property
    def embodiment_to_domain(self) -> tuple[tuple[str, int], ...]:
        if self.action_schema is None:
            return (("agibotworld", 15),)
        return tuple(sorted(self.action_schema.embodiment_to_domain.items()))

    @property
    def action_contract_sha256(self) -> str:
        if self.action_schema is None:
            return "none"
        return self.action_schema.contract_sha256

    @classmethod
    def from_od_config(
        cls,
        od_config: Any,
        *,
        require_explicit: bool = False,
    ) -> CosmosDreamsManifest:
        if require_explicit:
            artifact = _exported_artifact_source(od_config)
            if not artifact:
                raise ValueError(
                    "Cosmos-Dreams requires a causal manifest embedded in transformer/config.json; "
                    "deployment defaults are not a validated artifact."
                )
            missing_artifact_fields = sorted(_EXPORTED_ARTIFACT_FIELDS - set(artifact))
            unknown_artifact_fields = sorted(set(artifact) - _EXPORTED_ARTIFACT_FIELDS)
            if missing_artifact_fields:
                raise ValueError(
                    f"Cosmos-Dreams transformer artifact is incomplete; missing {', '.join(missing_artifact_fields)}."
                )
            if unknown_artifact_fields:
                raise ValueError(
                    f"Cosmos-Dreams transformer artifact contains unknown fields: {unknown_artifact_fields}."
                )
            _validate_deploy_overrides(od_config, artifact)
        else:
            artifact = _exported_artifact_source(od_config)
            if artifact:
                _validate_deploy_overrides(od_config, artifact)
        sources = [artifact] if artifact else _manifest_sources(od_config)
        fixed_step = _mapping(_first(sources, "fixed_step_sampler_config", default={}))
        resolution = _first(sources, "deploy_resolution", default=None)
        resolution_map = _mapping(resolution)
        resolution_list = resolution if isinstance(resolution, list | tuple) else None
        height = _first(sources, "height", "deploy_height", default=None)
        width = _first(sources, "width", "deploy_width", default=None)
        if height is None and resolution_map:
            height = resolution_map.get("height")
        if width is None and resolution_map:
            width = resolution_map.get("width")
        if resolution_list is not None and len(resolution_list) == 2:
            height = resolution_list[0] if height is None else height
            width = resolution_list[1] if width is None else width

        raw_t_list = fixed_step.get("t_list", _first(sources, "t_list", default=cls.t_list))
        raw_action_schema = _first(sources, "action_schema", default=None)
        action_schema = (
            CosmosDreamsActionSchema.model_validate(raw_action_schema) if raw_action_schema is not None else None
        )
        return cls(
            schema_version=int(_first(sources, "schema_version", default=2)),
            chunk_size=int(_first(sources, "chunk_size", "teacher_forcing_frames_per_chunk", default=4)),
            window_frames=int(_first(sources, "window_frames", "kv_cache_inference_size", default=96)),
            sink_frames=int(_first(sources, "sink_frames", "attention_sink_size", default=0)),
            text_cache_max_len=int(_first(sources, "text_cache_max_len", "ar_static_und_cache_max_len", default=512)),
            height=int(height if height is not None else 720),
            width=int(width if width is not None else 1280),
            latent_patch_size=int(_first(sources, "latent_patch_size", "patch_spatial", default=2)),
            vae_spatial_compression_factor=int(
                _first(sources, "vae_spatial_compression_factor", "latent_downsample_factor", default=16)
            ),
            temporal_compression_factor=int(_first(sources, "temporal_compression_factor", default=4)),
            temporal_modality_margin=int(
                _first(
                    sources,
                    "temporal_modality_margin",
                    "unified_3d_mrope_temporal_modality_margin",
                    default=15_000,
                )
            ),
            base_fps=float(_first(sources, "base_fps", default=24.0)),
            enable_fps_modulation=bool(_first(sources, "enable_fps_modulation", default=True)),
            attention_mode=str(_first(sources, "attention_mode", "joint_attn_implementation", default="three_way")),
            video_temporal_causal=bool(_first(sources, "video_temporal_causal", default=True)),
            sample_type=str(fixed_step.get("sample_type", _first(sources, "sample_type", default="sde"))),
            t_list=tuple(float(value) for value in raw_t_list),
            num_train_timesteps=int(
                fixed_step.get(
                    "num_train_timesteps",
                    _first(sources, "num_train_timesteps", default=1000),
                )
            ),
            checkpoint_id=str(_first(sources, "checkpoint_id", default="unknown")),
            checkpoint_iteration=int(_first(sources, "checkpoint_iteration", default=0)),
            checkpoint_hash=str(_first(sources, "checkpoint_hash", default="unknown")),
            action_schema=action_schema,
        )

    def require_exported_artifact(self) -> None:
        """Reject deployment-only defaults without the Stage-2 artifact data."""

        missing: list[str] = []
        if not self.checkpoint_id or self.checkpoint_id == "unknown":
            missing.append("checkpoint_id")
        if self.checkpoint_iteration <= 0:
            missing.append("checkpoint_iteration")
        if not self.checkpoint_hash or self.checkpoint_hash == "unknown":
            missing.append("checkpoint_hash")
        if self.action_schema is None:
            missing.append("action_schema")
        if missing:
            raise ValueError(
                "Cosmos-Dreams requires a validated exported artifact; missing "
                f"{', '.join(missing)} from its causal manifest."
            )

    def resolve_domain_name(self, name: str) -> int:
        normalized = str(name).strip().lower()
        try:
            return dict(self.embodiment_to_domain)[normalized]
        except KeyError as exc:
            raise ValueError(
                f"Unknown Cosmos-Dreams domain_name={name!r}; expected one of "
                f"{sorted(dict(self.embodiment_to_domain))}."
            ) from exc

    def resolve_embodiment(self, name: str | None, domain_id: int | None) -> str:
        if self.action_schema is None:
            raise ValueError("Cosmos-Dreams action_schema is unavailable.")
        return self.action_schema.resolve_embodiment(name, domain_id)

    @property
    def latent_height(self) -> int:
        return math.ceil(self.height / self.vae_spatial_compression_factor)

    @property
    def latent_width(self) -> int:
        return math.ceil(self.width / self.vae_spatial_compression_factor)

    @property
    def patch_grid(self) -> tuple[int, int]:
        return (
            math.ceil(self.latent_height / self.latent_patch_size),
            math.ceil(self.latent_width / self.latent_patch_size),
        )

    @property
    def vision_tokens_per_frame(self) -> int:
        grid_h, grid_w = self.patch_grid
        return grid_h * grid_w

    @property
    def tokens_per_frame(self) -> int:
        return self.action_tokens_per_frame + self.vision_tokens_per_frame

    @property
    def sampler_id(self) -> str:
        values = ",".join(f"{value:.12g}" for value in self.t_list)
        return f"{self.sample_type}:{values}:train={self.num_train_timesteps}"

    @property
    def digest(self) -> str:
        values = {field.name: getattr(self, field.name) for field in fields(self) if field.name != "action_schema"}
        values["action_schema"] = (
            self.action_schema.model_dump(mode="json", exclude_none=True) if self.action_schema is not None else None
        )
        payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()
