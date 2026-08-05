# SPDX-License-Identifier: Apache-2.0
"""Validated deployment manifest for Cosmos-Dreams checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any


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
    for key in ("cosmos_dreams", "causal_manifest", "interactive_config"):
        nested = _mapping(transformer_config.get(key))
        if nested:
            return nested
    return {}


def _validate_deploy_overrides(config: Any, artifact: dict[str, Any]) -> None:
    """Reject deploy values that contradict the signed artifact contract."""

    for attr in ("custom_pipeline_args", "model_config"):
        root = _mapping(getattr(config, attr, None))
        for key in ("cosmos_dreams", "causal_manifest", "interactive_config"):
            override = _mapping(root.get(key))
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

    schema_version: int = 1
    chunk_size: int = 4
    window_frames: int = 96
    sink_frames: int = 0
    text_cache_max_len: int = 512
    height: int = 720
    width: int = 1280
    latent_patch_size: int = 2
    vae_spatial_compression_factor: int = 16
    temporal_compression_factor: int = 4
    action_tokens_per_frame: int = 4
    max_action_dim: int = 64
    num_embodiment_domains: int = 32
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
    normalizer_id: str = "none"
    normalizer_source: str = "unknown"
    embodiment_to_domain: tuple[tuple[str, int], ...] = (("agibotworld", 15),)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                f"Cosmos-Dreams manifest schema_version must be 1, got {self.schema_version}"
            )
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
            "max_action_dim": self.max_action_dim,
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
                "Cosmos-Dreams checkpoint_iteration must be non-negative, "
                f"got {self.checkpoint_iteration}"
            )
        if self.attention_mode != "three_way":
            raise ValueError(
                "Cosmos-Dreams requires attention_mode='three_way', "
                f"got {self.attention_mode!r}"
            )
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
                "Cosmos-Dreams checkpoint_hash must be a 64-character SHA-256 digest, "
                f"got {self.checkpoint_hash!r}"
            )
        if self.checkpoint_hash != "unknown" and set(self.checkpoint_hash) == {"0"}:
            raise ValueError("Cosmos-Dreams checkpoint_hash cannot be the all-zero template value")
        domain_names = [name for name, _ in self.embodiment_to_domain]
        if not domain_names or any(not name for name in domain_names):
            raise ValueError("Cosmos-Dreams embodiment_to_domain must contain non-empty names")
        if len(domain_names) != len(set(domain_names)):
            raise ValueError(
                "Cosmos-Dreams embodiment_to_domain names must be unique, "
                f"got {domain_names}"
            )
        invalid_domains = {
            name: domain_id
            for name, domain_id in self.embodiment_to_domain
            if domain_id < 0 or domain_id >= self.num_embodiment_domains
        }
        if invalid_domains:
            raise ValueError(
                "Cosmos-Dreams embodiment domain IDs must be inside the exported table, "
                f"got {invalid_domains}"
            )

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
            missing_artifact_fields = [
                field
                for field in (
                    "checkpoint_id",
                    "checkpoint_iteration",
                    "checkpoint_hash",
                    "normalizer_id",
                    "normalizer_source",
                    "action_normalizer",
                )
                if artifact.get(field) in (None, "", "unknown", "none", 0)
            ]
            if missing_artifact_fields:
                raise ValueError(
                    "Cosmos-Dreams transformer artifact is incomplete; missing "
                    f"{', '.join(missing_artifact_fields)}."
                )
            _validate_deploy_overrides(od_config, artifact)
        sources = _manifest_sources(od_config)
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
        raw_domains = _first(
            sources,
            "embodiment_to_domain",
            default=dict(cls.embodiment_to_domain),
        )
        if not isinstance(raw_domains, dict):
            raise ValueError(
                "Cosmos-Dreams embodiment_to_domain must be a mapping, "
                f"got {type(raw_domains).__name__}"
            )
        return cls(
            schema_version=int(_first(sources, "schema_version", default=1)),
            chunk_size=int(
                _first(sources, "chunk_size", "teacher_forcing_frames_per_chunk", default=4)
            ),
            window_frames=int(_first(sources, "window_frames", "kv_cache_inference_size", default=96)),
            sink_frames=int(_first(sources, "sink_frames", "attention_sink_size", default=0)),
            text_cache_max_len=int(
                _first(sources, "text_cache_max_len", "ar_static_und_cache_max_len", default=512)
            ),
            height=int(height if height is not None else 720),
            width=int(width if width is not None else 1280),
            latent_patch_size=int(_first(sources, "latent_patch_size", "patch_spatial", default=2)),
            vae_spatial_compression_factor=int(
                _first(sources, "vae_spatial_compression_factor", "latent_downsample_factor", default=16)
            ),
            temporal_compression_factor=int(
                _first(sources, "temporal_compression_factor", default=4)
            ),
            action_tokens_per_frame=int(
                _first(sources, "action_tokens_per_frame", "temporal_compression_factor", default=4)
            ),
            max_action_dim=int(_first(sources, "max_action_dim", "action_dim", default=64)),
            num_embodiment_domains=int(_first(sources, "num_embodiment_domains", default=32)),
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
            attention_mode=str(
                _first(sources, "attention_mode", "joint_attn_implementation", default="three_way")
            ),
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
            normalizer_id=str(
                _first(sources, "normalizer_id", "action_normalizer_id", default="none")
            ),
            normalizer_source=str(_first(sources, "normalizer_source", default="unknown")),
            embodiment_to_domain=tuple(
                sorted((str(name).strip().lower(), int(domain_id)) for name, domain_id in raw_domains.items())
            ),
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
        if not self.normalizer_id or self.normalizer_id == "none":
            missing.append("normalizer_id")
        if not self.normalizer_source or self.normalizer_source == "unknown":
            missing.append("normalizer_source")
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
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()
