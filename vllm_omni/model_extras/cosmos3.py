# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

COSMOS3_PIPELINE_NAMES = frozenset(
    {
        "Cosmos3OmniDiffusersPipeline",
        "Cosmos3OmniPipeline",
    }
)
COSMOS3_DROID_POLICY_CHECKPOINT_NAMES = frozenset(
    {
        "cosmos3-nano-policy-droid",
        "cosmos3-edge-policy-droid",
    }
)


def _find_action_gen(config: Any) -> Any:
    if not isinstance(config, Mapping):
        return None
    if "action_gen" in config:
        return config["action_gen"]
    for value in config.values():
        resolved = _find_action_gen(value)
        if resolved is not None:
            return resolved
    return None


def resolve_cosmos3_action_gen(model: str | None, *, revision: str | None = None) -> bool | None:
    """Resolve the action-generation capability from checkpoint metadata.

    The API process cannot rely on the worker-only ``tf_model_config`` in
    out-of-process deployments. Read the same checkpoint config directly so
    route admission remains stable across replica and served-name topologies.
    ``None`` means the capability could not be determined.
    """
    if not model:
        return None

    from vllm.transformers_utils.config import get_hf_file_to_dict

    for config_path in ("transformer/config.json", "config.json"):
        try:
            config = get_hf_file_to_dict(config_path, model, revision=revision)
        except Exception as exc:
            logger.debug(
                "Could not inspect Cosmos3 action capability in %s for %s: %s",
                config_path,
                model,
                exc,
            )
            continue
        action_gen = _find_action_gen(config)
        if action_gen is None:
            continue
        if isinstance(action_gen, str):
            return action_gen.strip().lower() in {"1", "true", "yes", "on"}
        return bool(action_gen)
    return None


def is_cosmos3_droid_policy_checkpoint(model: str | None) -> bool:
    """Return whether *model* names one of the released DROID policies."""
    if not model:
        return False
    checkpoint_name = str(model).rstrip("/").rsplit("/", 1)[-1].lower()
    return checkpoint_name in COSMOS3_DROID_POLICY_CHECKPOINT_NAMES


COSMOS3_EXTRA_BODY_PARAMS = frozenset(
    {
        "flow_shift",
        "max_sequence_length",
        "use_resolution_template",
        "use_duration_template",
        "use_system_prompt",
        "system_prompt",
        "negative_prompt",
        "guardrails",
        "condition_frame_indexes_vision",
        "condition_video_keep",
        "generate_sound",
        "sound_gen",
        "sound_duration",
        "audio_duration",
        "action_mode",
        "action",
        "domain_name",
        "domain_id",
        "raw_action_dim",
        "action_chunk_size",
        "action_space",
        "action_fps",
        "image_height",
        "image_width",
        "history_length",
        "conditioning_fps",
        "resolution",
        "image_size",
        "use_state",
        "format_prompt_as_json",
        "observation",
        "robot_obs",
        "deterministic_seed",
        "session_id",
    }
)
COSMOS3_EXTRA_OUTPUT_PARAMS = frozenset(
    {
        "action",
        "raw_action_dim",
        "domain_id",
        "action_mode",
    }
)
