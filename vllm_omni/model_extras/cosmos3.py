# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import math
from typing import Any

from PIL import Image


def resolve_image_edit_size(
    source_size: tuple[int, int],
    width: int | None = None,
    height: int | None = None,
    *,
    width_not_provided: bool = False,
    height_not_provided: bool = False,
) -> tuple[int, int, bool]:
    """Return (width, height, is_explicit) for offline and API edit requests.

    Automatic sizes preserve the source within 1 megapixel, rounded down to
    32-pixel multiples. Dimensions below 32 grow to the minimum. Explicit
    sizes override the budget and are applied with resize and center crop.
    """
    width = None if width_not_provided else width
    height = None if height_not_provided else height
    if width is not None or height is not None:
        if width is None or height is None:
            raise ValueError("Cosmos3 image editing requires both width and height, or neither for automatic sizing.")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value % 32
            for value in (width, height)
        ):
            raise ValueError("Cosmos3 image editing width and height must be positive integers divisible by 32.")
        return width, height, True

    source_width, source_height = source_size
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Cosmos3 source image dimensions must be positive.")
    max_pixels = 1_048_576
    scale = min(1.0, math.sqrt(max_pixels / (source_width * source_height)))
    width = max(32, int(source_width * scale) // 32 * 32)
    height = max(32, int(source_height * scale) // 32 * 32)
    # The 32-pixel minimum can exceed the budget for extreme aspect ratios.
    while width * height > max_pixels:
        if width >= height and width > 32:
            width -= 32
        elif height > 32:
            height -= 32
        else:
            break
    return width, height, False


def build_image_to_image_prompt(
    prompt: str,
    negative_prompt: str | None,
    input_image: Image.Image | list[Image.Image],
    height: int | None = None,
    width: int | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "prompt": prompt,
        "modalities": ["image"],
        "multi_modal_data": {"image": input_image},
    }
    if negative_prompt is not None:
        result["negative_prompt"] = negative_prompt
    return result


COSMOS3_EXTRA_BODY_PARAMS = frozenset(
    {
        "flow_shift",
        "max_sequence_length",
        "use_resolution_template",
        "use_duration_template",
        "negative_metadata_mode",
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
