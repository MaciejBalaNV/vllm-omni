# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Canonical Cosmos3 prompt formatting and tokenization, without model allocation."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

logger = logging.getLogger(__name__)


def checkpoint_identity(model: str, revision: str | None) -> dict[str, str]:
    """Identity checked before GEN accepts another stage's conditioning."""
    path = Path(model)
    return {"model": str(path.resolve()) if path.exists() else model, "revision": revision or "main"}


VIDEO_RES_SIZE_INFO: dict[str, dict[str, tuple[int, int]]] = {
    "256": {
        "1,1": (256, 256),
        "4,3": (320, 256),
        "3,4": (256, 320),
        "16,9": (320, 192),
        "9,16": (192, 320),
    },
    "480": {
        "1,1": (640, 640),
        "4,3": (736, 544),
        "3,4": (544, 736),
        "16,9": (832, 480),
        "9,16": (480, 832),
    },
    "704": {
        "1,1": (960, 960),
        "4,3": (1088, 832),
        "3,4": (832, 1088),
        "16,9": (1280, 704),
        "9,16": (704, 1280),
    },
    "720": {
        "1,1": (960, 960),
        "4,3": (1104, 832),
        "3,4": (832, 1104),
        "16,9": (1280, 720),
        "9,16": (720, 1280),
    },
}


COSMOS3_DURATION_TEMPLATE = "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
COSMOS3_RESOLUTION_TEMPLATE = "This video is of {height}x{width} resolution."
COSMOS3_IMAGE_RESOLUTION_TEMPLATE = "This image is of {height}x{width} resolution."
COSMOS3_INVERSE_DURATION_TEMPLATE = "The video is not {duration:.1f} seconds long and is not of {fps:.0f} FPS."
COSMOS3_INVERSE_RESOLUTION_TEMPLATE = "This video is not of {height}x{width} resolution."
COSMOS3_INVERSE_IMAGE_RESOLUTION_TEMPLATE = "This image is not of {height}x{width} resolution."
# NOTE: Intentional typo in "give" instead of "given" to match training setup.
COSMOS3_SYSTEM_PROMPT = "You are a helpful assistant who will generate videos from a give prompt."
COSMOS3_T2I_SYSTEM_PROMPT = "You are a helpful assistant who will generate images from a give prompt."


def _ceil_video_num_frames(num_frames: int, temporal_compression_factor: int) -> int:
    """Round a video length up to the causal VAE's ``factor * k + 1`` grid."""
    if temporal_compression_factor <= 0:
        raise ValueError(f"Cosmos3 temporal_compression_factor must be positive, got {temporal_compression_factor}.")
    if num_frames <= 1:
        return num_frames
    latent_frames = math.ceil((num_frames - 1) / temporal_compression_factor)
    return latent_frames * temporal_compression_factor + 1


def _format_json_object_prompt(
    prompt: str,
    *,
    num_frames: int,
    frame_rate: float,
    height: int,
    width: int,
    aspect_ratio: str | None,
) -> str | None:
    """Match Imaginaire4 metadata injection for JSON-object prompts."""
    try:
        prompt_obj = json.loads(prompt)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(prompt_obj, dict):
        return None

    metadata: dict[str, Any] = {}
    if num_frames > 1:
        duration_seconds = int(num_frames / frame_rate) if frame_rate > 0 else 0
        metadata.update({"duration": f"{duration_seconds}s", "fps": float(frame_rate)})
    else:
        prompt_obj.pop("duration", None)
        prompt_obj.pop("fps", None)
    metadata["resolution"] = {"H": int(height), "W": int(width)}
    if aspect_ratio is not None:
        metadata["aspect_ratio"] = aspect_ratio

    prompt_obj.update(metadata)
    return json.dumps(prompt_obj)


def _json_object_aspect_ratio(prompt: str) -> Any | None:
    """Return an aspect-ratio field from a JSON-object prompt, if present."""
    try:
        prompt_obj = json.loads(prompt)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(prompt_obj, dict):
        return None
    return prompt_obj.get("aspect_ratio")


def _normalize_aspect_ratio(value: Any) -> str | None:
    """Normalize supported ratio spellings for comparison and prompt metadata."""
    if value is None:
        return None
    parts = str(value).strip().replace(":", ",").split(",")
    if len(parts) != 2:
        return None
    try:
        width, height = (int(part.strip()) for part in parts)
    except ValueError:
        return None
    if width <= 0 or height <= 0:
        return None
    divisor = math.gcd(width, height)
    return f"{width // divisor},{height // divisor}"


def _aspect_ratio_for_dimensions(width: int, height: int) -> str:
    """Return the nearest supported Cosmos aspect-ratio label."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Cosmos3 canvas dimensions must be positive, got {width}x{height}.")
    canvas_ratio = width / height

    def ratio_value(label: str) -> float:
        ratio_width, ratio_height = (int(part) for part in label.split(","))
        return ratio_width / ratio_height

    return min(
        (label for sizes in VIDEO_RES_SIZE_INFO.values() for label in sizes),
        key=lambda label: abs(canvas_ratio - ratio_value(label)),
    )


COSMOS3_T2V_DEFAULT_HEIGHT = 720
COSMOS3_T2V_DEFAULT_WIDTH = 1280
COSMOS3_T2V_DEFAULT_NUM_FRAMES = 189
COSMOS3_T2V_DEFAULT_NUM_INFERENCE_STEPS = 35
COSMOS3_T2V_DEFAULT_GUIDANCE_SCALE = 6.0
COSMOS3_VIDEO_DEFAULT_FLOW_SHIFT = 10.0

COSMOS3_T2I_DEFAULT_HEIGHT = 1024
COSMOS3_T2I_DEFAULT_WIDTH = 1024
COSMOS3_T2I_DEFAULT_NUM_INFERENCE_STEPS = 50
COSMOS3_T2I_DEFAULT_GUIDANCE_SCALE = 7.0
COSMOS3_T2I_DEFAULT_FLOW_SHIFT = 3.0
COSMOS3_T2I_DEFAULT_GUIDANCE_INTERVAL: tuple[float, float] = (400.0, 1000.0)


class Cosmos3Conditioning:
    @staticmethod
    def _apply_metadata_templates(
        prompt: str,
        num_frames: int,
        frame_rate: float,
        height: int,
        width: int,
        duration_template: str | None = COSMOS3_DURATION_TEMPLATE,
        resolution_template: str | None = COSMOS3_RESOLUTION_TEMPLATE,
        force_duration_template: bool = False,
    ) -> str:
        """
        Append duration and resolution metadata to a prompt.
        """
        prompt = prompt.strip()
        if duration_template is None and resolution_template is None:
            return prompt

        parts: list[str] = []
        head = prompt.rstrip(".").strip()
        if head:
            parts.append(head)
        if duration_template is not None and (num_frames > 1 or force_duration_template):
            duration = num_frames / frame_rate
            parts.append(duration_template.format(duration=duration, fps=frame_rate).rstrip("."))
        if resolution_template is not None:
            parts.append(resolution_template.format(height=height, width=width).rstrip("."))
        if not parts:
            return ""
        return ". ".join(parts) + "."

    # -- Tokenization --------------------------------------------------------

    def _tokenize_prompt(
        self,
        text: str,
        max_sequence_length: int,
        use_system_prompt: bool = False,
        system_prompt: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a prompt using the Qwen2 chat template.

        Returns (input_ids, attention_mask) as [1, S] tensors on device.
        """
        conversations = []
        if use_system_prompt:
            conversations.append(
                {
                    "role": "system",
                    "content": system_prompt or COSMOS3_SYSTEM_PROMPT,
                }
            )
        conversations.append({"role": "user", "content": text})

        token_ids = self._normalize_token_ids(
            self.tokenizer.apply_chat_template(conversations, tokenize=True, add_generation_prompt=True)
        )
        original_token_count = len(token_ids)
        if original_token_count > max_sequence_length and (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        ):
            logger.warning(
                "Cosmos3 prompt token_ids shortened to max_sequence_length: "
                "original_token_count=%d, max_sequence_length=%d, removed_token_count=%d",
                original_token_count,
                max_sequence_length,
                original_token_count - max_sequence_length,
            )
        token_ids = token_ids[:max_sequence_length]
        token_ids.append(self.tokenizer.eos_token_id)  # 151645
        token_ids.append(self.tokenizer.convert_tokens_to_ids("<|vision_start|>"))  # 151652
        seq_len = len(token_ids)

        # No right-padding: the prompt is tokenized to its natural length.
        # The UND pathway uses causal self-attention with no padding mask and
        # the GEN cross-attention K/V is trimmed to the real text length, so
        # padding to a fixed length only added dead compute and never changed
        # the output.  ``max_sequence_length`` is kept purely as a truncation
        # cap (above).  The mask is therefore all ones.
        attention_mask = [1] * seq_len

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long, device=self.device)
        return input_ids, attention_mask

    @staticmethod
    def _normalize_token_ids(tokenized_output: object) -> list[int]:
        """Normalize tokenizer outputs into a flat ``list[int]``.

        Different Transformers/tokenizers versions can return ``list[int]``,
        a mapping/BatchEncoding with ``input_ids``, tensors, or
        ``tokenizers.Encoding`` objects from ``apply_chat_template``.
        """
        token_ids = tokenized_output
        while True:
            if isinstance(token_ids, dict) and "input_ids" in token_ids:
                token_ids = token_ids["input_ids"]
            elif hasattr(token_ids, "input_ids"):
                token_ids = token_ids.input_ids
            elif hasattr(token_ids, "ids"):
                token_ids = token_ids.ids
            elif hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()
            elif isinstance(token_ids, tuple):
                token_ids = list(token_ids)
            elif isinstance(token_ids, list) and len(token_ids) == 1:
                first = token_ids[0]
                if isinstance(first, list | tuple):
                    token_ids = list(first)
                elif hasattr(first, "ids") or hasattr(first, "input_ids"):
                    token_ids = first
                elif hasattr(first, "tolist"):
                    first_list = first.tolist()
                    if isinstance(first_list, list | tuple):
                        token_ids = list(first_list)
                    else:
                        break
                else:
                    break
            else:
                break

        if not isinstance(token_ids, list):
            raise TypeError(
                "Cosmos3 tokenizer must return token IDs as a list-like value; "
                f"got {type(token_ids).__name__}: {token_ids!r}"
            )

        normalized_ids = []
        for idx, token_id in enumerate(token_ids):
            if hasattr(token_id, "item"):
                token_id = token_id.item()
            try:
                normalized_ids.append(int(token_id))
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "Cosmos3 tokenizer returned a non-integer token at "
                    f"index {idx}: {type(token_id).__name__}: {token_id!r}"
                ) from exc
        return normalized_ids

    def _format_and_tokenize_prompts(
        self,
        prompt: str,
        negative_prompt: str,
        num_frames: int,
        frame_rate: float,
        height: int,
        width: int,
        max_sequence_length: int,
        sp: OmniDiffusionSamplingParams,
        use_system_prompt: bool = False,
        is_t2i: bool = False,
        system_prompt: str | None = None,
        prompt_suffix: str | None = None,
        use_duration_template: bool | None = None,
        use_resolution_template: bool | None = None,
        negative_metadata_mode: str | None = None,
        aspect_ratio_override: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Format prompts with metadata templates and tokenize.

        Returns (cond_ids, cond_mask, uncond_ids, uncond_mask).

        For T2I (``is_t2i=True``) the duration template is suppressed (no FPS
        or duration concept for a single image) and the image-flavored
        resolution template is used. ``prompt_suffix`` is appended only to the
        positive prompt, after metadata formatting. ``negative_metadata_mode``
        controls whether the negative branch receives no metadata, the same
        metadata as the positive branch, or inverse metadata.
        """
        # Route cosmos3-specific controls through ``_get_sp_param`` so they
        # are picked up from ``extra_args`` (OpenAI endpoint path) as well
        # as from direct attributes.
        if use_duration_template is None:
            use_duration_template = bool(self._get_sp_param(sp, "use_duration_template", False))
        if use_resolution_template is None:
            use_resolution_template = bool(self._get_sp_param(sp, "use_resolution_template", False))
        if negative_metadata_mode is None:
            negative_metadata_mode = str(self._get_sp_param(sp, "negative_metadata_mode", "inverse"))
        negative_metadata_mode = negative_metadata_mode.strip().lower()
        if negative_metadata_mode not in {"none", "same", "inverse"}:
            raise ValueError(
                "Cosmos3 negative_metadata_mode must be one of 'none', 'same', or 'inverse'; "
                f"got {negative_metadata_mode!r}."
            )

        use_duration_template = use_duration_template and not is_t2i
        dur_tmpl = COSMOS3_DURATION_TEMPLATE if use_duration_template else None
        if use_resolution_template:
            res_tmpl = COSMOS3_IMAGE_RESOLUTION_TEMPLATE if is_t2i else COSMOS3_RESOLUTION_TEMPLATE
        else:
            res_tmpl = None
        requested_aspect_ratio = self._get_sp_param(sp, "aspect_ratio", None)
        prompt_aspect_ratio = _json_object_aspect_ratio(prompt)
        metadata_aspect_ratio = requested_aspect_ratio if requested_aspect_ratio is not None else prompt_aspect_ratio
        canvas_aspect_ratio = aspect_ratio_override or _aspect_ratio_for_dimensions(width, height)
        normalized_metadata_ratio = _normalize_aspect_ratio(metadata_aspect_ratio)
        has_aspect_ratio_conflict = (
            aspect_ratio_override is None
            and metadata_aspect_ratio is not None
            and normalized_metadata_ratio != canvas_aspect_ratio
        )
        if has_aspect_ratio_conflict:
            aspect_ratio_source = (
                "requested aspect_ratio" if requested_aspect_ratio is not None else "JSON prompt aspect_ratio"
            )
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                logger.warning(
                    "Cosmos3 %s=%r conflicts with the generated %dx%d canvas (WxH); using %s for prompt metadata.",
                    aspect_ratio_source,
                    metadata_aspect_ratio,
                    width,
                    height,
                    canvas_aspect_ratio,
                )
            metadata_aspect_ratio = canvas_aspect_ratio
        elif aspect_ratio_override is not None:
            metadata_aspect_ratio = aspect_ratio_override

        json_prompt = _format_json_object_prompt(
            prompt,
            num_frames=num_frames,
            frame_rate=frame_rate,
            height=height,
            width=width,
            aspect_ratio=metadata_aspect_ratio,
        )
        if json_prompt is not None:
            prompt = json_prompt
        else:
            prompt = self._apply_metadata_templates(
                prompt,
                num_frames,
                frame_rate,
                height,
                width,
                duration_template=dur_tmpl,
                resolution_template=res_tmpl,
            )
        if prompt_suffix:
            prompt = f"{prompt.rstrip()} {prompt_suffix.lstrip()}".strip()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Final prompt: '%s'", prompt)

        if negative_metadata_mode == "none":
            negative_dur_tmpl = None
            negative_res_tmpl = None
        elif negative_metadata_mode == "same":
            negative_dur_tmpl = dur_tmpl
            negative_res_tmpl = res_tmpl
        else:
            negative_dur_tmpl = COSMOS3_INVERSE_DURATION_TEMPLATE if dur_tmpl else None
            if res_tmpl is None:
                negative_res_tmpl = None
            elif is_t2i:
                negative_res_tmpl = COSMOS3_INVERSE_IMAGE_RESOLUTION_TEMPLATE
            else:
                negative_res_tmpl = COSMOS3_INVERSE_RESOLUTION_TEMPLATE
        negative_prompt = self._apply_metadata_templates(
            negative_prompt,
            num_frames,
            frame_rate,
            height,
            width,
            duration_template=negative_dur_tmpl,
            resolution_template=negative_res_tmpl,
            force_duration_template=negative_metadata_mode == "inverse",
        )

        if system_prompt is None:
            default_sys_prompt = COSMOS3_T2I_SYSTEM_PROMPT if is_t2i else COSMOS3_SYSTEM_PROMPT
            system_prompt = self._get_sp_param(sp, "system_prompt", default_sys_prompt) or default_sys_prompt
        cond_ids, cond_mask = self._tokenize_prompt(
            prompt, max_sequence_length, use_system_prompt, system_prompt=system_prompt
        )
        uncond_ids, uncond_mask = self._tokenize_prompt(
            negative_prompt, max_sequence_length, use_system_prompt, system_prompt=system_prompt
        )
        return cond_ids, cond_mask, uncond_ids, uncond_mask

    @staticmethod
    def _get_sp_param(sp: OmniDiffusionSamplingParams, key: str, default: Any = None) -> Any:
        """Read a runtime control from sampling params.

        Order of precedence:
            1. ``sp.extra_args[key]`` - preferred path; the OpenAI image/video
               endpoints surface custom controls here (see e.g.
               ``serving_video.py`` writing ``extra_args['flow_shift']``).
            2. direct attribute on ``sp`` - backward compat for callers that
               set attributes directly.
            3. ``default``.

        Skipping this helper would cause API-driven overrides like
        ``request.flow_shift`` (forwarded as ``extra_args['flow_shift']``) to
        be silently ignored.
        """
        extra = sp.extra_args or {}
        if extra.get(key) is not None:
            return extra[key]
        val = getattr(sp, key, None)
        if val is not None:
            return val
        return default
