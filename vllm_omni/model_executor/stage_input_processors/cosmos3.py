# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Canonical text-only conditioning admission for the shared Cosmos3 UND stage."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from PIL import Image
from transformers import AutoTokenizer

from vllm_omni.diffusion.models.cosmos3.conditioning import (
    COSMOS3_T2I_DEFAULT_GUIDANCE_SCALE,
    COSMOS3_T2I_DEFAULT_HEIGHT,
    COSMOS3_T2I_DEFAULT_WIDTH,
    COSMOS3_T2V_DEFAULT_GUIDANCE_SCALE,
    COSMOS3_T2V_DEFAULT_HEIGHT,
    COSMOS3_T2V_DEFAULT_NUM_FRAMES,
    COSMOS3_T2V_DEFAULT_WIDTH,
    Cosmos3Conditioning,
    _ceil_video_num_frames,
)
from vllm_omni.model_executor.stage_input_processors.bagel import ExpandedPrompt

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams

    from vllm_omni.distributed.omni_connectors.kv_transfer_manager import OmniKVTransferManager


class Cosmos3PromptTransform(Cosmos3Conditioning):
    """Instantiated once per frontend with the same checkpoint revision as UND."""

    def __init__(self, config: VllmConfig):
        mc = config.model_config
        self.tokenizer = AutoTokenizer.from_pretrained(
            mc.model, subfolder="text_tokenizer", revision=mc.revision, trust_remote_code=mc.trust_remote_code
        )
        self.device = torch.device("cpu")

    def __call__(self, prompt: dict | str, sampling_params_list: list) -> dict | str:
        if isinstance(prompt, dict) and "img2img" in prompt.get("modalities", ()):
            raise ValueError("Shared Cosmos3 UND does not support image editing")
        if not isinstance(prompt, dict) or not set(prompt.get("modalities", ())) & {"image", "video"}:
            return prompt
        if len(sampling_params_list) != 2:
            raise ValueError("Shared Cosmos3 UND requires two stages")
        sp = copy.copy(sampling_params_list[1])
        sp.extra_args = dict(sp.extra_args or {})
        sampling_params_list[1] = sp
        if set(prompt["modalities"]) not in ({"image"}, {"video"}):
            raise ValueError("Request exactly one of image or video generation")
        if sp.lora_request is not None:
            raise ValueError("Shared Cosmos3 UND does not support LoRA")
        mm = prompt.get("multi_modal_data") or {}
        is_image = "image" in prompt["modalities"]
        if mm.get("video") is not None or mm.get("audio") is not None:
            raise ValueError("Shared Cosmos3 UND supports T2I, T2V and I2V only")
        if (sp.num_outputs_per_prompt or 1) != 1:
            raise ValueError("Shared Cosmos3 UND requires one output per request")
        for key in (
            "generate_sound",
            "sound_gen",
            "action",
            "action_mode",
            "observation",
            "robot_obs",
            "control_path",
            "transfer",
        ):
            if self._get_sp_param(sp, key) or prompt.get(key):
                raise ValueError(f"Shared Cosmos3 UND does not support {key}")
        image = mm.get("image")
        if isinstance(image, list):
            if len(image) != 1:
                raise ValueError("I2V requires one reference image")
            image = image[0]
        if image is not None:
            if is_image:
                raise ValueError("Image editing is not supported by shared Cosmos3 UND")
            if not isinstance(image, Image.Image):
                raise ValueError("I2V admission requires a decoded PIL reference image")
            ratio = image.height / image.width
            sp.height = (
                sp.height or round(math.sqrt(COSMOS3_T2V_DEFAULT_HEIGHT * COSMOS3_T2V_DEFAULT_WIDTH * ratio)) // 16 * 16
            )
            sp.width = (
                sp.width or round(math.sqrt(COSMOS3_T2V_DEFAULT_HEIGHT * COSMOS3_T2V_DEFAULT_WIDTH / ratio)) // 16 * 16
            )
        sp.height = sp.height or (COSMOS3_T2I_DEFAULT_HEIGHT if is_image else COSMOS3_T2V_DEFAULT_HEIGHT)
        sp.width = sp.width or (COSMOS3_T2I_DEFAULT_WIDTH if is_image else COSMOS3_T2V_DEFAULT_WIDTH)
        sp.num_frames = 1 if is_image else _ceil_video_num_frames(sp.num_frames or COSMOS3_T2V_DEFAULT_NUM_FRAMES, 4)
        fps = self._get_sp_param(sp, "resolved_frame_rate") or self._get_sp_param(sp, "frame_rate") or 24.0
        if min(sp.height, sp.width, sp.num_frames, fps) <= 0:
            raise ValueError("Cosmos3 dimensions, frames and FPS must be positive")
        sp.extra_args["resolved_frame_rate"] = fps
        guidance = (
            float(sp.guidance_scale)
            if sp.guidance_scale is not None
            else (COSMOS3_T2I_DEFAULT_GUIDANCE_SCALE if is_image else COSMOS3_T2V_DEFAULT_GUIDANCE_SCALE)
        )
        sp.guidance_scale = guidance
        sp.guidance_scale_provided = True
        negative = prompt.get("negative_prompt")
        if negative is None:
            negative = self._get_sp_param(sp, "negative_prompt", "")
        cap = self._get_sp_param(sp, "max_sequence_length", 4096) or 4096
        positive_ids, _, negative_ids, _ = self._format_and_tokenize_prompts(
            prompt.get("prompt", ""),
            negative,
            sp.num_frames,
            fps,
            sp.height,
            sp.width,
            cap,
            sp,
            bool(self._get_sp_param(sp, "use_system_prompt", False)),
            is_t2i=is_image,
        )
        positive_ids = positive_ids[0].tolist()
        negative_ids = negative_ids[0].tolist()
        sp.extra_args["cosmos3_expected_lengths"] = {"positive": len(positive_ids), "negative": len(negative_ids)}
        ar = sampling_params_list[0].clone()
        ar.max_tokens = 1
        ar.min_tokens = 0
        ar.extra_args = dict(ar.extra_args or {})
        ar.extra_args["cosmos3_negative_ids"] = negative_ids if guidance > 1 else None
        sampling_params_list[0] = ar
        return {
            "prompt_token_ids": positive_ids,
            "additional_information": {
                "cosmos3_conditioning_length": len(positive_ids),
                "omni_wait_for_processed_kv_output": True,
            },
        }


def expand_cfg_prompts(prompt: dict | str, sampling_params: SamplingParams) -> list[ExpandedPrompt]:
    tokens = (sampling_params.extra_args or {}).get("cosmos3_negative_ids")
    if tokens is None:
        return []
    return [
        ExpandedPrompt(
            prompt={"prompt_token_ids": tokens, "additional_information": {"cosmos3_conditioning_length": len(tokens)}},
            role="cfg_text",
            request_id_suffix="__cosmos3_negative",
            sampling_params_override={"max_tokens": 1},
        )
    ]


@dataclass
class DenseKV:
    key_cache: list[torch.Tensor]
    value_cache: list[torch.Tensor]


def collect_cfg_kv_caches(
    request_id: str,
    cfg_request_ids: dict[str, str],
    kv_transfer_manager: OmniKVTransferManager,
    target_device: torch.device | None = None,
) -> dict:
    result = {}
    for role, companion_id in cfg_request_ids.items():
        data, _ = kv_transfer_manager.receive_kv_cache_for_request(companion_id, target_device)
        if not data or "layer_blocks" not in data:
            raise ValueError(f"Missing Cosmos3 {role} conditioning for {request_id}")
        result[f"{role}_past_key_values"] = DenseKV(**data["layer_blocks"])
        result[f"{role}_kv_metadata"] = data.get("metadata", {})
    return result
