# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers.utils import export_to_video

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

SUPPORTED_MODEL_MODES = {"image2video", "text2video"}
SUPPORTED_RESOLUTIONS = {"480": (832, 480)}


def _safe_camera_name(camera: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", camera).strip("_") or "camera"


def _safe_sample_name(name: str, sample_index: int) -> str:
    safe_name = _safe_camera_name(name)
    if safe_name in {".", ".."}:
        return f"sample_{sample_index:04d}"
    return safe_name


def _load_requests(input_path: Path) -> list[dict[str, Any]]:
    if input_path.suffix.lower() == ".jsonl":
        requests = []
        for line_number, line in enumerate(input_path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {input_path} line {line_number}: {exc}") from exc
            if not isinstance(request, dict):
                raise TypeError(f"{input_path} line {line_number} must contain a JSON object.")
            requests.append(request)
        if not requests:
            raise ValueError(f"Input JSONL file is empty: {input_path}")
        return requests

    request = json.loads(input_path.read_text())
    if not isinstance(request, dict):
        raise TypeError(f"Input JSON must contain one object, got {type(request).__name__}.")
    return [request]


def _resolve_model_mode(request: dict[str, Any], views: list[dict[str, Any]]) -> str:
    vision_present = [view.get("vision_path", view.get("vision")) is not None for view in views]
    inferred_mode = "image2video" if any(vision_present) else "text2video"
    model_mode = str(request.get("model_mode", inferred_mode)).strip().lower()
    if model_mode not in SUPPORTED_MODEL_MODES:
        raise ValueError(
            f"Unsupported model_mode {model_mode!r}; expected one of {sorted(SUPPORTED_MODEL_MODES)}."
        )
    if model_mode == "image2video" and not all(vision_present):
        raise ValueError("model_mode='image2video' requires vision input for every camera view.")
    if model_mode == "text2video" and any(vision_present):
        raise ValueError("model_mode='text2video' must not include per-camera vision inputs.")
    return model_mode


def _resolve_resolution(request: dict[str, Any], multiview: dict[str, Any]) -> tuple[str, int, int]:
    top_level = request.get("resolution")
    nested = multiview.get("resolution")
    if top_level is not None and nested is not None and str(top_level) != str(nested):
        raise ValueError(
            "Conflicting Cosmos3 multiview resolutions: "
            f"top-level resolution={top_level!r}, multiview.resolution={nested!r}."
        )
    resolution = str(nested if nested is not None else top_level if top_level is not None else "480")
    if resolution not in SUPPORTED_RESOLUTIONS:
        raise ValueError(
            f"Unsupported Cosmos3 multiview resolution {resolution!r}; "
            f"expected one of {sorted(SUPPORTED_RESOLUTIONS)}."
        )
    width, height = SUPPORTED_RESOLUTIONS[resolution]
    return resolution, width, height


def _resolve_seed(request: dict[str, Any], base_seed: int, sample_index: int) -> int:
    value = request.get("seed", base_seed + sample_index)
    if isinstance(value, bool):
        raise TypeError("seed must be an integer, not a boolean.")
    return int(value)


def _extract_payload(value: Any) -> tuple[Any, dict[str, Any]]:
    if isinstance(value, list) and len(value) == 1:
        return _extract_payload(value[0])
    if isinstance(value, OmniRequestOutput):
        if value.images:
            return _extract_payload(value.images[0] if len(value.images) == 1 else value.images)
        raise ValueError("Cosmos3 multiview inference returned no video output.")
    if isinstance(value, dict):
        metadata = value.get("metadata") if isinstance(value.get("metadata"), dict) else {}
        payload = value.get("payload") if isinstance(value.get("payload"), dict) else value
        if "video" in payload:
            return payload["video"], metadata
    return value, {}


def _frame_list(video: Any) -> list[Any]:
    if isinstance(video, torch.Tensor):
        tensor = video.detach().cpu()
        if tensor.ndim == 5:
            tensor = tensor[0]
        if tensor.ndim == 4 and tensor.shape[0] in (3, 4):
            tensor = tensor.permute(1, 2, 3, 0)
        if tensor.is_floating_point() and tensor.numel() and tensor.min() < 0:
            tensor = tensor.mul(0.5).add(0.5)
        return list(tensor.clamp(0, 1).numpy())
    if isinstance(video, np.ndarray):
        array = video[0] if video.ndim == 5 else video
        if np.issubdtype(array.dtype, np.integer):
            array = array.astype(np.float32) / 255.0
        return list(array)
    if isinstance(video, list):
        if len(video) == 1 and isinstance(video[0], list):
            return video[0]
        if len(video) == 1 and isinstance(video[0], np.ndarray) and video[0].ndim == 4:
            return list(video[0])
        return video
    raise TypeError(f"Unsupported multiview video output type: {type(video).__name__}.")


def _run_request(
    omni: Omni,
    request: dict[str, Any],
    *,
    output_dir: Path,
    seed: int,
    fallback_negative_prompt: str | None,
) -> dict[str, Any]:
    multiview_value = request.get("multiview")
    if not isinstance(multiview_value, dict):
        raise ValueError("Input JSON must contain a multiview object.")
    multiview = dict(multiview_value)
    views_value = multiview.get("views")
    if not isinstance(views_value, list) or not views_value:
        raise ValueError("Input JSON must contain multiview.views.")
    if not all(isinstance(view, dict) for view in views_value):
        raise TypeError("Every multiview.views entry must be an object.")
    views: list[dict[str, Any]] = views_value

    model_mode = _resolve_model_mode(request, views)
    resolution, width, height = _resolve_resolution(request, multiview)
    # Keep the resolved value with the variant-owned multiview parameters so
    # top-level Imaginaire inputs and native vLLM-Omni inputs behave identically.
    multiview["resolution"] = resolution
    num_frames = int(multiview.get("num_frames", request.get("num_frames", 93)))
    fps = float(request.get("fps", 10))

    extra_args = {
        "multiview": multiview,
        "resolution": resolution,
        "wsm": request.get("wsm", {}),
    }
    sampling_params = OmniDiffusionSamplingParams(
        height=height,
        width=width,
        num_frames=num_frames,
        fps=fps,
        num_inference_steps=int(request.get("num_inference_steps", 35)),
        guidance_scale=float(request.get("guidance_scale", 6.0)),
        seed=seed,
        extra_args=extra_args,
    )
    prompt = {
        "prompt": str(request.get("prompt", "")),
        "modalities": ["video"],
    }
    if request.get("negative_prompt") is not None:
        prompt["negative_prompt"] = request["negative_prompt"]
    elif fallback_negative_prompt is not None:
        prompt["negative_prompt"] = fallback_negative_prompt

    result = omni.generate(prompt, sampling_params)
    video, metadata = _extract_payload(result)
    frames = _frame_list(video)

    frames_per_view = int(metadata.get("multiview", {}).get("frames_per_view", sampling_params.num_frames))
    cameras = metadata.get("multiview", {}).get("cameras") or [view["camera_key"] for view in views]
    output_fps = float(metadata.get("multiview", {}).get("fps", sampling_params.fps or 10))
    if len(frames) != len(cameras) * frames_per_view:
        raise ValueError(f"Expected {len(cameras) * frames_per_view} camera-major frames, got {len(frames)}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    files_by_camera = {}
    for index, camera in enumerate(cameras):
        camera_frames = frames[index * frames_per_view : (index + 1) * frames_per_view]
        output_path = output_dir / f"vision_view{index:02d}_{_safe_camera_name(camera)}.mp4"
        export_to_video(camera_frames, str(output_path), fps=output_fps)
        files_by_camera[camera] = [str(output_path)]

    manifest = {
        "name": request.get("name"),
        "model_mode": model_mode,
        "resolution": resolution,
        "seed": seed,
        "prompt": prompt["prompt"],
        "multiview_cameras": cameras,
        "frames_per_view": frames_per_view,
        "fps": output_fps,
        "files_by_camera": files_by_camera,
    }
    (output_dir / "sample_outputs.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Exported Cosmos3 Multiview-AV Diffusers directory")
    parser.add_argument("--input", required=True, type=Path, help="Multiview JSON or JSONL file")
    parser.add_argument("--output-dir", type=Path, default=Path("cosmos3_multiview_output"))
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed; JSONL records use seed + record index unless the record provides seed",
    )
    parser.add_argument(
        "--negative-prompt-json",
        type=Path,
        help=(
            "Structured negative prompt to serialize with json.dumps defaults. The pipeline ships no default "
            "negative prompt, so reference-parity runs must supply the reference one here."
        ),
    )
    args = parser.parse_args()

    requests = _load_requests(args.input)
    fallback_negative_prompt = None
    if args.negative_prompt_json is not None:
        # Default separators (", " and ": ") and the file's key order are part
        # of the reference's serialization, so keep json.dumps unconfigured.
        fallback_negative_prompt = json.dumps(json.loads(args.negative_prompt_json.read_text()))

    omni = Omni(
        model=args.model,
        dtype="bfloat16",
        model_class_name="Cosmos3MultiviewPipeline",
        enforce_eager=False,
        ulysses_degree=1,
        ring_degree=1,
        cfg_parallel_size=1,
        diffusion_compile_granularity="regional",
        diffusion_compile_dynamic=False,
    )

    sample_names = [
        _safe_sample_name(str(request.get("name") or f"sample_{sample_index:04d}"), sample_index)
        for sample_index, request in enumerate(requests)
    ]
    if len(sample_names) != len(set(sample_names)):
        duplicates = sorted({name for name in sample_names if sample_names.count(name) > 1})
        raise ValueError(f"Duplicate output sample names after sanitization: {duplicates}.")

    is_batch = len(requests) > 1
    manifests = []
    for sample_index, (request, sample_name) in enumerate(zip(requests, sample_names, strict=True)):
        output_dir = args.output_dir / sample_name if is_batch else args.output_dir
        seed = _resolve_seed(request, args.seed, sample_index)
        print(f"[{sample_index + 1}/{len(requests)}] Generating {sample_name!r} with seed {seed}...")
        manifest = _run_request(
            omni,
            request,
            output_dir=output_dir,
            seed=seed,
            fallback_negative_prompt=fallback_negative_prompt,
        )
        manifests.append({**manifest, "output_dir": str(output_dir)})

    if is_batch:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "sample_outputs.jsonl").write_text(
            "".join(json.dumps(manifest) + "\n" for manifest in manifests)
        )


if __name__ == "__main__":
    main()
