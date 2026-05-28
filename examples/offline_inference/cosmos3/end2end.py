# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import hashlib
import json
import os
import statistics
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import PIL.Image
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import COSMOS3_VAE_TEMPORAL_COMPRESSION
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

DEFAULT_NEGATIVE_PROMPT = "blurry, distorted, low quality"
TASK_DEFAULTS = {
    "t2i": {
        "height": 960,
        "width": 960,
        "num_frames": None,
        "num_inference_steps": 50,
        "guidance_scale": 4.0,
        "flow_shift": 3.0,
        "fps": 24,
        "output": "cosmos3_t2i.png",
    },
    "t2v": {
        "height": 720,
        "width": 1280,
        "num_frames": 189,
        "num_inference_steps": 35,
        "guidance_scale": 6.0,
        "flow_shift": 10.0,
        "fps": 24,
        "output": "cosmos3_t2v.mp4",
    },
    "i2v": {
        "height": 720,
        "width": 1280,
        "num_frames": 189,
        "num_inference_steps": 35,
        "guidance_scale": 6.0,
        "flow_shift": 10.0,
        "fps": 24,
        "output": "cosmos3_i2v.mp4",
    },
    "v2v": {
        "height": 720,
        "width": 1280,
        "num_frames": 189,
        "num_inference_steps": 35,
        "guidance_scale": 6.0,
        "flow_shift": 10.0,
        "fps": 24,
        "output": "cosmos3_v2v.mp4",
    },
    "t2v_sound": {
        "height": 720,
        "width": 1280,
        "num_frames": 189,
        "num_inference_steps": 35,
        "guidance_scale": 6.0,
        "flow_shift": 10.0,
        "fps": 24,
        "output": "cosmos3_t2v_sound.mp4",
    },
    "action_policy": {
        "height": 480,
        "width": 640,
        "num_frames": 17,
        "num_inference_steps": 30,
        "guidance_scale": 1.0,
        "flow_shift": 5.0,
        "fps": 24,
        "output": "cosmos3_action_policy.mp4",
    },
    "action_forward_dynamics": {
        "height": 480,
        "width": 640,
        "num_frames": 17,
        "num_inference_steps": 30,
        "guidance_scale": 1.0,
        "flow_shift": 5.0,
        "fps": 5,
        "output": "cosmos3_action_forward_dynamics.mp4",
    },
    "action_inverse_dynamics": {
        "height": 480,
        "width": 640,
        "num_frames": 17,
        "num_inference_steps": 30,
        "guidance_scale": 1.0,
        "flow_shift": 5.0,
        "fps": 5,
        "output": "cosmos3_action_inverse_dynamics.mp4",
    },
}

_INPUTS_DIR = Path(__file__).resolve().parent / "inputs"
_TASK_ACTION_MODES = {
    "action_policy": "policy",
    "action_forward_dynamics": "forward_dynamics",
    "action_inverse_dynamics": "inverse_dynamics",
}
_ACTION_TASKS = set(_TASK_ACTION_MODES)
_VIDEO_INPUT_TASKS = {"v2v", "action_inverse_dynamics"}
_IMAGE_INPUT_TASKS = {"i2v", "action_policy", "action_forward_dynamics"}
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
_CACHE_DIR = Path(
    os.environ.get(
        "COSMOS3_EXAMPLE_CACHE",
        str(Path(tempfile.gettempdir()) / "cosmos3_examples"),
    )
)
_JSON_TO_ATTR = {
    "prompt": "prompt",
    "negative_prompt": "negative_prompt",
    "vision_path": "vision_path",
    "action_path": "action_path",
    "height": "height",
    "width": "width",
    "num_frames": "num_frames",
    "num_inference_steps": "num_inference_steps",
    "guidance_scale": "guidance_scale",
    "flow_shift": "flow_shift",
    "fps": "fps",
    "seed": "seed",
    "action_mode": "action_mode",
    "action_chunk_size": "action_chunk_size",
    "raw_action_dim": "raw_action_dim",
    "domain_name": "domain_name",
    "domain_id": "domain_id",
    "generate_sound": "generate_sound",
    "sound_duration": "sound_duration",
    "condition_frame_indexes_vision": "condition_frame_indexes_vision",
    "condition_video_keep": "condition_video_keep",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cosmos3 offline inference examples.")
    parser.add_argument(
        "--model",
        default=os.environ.get("COSMOS3_MODEL", "nvidia/Cosmos3-Nano"),
        help="Cosmos3 checkpoint (Hugging Face repo id or local Diffusers-format path). "
        "Defaults to COSMOS3_MODEL when set, otherwise nvidia/Cosmos3-Nano.",
    )
    parser.add_argument(
        "--task",
        choices=sorted(TASK_DEFAULTS),
        default="t2v",
        help="Cosmos3 example task to run.",
    )
    parser.add_argument(
        "--input-json",
        default=None,
        help="Path to a JSON or JSONL input file (e.g. inputs/t2v.json). When given, every recognized "
        "field overrides the matching default; explicit CLI flags still win. Use JSONL to batch multiple "
        "generations in one invocation (e.g. inputs/action_forward_dynamics_camera.jsonl).",
    )
    parser.add_argument(
        "--prompt",
        default="A small warehouse robot moves a blue box across a clean floor.",
        help="Text prompt. Overrides any prompt loaded from --input-json.",
    )
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt.")
    parser.add_argument(
        "--image",
        default=None,
        help="Input image path for i2v / image-input action tasks. Alias for --vision-path.",
    )
    parser.add_argument(
        "--vision-path",
        default=None,
        help="Vision input as a local path or http(s) URL. Image file for i2v / policy; image or video file "
        "for v2v / forward_dynamics; video file for inverse_dynamics. If a video is supplied for i2v / policy, "
        "the first frame is extracted automatically (requires imageio).",
    )
    parser.add_argument(
        "--action-path",
        default=None,
        help="Local path or URL to an action JSON for forward_dynamics tasks.",
    )
    parser.add_argument(
        "--action-mode",
        default=None,
        choices=["forward_dynamics", "inverse_dynamics", "policy"],
        help="Override action_mode. Defaults are derived from --task.",
    )
    parser.add_argument(
        "--generate-sound",
        action="store_true",
        help="Enable sound generation.",
    )
    parser.add_argument("--output", default=None, help="Output PNG or MP4 path. Default depends on --task.")
    parser.add_argument(
        "--action-output",
        default=None,
        help="Action JSON path for inverse_dynamics / action_policy outputs. "
        "Defaults to the video output stem plus _action.json.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--height", type=int, default=None, help="Output height. Default depends on --task.")
    parser.add_argument("--width", type=int, default=None, help="Output width. Default depends on --task.")
    parser.add_argument("--num-frames", type=int, default=None, help="Video frames. Default depends on --task.")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Sampling steps. Default depends on --task.",
    )
    parser.add_argument("--guidance-scale", type=float, default=None, help="CFG scale. Default depends on --task.")
    parser.add_argument(
        "--flow-shift",
        type=float,
        default=None,
        help="Flow-matching scheduler shift. Default depends on --task (cosmos3-internal: 3.0 t2i / 10.0 t2v/i2v / 5.0 action).",
    )
    parser.add_argument("--fps", type=int, default=None, help="Output video fps. Default depends on --task.")
    parser.add_argument(
        "--sound-duration",
        type=float,
        default=None,
        help="Audio duration in seconds for t2v_sound. Defaults to generated video duration.",
    )
    parser.add_argument(
        "--condition-frame-indexes-vision",
        default="0,1",
        help="Comma-separated latent frame indexes conditioned by a V2V source video.",
    )
    parser.add_argument(
        "--condition-video-keep",
        default="first",
        choices=["first", "last"],
        help="Use the first or last source frames when trimming a V2V source video.",
    )
    parser.add_argument(
        "--audio-sample-rate",
        type=int,
        default=24000,
        help="Fallback sample rate used when muxing audio if the model does not return one.",
    )
    parser.add_argument(
        "--domain-name",
        default="bridge_orig_lerobot",
        help="Cosmos3 action embodiment name for action_policy.",
    )
    parser.add_argument("--domain-id", type=int, default=None, help="Cosmos3 action embodiment id.")
    parser.add_argument(
        "--raw-action-dim",
        type=int,
        default=2,
        help="Number of action dimensions to keep for action_policy.",
    )
    parser.add_argument(
        "--action-chunk-size",
        type=int,
        default=16,
        help="Number of action steps for action_policy.",
    )
    parser.add_argument(
        "--cache-backend",
        type=str,
        default=None,
        choices=["cache_dit"],
        help="Cache backend for supported Cosmos3 generation paths.",
    )
    parser.add_argument(
        "--disable-guardrails",
        "--no-guardrails",
        dest="disable_guardrails",
        action="store_true",
        help="Disable Cosmos3 text/video guardrails for this inference run.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark mode: discard one warmup generation, then time benchmark generations.",
    )
    parser.add_argument(
        "--benchmark-generations",
        type=int,
        default=None,
        metavar="N",
        help="Number of timed generations for benchmark mode. Providing this flag implies --benchmark.",
    )
    parser.add_argument("--enable-layerwise-offload", action="store_true", help="Enable layerwise offload.")
    parser.add_argument("--vae-use-slicing", action="store_true", help="Enable VAE slicing.")
    parser.add_argument("--vae-use-tiling", action="store_true", help="Enable VAE tiling.")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile.")
    parser.add_argument("--ulysses-degree", type=int, default=1, help="Ulysses sequence parallel degree.")
    parser.add_argument("--ring-degree", type=int, default=1, help="Ring sequence parallel degree.")
    parser.add_argument("--cfg-parallel-size", type=int, default=1, choices=[1, 2], help="CFG parallel size.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--vae-patch-parallel-size", type=int, default=1, help="VAE patch parallel size.")
    parser.add_argument("--use-hsdp", action="store_true", help="Enable HSDP.")
    parser.add_argument("--hsdp-shard-size", type=int, default=1, help="HSDP shard size.")
    parser.add_argument("--hsdp-replicate-size", type=int, default=1, help="HSDP replicate size.")
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["fp8", "mxfp8", "int8", "gguf"],
        help="Transformer quantization method.",
    )
    return parser.parse_args()


def _cache_config(cache_backend: str | None) -> dict[str, Any] | None:
    if cache_backend != "cache_dit":
        return None
    return {
        "Fn_compute_blocks": 1,
        "Bn_compute_blocks": 0,
        "max_warmup_steps": 4,
        "max_cached_steps": 20,
        "residual_diff_threshold": 0.24,
        "max_continuous_cached_steps": 3,
        "enable_taylorseer": False,
        "taylorseer_order": 1,
        "scm_steps_mask_policy": None,
        "scm_steps_policy": "dynamic",
    }


def _is_url(value: str) -> bool:
    return urllib.parse.urlparse(value).scheme in {"http", "https"}


def _is_video_path(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    target = parsed.path if parsed.scheme else value
    return Path(target).suffix.lower() in _VIDEO_EXTENSIONS


def _resolve_local_path(path_or_url: str) -> str:
    if not _is_url(path_or_url):
        return path_or_url
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(urllib.parse.urlparse(path_or_url).path).suffix or ""
    digest = hashlib.sha256(path_or_url.encode("utf-8")).hexdigest()[:16]
    target = _CACHE_DIR / f"{digest}{suffix}"
    if not target.exists():
        print(f"Downloading {path_or_url} -> {target}")
        with urllib.request.urlopen(path_or_url) as response, open(target, "wb") as fh:
            fh.write(response.read())
    return str(target)


def _first_video_frame(video_path: str) -> PIL.Image.Image:
    try:
        import imageio.v3 as iio  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Extracting the first frame of a video for an image-input task requires imageio. "
            "Install with `pip install imageio[ffmpeg]` or pass a still image via --vision-path."
        ) from exc
    frame = np.asarray(iio.imread(video_path, index=0))
    return PIL.Image.fromarray(frame).convert("RGB")


def _load_video_frames_from(path_or_url: str, max_frames: int) -> list[PIL.Image.Image]:
    if max_frames <= 0:
        raise ValueError(f"max_frames must be positive, got {max_frames}.")

    try:
        import imageio.v3 as iio  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Loading video frames for Cosmos3 action video input requires imageio. "
            "Install with `pip install imageio[ffmpeg]`."
        ) from exc

    local = _resolve_local_path(path_or_url)
    frames: list[PIL.Image.Image] = []
    for frame in iio.imiter(local):
        frames.append(PIL.Image.fromarray(np.asarray(frame)).convert("RGB"))
        if len(frames) >= max_frames:
            break
    if not frames:
        raise ValueError(f"Cosmos3 action video input contains no frames: {path_or_url}")
    return frames


def _parse_condition_frame_indexes_vision(value: Any) -> list[int]:
    if value is None:
        return [0, 1]
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, int):
        values = [value]
    else:
        values = list(value)
    indexes = sorted({int(index) for index in values})
    if not indexes or any(index < 0 for index in indexes):
        raise ValueError(f"condition_frame_indexes_vision must contain non-negative indexes, got {value!r}.")
    return indexes


def _condition_video_pixel_frames(condition_frame_indexes_vision: list[int]) -> int:
    return max(condition_frame_indexes_vision) * COSMOS3_VAE_TEMPORAL_COMPRESSION + 1


def _load_image_from(path_or_url: str) -> PIL.Image.Image:
    local = _resolve_local_path(path_or_url)
    if _is_video_path(path_or_url):
        return _first_video_frame(local)
    return PIL.Image.open(local).convert("RGB")


def _load_input_records(path: str) -> list[dict[str, Any]]:
    src = Path(path)
    if not src.exists():
        candidate = _INPUTS_DIR / path
        if candidate.exists():
            src = candidate
    if not src.exists():
        raise FileNotFoundError(f"Input JSON file not found: {path}")
    text = src.read_text(encoding="utf-8").strip()
    if src.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return [json.loads(text)]


def _cli_provided_attrs(argv: list[str]) -> set[str]:
    provided: set[str] = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        flag = token.split("=", 1)[0][2:]
        provided.add(flag.replace("-", "_"))
    return provided


def _apply_record(record: dict[str, Any], args: argparse.Namespace, cli_set: set[str]) -> None:
    # --image and --vision-path are aliases for the same visual input. A CLI
    # value for either should suppress a JSON override of the other.
    effective_cli_set = set(cli_set)
    if "image" in effective_cli_set or "vision_path" in effective_cli_set:
        effective_cli_set |= {"image", "vision_path"}
    for key, value in record.items():
        attr = _JSON_TO_ATTR.get(key)
        if attr is None:
            print(f"Ignoring unknown input-json field: {key}")
            continue
        if attr in effective_cli_set:
            continue
        if attr == "generate_sound" and not bool(value):
            continue
        setattr(args, attr, value)


def _first_output(outputs: Any) -> Any:
    if isinstance(outputs, list):
        if not outputs:
            raise ValueError("No output generated.")
        return outputs[0]
    return outputs


def _inner_output(output: Any) -> Any:
    if isinstance(output, OmniRequestOutput) and output.is_pipeline_output and output.request_output is not None:
        return output.request_output
    return output


def _extract_images(outputs: Any) -> list[Any]:
    output = _inner_output(_first_output(outputs))
    if isinstance(output, OmniRequestOutput) and output.images:
        return output.images
    images = getattr(output, "images", None)
    if images:
        return images
    raise ValueError("No images found in output.")


def _extract_video_audio_action(outputs: Any) -> tuple[Any, Any | None, int | None, dict[str, Any]]:
    outer = _first_output(outputs)
    output = _inner_output(outer)
    audio = None
    audio_sample_rate = None
    action = {}

    for candidate in (outer, output):
        if isinstance(candidate, OmniRequestOutput):
            if candidate.multimodal_output:
                audio = audio or candidate.multimodal_output.get("audio")
                audio_sample_rate = audio_sample_rate or candidate.multimodal_output.get("audio_sample_rate")
            if candidate.custom_output:
                action.update(candidate.custom_output)

    videos = None
    if isinstance(output, OmniRequestOutput):
        if output.multimodal_output:
            videos = output.multimodal_output.get("video")
            audio = audio or output.multimodal_output.get("audio")
            audio_sample_rate = audio_sample_rate or output.multimodal_output.get("audio_sample_rate")
        if videos is None and output.images:
            videos = output.images
    else:
        videos = getattr(output, "images", None)
        mm = getattr(output, "multimodal_output", None)
        if mm:
            videos = videos or mm.get("video")
            audio = audio or mm.get("audio")
            audio_sample_rate = audio_sample_rate or mm.get("audio_sample_rate")

    if isinstance(videos, list) and len(videos) == 1:
        first = videos[0]
        if isinstance(first, tuple) and len(first) == 2:
            videos, audio = first
        elif isinstance(first, dict):
            audio = audio or first.get("audio")
            audio_sample_rate = audio_sample_rate or first.get("audio_sample_rate")
            videos = first.get("frames") or first.get("video")
        elif isinstance(first, list):
            videos = first

    if isinstance(videos, tuple) and len(videos) == 2:
        videos, audio = videos
    elif isinstance(videos, dict):
        audio = audio or videos.get("audio")
        audio_sample_rate = audio_sample_rate or videos.get("audio_sample_rate")
        videos = videos.get("frames") or videos.get("video")

    if videos is None:
        raise ValueError("No video frames found in output.")
    return videos, audio, audio_sample_rate, action


def _normalize_frame(frame: Any) -> Any:
    if isinstance(frame, torch.Tensor):
        frame_tensor = frame.detach().cpu()
        if frame_tensor.dim() == 4 and frame_tensor.shape[0] == 1:
            frame_tensor = frame_tensor[0]
        if frame_tensor.dim() == 3 and frame_tensor.shape[0] in (3, 4):
            frame_tensor = frame_tensor.permute(1, 2, 0)
        if frame_tensor.is_floating_point():
            frame_tensor = frame_tensor.clamp(-1, 1) * 0.5 + 0.5
        return frame_tensor.float().numpy()
    if isinstance(frame, np.ndarray):
        frame_array = frame
        if frame_array.ndim == 4 and frame_array.shape[0] == 1:
            frame_array = frame_array[0]
        if np.issubdtype(frame_array.dtype, np.integer):
            frame_array = frame_array.astype(np.float32) / 255.0
        return frame_array
    if isinstance(frame, PIL.Image.Image):
        return np.asarray(frame).astype(np.float32) / 255.0
    return frame


def _ensure_frame_list(video: Any) -> Any:
    if isinstance(video, list):
        if not video:
            return video
        first = video[0]
        if isinstance(first, np.ndarray):
            if first.ndim == 5:
                return list(first[0])
            if first.ndim == 4:
                return list(first)
            if first.ndim == 3:
                return video
        return video
    if isinstance(video, np.ndarray):
        if video.ndim == 5:
            return list(video[0])
        if video.ndim == 4:
            return list(video)
        if video.ndim == 3:
            return [video]
    return video


def _video_to_array(video: Any) -> Any:
    if isinstance(video, torch.Tensor):
        video_tensor = video.detach().cpu()
        if video_tensor.dim() == 5:
            if video_tensor.shape[1] in (3, 4):
                video_tensor = video_tensor[0].permute(1, 2, 3, 0)
            else:
                video_tensor = video_tensor[0]
        elif video_tensor.dim() == 4 and video_tensor.shape[0] in (3, 4):
            video_tensor = video_tensor.permute(1, 2, 3, 0)
        if video_tensor.is_floating_point():
            video_tensor = video_tensor.clamp(-1, 1) * 0.5 + 0.5
        return video_tensor.float().numpy()
    if isinstance(video, np.ndarray):
        video_array = video
        if video_array.ndim == 5:
            video_array = video_array[0]
        if np.issubdtype(video_array.dtype, np.integer):
            video_array = video_array.astype(np.float32) / 255.0
        return video_array
    if isinstance(video, list):
        if not video:
            raise ValueError("No video frames found in output.")
        return [_normalize_frame(frame) for frame in video]
    return video


def _save_video(video: Any, output_path: Path, fps: int, audio: Any | None, audio_sample_rate: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_array = _ensure_frame_list(_video_to_array(video))

    if audio is not None:
        from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes

        frames_np = np.stack(video_array, axis=0) if isinstance(video_array, list) else np.asarray(video_array)
        if frames_np.ndim == 4 and frames_np.shape[-1] == 4:
            frames_np = frames_np[..., :3]
        frames_u8 = (np.clip(frames_np, 0.0, 1.0) * 255).round().clip(0, 255).astype("uint8")

        audio_np = audio
        if isinstance(audio_np, list):
            audio_np = audio_np[0] if audio_np else None
        if isinstance(audio_np, torch.Tensor):
            audio_np = audio_np.detach().cpu().float().numpy()
        if isinstance(audio_np, np.ndarray):
            audio_np = np.squeeze(audio_np).astype(np.float32)

        video_bytes = mux_video_audio_bytes(
            frames_u8,
            audio_np,
            fps=float(fps),
            audio_sample_rate=audio_sample_rate,
        )
        output_path.write_bytes(video_bytes)
        return

    try:
        from diffusers.utils import export_to_video
    except ImportError as exc:
        raise ImportError("diffusers is required for export_to_video.") from exc
    export_to_video(video_array, str(output_path), fps=fps)


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _build_omni(args: argparse.Namespace) -> Omni:
    parallel_config = DiffusionParallelConfig(
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        cfg_parallel_size=args.cfg_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        vae_patch_parallel_size=args.vae_patch_parallel_size,
        use_hsdp=args.use_hsdp,
        hsdp_shard_size=args.hsdp_shard_size,
        hsdp_replicate_size=args.hsdp_replicate_size,
    )
    kwargs: dict[str, Any] = {
        "model": args.model,
        "model_class_name": "Cosmos3OmniDiffusersPipeline",
        "enable_layerwise_offload": args.enable_layerwise_offload,
        "vae_use_slicing": args.vae_use_slicing,
        "vae_use_tiling": args.vae_use_tiling,
        "enforce_eager": args.enforce_eager,
        "parallel_config": parallel_config,
        "cache_backend": args.cache_backend,
        "cache_config": _cache_config(args.cache_backend),
    }
    if args.disable_guardrails:
        kwargs["model_config"] = {"guardrails": False}
    if args.quantization is not None:
        kwargs["quantization"] = args.quantization
    return Omni(**kwargs)


def _resolve_action_mode(task: str, args: argparse.Namespace) -> str | None:
    if getattr(args, "action_mode", None):
        return args.action_mode
    return _TASK_ACTION_MODES.get(task)


def _build_prompt_and_extra(
    args: argparse.Namespace,
    task: str,
    action_mode: str | None,
    num_frames: int | None,
    flow_shift: float | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    vision_path = args.vision_path or args.image

    prompt: dict[str, Any] = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "modalities": ["image"] if task == "t2i" else ["video"],
    }

    if task in _VIDEO_INPUT_TASKS:
        if not vision_path:
            raise ValueError(f"--vision-path (video) is required for {task}.")
        if task == "v2v":
            condition_frame_indexes_vision = _parse_condition_frame_indexes_vision(
                getattr(args, "condition_frame_indexes_vision", None)
            )
            max_frames = _condition_video_pixel_frames(condition_frame_indexes_vision)
        else:
            max_frames = int(num_frames if num_frames is not None else args.action_chunk_size + 1)
        prompt["multi_modal_data"] = {"video": _load_video_frames_from(vision_path, max_frames)}
    elif task == "action_forward_dynamics" and vision_path and _is_video_path(vision_path):
        prompt["multi_modal_data"] = {"video": _load_video_frames_from(vision_path, args.action_chunk_size + 1)}
    elif task in _IMAGE_INPUT_TASKS:
        if not vision_path:
            raise ValueError(f"--vision-path (image) is required for {task}.")
        prompt["multi_modal_data"] = {"image": _load_image_from(vision_path)}
    elif vision_path:
        prompt["multi_modal_data"] = {"image": _load_image_from(vision_path)}

    extra_args: dict[str, Any] = {}

    if flow_shift is not None:
        extra_args["flow_shift"] = float(flow_shift)
    if args.disable_guardrails:
        extra_args["guardrails"] = False

    if task == "v2v":
        extra_args["condition_frame_indexes_vision"] = _parse_condition_frame_indexes_vision(
            getattr(args, "condition_frame_indexes_vision", None)
        )
        extra_args["condition_video_keep"] = getattr(args, "condition_video_keep", "first")

    sound_enabled = bool(getattr(args, "generate_sound", False)) or task == "t2v_sound"
    if sound_enabled and action_mode is not None:
        raise ValueError("Cosmos3 does not support action modes combined with sound generation.")
    if sound_enabled:
        prompt["generate_sound"] = True
        extra_args["generate_sound"] = True
        if args.sound_duration is not None:
            extra_args["sound_duration"] = args.sound_duration

    if action_mode is not None:
        extra_args["action_mode"] = action_mode
        extra_args["action_chunk_size"] = args.action_chunk_size
        if action_mode in {"policy", "inverse_dynamics"}:
            extra_args["raw_action_dim"] = args.raw_action_dim
        elif args.raw_action_dim is not None:
            extra_args["raw_action_dim"] = args.raw_action_dim
        if args.domain_id is not None:
            extra_args["domain_id"] = args.domain_id
        else:
            extra_args["domain_name"] = args.domain_name
        if action_mode == "forward_dynamics":
            if not args.action_path:
                raise ValueError("--action-path is required for forward_dynamics.")
            extra_args["action_path"] = _resolve_local_path(args.action_path)
        elif args.action_path:
            extra_args["action_path"] = _resolve_local_path(args.action_path)

    return prompt, extra_args


def _benchmark_generation_count(args: argparse.Namespace) -> int | None:
    if args.benchmark_generations is None:
        if not args.benchmark:
            return None
        return 10
    if args.benchmark_generations < 1:
        raise ValueError(f"--benchmark-generations must be >= 1, got {args.benchmark_generations}.")
    return args.benchmark_generations


def _resolve_generation_options(args: argparse.Namespace, task: str) -> dict[str, Any]:
    defaults = TASK_DEFAULTS[task]
    height = args.height or defaults["height"]
    width = args.width or defaults["width"]
    num_frames = args.num_frames if args.num_frames is not None else defaults["num_frames"]
    num_inference_steps = args.num_inference_steps or defaults["num_inference_steps"]
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else defaults["guidance_scale"]
    fps = args.fps or defaults["fps"]
    flow_shift = args.flow_shift if args.flow_shift is not None else defaults.get("flow_shift")
    return {
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "fps": fps,
        "flow_shift": flow_shift,
    }


def _build_sampling(
    args: argparse.Namespace,
    options: dict[str, Any],
    extra_args: dict[str, Any],
) -> OmniDiffusionSamplingParams:
    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)
    return OmniDiffusionSamplingParams(
        height=options["height"],
        width=options["width"],
        generator=generator,
        guidance_scale=options["guidance_scale"],
        num_inference_steps=options["num_inference_steps"],
        num_frames=options["num_frames"],
        frame_rate=float(options["fps"]),
        extra_args=dict(extra_args),
    )


def _prompt_for_request(prompt: dict[str, Any]) -> dict[str, Any]:
    request_prompt = dict(prompt)
    if isinstance(request_prompt.get("multi_modal_data"), dict):
        request_prompt["multi_modal_data"] = dict(request_prompt["multi_modal_data"])
    if isinstance(request_prompt.get("additional_information"), dict):
        request_prompt["additional_information"] = dict(request_prompt["additional_information"])
    return request_prompt


def _synchronize_device() -> None:
    try:
        current_omni_platform.synchronize()
    except (AttributeError, NotImplementedError, RuntimeError, AssertionError):
        pass


def _print_generation_configuration(
    args: argparse.Namespace,
    task: str,
    options: dict[str, Any],
    action_mode: str | None,
    record_index: int | None,
    benchmark_generations: int | None = None,
) -> None:
    print("Cosmos3 generation configuration:")
    print(f"  Task: {task}")
    if action_mode:
        print(f"  Action mode: {action_mode}")
    if record_index is not None:
        print(f"  Record: {record_index}")
    print(f"  Model: {args.model}")
    print(f"  Size: {options['width']}x{options['height']}")
    if options["num_frames"] is not None:
        print(f"  Frames: {options['num_frames']}")
    print(f"  Steps: {options['num_inference_steps']}")
    print(f"  Guidance scale: {options['guidance_scale']}")
    if options["flow_shift"] is not None:
        print(f"  Flow shift: {options['flow_shift']}")
    if args.disable_guardrails:
        print("  Guardrails: disabled")
    if benchmark_generations is not None:
        print("  Benchmark mode: enabled")
        print("  Warmup generations: 1")
        print(f"  Timed generations: {benchmark_generations}")
        print("  Output saving: disabled")


def _generate_once(
    omni: Omni,
    prompt: dict[str, Any],
    args: argparse.Namespace,
    options: dict[str, Any],
    extra_args: dict[str, Any],
    *,
    use_tqdm: bool = True,
) -> tuple[Any, float]:
    sampling = _build_sampling(args, options, extra_args)
    request_prompt = _prompt_for_request(prompt)
    _synchronize_device()
    start = time.perf_counter()
    outputs = omni.generate(request_prompt, sampling, use_tqdm=use_tqdm)
    _synchronize_device()
    elapsed = time.perf_counter() - start
    return outputs, elapsed


def _prepare_generation(
    args: argparse.Namespace,
    task: str,
) -> tuple[dict[str, Any], str | None, dict[str, Any], dict[str, Any]]:
    options = _resolve_generation_options(args, task)

    action_mode = _resolve_action_mode(task, args)
    prompt, extra_args = _build_prompt_and_extra(
        args,
        task,
        action_mode,
        options["num_frames"],
        options["flow_shift"],
    )
    return options, action_mode, prompt, extra_args


def _run_one(
    omni: Omni,
    args: argparse.Namespace,
    task: str,
    output_path: Path,
    record_index: int | None = None,
) -> None:
    options, action_mode, prompt, extra_args = _prepare_generation(args, task)
    _print_generation_configuration(args, task, options, action_mode, record_index)

    outputs, elapsed = _generate_once(omni, prompt, args, options, extra_args)
    print(f"Total generation time: {elapsed:.4f} seconds")

    if task == "t2i":
        images = _extract_images(outputs)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(output_path)
        print(f"Saved image to {output_path}")
        return

    video, audio, returned_sample_rate, action = _extract_video_audio_action(outputs)
    _save_video(
        video,
        output_path,
        fps=options["fps"],
        audio=audio,
        audio_sample_rate=returned_sample_rate or args.audio_sample_rate,
    )
    print(f"Saved video to {output_path}")

    if action_mode in {"policy", "inverse_dynamics"} and action:
        action_out = (
            Path(args.action_output) if args.action_output else output_path.with_name(f"{output_path.stem}_action.json")
        )
        action_out.parent.mkdir(parents=True, exist_ok=True)
        action_out.write_text(json.dumps(_jsonable(action), indent=2) + "\n", encoding="utf-8")
        print(f"Saved action metadata to {action_out}")


def _run_benchmark(
    omni: Omni,
    args: argparse.Namespace,
    task: str,
    benchmark_generations: int,
    record_index: int | None = None,
) -> None:
    options, action_mode, prompt, extra_args = _prepare_generation(args, task)
    _print_generation_configuration(
        args,
        task,
        options,
        action_mode,
        record_index,
        benchmark_generations=benchmark_generations,
    )

    print("Running warmup generation...")
    outputs, warmup_elapsed = _generate_once(omni, prompt, args, options, extra_args, use_tqdm=False)
    del outputs
    print(f"Warmup generation discarded: {warmup_elapsed:.4f} seconds")

    print(f"Running {benchmark_generations} timed generation(s)...")
    generation_times: list[float] = []
    for _ in range(benchmark_generations):
        outputs, elapsed = _generate_once(omni, prompt, args, options, extra_args, use_tqdm=False)
        del outputs
        generation_times.append(elapsed)

    total_elapsed = sum(generation_times)
    average_elapsed = statistics.mean(generation_times)
    throughput = benchmark_generations / total_elapsed if total_elapsed > 0 else float("inf")
    print(f"Benchmark total generation time: {total_elapsed:.4f} seconds")
    print(f"Benchmark average generation time: {average_elapsed:.4f} seconds")
    print(f"Benchmark throughput: {throughput:.4f} generations/second")


def _record_output_path(base: Path, index: int, total: int) -> Path:
    if total <= 1:
        return base
    return base.with_name(f"{base.stem}_{index}{base.suffix}")


def main() -> None:
    args = parse_args()
    cli_set = _cli_provided_attrs(sys.argv[1:])
    benchmark_generations = _benchmark_generation_count(args)

    records: list[dict[str, Any]] = [{}]
    if args.input_json:
        records = _load_input_records(args.input_json)
        if not records:
            raise ValueError(f"--input-json {args.input_json} contained no records.")

    omni = _build_omni(args)

    base_output = Path(args.output or TASK_DEFAULTS[args.task]["output"])

    for index, record in enumerate(records):
        record_args = argparse.Namespace(**vars(args))
        if record:
            _apply_record(record, record_args, cli_set)
        record_index = index if len(records) > 1 else None
        if benchmark_generations is not None:
            _run_benchmark(
                omni,
                record_args,
                args.task,
                benchmark_generations,
                record_index=record_index,
            )
        else:
            output_path = _record_output_path(base_output, index, len(records))
            _run_one(
                omni,
                record_args,
                args.task,
                output_path,
                record_index=record_index,
            )


if __name__ == "__main__":
    main()
