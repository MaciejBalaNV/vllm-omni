# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Upload a Cosmos multiview JSON manifest's local conditioning files."""

import argparse
import copy
import json
import math
import mimetypes
import os
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import httpx

# Keep this client usable with only httpx installed. A contract test checks this
# table against the inference runtime's VIDEO_RES_SIZE_INFO.
SUPPORTED_RESOLUTIONS = {
    "480": {
        "1,1": (640, 640),
        "4,3": (736, 544),
        "3,4": (544, 736),
        "16,9": (832, 480),
        "9,16": (480, 832),
    },
    "720": {
        "1,1": (960, 960),
        "4,3": (1104, 832),
        "3,4": (832, 1104),
        "16,9": (1280, 720),
        "9,16": (720, 1280),
    },
}


def normalize_aspect_ratio(value: Any) -> str:
    if value is None or value == "auto":
        return "auto"
    parts = str(value).strip().replace(":", ",").split(",")
    if len(parts) == 2:
        try:
            width, height = (int(part.strip()) for part in parts)
        except ValueError:
            pass
        else:
            if width > 0 and height > 0:
                divisor = math.gcd(width, height)
                ratio = f"{width // divisor},{height // divisor}"
                if ratio in SUPPORTED_RESOLUTIONS["480"]:
                    return ratio
    raise ValueError(
        f"Unsupported Cosmos3 multiview aspect_ratio={value!r}; expected auto, 1:1, 4:3, 3:4, 16:9, or 9:16."
    )


def prepare_request(
    manifest: dict[str, Any],
    base_dir: Path,
    *,
    resolution_override: str | None = None,
    aspect_ratio_override: str | None = None,
) -> tuple[dict[str, str], list[Path]]:
    """Accept an offline manifest or a video API request containing extra_params."""
    extra = copy.deepcopy(manifest.get("extra_params", {}))
    if "multiview" not in extra:
        extra["multiview"] = copy.deepcopy(manifest["multiview"])
    extra.setdefault("wsm", manifest.get("wsm", True))
    if resolution_override is None:
        declarations = {
            str(value)
            for value in (manifest.get("resolution"), extra.get("resolution"), extra["multiview"].get("resolution"))
            if value is not None
        }
        if len(declarations) > 1:
            raise ValueError(f"Conflicting Cosmos3 multiview resolutions: {sorted(declarations)}.")
        resolution = next(iter(declarations), "480")
    else:
        resolution = str(resolution_override)
    if resolution not in SUPPORTED_RESOLUTIONS:
        raise ValueError(
            f"Unsupported Cosmos3 multiview resolution {resolution!r}; expected one of {sorted(SUPPORTED_RESOLUTIONS)}."
        )
    if aspect_ratio_override is None:
        declarations = {
            normalize_aspect_ratio(value)
            for value in (
                manifest.get("aspect_ratio"),
                extra.get("aspect_ratio"),
                extra["multiview"].get("aspect_ratio"),
            )
            if value is not None
        }
        if len(declarations) > 1:
            raise ValueError(f"Conflicting Cosmos3 multiview aspect ratios: {sorted(declarations)}.")
        aspect_ratio = next(iter(declarations), "auto")
    else:
        aspect_ratio = normalize_aspect_ratio(aspect_ratio_override)
    geometry_override = resolution_override is not None or aspect_ratio_override is not None
    dimensions = {
        key: str(manifest[key])
        for key in ("width", "height")
        if not geometry_override and manifest.get(key) is not None
    }
    if aspect_ratio != "auto":
        width, height = SUPPORTED_RESOLUTIONS[resolution][aspect_ratio]
        for key, expected in (("width", width), ("height", height)):
            if key in dimensions and int(dimensions[key]) != expected:
                raise ValueError(
                    f"Cosmos3 multiview resolution={resolution!r} requires {key}={expected}, got {dimensions[key]} "
                    f"for aspect_ratio={aspect_ratio!r}."
                )
            dimensions[key] = str(expected)
    extra["resolution"] = resolution
    extra["multiview"]["resolution"] = resolution
    extra["aspect_ratio"] = aspect_ratio
    extra["multiview"]["aspect_ratio"] = aspect_ratio
    paths = []
    for view in extra["multiview"]["views"]:
        for role in ("control", "vision"):
            if f"{role}_reference_index" in view:
                raise ValueError("The client expects local media paths, not pre-existing upload indexes.")
            fields = [field for field in (f"{role}_path", role) if view.get(field) is not None]
            if not fields:
                continue
            if len(fields) != 1:
                raise ValueError(f"Specify only one of {role} and {role}_path per camera.")
            value = view.pop(fields[0])
            if not isinstance(value, str):
                raise ValueError(f"{role} must be a local file path for HTTP uploads.")
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = base_dir / path
            if not path.is_file():
                raise FileNotFoundError(path)
            view[f"{role}_reference_index"] = len(paths)
            paths.append(path)
    data = {
        "prompt": str(manifest.get("prompt", "")),
        "extra_params": json.dumps(extra),
        **dimensions,
    }
    for key in (
        "model",
        "fps",
        "num_frames",
        "num_inference_steps",
        "guidance_scale",
        "flow_shift",
        "seed",
        "negative_prompt",
    ):
        if manifest.get(key) is not None:
            data[key] = str(manifest[key])
    for alias, key in (("num_steps", "num_inference_steps"), ("guidance", "guidance_scale"), ("shift", "flow_shift")):
        if key not in data and manifest.get(alias) is not None:
            data[key] = str(manifest[alias])
    return data, paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--server", default="http://localhost:8091")
    parser.add_argument("--sync", action="store_true", help="Use /v1/videos/sync instead of a background job")
    parser.add_argument(
        "--num-inference-steps", type=int, help="Number of diffusion steps, overriding the manifest value"
    )
    parser.add_argument("--num-frames", type=int, help="Number of video frames, overriding the manifest value")
    parser.add_argument(
        "--resolution",
        choices=tuple(SUPPORTED_RESOLUTIONS),
        help="Video resolution bucket (480 or 720), overriding the manifest value; defaults to 480",
    )
    parser.add_argument(
        "--aspect-ratio",
        type=normalize_aspect_ratio,
        help="auto, 1:1, 4:3, 3:4, 16:9, or 9:16; overrides the manifest (default: detect from first WSM input)",
    )
    parser.add_argument("--output", type=Path, default=Path("multiview.mp4"))
    parser.add_argument("--timeout", type=float, default=3600, help="HTTP and job polling timeout in seconds")
    args = parser.parse_args()
    for key in ("num_inference_steps", "num_frames"):
        value = getattr(args, key)
        if value is not None and value < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    manifest = json.loads(args.manifest.read_text())
    data, paths = prepare_request(
        manifest,
        args.manifest.resolve().parent,
        resolution_override=args.resolution,
        aspect_ratio_override=args.aspect_ratio,
    )
    for key in ("num_inference_steps", "num_frames"):
        if getattr(args, key) is not None:
            data[key] = str(getattr(args, key))
    api_key = os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    server = args.server.rstrip("/")
    with httpx.Client(timeout=args.timeout, headers=headers) as client:
        with ExitStack() as stack:
            files = [
                (
                    "input_references",
                    (
                        path.name,
                        stack.enter_context(path.open("rb")),
                        mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                    ),
                )
                for path in paths
            ]
            response = client.post(server + "/v1/videos" + ("/sync" if args.sync else ""), data=data, files=files)
            response.raise_for_status()
        if not args.sync:
            job = response.json()
            job_url = server + "/v1/videos/" + job["id"]
            print(f"Submitted {job['id']}", flush=True)
            deadline = time.monotonic() + args.timeout
            while job["status"] not in ("completed", "failed"):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Polling timed out; the job is still available at {job_url}")
                time.sleep(1)
                response = client.get(job_url)
                # Failed jobs use an error HTTP status while still returning job metadata.
                if response.is_error:
                    try:
                        failed_job = response.json()
                    except ValueError:
                        response.raise_for_status()
                    if isinstance(failed_job, dict) and failed_job.get("status") == "failed":
                        raise RuntimeError(f"Video generation failed: {failed_job.get('error')}")
                    response.raise_for_status()
                job = response.json()
            if job["status"] == "failed":
                raise RuntimeError(f"Video generation failed: {job.get('error')}")
            response = client.get(job_url + "/content")
            response.raise_for_status()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(response.content)
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
