# Cosmos Multiview-AV uploads

## Multi-GPU serving

Use the existing engine parallelism flags; the video API payload is unchanged:

```bash
vllm serve /models/cosmos3-multiview --omni \
  --model-class-name Cosmos3MultiviewPipeline --num-gpus 4 \
  --cfg-parallel-size 2 --ulysses-degree 2 --port 8091
```

For HSDP on those same four GPUs, add `--use-hsdp --hsdp-shard-size 4`.
For TP2 x CP2, replace `--cfg-parallel-size 2` with `--tensor-parallel-size 2`.
HSDP and TP are mutually exclusive. Single-GPU execution remains the default.
Triton and FA4 use the same sparse visibility rules; FA4 requires datacenter
Blackwell and the optional `fa4` extra.

See the
[offline script](https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/multiview_video/cosmos3_multiview.py)
for usage examples.

## Upload contract

`POST /v1/videos` and `POST /v1/videos/sync` accept all camera inputs in the HTTP request.
See the [video API reference](../../../serving/videos_api.md) for endpoint and response details.
Send the files as repeated `input_references` parts and the camera manifest as
a JSON-encoded `extra_params` form field. Each view's
`control_reference_index` or `vision_reference_index` is a zero-based index
into the uploaded file list. Filenames do not identify cameras; duplicate
filenames are allowed. The server substitutes temporary paths before inference.

For example, a view can reference control upload 0 and vision upload 11:

```json
{
  "camera_key": "camera_front_wide_120fov",
  "control_reference_index": 0,
  "vision_reference_index": 11
}
```

Include all eleven views in `extra_params.multiview.views`, in this order:

```text
camera_front_wide_120fov
camera_cross_right_120fov
camera_rear_right_70fov
camera_rear_tele_30fov
camera_rear_left_70fov
camera_cross_left_120fov
camera_front_tele_30fov
camera_front_fisheye_200fov
camera_left_fisheye_200fov
camera_right_fisheye_200fov
camera_rear_fisheye_200fov
```

Set `extra_params.wsm` to `true` or `{}`. Every camera needs a control input;
vision inputs must cover every camera or be omitted entirely. Within each
role, use all videos or all images. Uploads support MP4, MOV, MKV, WebM and
the existing control image formats (BMP, GIF, JPEG, PNG, TIFF, WebP).

Every upload must be referenced exactly once, and indexes must be integers.
You may retain existing server-side `control_path`/`vision_path` entries for
other inputs, but cannot provide a path and an upload index for the same
camera and role. Do not combine this upload mode with `input_reference`,
`image_reference`, `video_reference`, `audio_reference`, `control_reference`,
or `control_type`.

The online client accepts an existing offline JSON manifest containing
`prompt`, `wsm`, and `multiview`, or a video API manifest containing
`extra_params`. It opens local `control_path`/`vision_path` files (also accepting
the `control`/`vision` path aliases) and replaces them with upload indexes:

```bash
python examples/online_serving/multiview_video/cosmos3_multiview_client.py \
  request.json --server http://localhost:8091 --output multiview.mp4

# Use the synchronous endpoint:
python examples/online_serving/multiview_video/cosmos3_multiview_client.py \
  request.json --server http://localhost:8091 --sync --output multiview.mp4

# Override the manifest's frame count, resolution, and aspect ratio:
python examples/online_serving/multiview_video/cosmos3_multiview_client.py \
  request.json --num-frames 29 --resolution 720 --aspect-ratio 9:16 --output multiview.mp4
```

`--num-frames` must be positive. `--resolution` selects `480` (default) or `720`.
`--aspect-ratio` accepts `auto` (default), `1:1`, `4:3`, `3:4`, `16:9`, or `9:16`;
comma spellings are also accepted. All eleven cameras share the selected size:

| Aspect ratio | 480p (width × height) | 720p (width × height) |
|---|---|---|
| 1:1 | 640 × 640 | 960 × 960 |
| 4:3 | 736 × 544 | 1104 × 832 |
| 3:4 | 544 × 736 | 832 × 1104 |
| 16:9 | 832 × 480 | 1280 × 720 |
| 9:16 | 480 × 832 | 720 × 1280 |

Automatic mode selects the nearest Cosmos3 bucket from the original dimensions
of the first camera's WSM input (`camera_front_wide_120fov`), using the first
frame for video. Detection happens in the pipeline after uploaded references
have been resolved. Other cameras and optional vision inputs do not determine
the ratio; all are resized and center-cropped to the same output size. Input
pixel count does not select the resolution tier. An unreadable first WSM input
fails generation. Specify `--aspect-ratio 16:9` to retain the previous fixed
landscape behavior.

Offline manifests accept top-level `resolution`/`aspect_ratio` or their fields
inside `multiview`; API manifests also accept them inside `extra_params`.
Duplicate declarations must agree after normalization. Each CLI override
replaces declarations for its own setting and clears stale width/height
constraints. Automatic mode sends no computed width/height; user-supplied
constraints are retained when there is no geometry override and must match the
pipeline's resolved size. The client does not decode media or edit the manifest.

For direct HTTP requests, use the existing `aspect_ratio` form field or
`extra_params.aspect_ratio` / `extra_params.multiview.aspect_ratio`. Nested
multiview fields take precedence over `extra_params` fields, which take
precedence over the top-level aspect-ratio field. Defaults are `"480"` and
`"auto"`. For portrait output at 720p, set
`extra_params.multiview.resolution` to `"720"` and
`extra_params.multiview.aspect_ratio` to `"9:16"`, including all eleven camera
entries as described above. Omit width/height or explicitly supply width `720`
and height `1280`. Only canonical 480p and 720p buckets are supported.
Single-dimension constraints are also checked by the pipeline.

720p uses approximately 2.3 times as many spatial tokens as 480p. Measure
latency and peak memory for the desired clip length and execution topology.

Relative input paths resolve against the manifest's directory. The default
client submits a background job, polls it, and downloads the existing video
output. Set `VLLM_API_KEY` or `OPENAI_API_KEY` when the server requires a bearer
token. Output packaging and per-camera conditioning settings are unchanged.

Each uploaded file must be nonempty and no larger than 512 MiB; at most 22
files are accepted. Invalid manifests return HTTP 400 before inference.
Media decoding happens during generation, so a corrupt clip can produce a
failed asynchronous job even after the upload was accepted.

Uploaded files are retained until the job finishes or is cancelled, and are
removed on errors and synchronous timeouts. Existing caller-owned paths are
never included in upload cleanup. API and inference workers must share access
to the temporary filesystem. Configure proxy body-size limits and temporary
disk capacity for concurrent requests: the maximum payload is 11 GiB, and
multipart spooling plus persisted inputs can temporarily require extra disk
space. Application file limits are checked after multipart parsing; they do
not replace ingress limits. This API does not provide resumable uploads or
cross-host file transport.
