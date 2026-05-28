# Cosmos3

This example shows Cosmos3 online serving with `Cosmos3OmniDiffusersPipeline` for text-to-image, text-to-video, image-to-video, video-to-video, video-with-sound, and action generation.

The server defaults to the `nvidia/Cosmos3-Nano` Hugging Face repo. Override the checkpoint by exporting `MODEL` or `COSMOS3_MODEL` to a local Diffusers-format checkpoint.

```bash
cd examples/online_serving/cosmos3
bash run_server.sh
```

`run_server.sh` accepts these environment overrides:

- `MODEL`: checkpoint path or Hugging Face repo, defaults to `nvidia/Cosmos3-Nano` (or `COSMOS3_MODEL` if set)
- `PORT`: server port, defaults to `8091`
- `HOST`: optional bind host
- `MODEL_CLASS_NAME`: diffusion pipeline class, defaults to `Cosmos3OmniDiffusersPipeline`
- `SERVED_MODEL_NAME`: optional OpenAI model name alias
- `CACHE_BACKEND`: set to `cache_dit` to enable Cache-DiT
- `ENABLE_CACHE_DIT_SUMMARY`: set to `1` to log Cache-DiT summaries
- `ENABLE_LAYERWISE_OFFLOAD`: set to `1` to enable layerwise offload
- `CFG_PARALLEL_SIZE`, `TENSOR_PARALLEL_SIZE`, `ULYSSES_DEGREE`, `RING_DEGREE`, `VAE_PATCH_PARALLEL_SIZE`, `USE_HSDP`: parallel execution controls
- `VAE_USE_SLICING`, `VAE_USE_TILING`, `HSDP_SHARD_SIZE`, `HSDP_REPLICATE_SIZE`, `ENFORCE_EAGER`: optional runtime controls
- `ALLOWED_LOCAL_MEDIA_PATH`: local media access path, defaults to `/`
- `DEPLOY_CONFIG`, `DEFAULT_SAMPLING_PARAMS`: optional server config overrides

## Disabling guardrails

Cosmos3 ships with safety guardrails that check prompts and apply generated-output face blurring. Two override paths are available depending on whether you want to skip the guardrails globally or on a single request.

### Server-wide (skip loading guardrail models entirely)

Start the server with `--deploy-config cosmos3_no_guardrails.yaml`, which sets `model_config.guardrails: false` on the diffusion stage so the guardrail models are never loaded:

```bash
vllm serve nvidia/Cosmos3-Nano --omni \
  --model-class-name Cosmos3OmniDiffusersPipeline \
  --deploy-config examples/online_serving/cosmos3/cosmos3_no_guardrails.yaml \
  --port 8091
```

Other CLI flags (parallelism, cache backend, layerwise offload, etc.) are still honored; the YAML only overrides the guardrail toggle. When this path is used, per-request overrides cannot turn guardrails back on — the underlying models are not in memory.

### Per-request (skip checks for a single generation)

When the server has guardrails enabled, an individual request can opt out by passing `guardrails: false` inside `extra_params`. The server merges `extra_params` into the pipeline's `extra_args`, and the guardrail gate reads `extra_args["guardrails"]` as a per-request override:

```bash
curl -sS -X POST "${BASE_URL}/v1/videos/sync" \
  --form-string "prompt=..." \
  --form-string 'extra_params={"guardrails": false}' \
  -o cosmos3_no_check.mp4
```

> Use `curl --form-string` instead of `-F` for free-form text fields like `prompt`, `negative_prompt`, and `extra_params`. With `-F`, curl treats `;` as a separator for content-type/filename hints, so a prompt containing a semicolon would be silently truncated. `--form-string` sends the value literally.

For action-mode requests, fold the override into the existing `extra_params` object alongside `action_mode`, `domain_name`, and the rest. Anything other than `false` (or a missing field) keeps the default behavior.

## Curl scripts

Each script sources its prompt and sampling parameters from the canonical input file shared with the offline example at `../../offline_inference/cosmos3/inputs/`. Override the input file with `INPUT_JSON=` (or `INPUT_JSONL=` for the camera variant) or the parent directory with `INPUTS_DIR=`.

Companion vision and action assets are auto-downloaded from `nvidia-cosmos/cosmos-dependencies` on first run, so the scripts work out of the box once the server is up. Image-input action modes (`policy`, `forward_dynamics`) extract the first frame of the source `.mp4` via `ffmpeg`, which is already a Cosmos3 system dependency.

## Text-to-Image

```bash
bash run_curl_t2i.sh
```

Calls `POST /v1/images/generations`, which selects Cosmos3 text-to-image through `modalities=["image"]` internally.

## Text-to-Video

```bash
bash run_curl_t2v.sh
```

## Image-to-Video

The companion image (`robot_153.jpg`) is auto-downloaded on first run. To use your own image:

```bash
IMAGE_PATH=/path/to/your.jpg bash run_curl_i2v.sh
```

## Video-to-Video

The companion video (`robot_pouring.mp4`) is auto-downloaded on first run. The script uploads it through `input_reference` and sends `condition_frame_indexes_vision` / `condition_video_keep` through `extra_params`, so the server only decodes the source frames needed for V2V conditioning.

```bash
bash run_curl_v2v.sh
```

To use your own source video:

```bash
VIDEO_PATH=/path/to/source.mp4 bash run_curl_v2v.sh
```

## Video With Sound

```bash
bash run_curl_t2v_sound.sh
```

The script reads `sound_duration` from `inputs/t2v_sound.json` and posts `generate_sound=true` to `/v1/videos/sync`.

## Action — Policy

Policy mode returns a video plus a predicted action chunk; both are saved.

Robot (`bridge_orig_lerobot`, `raw_action_dim=10`):

```bash
bash run_curl_action_policy.sh
```

Autonomous vehicle (`raw_action_dim=9`, "Please go backward"):

```bash
bash run_curl_action_policy_av.sh
```

## Action — Forward Dynamics

Forward-dynamics scripts download both the source vision asset and the matching `action_path` JSON. The action JSON is passed as `action_path` inside `extra_params`, so it must be readable by the server process — that works out of the box on a same-host deployment with the default `ALLOWED_LOCAL_MEDIA_PATH=/`. For cross-host setups, share the file (e.g. via a mounted volume) or inline the action data into `extra_params` instead.

Robot:

```bash
bash run_curl_action_forward_dynamics_robot.sh
```

Autonomous vehicle:

```bash
bash run_curl_action_forward_dynamics_av.sh
```

Camera-pose (two scenes — `SCENE_INDEX=0` for mountain (default), `SCENE_INDEX=1` for solar):

```bash
bash run_curl_action_forward_dynamics_camera.sh
SCENE_INDEX=1 bash run_curl_action_forward_dynamics_camera.sh
```

## Action — Inverse Dynamics

Inverse-dynamics scripts upload the full source video through `input_reference`, poll the async `/v1/videos` job, and save both the generated video and returned action JSON.

```bash
bash run_curl_action_inverse_dynamics_robot.sh
bash run_curl_action_inverse_dynamics_av.sh
```

## Common script overrides

Every curl script accepts a small set of env overrides:

- `BASE_URL`: server URL, defaults to `http://localhost:8091`
- `OUTPUT_PATH`: where to save the generated image / video
- `ACTION_OUTPUT_PATH`: where to save predicted action JSON (policy / inverse_dynamics)
- `INPUT_JSON` / `INPUT_JSONL` (camera) / `INPUTS_DIR`: alternate source for prompt and sampling parameters
- `IMAGE_PATH` / `VIDEO_PATH`: pre-existing vision asset (skip auto-download / frame-extraction)
- `ACTION_PATH` (forward-dynamics): pre-existing action JSON on the server's filesystem
- `CONDITION_FRAME_INDEXES_VISION`, `CONDITION_VIDEO_KEEP` (V2V): override source-video conditioning controls
- `POLL_INTERVAL` (async scripts): seconds between status checks

Async scripts use `POST /v1/videos` so they can download the MP4 once the job completes and save the action JSON returned in the status response.
