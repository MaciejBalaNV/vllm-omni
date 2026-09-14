# Cosmos3 Multiview-AV

Cosmos3 Multiview-AV generates the eleven fixed MADS camera views in one
bidirectional denoising pass. It uses the regular Cosmos3 Nano architecture and
weights plus camera-major VAE processing and a weight-free sparse attention
mask. The runtime defaults to single-GPU and sequential-CFG execution and PyTorch
FlexAttention's Triton backend, with an opt-in FlashAttention-4 backend on
Blackwell (see [Sparse attention backend](#sparse-attention-backend)).

## Export contract

Export the `wsm_transfer_nano_480p_11view_decomposed_attn_16n` checkpoint with
the normal Cosmos3 EMA-to-Diffusers conversion. No multiview-only weight keys
are expected. Update `model_index.json` to use:

```json
{"_class_name": "Cosmos3MultiviewPipeline"}
```

Add the following fields to `transformer/config.json` (preserve all existing
Cosmos3 Nano fields):

```json
{
  "backbone_type": "cosmos3_multiview",
  "multiview": {
    "causal_training_strategy": "none",
    "attention_scope": "decomposed",
    "decomposed_temporal_window_seconds": null,
    "control_attends_sensor": false,
    "align_temporal_positions_across_views": false,
    "backend": "triton",
    "max_views": 11,
    "share_vision_temporal_positions": true,
    "cameras": [
      "camera_front_wide_120fov",
      "camera_cross_right_120fov",
      "camera_rear_right_70fov",
      "camera_rear_tele_30fov",
      "camera_rear_left_70fov",
      "camera_cross_left_120fov",
      "camera_front_tele_30fov",
      "camera_front_fisheye_200fov",
      "camera_left_fisheye_200fov",
      "camera_right_fisheye_200fov",
      "camera_rear_fisheye_200fov"
    ]
  }
}
```

The scheduler directory must describe the regular FlowUniPC scheduler. The
request defaults to 35 steps, guidance 6.0, flow shift 10, and 480p (832×480).
Resolution, frame rate, and per-camera frame count are request-driven. When
omitted, fps defaults to 30 and num_frames to 201.

30 FPS is the training rate: the MADS WSM transfer recipes read their clips at
native 30 FPS and stamp "30 FPS" into the training captions, so the
fps-modulated temporal mRoPE and the prompt metadata are on-distribution only
there. Other rates are accepted with a warning outside [10, 30], and frame
counts are rounded up to the VAE's `4k+1` grid (200 becomes 201) instead of
being rejected.

## Resolution

Every camera uses the same selected landscape size:

| `resolution` | Per-camera output (width × height) |
|---|---|
| `"480"` (default) | 832 × 480 |
| `"720"` | 1280 × 720 |

For offline JSON/JSONL input, set `resolution` at the top level or inside
`multiview`. Integer values `480` and `720` are also accepted. If both fields
are present, their values must agree. `--resolution` overrides both fields for
every record:

```bash
python examples/offline_inference/multiview_video/cosmos3_multiview.py \
  --model /models/cosmos3-multiview-av --input /data/mv_i2v_wsm.json \
  --resolution 720 --num-frames 29 --output-dir outputs/mv_720
```

Direct pipeline requests resolve `extra_args.multiview.resolution` first,
then `extra_args.resolution`, and otherwise use `"480"`. The generic image
resolution field is not used. Explicit sampling width/height must match the
selected bucket; arbitrary dimensions and other resolution buckets are rejected.
The [online client](../../docs/user_guide/examples/online_serving/cosmos3_multiview.md)
also supports `--resolution 720`.

Vision and WSM inputs are resized and center-cropped to the selected output
size. With spatial compression 16 and transformer patch size 2, 720p uses
45×80 latents and 23×40 patches. Transformer padding is cropped back before
VAE decode, preserving the exact 1280×720 output. This produces approximately
2.36 times as many spatial tokens as 480p; memory and latency depend on clip
length, attention backend, and execution topology.

## Sparse attention backend

`multiview.backend` selects the kernel that consumes the sparse block map. Both
backends are built from the same run-level projection of the visibility
predicate, so they agree on which token pairs are visible; they differ only in
block geometry and floating-point rounding.

| `backend` | Kernel | Sparse block `(q, kv)` | Requirements |
|---|---|---|---|
| `"triton"` (default) | PyTorch FlexAttention, Triton template | 64 × 64 | Any CUDA GPU |
| `"fa4"` | FlashAttention-4 CuTe | 256 × 128 | SM100 (Blackwell), CUDA 13, `pip install 'vllm-omni[fa4]'` |

Because the backend changes only how the mask is executed, it can be overridden
per run without editing the checkpoint — useful for A/B measurement:

```bash
VLLM_OMNI_COSMOS3_MULTIVIEW_BACKEND=fa4 \
  python examples/offline_inference/multiview_video/cosmos3_multiview.py \
  --model /models/cosmos3-multiview-av --input /data/mv_i2v_wsm.json
```

The environment variable wins over `transformer/config.json`; an unset or empty
value falls back to the checkpoint. An unknown name fails at load time rather
than on the first generated frame. Parity thresholds are backend-specific:
goldens taken on Triton must be re-calibrated before they are used to gate the
FA4 path.

### Prompt length is capped by the variant, not the request

The sparse attention pads its text (UND) stream to a fixed capacity so the
compiled kernel sees one input shape for the life of the process. A pad that
tracked each prompt's length would resize the packed key tensor, and the kernel
is compiled with `dynamic=False`, so every distinct prompt length would cost a
recompile — and past Dynamo's default limit of eight the whole attention falls
back to eager FlexAttention, which cannot fit its score matrix at this
sequence length.

Requests may therefore *lower* `max_sequence_length` but not raise it past
`COSMOS3_MULTIVIEW_MAX_SEQUENCE_LENGTH` (4096); a larger value is rejected at
admission. Raise that constant if a golden fixture ever shows the reference
negative prompt being truncated. The padding itself is numerically free: pad
keys are excluded from every real query by the visibility predicate.

## Input and run

The input JSON needs to have the following structure: Each view must appear in
the exact exported camera order and provide `control_path`. For i2v_wsm, every
view also provides `vision_path`; set `condition_video_as_image: true` to use
only its first frame. A top-level empty `wsm` object selects the only supported
control hint.

```bash
python examples/offline_inference/multiview_video/cosmos3_multiview.py \
  --model /models/cosmos3-multiview-av \
  --input /data/mv_i2v_wsm.json \
  --negative-prompt-json recipes/cosmos3/negative_prompt.json \
  --output-dir outputs/mv_i2v_wsm \
  --seed 42 --fps 30 --num-frames 200
```

`--negative-prompt-json` applies the required serialization for you; a
`negative_prompt` string in the input JSON takes precedence over it. Omit both
only for runs where reference parity does not matter.

`--fps` and `--num-frames` override every record, so one input file can be run
at several rates or lengths without editing it. Records may also use the field
names `guidance`, `num_steps`, and `shift` as aliases for `guidance_scale`,
`num_inference_steps`, and `flow_shift`; the vLLM-Omni names win when both are
present.

By default the negative prompt carries the same duration/FPS and resolution
sentences as the positive prompt; set `negative_metadata_mode` in the request's
extra args to change it.

The example writes `vision_viewNN_<camera>.mp4` for all eleven cameras plus
`sample_outputs.json`, including the selected resolution. Strict Ulysses CP,
CFG parallelism, TP, and HSDP use the existing engine flags; HSDP and TP cannot
be combined. See the offline script's usage examples. Cache-DiT, session state,
LiDAR, camera subsets, and reordered cameras are rejected in v1.

## Verification

Run the CPU contract suite:

```bash
pytest -q \
  tests/diffusion/models/cosmos3/test_multiview_flex_attention.py \
  tests/diffusion/models/cosmos3/test_cosmos3_multiview_pipeline.py \
  tests/diffusion/models/cosmos3/test_cosmos3_transformer.py \
  tests/examples/offline_inference/test_cosmos3_multiview.py \
  tests/model_extras/test_cosmos3_multiview_uploads.py \
  tests/model_extras/test_model_extras.py \
  tests/model_tests/diffusion/test_alignment.py
```

On CUDA, run `test_multiview_recompile.py` and, on supported Blackwell hardware,
`test_multiview_fa4.py` from the same model test directory. These cover warmed
480p/720p attention geometries and backend parity. Run the existing distributed
multiview tests for the deployment's parallel configuration.

For checkpoint validation, generate 29-frame clips in WSM-only and
vision-conditioned modes at both resolutions with the same seed and settings.
Check all eleven exported videos for camera order, frame count, and exact
dimensions. Follow with a representative 201-frame 720p generation on sufficient
hardware. Record the backend, GPU topology, steps, cold/warm latency, and peak
memory; no fixed memory or latency target is implied by resolution support.
