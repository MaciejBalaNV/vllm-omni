# Cosmos-Dreams keyboard/WebRTC demo

This is the Phase-3 single-user serving surface for the distilled
`interact_8b_tfdcm_chunk4_agibot` artifact. It is a stateful chunk loop, not a
`/v1/videos` client: one browser session owns one AR-Diffusion KV session, one
AgiBot controller, and one causal Wan decoder feature cache.

The first tick sends the seed image and returns the seed plus 16 generated RGB
frames (`17` total). Every later tick sends only raw `float32[16,29]` AgiBot
actions and returns 16 new frames. The pipeline alone applies the artifact's
normalizer and 64-D padding. Live execution never loads jsonl/pickle episodes,
re-decodes latent history, or writes/reads an MP4 in the inference loop.

The gateway uses the shared typed AR-Diffusion session path. Each raw action
chunk is serialized as the schema-tagged `robot_action.v1` control of an
`ARDiffusionTickRequest`; the model adapter reconstructs and validates the
`float32[16,29]` tensor. A tick commits only after its session/request/event/
chunk metadata matches the standard `metadata.ar_diffusion` response. Reset,
disconnect, and close use worker lifecycle RPCs rather than cleanup inference
requests.

## Scene bundle

Copy `scene.example.json.template` to a `.json` file, provide a 720×1280 seed
PNG, and replace the identity
head/wrist transforms with transforms from the same calibrated AgiBot FK and
OpenCV coordinate conversion used to build the checkpoint's training actions.
Identity matrices are placeholders for format illustration, not a calibrated
robot scene. The bundle is immutable for a session; changing prompt, FPS,
domain, resolution, or controller scene requires Reset and a new seed.

## Deterministic controller output

The schedule CLI uses the exact same resampler and controller as the browser:

```bash
python examples/online_serving/cosmos_dreams/key_schedule.py \
  --scene /scenes/agibot_scene.json \
  --schedule 'right+w:8,space:1,none:7' \
  --output /tmp/agibot_action.pt
```

The output is raw `[16,29]`; it is suitable for controller tests or replay and
contains no normalizer/padding transformation.

## Run

Install the optional WebRTC dependencies, then start the model and gateway in
one process. Use a deployment environment without the repository's `dev`
extra: its ComfyUI test dependency pins PyAV 18, while current `aiortc` releases
require PyAV below 18.

```bash
pip install -e '.[cosmos-dreams-demo]'

python examples/online_serving/cosmos_dreams/webrtc_demo.py \
  --model /exports/cosmos-dreams-diffusers \
  --scene /scenes/agibot_scene.json \
  --deploy-config vllm_omni/deploy/cosmos_dreams.yaml \
  --host 0.0.0.0 --port 8080
```

Open `http://HOST:8080`. Keys `1/2/3` select left wrist, right wrist, or head;
`W/S A/D R/F` translate in the selected local OpenCV frame; `I/K J/L U/O`
rotate; Space toggles the selected wrist gripper. The server ignores browser
auto-repeat, releases stuck keys on blur, supports heartbeat/reset/disconnect,
and applies backpressure through a bounded RGB frame queue.

For a model-parity replay that bypasses keyboard translation, preload raw
action chunks:

```bash
python examples/online_serving/cosmos_dreams/webrtc_demo.py \
  --model /exports/cosmos-dreams-diffusers \
  --scene /scenes/agibot_scene.json \
  --replay-actions /fixtures/raw_agibot_chunks.pt
```

The gateway reports denoise, clean-cache commit, incremental VAE decode,
request wall time, RGB/encoder handoff, enqueue, and browser-acknowledged
presentation latency per chunk. Cold chunk 0 is separate from the warm p50/p90 summary. At
30 fps, the 16-frame playback budget is 533 ms. If warm p90 exceeds that
budget, describe the run as buffered/non-realtime or lower the advertised
playback rate; the queue is bounded and does not hide accumulating lag.

`/v1/videos` and the offline jsonl/pickle runner remain evaluation/replay
surfaces. A 17-frame fixture cannot be extended to 65 frames unless the missing
raw action/state rows are available.
