#!/bin/bash
# Cosmos3 text-to-video example using the sync video API.
#
# The prompt is loaded from the canonical input JSON shared with the offline
# example so updates only need to happen in one place.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUTS_DIR="${INPUTS_DIR:-${SCRIPT_DIR}/../../offline_inference/cosmos3/inputs}"
INPUT_JSON="${INPUT_JSON:-${INPUTS_DIR}/t2v.json}"

BASE_URL="${BASE_URL:-http://localhost:8091}"
OUTPUT_PATH="${OUTPUT_PATH:-cosmos3_t2v.mp4}"

if [ ! -f "${INPUT_JSON}" ]; then
  echo "Missing input JSON: ${INPUT_JSON}" >&2
  exit 1
fi

PROMPT="$(jq -r '.prompt' "${INPUT_JSON}")"
HEIGHT="$(jq -r '.height // 720' "${INPUT_JSON}")"
WIDTH="$(jq -r '.width // 1280' "${INPUT_JSON}")"
NUM_FRAMES="$(jq -r '.num_frames // 189' "${INPUT_JSON}")"
FPS="$(jq -r '.fps // 24' "${INPUT_JSON}")"
NUM_INFERENCE_STEPS="$(jq -r '.num_inference_steps // 35' "${INPUT_JSON}")"
GUIDANCE_SCALE="$(jq -r '.guidance_scale // 6.0' "${INPUT_JSON}")"
FLOW_SHIFT="$(jq -r '.flow_shift // 10.0' "${INPUT_JSON}")"

curl -sS -X POST "${BASE_URL}/v1/videos/sync" \
  --form-string "prompt=${PROMPT}" \
  --form-string "negative_prompt=blurry, distorted, low quality" \
  -F "size=${WIDTH}x${HEIGHT}" \
  -F "num_frames=${NUM_FRAMES}" \
  -F "fps=${FPS}" \
  -F "num_inference_steps=${NUM_INFERENCE_STEPS}" \
  -F "guidance_scale=${GUIDANCE_SCALE}" \
  -F "flow_shift=${FLOW_SHIFT}" \
  -F "seed=42" \
  -o "${OUTPUT_PATH}"

echo "Saved video to ${OUTPUT_PATH}"
