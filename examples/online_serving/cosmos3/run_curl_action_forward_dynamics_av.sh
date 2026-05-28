#!/bin/bash
# Cosmos3 forward-dynamics example (autonomous vehicle, image input + action).
#
# See run_curl_action_forward_dynamics_robot.sh for notes on how action_path
# is consumed by the server.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUTS_DIR="${INPUTS_DIR:-${SCRIPT_DIR}/../../offline_inference/cosmos3/inputs}"
INPUT_JSON="${INPUT_JSON:-${INPUTS_DIR}/action_forward_dynamics_av.json}"

BASE_URL="${BASE_URL:-http://localhost:8091}"
OUTPUT_PATH="${OUTPUT_PATH:-cosmos3_forward_dynamics_av.mp4}"
IMAGE_PATH="${IMAGE_PATH:-av_vision_25_frame0.jpg}"
VIDEO_PATH="${VIDEO_PATH:-av_vision_25.mp4}"
ACTION_PATH="${ACTION_PATH:-$(pwd)/av_action_25.json}"

if [ ! -f "${INPUT_JSON}" ]; then
  echo "Missing input JSON: ${INPUT_JSON}" >&2
  exit 1
fi

PROMPT="$(jq -r '.prompt' "${INPUT_JSON}")"
VISION_URL="$(jq -r '.vision_path' "${INPUT_JSON}")"
ACTION_URL="$(jq -r '.action_path' "${INPUT_JSON}")"
DOMAIN_NAME="$(jq -r '.domain_name' "${INPUT_JSON}")"
RAW_ACTION_DIM="$(jq -r '.raw_action_dim' "${INPUT_JSON}")"
ACTION_CHUNK_SIZE="$(jq -r '.action_chunk_size' "${INPUT_JSON}")"
NUM_FRAMES="$(jq -r '.num_frames // 61' "${INPUT_JSON}")"
FPS="$(jq -r '.fps // 10' "${INPUT_JSON}")"
HEIGHT="$(jq -r '.height // 480' "${INPUT_JSON}")"
WIDTH="$(jq -r '.width // 640' "${INPUT_JSON}")"
NUM_INFERENCE_STEPS="$(jq -r '.num_inference_steps // 30' "${INPUT_JSON}")"
GUIDANCE_SCALE="$(jq -r '.guidance_scale // 1.0' "${INPUT_JSON}")"
FLOW_SHIFT="$(jq -r '.flow_shift // 5.0' "${INPUT_JSON}")"
SEED="$(jq -r '.seed // 0' "${INPUT_JSON}")"

if [ ! -f "${IMAGE_PATH}" ]; then
  if [ ! -f "${VIDEO_PATH}" ]; then
    echo "Downloading ${VISION_URL} -> ${VIDEO_PATH}"
    curl -sSL "${VISION_URL}" -o "${VIDEO_PATH}"
  fi
  echo "Extracting first frame -> ${IMAGE_PATH}"
  ffmpeg -y -loglevel error -i "${VIDEO_PATH}" -vf "select=eq(n\,0)" -vframes 1 "${IMAGE_PATH}"
fi

if [ ! -f "${ACTION_PATH}" ]; then
  echo "Downloading ${ACTION_URL} -> ${ACTION_PATH}"
  curl -sSL "${ACTION_URL}" -o "${ACTION_PATH}"
fi

EXTRA_PARAMS="$(jq -nc \
  --arg domain "${DOMAIN_NAME}" \
  --argjson dim "${RAW_ACTION_DIM}" \
  --argjson chunk "${ACTION_CHUNK_SIZE}" \
  --arg action_path "${ACTION_PATH}" \
  '{action_mode:"forward_dynamics", domain_name:$domain, raw_action_dim:$dim, action_chunk_size:$chunk, action_path:$action_path}')"

curl -sS -X POST "${BASE_URL}/v1/videos/sync" \
  --form-string "prompt=${PROMPT}" \
  -F "input_reference=@${IMAGE_PATH}" \
  -F "size=${WIDTH}x${HEIGHT}" \
  -F "num_frames=${NUM_FRAMES}" \
  -F "fps=${FPS}" \
  -F "num_inference_steps=${NUM_INFERENCE_STEPS}" \
  -F "guidance_scale=${GUIDANCE_SCALE}" \
  -F "flow_shift=${FLOW_SHIFT}" \
  --form-string "extra_params=${EXTRA_PARAMS}" \
  -F "seed=${SEED}" \
  -o "${OUTPUT_PATH}"

echo "Saved video to ${OUTPUT_PATH}"
