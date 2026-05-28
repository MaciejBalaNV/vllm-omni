#!/bin/bash
# Cosmos3 forward-dynamics example (camera_pose domain, image input + action).
#
# The JSONL input ships two scenes: mountain (index 0) and solar (index 1).
# Pick one with SCENE_INDEX. Vision input is a still PNG, so no ffmpeg step.
# Forward-dynamics returns only a video (no predicted action), so this uses
# the sync video endpoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUTS_DIR="${INPUTS_DIR:-${SCRIPT_DIR}/../../offline_inference/cosmos3/inputs}"
INPUT_JSONL="${INPUT_JSONL:-${INPUTS_DIR}/action_forward_dynamics_camera.jsonl}"
SCENE_INDEX="${SCENE_INDEX:-0}"

BASE_URL="${BASE_URL:-http://localhost:8091}"
OUTPUT_PATH="${OUTPUT_PATH:-cosmos3_forward_dynamics_camera.mp4}"

if [ ! -f "${INPUT_JSONL}" ]; then
  echo "Missing input JSONL: ${INPUT_JSONL}" >&2
  exit 1
fi

RECORD="$(awk "NR==$((SCENE_INDEX + 1))" "${INPUT_JSONL}")"
if [ -z "${RECORD}" ]; then
  echo "SCENE_INDEX=${SCENE_INDEX} out of range for ${INPUT_JSONL}" >&2
  exit 1
fi

PROMPT="$(echo "${RECORD}" | jq -r '.prompt')"
VISION_URL="$(echo "${RECORD}" | jq -r '.vision_path')"
ACTION_URL="$(echo "${RECORD}" | jq -r '.action_path')"
DOMAIN_NAME="$(echo "${RECORD}" | jq -r '.domain_name')"
ACTION_CHUNK_SIZE="$(echo "${RECORD}" | jq -r '.action_chunk_size')"
NUM_FRAMES="$(echo "${RECORD}" | jq -r '.num_frames // 61')"
FPS="$(echo "${RECORD}" | jq -r '.fps // 30')"
HEIGHT="$(echo "${RECORD}" | jq -r '.height // 480')"
WIDTH="$(echo "${RECORD}" | jq -r '.width // 640')"
NUM_INFERENCE_STEPS="$(echo "${RECORD}" | jq -r '.num_inference_steps // 30')"
GUIDANCE_SCALE="$(echo "${RECORD}" | jq -r '.guidance_scale // 1.0')"
FLOW_SHIFT="$(echo "${RECORD}" | jq -r '.flow_shift // 5.0')"
SEED="$(echo "${RECORD}" | jq -r '.seed // 0')"

IMAGE_PATH="${IMAGE_PATH:-camera_scene_${SCENE_INDEX}.png}"
ACTION_PATH="${ACTION_PATH:-$(pwd)/camera_action_44.json}"

if [ ! -f "${IMAGE_PATH}" ]; then
  echo "Downloading ${VISION_URL} -> ${IMAGE_PATH}"
  curl -sSL "${VISION_URL}" -o "${IMAGE_PATH}"
fi

if [ ! -f "${ACTION_PATH}" ]; then
  echo "Downloading ${ACTION_URL} -> ${ACTION_PATH}"
  curl -sSL "${ACTION_URL}" -o "${ACTION_PATH}"
fi

EXTRA_PARAMS="$(jq -nc \
  --arg domain "${DOMAIN_NAME}" \
  --argjson chunk "${ACTION_CHUNK_SIZE}" \
  --arg action_path "${ACTION_PATH}" \
  '{action_mode:"forward_dynamics", domain_name:$domain, action_chunk_size:$chunk, action_path:$action_path}')"

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
