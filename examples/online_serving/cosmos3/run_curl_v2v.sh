#!/bin/bash
# Cosmos3 video-to-video example using the sync video API.
#
# The prompt and V2V conditioning controls are loaded from the canonical input
# JSON shared with the offline example. The companion video is auto-downloaded
# if missing.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUTS_DIR="${INPUTS_DIR:-${SCRIPT_DIR}/../../offline_inference/cosmos3/inputs}"
INPUT_JSON="${INPUT_JSON:-${INPUTS_DIR}/v2v.json}"

BASE_URL="${BASE_URL:-http://localhost:8091}"
OUTPUT_PATH="${OUTPUT_PATH:-cosmos3_v2v.mp4}"
VIDEO_PATH="${VIDEO_PATH:-robot_pouring.mp4}"

if [ ! -f "${INPUT_JSON}" ]; then
  echo "Missing input JSON: ${INPUT_JSON}" >&2
  exit 1
fi

PROMPT="$(jq -r '.prompt' "${INPUT_JSON}")"
NEGATIVE_PROMPT="$(jq -r '.negative_prompt // ""' "${INPUT_JSON}")"
VISION_URL="$(jq -r '.vision_path' "${INPUT_JSON}")"
HEIGHT="$(jq -r '.height // 720' "${INPUT_JSON}")"
WIDTH="$(jq -r '.width // 1280' "${INPUT_JSON}")"
NUM_FRAMES="$(jq -r '.num_frames // 189' "${INPUT_JSON}")"
FPS="$(jq -r '.fps // 24' "${INPUT_JSON}")"
NUM_INFERENCE_STEPS="$(jq -r '.num_inference_steps // 35' "${INPUT_JSON}")"
GUIDANCE_SCALE="$(jq -r '.guidance_scale // 6.0' "${INPUT_JSON}")"
FLOW_SHIFT="$(jq -r '.flow_shift // 10.0' "${INPUT_JSON}")"
SEED="$(jq -r '.seed // 42' "${INPUT_JSON}")"
JSON_CONDITION_FRAME_INDEXES_VISION="$(
  jq -c '.condition_frame_indexes_vision // [0, 1]' "${INPUT_JSON}"
)"
JSON_CONDITION_VIDEO_KEEP="$(jq -r '.condition_video_keep // "first"' "${INPUT_JSON}")"

CONDITION_FRAME_INDEXES_VISION="${CONDITION_FRAME_INDEXES_VISION:-${JSON_CONDITION_FRAME_INDEXES_VISION}}"
CONDITION_VIDEO_KEEP="${CONDITION_VIDEO_KEEP:-${JSON_CONDITION_VIDEO_KEEP}}"

if [ ! -f "${VIDEO_PATH}" ]; then
  echo "Downloading ${VISION_URL} -> ${VIDEO_PATH}"
  curl -sSL "${VISION_URL}" -o "${VIDEO_PATH}"
fi

EXTRA_PARAMS="$(jq -nc \
  --argjson indexes "${CONDITION_FRAME_INDEXES_VISION}" \
  --arg keep "${CONDITION_VIDEO_KEEP}" \
  '{condition_frame_indexes_vision:$indexes, condition_video_keep:$keep}')"

curl -sS -X POST "${BASE_URL}/v1/videos/sync" \
  --form-string "prompt=${PROMPT}" \
  --form-string "negative_prompt=${NEGATIVE_PROMPT}" \
  -F "input_reference=@${VIDEO_PATH}" \
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
