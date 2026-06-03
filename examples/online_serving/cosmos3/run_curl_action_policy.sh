#!/bin/bash
# Cosmos3 action policy example (bridge_orig_lerobot, image input).
#
# Cosmos3 policy mode consumes an image plus a language instruction and
# generates a video together with the predicted action sequence. The example
# image is the first frame of bridge_0.mp4 (cosmos-dependencies), extracted
# locally with ffmpeg so the request matches the prompt scene.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUTS_DIR="${INPUTS_DIR:-${SCRIPT_DIR}/../../offline_inference/cosmos3/inputs}"
INPUT_JSON="${INPUT_JSON:-${INPUTS_DIR}/action_policy_robot.json}"

BASE_URL="${BASE_URL:-http://localhost:8091}"
OUTPUT_PATH="${OUTPUT_PATH:-cosmos3_action_policy.mp4}"
ACTION_OUTPUT_PATH="${ACTION_OUTPUT_PATH:-cosmos3_action_policy_action.json}"
IMAGE_PATH="${IMAGE_PATH:-bridge_0_frame0.jpg}"
VIDEO_PATH="${VIDEO_PATH:-bridge_0.mp4}"
POLL_INTERVAL="${POLL_INTERVAL:-2}"

if [ ! -f "${INPUT_JSON}" ]; then
  echo "Missing input JSON: ${INPUT_JSON}" >&2
  exit 1
fi

PROMPT="$(jq -r '.prompt' "${INPUT_JSON}")"
VISION_URL="$(jq -r '.vision_path' "${INPUT_JSON}")"
DOMAIN_NAME="$(jq -r '.domain_name' "${INPUT_JSON}")"
RAW_ACTION_DIM="$(jq -r '.raw_action_dim' "${INPUT_JSON}")"
ACTION_CHUNK_SIZE="$(jq -r '.action_chunk_size' "${INPUT_JSON}")"
NUM_FRAMES="$(jq -r '.num_frames // 17' "${INPUT_JSON}")"
FPS="$(jq -r '.fps // 5' "${INPUT_JSON}")"
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

EXTRA_PARAMS="$(jq -nc \
  --arg domain "${DOMAIN_NAME}" \
  --argjson dim "${RAW_ACTION_DIM}" \
  --argjson chunk "${ACTION_CHUNK_SIZE}" \
  '{action_mode:"policy", domain_name:$domain, raw_action_dim:$dim, action_chunk_size:$chunk}')"

create_response=$(
  curl -sS -X POST "${BASE_URL}/v1/videos" \
    -H "Accept: application/json" \
    --form-string "prompt=${PROMPT}" \
    -F "input_reference=@${IMAGE_PATH}" \
    -F "size=${WIDTH}x${HEIGHT}" \
    -F "num_frames=${NUM_FRAMES}" \
    -F "fps=${FPS}" \
    -F "num_inference_steps=${NUM_INFERENCE_STEPS}" \
    -F "guidance_scale=${GUIDANCE_SCALE}" \
    -F "flow_shift=${FLOW_SHIFT}" \
    --form-string "extra_params=${EXTRA_PARAMS}" \
    -F "seed=${SEED}"
)

video_id="$(echo "${create_response}" | jq -r '.id')"
if [ -z "${video_id}" ] || [ "${video_id}" = "null" ]; then
  echo "Failed to create video job:"
  echo "${create_response}" | jq .
  exit 1
fi

echo "Created video job ${video_id}"
while true; do
  status_response="$(curl -sS "${BASE_URL}/v1/videos/${video_id}")"
  status="$(echo "${status_response}" | jq -r '.status')"

  case "${status}" in
    queued|in_progress)
      echo "Video job ${video_id} status: ${status}"
      sleep "${POLL_INTERVAL}"
      ;;
    completed)
      echo "${status_response}" | jq '.data[0].action' > "${ACTION_OUTPUT_PATH}"
      break
      ;;
    failed)
      echo "Video generation failed:"
      echo "${status_response}" | jq .
      exit 1
      ;;
    *)
      echo "Unexpected status response:"
      echo "${status_response}" | jq .
      exit 1
      ;;
  esac
done

curl -sS -L "${BASE_URL}/v1/videos/${video_id}/content" -o "${OUTPUT_PATH}"

echo "Saved video to ${OUTPUT_PATH}"
echo "Saved action metadata to ${ACTION_OUTPUT_PATH}"
