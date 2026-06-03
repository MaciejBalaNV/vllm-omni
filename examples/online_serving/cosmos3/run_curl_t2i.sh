#!/bin/bash
# Cosmos3 text-to-image example using the images API.
#
# The prompt is loaded from the canonical input JSON shared with the offline
# example so updates only need to happen in one place.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUTS_DIR="${INPUTS_DIR:-${SCRIPT_DIR}/../../offline_inference/cosmos3/inputs}"
INPUT_JSON="${INPUT_JSON:-${INPUTS_DIR}/t2i.json}"

BASE_URL="${BASE_URL:-http://localhost:8091}"
OUTPUT_PATH="${OUTPUT_PATH:-cosmos3_t2i.png}"

if [ ! -f "${INPUT_JSON}" ]; then
  echo "Missing input JSON: ${INPUT_JSON}" >&2
  exit 1
fi

PROMPT="$(jq -r '.prompt' "${INPUT_JSON}")"
HEIGHT="$(jq -r '.height // 960' "${INPUT_JSON}")"
WIDTH="$(jq -r '.width // 960' "${INPUT_JSON}")"
NUM_INFERENCE_STEPS="$(jq -r '.num_inference_steps // 50' "${INPUT_JSON}")"
GUIDANCE_SCALE="$(jq -r '.guidance_scale // 4.0' "${INPUT_JSON}")"
FLOW_SHIFT="$(jq -r '.flow_shift // 3.0' "${INPUT_JSON}")"

curl -sS -X POST "${BASE_URL}/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d "$(jq -nc \
        --arg prompt "${PROMPT}" \
        --arg negative "blurry, distorted, low quality" \
        --arg size "${WIDTH}x${HEIGHT}" \
        --argjson steps "${NUM_INFERENCE_STEPS}" \
        --argjson guidance "${GUIDANCE_SCALE}" \
        --argjson flow_shift "${FLOW_SHIFT}" \
        '{prompt:$prompt,
          size:$size,
          n:1,
          num_inference_steps:$steps,
          guidance_scale:$guidance,
          flow_shift:$flow_shift,
          negative_prompt:$negative,
          seed:42}')" \
  | jq -r '.data[0].b64_json' | base64 -d > "${OUTPUT_PATH}"

echo "Saved image to ${OUTPUT_PATH}"
