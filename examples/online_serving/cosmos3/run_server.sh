#!/bin/bash
# Cosmos3 online serving startup script.

set -euo pipefail

MODEL="${MODEL:-${COSMOS3_MODEL:-nvidia/Cosmos3-Nano}}"
PORT="${PORT:-8091}"
HOST="${HOST:-}"
MODEL_CLASS_NAME="${MODEL_CLASS_NAME:-Cosmos3OmniDiffusersPipeline}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"
CACHE_BACKEND="${CACHE_BACKEND:-none}"
ENABLE_CACHE_DIT_SUMMARY="${ENABLE_CACHE_DIT_SUMMARY:-0}"
ENABLE_LAYERWISE_OFFLOAD="${ENABLE_LAYERWISE_OFFLOAD:-0}"
CFG_PARALLEL_SIZE="${CFG_PARALLEL_SIZE:-1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
ULYSSES_DEGREE="${ULYSSES_DEGREE:-1}"
RING_DEGREE="${RING_DEGREE:-1}"
VAE_PATCH_PARALLEL_SIZE="${VAE_PATCH_PARALLEL_SIZE:-1}"
VAE_USE_SLICING="${VAE_USE_SLICING:-0}"
VAE_USE_TILING="${VAE_USE_TILING:-0}"
USE_HSDP="${USE_HSDP:-0}"
HSDP_SHARD_SIZE="${HSDP_SHARD_SIZE:-}"
HSDP_REPLICATE_SIZE="${HSDP_REPLICATE_SIZE:-}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-/}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-}"
DEFAULT_SAMPLING_PARAMS="${DEFAULT_SAMPLING_PARAMS:-}"

args=(
  vllm serve "${MODEL}"
  --omni
  --port "${PORT}"
  --allowed-local-media-path "${ALLOWED_LOCAL_MEDIA_PATH}"
  --cfg-parallel-size "${CFG_PARALLEL_SIZE}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
)

if [ -n "${HOST}" ]; then
  args+=(--host "${HOST}")
fi

if [ -n "${MODEL_CLASS_NAME}" ]; then
  args+=(--model-class-name "${MODEL_CLASS_NAME}")
fi

if [ -n "${SERVED_MODEL_NAME}" ]; then
  args+=(--served-model-name "${SERVED_MODEL_NAME}")
fi

if [ -n "${DEPLOY_CONFIG}" ]; then
  args+=(--deploy-config "${DEPLOY_CONFIG}")
fi

if [ -n "${DEFAULT_SAMPLING_PARAMS}" ]; then
  args+=(--default-sampling-params "${DEFAULT_SAMPLING_PARAMS}")
fi

if [ "${ULYSSES_DEGREE}" != "1" ]; then
  args+=(--usp "${ULYSSES_DEGREE}")
fi

if [ "${RING_DEGREE}" != "1" ]; then
  args+=(--ring "${RING_DEGREE}")
fi

if [ "${VAE_PATCH_PARALLEL_SIZE}" != "1" ]; then
  args+=(--vae-patch-parallel-size "${VAE_PATCH_PARALLEL_SIZE}")
fi

if [ "${CACHE_BACKEND}" != "none" ]; then
  args+=(--cache-backend "${CACHE_BACKEND}")
fi

if [ "${ENABLE_CACHE_DIT_SUMMARY}" != "0" ]; then
  args+=(--enable-cache-dit-summary)
fi

if [ "${ENABLE_LAYERWISE_OFFLOAD}" != "0" ]; then
  args+=(--enable-layerwise-offload)
fi

if [ "${VAE_USE_SLICING}" != "0" ]; then
  args+=(--vae-use-slicing)
fi

if [ "${VAE_USE_TILING}" != "0" ]; then
  args+=(--vae-use-tiling)
fi

if [ "${ENFORCE_EAGER}" != "0" ]; then
  args+=(--enforce-eager)
fi

if [ "${USE_HSDP}" != "0" ]; then
  args+=(--use-hsdp)
fi

if [ -n "${HSDP_SHARD_SIZE}" ]; then
  args+=(--hsdp-shard-size "${HSDP_SHARD_SIZE}")
fi

if [ -n "${HSDP_REPLICATE_SIZE}" ]; then
  args+=(--hsdp-replicate-size "${HSDP_REPLICATE_SIZE}")
fi

if [ "$#" -gt 0 ]; then
  args+=("$@")
fi

echo "Starting Cosmos3 server on port ${PORT}"
echo "Model: ${MODEL}"
exec "${args[@]}"
