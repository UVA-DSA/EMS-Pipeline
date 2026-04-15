#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
  exec "$@"
fi

APP_HOME="${APP_HOME:-/workspace/EgoEMS}"
BACKEND="${BACKEND:-detr-trt}"
ENGINE_PATH="${ENGINE_PATH:-${APP_HOME}/Tools/inference/checkpoints/ems_finetuned_detr_trt.ts}"
ACTIVITY_ENGINE_PATH="${ACTIVITY_ENGINE_PATH:-}"
ACTIVITY_FEATURE_ENGINE_PATH="${ACTIVITY_FEATURE_ENGINE_PATH:-}"
ACTIVITY_CLASS_MAP_PATH="${ACTIVITY_CLASS_MAP_PATH:-${APP_HOME}/Tools/inference/conversion/action_class_mapping.json}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-8000}"
DEVICE="${DEVICE:-cuda}"
WARMUP="${WARMUP:-3}"
DETR_VERSION="${DETR_VERSION:-ems}"
DETECTION_THRESHOLD="${DETECTION_THRESHOLD:-0.7}"
ENGINE_HEIGHT="${ENGINE_HEIGHT:-480}"
ENGINE_WIDTH="${ENGINE_WIDTH:-640}"
ACTIVITY_WINDOW_SIZE="${ACTIVITY_WINDOW_SIZE:-30}"
ACTIVITY_STRIDE="${ACTIVITY_STRIDE:-1}"
ACTIVITY_MODEL_SEQ_LEN="${ACTIVITY_MODEL_SEQ_LEN:-}"
ACTIVITY_RESIZE_SHORT_SIDE="${ACTIVITY_RESIZE_SHORT_SIDE:-256}"
ACTIVITY_CENTER_CROP_SIZE="${ACTIVITY_CENTER_CROP_SIZE:-224}"
ACTIVITY_FEATURE_WEIGHTS="${ACTIVITY_FEATURE_WEIGHTS:-imagenet1k_v1}"
SERVER_ENTRYPOINT="${APP_HOME}/Tools/inference/servers/container_inference_server.py"

if [[ ! -d "${APP_HOME}" ]]; then
  echo "Error: APP_HOME does not exist: ${APP_HOME}" >&2
  echo "This image normally includes the inference app at that path." >&2
  echo "If you overrode APP_HOME, point it at a valid baked or mounted app directory." >&2
  exit 1
fi

if [[ ! -f "${SERVER_ENTRYPOINT}" ]]; then
  echo "Error: server entrypoint not found: ${SERVER_ENTRYPOINT}" >&2
  echo "The image may be incomplete, or APP_HOME may be pointing at the wrong directory." >&2
  exit 1
fi

if [[ ! -f "${ENGINE_PATH}" ]]; then
  echo "Error: TensorRT engine not found: ${ENGINE_PATH}" >&2
  echo "The default image bakes in the default engine path above." >&2
  echo "If you overrode ENGINE_PATH, verify that the target file exists." >&2
  exit 1
fi

if [[ -n "${ACTIVITY_ENGINE_PATH}" && ! -f "${ACTIVITY_ENGINE_PATH}" ]]; then
  echo "Error: activity TensorRT engine not found: ${ACTIVITY_ENGINE_PATH}" >&2
  echo "If you overrode ACTIVITY_ENGINE_PATH, verify that the target file exists." >&2
  exit 1
fi

if [[ -n "${ACTIVITY_FEATURE_ENGINE_PATH}" && ! -f "${ACTIVITY_FEATURE_ENGINE_PATH}" ]]; then
  echo "Error: activity feature TensorRT engine not found: ${ACTIVITY_FEATURE_ENGINE_PATH}" >&2
  echo "If you overrode ACTIVITY_FEATURE_ENGINE_PATH, verify that the target file exists." >&2
  exit 1
fi

echo "[server] starting container inference server"
echo "[server] app home   : ${APP_HOME}"
echo "[server] backend    : ${BACKEND}"
echo "[server] engine     : ${ENGINE_PATH}"
echo "[server] host:port  : ${SERVER_HOST}:${SERVER_PORT}"
echo "[server] device     : ${DEVICE}"
echo "[server] detr ver   : ${DETR_VERSION}"
echo "[server] threshold  : ${DETECTION_THRESHOLD}"
echo "[server] input size : ${ENGINE_WIDTH}x${ENGINE_HEIGHT}"
if [[ -n "${ACTIVITY_ENGINE_PATH}" ]]; then
  echo "[server] activity   : enabled"
  echo "[server] act engine : ${ACTIVITY_ENGINE_PATH}"
  if [[ -n "${ACTIVITY_FEATURE_ENGINE_PATH}" ]]; then
    echo "[server] act feat   : ${ACTIVITY_FEATURE_ENGINE_PATH}"
  else
    echo "[server] act feat   : pytorch resnet50"
  fi
  echo "[server] act window : ${ACTIVITY_WINDOW_SIZE}"
  echo "[server] act stride : ${ACTIVITY_STRIDE}"
else
  echo "[server] activity   : disabled"
fi

args=(
  --backend "${BACKEND}"
  --engine "${ENGINE_PATH}"
  --host "${SERVER_HOST}"
  --port "${SERVER_PORT}"
  --device "${DEVICE}"
  --warmup "${WARMUP}"
  --detr-version "${DETR_VERSION}"
  --threshold "${DETECTION_THRESHOLD}"
  --engine-height "${ENGINE_HEIGHT}"
  --engine-width "${ENGINE_WIDTH}"
)

if [[ -n "${ACTIVITY_ENGINE_PATH}" ]]; then
  args+=(
    --activity-engine "${ACTIVITY_ENGINE_PATH}"
    --activity-class-map "${ACTIVITY_CLASS_MAP_PATH}"
    --activity-window-size "${ACTIVITY_WINDOW_SIZE}"
    --activity-stride "${ACTIVITY_STRIDE}"
    --activity-resize-short-side "${ACTIVITY_RESIZE_SHORT_SIDE}"
    --activity-center-crop-size "${ACTIVITY_CENTER_CROP_SIZE}"
    --activity-feature-weights "${ACTIVITY_FEATURE_WEIGHTS}"
  )

  if [[ -n "${ACTIVITY_FEATURE_ENGINE_PATH}" ]]; then
    args+=(
      --activity-feature-engine "${ACTIVITY_FEATURE_ENGINE_PATH}"
    )
  fi

  if [[ -n "${ACTIVITY_MODEL_SEQ_LEN}" ]]; then
    args+=(
      --activity-model-seq-len "${ACTIVITY_MODEL_SEQ_LEN}"
    )
  fi
fi

exec python "${SERVER_ENTRYPOINT}" \
  "${args[@]}"
