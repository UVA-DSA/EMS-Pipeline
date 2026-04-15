#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/workspace/EgoEMS/Tools/inference/checkpoints/mtrsap_30frames_window_resnet.pt}"
OUTPUT_PATH="${OUTPUT_PATH:-/workspace/EgoEMS/Tools/inference/checkpoints/mtrsap_30frames_window_resnet_trt.ts}"

BATCH_SIZE="${BATCH_SIZE:-1}"
SEQ_LEN="${SEQ_LEN:-30}"
NHEAD="${NHEAD:-4}"
DROPOUT="${DROPOUT:-0.1}"
FP16="${FP16:-1}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: '${PYTHON_BIN}' is not available in PATH." >&2
  exit 1
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Error: checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

extra_args=()
if [[ "${FP16}" == "1" ]]; then
  extra_args+=(--fp16)
fi

exec "${PYTHON_BIN}" -u "${SCRIPT_DIR}/MTRSAP_model_to_tensorRT.py" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --output "${OUTPUT_PATH}" \
  --batch-size "${BATCH_SIZE}" \
  --seq-len "${SEQ_LEN}" \
  --nhead "${NHEAD}" \
  --dropout "${DROPOUT}" \
  "${extra_args[@]}" \
  "$@"
