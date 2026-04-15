#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
FEATURE_TYPE="${FEATURE_TYPE:-resnet}"
DEVICE="${DEVICE:-cuda}"

# ResNet runtime inference configuration.
SOURCE_FPS=30
MODEL_SEQ_LEN=30
WINDOW_LENGTH=1
STRIDE=0.5

VIDEO_PATH="${VIDEO_PATH:-${SCRIPT_DIR}/videos/ms1_cardiac_arrest_t4_ks2_5.273_12.523_ego.mp4}"
ENGINE_PATH="${ENGINE_PATH:-${REPO_ROOT}/Benchmarks/ActionRecognition/MTRSAP/checkpoints/resnet_segmentation/resnet_30s_segmentation_mtrsap_trt.ts}"
CSV_PATH="${CSV_PATH:-${SCRIPT_DIR}/mtrsap_inference_results.csv}"

# To tune speed/context for ResNet mode:
# - Increase STRIDE to run fewer inferences.
# - Reduce WINDOW_LENGTH for shorter temporal context.

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: '${PYTHON_BIN}' is not available in PATH." >&2
  exit 1
fi

for required_file in "${VIDEO_PATH}" "${ENGINE_PATH}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Error: required file not found: ${required_file}" >&2
    exit 1
  fi
done

exec "${PYTHON_BIN}" -u "${SCRIPT_DIR}/MTRSAP_inference.py" \
  --feature-type "${FEATURE_TYPE}" \
  --video-path "${VIDEO_PATH}" \
  --engine-path "${ENGINE_PATH}" \
  --csv-path "${CSV_PATH}" \
  --device "${DEVICE}" \
  --fps "${SOURCE_FPS}" \
  --window-seconds "${WINDOW_LENGTH}" \
  --stride-seconds "${STRIDE}" \
  --model-seq-len "${MODEL_SEQ_LEN}" \
  "$@"
