#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-mps-vlm"
OUT_DIR="${ROOT_DIR}/data/vlm_seatcontact"

SRC_A="${SRC_A:-${HOME}/Desktop/chair_dataset}"
SRC_B="${SRC_B:-${SRC_A}/v2}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[error] venv missing: ${VENV_DIR}"
  echo "Run: bash ${ROOT_DIR}/scripts/setup_mps_vlm_env.sh"
  exit 1
fi

source "${VENV_DIR}/bin/activate"

echo "[prepare] build VLM dataset"
if [[ ! -d "${SRC_A}" ]]; then
  echo "[error] dataset directory not found: ${SRC_A}"
  echo "Set SRC_A=/path/to/chair_dataset and run again."
  exit 1
fi

python "${ROOT_DIR}/scripts/build_vlm_dataset_from_labelme.py" \
  --src-dir "${SRC_A}" \
  --src-dir "${SRC_B}" \
  --out-dir "${OUT_DIR}" \
  --label seat_contact \
  --seed 42 \
  --train-ratio 0.85 \
  --val-ratio 0.10 \
  --prefer-last-source

echo "[prepare] validate jsonl"
python "${ROOT_DIR}/scripts/validate_vlm_jsonl.py" \
  --jsonl "${OUT_DIR}/train.jsonl" \
  --jsonl "${OUT_DIR}/val.jsonl" \
  --jsonl "${OUT_DIR}/test.jsonl"

cat <<EOF

[prepare] done
dataset dir:
  ${OUT_DIR}

next (dry-run only, no train):
  source "${VENV_DIR}/bin/activate"
  python "${ROOT_DIR}/scripts/train_vlm_lora_mps.py" \
    --model-name "HuggingFaceTB/SmolVLM-256M-Instruct" \
    --dry-run \
    --max-train-samples 8 \
    --max-eval-samples 4

actual training command:
  source "${VENV_DIR}/bin/activate"
  bash "${ROOT_DIR}/scripts/run_vlm_train_mps.sh"

strict full pipeline (recommended):
  bash "${ROOT_DIR}/scripts/retrain_vlm_strict_pipeline.sh"
EOF
