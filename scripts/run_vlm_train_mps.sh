#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-mps-vlm"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[error] venv missing: ${VENV_DIR}"
  exit 1
fi

source "${VENV_DIR}/bin/activate"

echo "[run] strict retrain pipeline"
echo "      (gate training -> gate inference quality -> full training -> best checkpoint selection by val quality)"

bash "${ROOT_DIR}/scripts/retrain_vlm_strict_pipeline.sh"
