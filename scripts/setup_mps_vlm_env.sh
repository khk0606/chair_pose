#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-mps-vlm"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[setup] root: ${ROOT_DIR}"
echo "[setup] python: $(${PYTHON_BIN} --version)"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[setup] creating venv at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

echo "[setup] upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

echo "[setup] installing requirements-mps-vlm.txt"
python -m pip install -r "${ROOT_DIR}/requirements-mps-vlm.txt"

echo "[setup] validating environment"
python "${ROOT_DIR}/scripts/check_mps_vlm_env.py"

cat <<EOF

[setup] done
activate:
  source "${VENV_DIR}/bin/activate"

quick check:
  python "${ROOT_DIR}/scripts/check_mps_vlm_env.py"
EOF
