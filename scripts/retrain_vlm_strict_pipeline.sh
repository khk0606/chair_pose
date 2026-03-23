#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-mps-vlm"
DATA_DIR="${ROOT_DIR}/data/vlm_seatcontact"

SRC_A="${SRC_A:-${HOME}/Desktop/chair_dataset}"
SRC_B="${SRC_B:-${HOME}/Desktop/chair_dataset/v2}"
BASE_MODEL="${BASE_MODEL:-HuggingFaceTB/SmolVLM-256M-Instruct}"

RUN_TAG="${RUN_TAG:-vlm_lora_mps_strict_v3}"
RUN_ROOT="${ROOT_DIR}/runs/${RUN_TAG}"
GATE_DIR="${RUN_ROOT}/gate"
FINAL_DIR="${RUN_ROOT}/final"
QUALITY_DIR="${RUN_ROOT}/val_quality"
BEST_SELECTION_JSON="${RUN_ROOT}/best_adapter_selection.json"
BEST_ADAPTER_TXT="${RUN_ROOT}/best_adapter_path.txt"
GATE_EVAL_JSON="${RUN_ROOT}/gate_eval_val.json"
FINAL_EVAL_JSON="${RUN_ROOT}/final_eval_test.json"

GATE_MAX_STEPS="${GATE_MAX_STEPS:-10}"
GATE_STEP_SCHEDULE="${GATE_STEP_SCHEDULE:-}"
RESUME_GATE_FROM="${RESUME_GATE_FROM:-}"
KEEP_GATE_DIR="${KEEP_GATE_DIR:-0}"
EPOCHS="${EPOCHS:-5}"
FULL_MAX_STEPS="${FULL_MAX_STEPS:--1}"
TRAIN_BS="${TRAIN_BS:-1}"
EVAL_BS="${EVAL_BS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
SAVE_STEPS="${SAVE_STEPS:-5}"
EVAL_STEPS="${EVAL_STEPS:-5}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-0}"
TEST_MAX_SAMPLES="${TEST_MAX_SAMPLES:-0}"
GATE_ONLY="${GATE_ONLY:-0}"
USE_GATE_AS_FINAL_START="${USE_GATE_AS_FINAL_START:-0}"
FINAL_RESUME_FROM="${FINAL_RESUME_FROM:-}"
GATE_EVAL_STEPS="${GATE_EVAL_STEPS:-target}"
GATE_SAVE_STEPS="${GATE_SAVE_STEPS:-target}"
GATE_AUTO_EXTEND="${GATE_AUTO_EXTEND:-1}"
GATE_AUTO_STEP_DELTA="${GATE_AUTO_STEP_DELTA:-10}"
GATE_AUTO_MAX_STEPS="${GATE_AUTO_MAX_STEPS:-70}"
GATE_RESUME_SCHEDULE_MODE="${GATE_RESUME_SCHEDULE_MODE:-preserve}"
GATE_TRAIN_RETRIES="${GATE_TRAIN_RETRIES:-3}"

GATE_MIN_VALID_JSON="${GATE_MIN_VALID_JSON:-0.80}"
GATE_MIN_SCHEMA_VALID="${GATE_MIN_SCHEMA_VALID:-0.70}"
GATE_MAX_PLACEHOLDER="${GATE_MAX_PLACEHOLDER:-0.10}"
GATE_MIN_IOU_SCHEMA="${GATE_MIN_IOU_SCHEMA:-0.18}"
GATE_MIN_IOU_ALL="${GATE_MIN_IOU_ALL:-0.12}"

if [[ -z "${GATE_STEP_SCHEDULE}" ]]; then
  GATE_STEP_SCHEDULE="${GATE_MAX_STEPS}"
fi

resolve_chair_dataset_root() {
  local candidate="$1"
  if [[ -d "${candidate}" ]]; then
    echo "${candidate}"
    return 0
  fi
  if command -v mdfind >/dev/null 2>&1; then
    while IFS= read -r hit; do
      [[ -n "${hit}" ]] || continue
      [[ "$(basename "${hit}")" == "chair_dataset" ]] || continue
      if [[ -d "${hit}" ]]; then
        echo "${hit}"
        return 0
      fi
    done < <(mdfind 'kMDItemFSName == "chair_dataset"c' 2>/dev/null || true)
  fi
  while IFS= read -r hit; do
    [[ -n "${hit}" ]] || continue
    if [[ -d "${hit}" ]]; then
      echo "${hit}"
      return 0
    fi
  done < <(find "${HOME}" -maxdepth 6 -type d -name chair_dataset 2>/dev/null || true)
  while IFS= read -r hit; do
    [[ -n "${hit}" ]] || continue
    [[ "$(basename "${hit}")" == "chair_dataset" ]] || continue
    if [[ -d "${hit}" ]]; then
      echo "${hit}"
      return 0
    fi
  done < <(mdfind -name chair_dataset 2>/dev/null || true)
  if [[ -d "${HOME}/Desktop/chair_dataset" ]]; then
    echo "${HOME}/Desktop/chair_dataset"
    return 0
  fi
  echo "${candidate}"
}

SRC_A="$(resolve_chair_dataset_root "${SRC_A}")"
if [[ ! -d "${SRC_B}" && -d "${SRC_A}/v2" ]]; then
  SRC_B="${SRC_A}/v2"
fi
if [[ ! -d "${SRC_A}" || ! -d "${SRC_B}" ]]; then
  echo "[error] chair dataset path not found."
  echo "  SRC_A=${SRC_A}"
  echo "  SRC_B=${SRC_B}"
  echo "Set path explicitly, e.g.:"
  echo "  SRC_A=\"/actual/chair_dataset\" SRC_B=\"/actual/chair_dataset/v2\" bash ${ROOT_DIR}/scripts/retrain_vlm_strict_pipeline.sh"
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[error] venv missing: ${VENV_DIR}"
  echo "Run: bash ${ROOT_DIR}/scripts/setup_mps_vlm_env.sh"
  exit 1
fi

PYTHON_BIN="${VENV_DIR}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[error] python not found in venv: ${PYTHON_BIN}"
  exit 1
fi
mkdir -p "${RUN_ROOT}" "${QUALITY_DIR}"

echo "[0/8] Environment check"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/check_mps_vlm_env.py"

echo "[1/8] Rebuild dataset with strict prompt"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/build_vlm_dataset_from_labelme.py" \
  --src-dir "${SRC_A}" \
  --src-dir "${SRC_B}" \
  --out-dir "${DATA_DIR}" \
  --label seat_contact \
  --seed 42 \
  --train-ratio 0.85 \
  --val-ratio 0.10 \
  --prefer-last-source | tee "${RUN_ROOT}/dataset_build.log"

echo "[2/8] Strict validate dataset"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/validate_vlm_jsonl.py" \
  --jsonl "${DATA_DIR}/train.jsonl" \
  --jsonl "${DATA_DIR}/val.jsonl" \
  --jsonl "${DATA_DIR}/test.jsonl" | tee "${RUN_ROOT}/dataset_validate.log"

echo "[3/8] Gate training/eval loop (step schedule: ${GATE_STEP_SCHEDULE})"

IFS=',' read -r -a _gate_steps_raw <<< "${GATE_STEP_SCHEDULE}"
declare -a GATE_STEPS=()
for raw in "${_gate_steps_raw[@]}"; do
  step="$(echo "${raw}" | tr -d '[:space:]')"
  if [[ -z "${step}" ]]; then
    continue
  fi
  if ! [[ "${step}" =~ ^[0-9]+$ ]]; then
    echo "[error] invalid gate step in GATE_STEP_SCHEDULE: ${step}"
    exit 1
  fi
  if [[ "${step}" -le 0 ]]; then
    echo "[error] gate step must be positive: ${step}"
    exit 1
  fi
  GATE_STEPS+=("${step}")
done

if [[ "${#GATE_STEPS[@]}" -eq 0 ]]; then
  echo "[error] no valid gate steps parsed from GATE_STEP_SCHEDULE=${GATE_STEP_SCHEDULE}"
  exit 1
fi
if ! [[ "${GATE_AUTO_STEP_DELTA}" =~ ^[0-9]+$ ]] || [[ "${GATE_AUTO_STEP_DELTA}" -le 0 ]]; then
  echo "[error] GATE_AUTO_STEP_DELTA must be a positive integer: ${GATE_AUTO_STEP_DELTA}"
  exit 1
fi
if ! [[ "${GATE_AUTO_MAX_STEPS}" =~ ^[0-9]+$ ]] || [[ "${GATE_AUTO_MAX_STEPS}" -le 0 ]]; then
  echo "[error] GATE_AUTO_MAX_STEPS must be a positive integer: ${GATE_AUTO_MAX_STEPS}"
  exit 1
fi
if ! [[ "${GATE_TRAIN_RETRIES}" =~ ^[0-9]+$ ]] || [[ "${GATE_TRAIN_RETRIES}" -lt 0 ]]; then
  echo "[error] GATE_TRAIN_RETRIES must be a non-negative integer: ${GATE_TRAIN_RETRIES}"
  exit 1
fi

if [[ "${KEEP_GATE_DIR}" != "1" && -z "${RESUME_GATE_FROM}" ]]; then
  rm -rf "${GATE_DIR}"
fi
mkdir -p "${GATE_DIR}"
: > "${RUN_ROOT}/gate_train.log"
: > "${RUN_ROOT}/gate_eval.log"

gate_resume="${RESUME_GATE_FROM}"
gate_passed=0
gate_pass_step=0
GATE_ADAPTER=""

declare -a GATE_STEPS_TO_RUN=("${GATE_STEPS[@]}")
gate_idx=0
while true; do
  if [[ "${gate_idx}" -ge "${#GATE_STEPS_TO_RUN[@]}" ]]; then
    if [[ "${GATE_AUTO_EXTEND}" == "1" && "${gate_passed}" -ne 1 ]]; then
      last_idx=$(( ${#GATE_STEPS_TO_RUN[@]} - 1 ))
      last_step="${GATE_STEPS_TO_RUN[$last_idx]}"
      next_step=$(( last_step + GATE_AUTO_STEP_DELTA ))
      if [[ "${next_step}" -le "${GATE_AUTO_MAX_STEPS}" ]]; then
        GATE_STEPS_TO_RUN+=("${next_step}")
        echo "[gate-auto] appended step=${next_step} (delta=${GATE_AUTO_STEP_DELTA}, max=${GATE_AUTO_MAX_STEPS})"
      else
        break
      fi
    else
      break
    fi
  fi

  target_step="${GATE_STEPS_TO_RUN[$gate_idx]}"
  gate_idx=$(( gate_idx + 1 ))
  echo "[gate] training to max_steps=${target_step} (resume=${gate_resume:-none})"
  gate_eval_steps="${target_step}"
  gate_save_steps="${target_step}"
  if [[ -n "${gate_resume}" && "${GATE_RESUME_SCHEDULE_MODE}" == "preserve" ]]; then
    resume_state_json="${gate_resume}/trainer_state.json"
    if [[ -f "${resume_state_json}" ]]; then
      resume_schedule="$("${PYTHON_BIN}" - "${resume_state_json}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
obj = json.loads(p.read_text(encoding="utf-8"))
print(f"{int(obj.get('eval_steps', 0))},{int(obj.get('save_steps', 0))}")
PY
)"
      resume_eval_steps="${resume_schedule%%,*}"
      resume_save_steps="${resume_schedule##*,}"
      if [[ "${resume_eval_steps}" =~ ^[0-9]+$ ]] && [[ "${resume_eval_steps}" -gt 0 ]]; then
        gate_eval_steps="${resume_eval_steps}"
      fi
      if [[ "${resume_save_steps}" =~ ^[0-9]+$ ]] && [[ "${resume_save_steps}" -gt 0 ]]; then
        gate_save_steps="${resume_save_steps}"
      fi
    fi
  fi
  if [[ "${GATE_EVAL_STEPS}" != "target" ]]; then
    gate_eval_steps="${GATE_EVAL_STEPS}"
  fi
  if [[ "${GATE_SAVE_STEPS}" != "target" ]]; then
    gate_save_steps="${GATE_SAVE_STEPS}"
  fi
  if ! [[ "${gate_eval_steps}" =~ ^[0-9]+$ ]] || [[ "${gate_eval_steps}" -le 0 ]]; then
    echo "[error] invalid gate eval steps: ${gate_eval_steps}"
    exit 1
  fi
  if ! [[ "${gate_save_steps}" =~ ^[0-9]+$ ]] || [[ "${gate_save_steps}" -le 0 ]]; then
    echo "[error] invalid gate save steps: ${gate_save_steps}"
    exit 1
  fi

  GATE_TRAIN_CMD=(
    "${PYTHON_BIN}" "${ROOT_DIR}/scripts/train_vlm_lora_mps.py"
    --model-name "${BASE_MODEL}"
    --train-jsonl "${DATA_DIR}/train.jsonl"
    --val-jsonl "${DATA_DIR}/val.jsonl"
    --output-dir "${GATE_DIR}"
    --epochs "${EPOCHS}"
    --train-batch-size "${TRAIN_BS}"
    --eval-batch-size "${EVAL_BS}"
    --grad-accum "${GRAD_ACCUM}"
    --logging-steps "${LOGGING_STEPS}"
    --eval-steps "${gate_eval_steps}"
    --save-steps "${gate_save_steps}"
    --save-total-limit 8
    --max-steps "${target_step}"
    --gradient-checkpointing
    --no-dataloader-pin-memory
  )
  if [[ -n "${gate_resume}" ]]; then
    GATE_TRAIN_CMD+=(--resume-from-checkpoint "${gate_resume}")
  fi
  train_ok=0
  train_attempt=0
  while [[ "${train_attempt}" -le "${GATE_TRAIN_RETRIES}" ]]; do
    train_attempt=$(( train_attempt + 1 ))
    echo "[gate] launch attempt ${train_attempt}/$(( GATE_TRAIN_RETRIES + 1 )) for max_steps=${target_step}"
    set +e
    "${GATE_TRAIN_CMD[@]}" | tee -a "${RUN_ROOT}/gate_train.log"
    train_rc=$?
    set -e
    if [[ "${train_rc}" -eq 0 ]]; then
      train_ok=1
      break
    fi
    latest_ckpt_after_fail="$(ls -1d "${GATE_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)"
    echo "[gate] train command failed (rc=${train_rc}). latest_ckpt=${latest_ckpt_after_fail:-none}"
    if [[ -n "${latest_ckpt_after_fail}" ]]; then
      gate_resume="${latest_ckpt_after_fail}"
      GATE_TRAIN_CMD=(
        "${PYTHON_BIN}" "${ROOT_DIR}/scripts/train_vlm_lora_mps.py"
        --model-name "${BASE_MODEL}"
        --train-jsonl "${DATA_DIR}/train.jsonl"
        --val-jsonl "${DATA_DIR}/val.jsonl"
        --output-dir "${GATE_DIR}"
        --epochs "${EPOCHS}"
        --train-batch-size "${TRAIN_BS}"
        --eval-batch-size "${EVAL_BS}"
        --grad-accum "${GRAD_ACCUM}"
        --logging-steps "${LOGGING_STEPS}"
        --eval-steps "${gate_eval_steps}"
        --save-steps "${gate_save_steps}"
        --save-total-limit 8
        --max-steps "${target_step}"
        --gradient-checkpointing
        --no-dataloader-pin-memory
        --resume-from-checkpoint "${gate_resume}"
      )
    fi
    sleep 3
  done

  if [[ "${train_ok}" -ne 1 ]]; then
    echo "[error] gate training failed after retries for target_step=${target_step}"
    exit 1
  fi

  current_gate_adapter="${GATE_DIR}/checkpoint-${target_step}"
  if [[ ! -d "${current_gate_adapter}" ]]; then
    current_gate_adapter="$(ls -1d "${GATE_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)"
  fi
  if [[ -z "${current_gate_adapter}" || ! -d "${current_gate_adapter}" ]]; then
    echo "[error] gate checkpoint not found after max_steps=${target_step}"
    exit 1
  fi

  gate_eval_step_json="${RUN_ROOT}/gate_eval_val_step${target_step}.json"
  GATE_EVAL_CMD=(
    "${PYTHON_BIN}" "${ROOT_DIR}/scripts/eval_vlm_polygon_quality.py"
    --base-model "${BASE_MODEL}"
    --adapter-dir "${current_gate_adapter}"
    --jsonl "${DATA_DIR}/val.jsonl"
    --output-json "${gate_eval_step_json}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --device auto
  )
  if [[ "${VAL_MAX_SAMPLES}" -gt 0 ]]; then
    GATE_EVAL_CMD+=(--max-samples "${VAL_MAX_SAMPLES}")
  fi
  echo "[gate] evaluating ${current_gate_adapter}"
  "${GATE_EVAL_CMD[@]}" | tee -a "${RUN_ROOT}/gate_eval.log"
  cp -f "${gate_eval_step_json}" "${GATE_EVAL_JSON}"

  if "${PYTHON_BIN}" - "${gate_eval_step_json}" \
    "${GATE_MIN_VALID_JSON}" \
    "${GATE_MIN_SCHEMA_VALID}" \
    "${GATE_MAX_PLACEHOLDER}" \
    "${GATE_MIN_IOU_SCHEMA}" \
    "${GATE_MIN_IOU_ALL}" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
obj = json.loads(p.read_text(encoding="utf-8"))
s = obj["summary"]

valid_json_rate = float(s["valid_json_rate"])
schema_valid_rate = float(s["schema_valid_rate"])
placeholder_rate = float(s["placeholder_rate"])
mean_iou_schema = float(s["mean_iou_schema_valid_only"])
mean_iou_all = float(s["mean_iou_all"])

min_valid_json = float(sys.argv[2])
min_schema_valid = float(sys.argv[3])
max_placeholder = float(sys.argv[4])
min_iou_schema = float(sys.argv[5])
min_iou_all = float(sys.argv[6])

print("[gate-summary]", json.dumps(s, ensure_ascii=False))

if valid_json_rate < min_valid_json:
    raise SystemExit(f"[gate-fail] valid_json_rate {valid_json_rate:.4f} < {min_valid_json:.4f}")
if schema_valid_rate < min_schema_valid:
    raise SystemExit(f"[gate-fail] schema_valid_rate {schema_valid_rate:.4f} < {min_schema_valid:.4f}")
if placeholder_rate > max_placeholder:
    raise SystemExit(f"[gate-fail] placeholder_rate {placeholder_rate:.4f} > {max_placeholder:.4f}")
if mean_iou_schema < min_iou_schema:
    raise SystemExit(f"[gate-fail] mean_iou_schema_valid_only {mean_iou_schema:.4f} < {min_iou_schema:.4f}")
if mean_iou_all < min_iou_all:
    raise SystemExit(f"[gate-fail] mean_iou_all {mean_iou_all:.4f} < {min_iou_all:.4f}")

print("[gate-pass] quality gate passed")
PY
  then
    gate_passed=1
    gate_pass_step="${target_step}"
    GATE_ADAPTER="${current_gate_adapter}"
    break
  else
    gate_resume="${current_gate_adapter}"
    echo "[gate] step=${target_step} failed; will continue to next target step."
  fi
done

if [[ "${gate_passed}" -ne 1 ]]; then
  echo "[error] gate failed for all attempted steps: ${GATE_STEPS_TO_RUN[*]}"
  echo "Try relaxing thresholds or increasing GATE_AUTO_MAX_STEPS (current: ${GATE_AUTO_MAX_STEPS})."
  exit 1
fi

echo "[gate] passed at step=${gate_pass_step}, adapter=${GATE_ADAPTER}"

if [[ "${GATE_ONLY}" == "1" ]]; then
  echo
  echo "[done] gate-only run complete"
  echo "  run root:      ${RUN_ROOT}"
  echo "  gate eval:     ${GATE_EVAL_JSON}"
  echo "  gate adapter:  ${GATE_ADAPTER}"
  echo "Set GATE_ONLY=0 (default) to continue with full training."
  exit 0
fi

if [[ -z "${FINAL_RESUME_FROM}" && "${USE_GATE_AS_FINAL_START}" == "1" ]]; then
  FINAL_RESUME_FROM="${GATE_ADAPTER}"
fi

echo "[6/8] Final training (full run)"
rm -rf "${FINAL_DIR}" "${QUALITY_DIR}"
mkdir -p "${QUALITY_DIR}"
FINAL_TRAIN_CMD=(
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/train_vlm_lora_mps.py"
  --model-name "${BASE_MODEL}"
  --train-jsonl "${DATA_DIR}/train.jsonl"
  --val-jsonl "${DATA_DIR}/val.jsonl"
  --output-dir "${FINAL_DIR}"
  --epochs "${EPOCHS}"
  --train-batch-size "${TRAIN_BS}"
  --eval-batch-size "${EVAL_BS}"
  --grad-accum "${GRAD_ACCUM}"
  --logging-steps "${LOGGING_STEPS}"
  --eval-steps "${EVAL_STEPS}"
  --save-steps "${SAVE_STEPS}"
  --save-total-limit "${SAVE_TOTAL_LIMIT}"
  --max-steps "${FULL_MAX_STEPS}"
  --gradient-checkpointing
  --no-dataloader-pin-memory
)

if [[ -n "${FINAL_RESUME_FROM}" ]]; then
  if [[ ! -d "${FINAL_RESUME_FROM}" ]]; then
    echo "[error] FINAL_RESUME_FROM not found: ${FINAL_RESUME_FROM}"
    exit 1
  fi
  echo "[final] resuming from checkpoint: ${FINAL_RESUME_FROM}"
  FINAL_TRAIN_CMD+=(--resume-from-checkpoint "${FINAL_RESUME_FROM}")
fi

"${FINAL_TRAIN_CMD[@]}" | tee "${RUN_ROOT}/final_train.log"

echo "[7/8] Evaluate all candidate adapters on val and select best"
declare -a CANDIDATES=()
if [[ -f "${FINAL_DIR}/adapter_config.json" ]]; then
  CANDIDATES+=("${FINAL_DIR}")
fi
while IFS= read -r ckpt; do
  CANDIDATES+=("${ckpt}")
done < <(ls -1d "${FINAL_DIR}"/checkpoint-* 2>/dev/null | sort -V || true)

if [[ "${#CANDIDATES[@]}" -eq 0 ]]; then
  echo "[error] no adapter candidates found in ${FINAL_DIR}"
  exit 1
fi

for adapter in "${CANDIDATES[@]}"; do
  name="$(basename "${adapter}")"
  if [[ "${adapter}" == "${FINAL_DIR}" ]]; then
    name="final_best_eval_loss"
  fi
  out_json="${QUALITY_DIR}/val_${name}.json"
  cmd=(
    "${PYTHON_BIN}" "${ROOT_DIR}/scripts/eval_vlm_polygon_quality.py"
    --base-model "${BASE_MODEL}"
    --adapter-dir "${adapter}"
    --jsonl "${DATA_DIR}/val.jsonl"
    --output-json "${out_json}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --device auto
  )
  if [[ "${VAL_MAX_SAMPLES}" -gt 0 ]]; then
    cmd+=(--max-samples "${VAL_MAX_SAMPLES}")
  fi
  "${cmd[@]}" | tee -a "${RUN_ROOT}/val_quality_eval.log"
done

"${PYTHON_BIN}" - "${QUALITY_DIR}" "${BEST_SELECTION_JSON}" "${BEST_ADAPTER_TXT}" <<'PY'
import json
import sys
from pathlib import Path

quality_dir = Path(sys.argv[1])
best_json = Path(sys.argv[2])
best_txt = Path(sys.argv[3])

records = []
for p in sorted(quality_dir.glob("val_*.json")):
    obj = json.loads(p.read_text(encoding="utf-8"))
    s = obj["summary"]
    score = (
        0.65 * float(s["mean_iou_schema_valid_only"])
        + 0.20 * float(s["mean_iou_all"])
        + 0.10 * float(s["schema_valid_rate"])
        + 0.05 * float(s["valid_json_rate"])
        - 0.20 * float(s["placeholder_rate"])
    )
    records.append(
        {
            "file": str(p),
            "adapter_dir": str(s["adapter_dir"]),
            "summary": s,
            "score": score,
        }
    )

if not records:
    raise SystemExit("[error] no val quality json files found")

records.sort(key=lambda x: x["score"], reverse=True)
best = records[0]
best_json.write_text(json.dumps({"best": best, "all": records}, ensure_ascii=False, indent=2), encoding="utf-8")
best_txt.write_text(best["adapter_dir"] + "\n", encoding="utf-8")

print("[selection-best]")
print(json.dumps(best, ensure_ascii=False, indent=2))
print(f"[saved] {best_json}")
print(f"[saved] {best_txt}")
PY

BEST_ADAPTER="$(cat "${BEST_ADAPTER_TXT}" | tr -d '\r\n')"
if [[ -z "${BEST_ADAPTER}" ]]; then
  echo "[error] empty best adapter path"
  exit 1
fi
echo "[selection] best adapter: ${BEST_ADAPTER}"

echo "[8/8] Final eval on test with selected adapter"
FINAL_EVAL_CMD=(
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/eval_vlm_polygon_quality.py"
  --base-model "${BASE_MODEL}"
  --adapter-dir "${BEST_ADAPTER}"
  --jsonl "${DATA_DIR}/test.jsonl"
  --output-json "${FINAL_EVAL_JSON}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --device auto
)
if [[ "${TEST_MAX_SAMPLES}" -gt 0 ]]; then
  FINAL_EVAL_CMD+=(--max-samples "${TEST_MAX_SAMPLES}")
fi
"${FINAL_EVAL_CMD[@]}" | tee "${RUN_ROOT}/final_eval.log"

echo
echo "[done] strict retrain pipeline complete"
echo "  run root:        ${RUN_ROOT}"
echo "  gate eval:       ${GATE_EVAL_JSON}"
echo "  val selection:   ${BEST_SELECTION_JSON}"
echo "  best adapter:    ${BEST_ADAPTER}"
echo "  final test eval: ${FINAL_EVAL_JSON}"
