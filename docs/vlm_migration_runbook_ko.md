# VLM 파인튜닝 이전/복구 런북 (맥북 간 이동용)

이 문서는 `chairpose` 프로젝트를 다른 노트북으로 옮겨서, 현재 진행 상태(게이트 체크포인트 포함)부터 다시 학습을 이어가기 위한 상세 가이드다.

## 1) 현재 원본 노트북의 실제 경로

- `chairpose` 프로젝트 루트  
  `/Users/ganghyeongyu/Documents/문서 - 강현규의 MacBook Pro/chairpose`
- 학습 데이터 루트  
  `/Users/ganghyeongyu/Desktop/데스크탑 - 강현규의 MacBook Pro/chair_dataset`
- 학습 데이터 v2  
  `/Users/ganghyeongyu/Desktop/데스크탑 - 강현규의 MacBook Pro/chair_dataset/v2`

주의:
- `MacBook Pro` 사이 공백은 일반 공백이 아니라 NBSP일 수 있다.
- 경로는 직접 타이핑보다 `복사/붙여넣기` 권장.

## 2) 다른 노트북으로 복사할 폴더

최소 권장(현재 게이트 재개 목적):
- `chairpose` 전체
- `chair_dataset` 전체

즉, 아래 2개를 통째로 복사:
- `/Users/ganghyeongyu/Documents/문서 - 강현규의 MacBook Pro/chairpose`
- `/Users/ganghyeongyu/Desktop/데스크탑 - 강현규의 MacBook Pro/chair_dataset`

왜 전체 복사?
- 코드 + 스크립트 + 현재 체크포인트 + 로그 + 데이터셋 JSONL 경로 이슈를 한 번에 보존 가능
- 일부만 복사하면 절대경로, 체크포인트 불일치로 재작업이 늘어남

## 3) 새 노트북에서 권장 폴더 구조

가능하면 아래처럼 맞추는 것이 가장 안전:
- `~/Documents/chairpose`
- `~/Desktop/chair_dataset`

정확히 동일 경로가 아니어도 되지만, 실행 시 `SRC_A`, `SRC_B`를 반드시 지정해야 한다.

## 4) 새 노트북 최초 1회 세팅

프로젝트 루트로 이동:

```bash
cd /Users/<YOUR_USER>/Documents/chairpose
```

가상환경/패키지 설치:

```bash
bash scripts/setup_mps_vlm_env.sh
```

환경 확인:

```bash
/Users/<YOUR_USER>/Documents/chairpose/.venv-mps-vlm/bin/python scripts/check_mps_vlm_env.py
```

정상 기준:
- `torch/transformers/accelerate/datasets/trl/peft` 모두 OK
- `mps built=True available=True tensor_test=True`
- `READY: True`

## 5) 현재 상태에서 게이트 학습 재개 (권장 커맨드)

아래 커맨드는:
- 이전 `checkpoint-*` 중 최신에서 이어서 시작
- 게이트 실패 시 자동 스텝 확장 (`20 -> 30 -> ... -> 70`)
- 프로세스 킬/실패 시 자동 재시도

```bash
ROOT="/Users/<YOUR_USER>/Documents/chairpose"
SRC_A="/Users/<YOUR_USER>/Desktop/chair_dataset"
SRC_B="$SRC_A/v2"
RUN_TAG="vlm_lora_mps_strict_v4"
RESUME_CKPT="$(ls -1d "$ROOT/runs/$RUN_TAG/gate"/checkpoint-* 2>/dev/null | sort -V | tail -n 1)"

RUN_TAG="$RUN_TAG" \
SRC_A="$SRC_A" SRC_B="$SRC_B" \
GATE_ONLY=1 \
KEEP_GATE_DIR=1 \
RESUME_GATE_FROM="$RESUME_CKPT" \
GATE_STEP_SCHEDULE=20 \
GATE_AUTO_EXTEND=1 \
GATE_AUTO_STEP_DELTA=10 \
GATE_AUTO_MAX_STEPS=70 \
GATE_TRAIN_RETRIES=5 \
GATE_RESUME_SCHEDULE_MODE=preserve \
bash "$ROOT/scripts/retrain_vlm_strict_pipeline.sh" \
2>&1 | tee "$ROOT/runs/$RUN_TAG/launcher_gate_resume.log"
```

## 6) 게이트 통과 후 본학습 시작

게이트 통과 로그가 나오면(예: `[gate-pass] quality gate passed`) 다음 실행:

```bash
ROOT="/Users/<YOUR_USER>/Documents/chairpose"
SRC_A="/Users/<YOUR_USER>/Desktop/chair_dataset"
SRC_B="$SRC_A/v2"
RUN_TAG="vlm_lora_mps_strict_v4"
BEST_GATE="$(ls -1d "$ROOT/runs/$RUN_TAG/gate"/checkpoint-* | sort -V | tail -n 1)"
STEP="${BEST_GATE##*-}"

RUN_TAG="$RUN_TAG" \
SRC_A="$SRC_A" SRC_B="$SRC_B" \
KEEP_GATE_DIR=1 \
RESUME_GATE_FROM="$BEST_GATE" \
GATE_STEP_SCHEDULE="$STEP" \
USE_GATE_AS_FINAL_START=1 \
FINAL_RESUME_FROM="$BEST_GATE" \
bash "$ROOT/scripts/retrain_vlm_strict_pipeline.sh" \
2>&1 | tee "$ROOT/runs/$RUN_TAG/launcher_full.log"
```

## 7) 출력 파일/결과 확인 포인트

런 루트:
- `runs/vlm_lora_mps_strict_v4/`

중요 파일:
- `launcher_gate_resume.log`: 게이트 재개 전체 로그
- `gate_eval_val_step*.json`: 게이트 평가 결과
- `gate/checkpoint-*`: 게이트 체크포인트
- `final_train.log`: 본학습 로그
- `best_adapter_selection.json`: val 기준 베스트 어댑터 선택 결과
- `final_eval_test.json`: 최종 테스트 평가

## 8) 자주 발생한 이슈와 해결

### A. 경로가 있는데 못 찾는 문제
- 원인: `문서 - ...`, `데스크탑 - ...`처럼 로컬라이즈된 폴더명 + 특수 공백(NBSP)
- 해결: 경로를 Finder에서 복사해 붙여넣기, 또는 `SRC_A/SRC_B`를 명시 지정

### B. `FileNotFoundError: src-dir not found`
- 원인: 새 노트북에서 `chair_dataset` 실제 경로 불일치
- 해결: `SRC_A="/실제/chair_dataset"` `SRC_B="$SRC_A/v2"`로 강제 지정

### C. `Killed: 9`
- 원인: 보통 메모리/시스템 압박으로 프로세스 강제 종료
- 완화:
  - 다른 앱 종료
  - 디스크 여유 최소 15~20GB 확보
  - 자동 재시도(`GATE_TRAIN_RETRIES`) 활용

### D. `eval_steps/save_steps mismatch` 경고
- 원인: 재개 체크포인트의 trainer_state와 현재 인자 불일치
- 조치: 치명적 오류 아님. 현재 스크립트는 `GATE_RESUME_SCHEDULE_MODE=preserve`로 완화됨

## 9) 다른 노트북 Codex에 붙여넣을 프롬프트 템플릿

아래를 새 노트북 Codex에 그대로 붙여넣고 `<...>`만 바꿔 사용:

```text
프로젝트를 이어서 진행해야 해.

프로젝트 루트:
/Users/<YOUR_USER>/Documents/chairpose

데이터셋 루트:
/Users/<YOUR_USER>/Desktop/chair_dataset
/Users/<YOUR_USER>/Desktop/chair_dataset/v2

목표:
1) 환경 점검(venv, mps, 필수 패키지)
2) runs/vlm_lora_mps_strict_v4의 최신 gate checkpoint 확인
3) 게이트 학습을 최신 checkpoint에서 재개
4) 게이트 통과 여부 보고
5) 통과하면 본학습 명령까지 준비

조건:
- 경로 자동탐색보다 위 절대경로를 우선 사용
- 실패 시 원인과 재시도 명령을 바로 제시
- 로그 파일 경로를 항상 명시
- 기존 checkpoint는 절대 삭제하지 말 것
```

## 10) 빠른 체크리스트

- [ ] `chairpose` 전체 복사 완료
- [ ] `chair_dataset` 전체 복사 완료
- [ ] `setup_mps_vlm_env.sh` 실행 완료
- [ ] `check_mps_vlm_env.py` READY True 확인
- [ ] `runs/vlm_lora_mps_strict_v4/gate/checkpoint-*` 존재 확인
- [ ] 게이트 재개 로그 생성 확인 (`launcher_gate_resume.log`)
- [ ] 게이트 통과 후 본학습 실행

