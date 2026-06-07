# VLM 작업 환경 이전 런북

새 Mac 또는 새 작업 디렉터리에서 ChairPose VLM 학습을 재개하기 위한 체크리스트입니다.

## 이전할 항목

- Git 저장소
- LabelMe 원본 이미지와 JSON
- 필요한 LoRA checkpoint
- 로컬에서 사용하는 외부 model weight

`runs/`, `data/`, `.venv-mps-vlm/`, `*.pt`, `*.safetensors`는 Git에서 제외되므로
필요한 파일은 별도 저장소나 외장 디스크로 이전해야 합니다.

## 1. 저장소 준비

```bash
git clone https://github.com/khk0606/chair_pose.git
cd chair_pose
```

기존 복사본을 사용한다면 다음을 확인합니다.

```bash
git status
git remote -v
```

## 2. Python 환경 재생성

가상환경 폴더를 복사하지 말고 새 장치에서 다시 만듭니다.

```bash
bash scripts/setup_mps_vlm_env.sh
source .venv-mps-vlm/bin/activate
python scripts/check_mps_vlm_env.py
```

`READY: True`와 `mps available=True`를 확인합니다.

## 3. 데이터 위치 지정

데이터셋은 저장소 밖에 둘 수 있습니다.

```bash
export SRC_A=/path/to/chair_dataset
```

필수 구조:

```text
chair_dataset/
├── sample_001.jpg
├── sample_001.json
├── sample_002.png
└── sample_002.json
```

각 JSON에는 `seat_contact` polygon이 있어야 합니다.

## 4. 학습 데이터 재생성

```bash
python scripts/build_vlm_dataset_from_labelme.py \
  --src-dir "$SRC_A" \
  --out-dir data/vlm_seatcontact \
  --label seat_contact \
  --target-shape bbox \
  --coord-mode grid \
  --grid-size 16 \
  --crop-mode chair \
  --target-width 224 \
  --target-height 224 \
  --write-resized-images
```

```bash
python scripts/validate_vlm_jsonl.py \
  --jsonl data/vlm_seatcontact/train.jsonl \
  --jsonl data/vlm_seatcontact/val.jsonl \
  --target-shape bbox \
  --coord-mode grid
```

## 5. 체크포인트 복원

백업한 checkpoint를 `runs/<run-name>/gate/checkpoint-<step>` 아래에 배치합니다.

확인할 파일:

- `adapter_config.json`
- `adapter_model.safetensors`
- `trainer_state.json`
- `training_args.bin`

이어 학습하려면:

```bash
python scripts/train_vlm_lora_mps.py \
  --model-name HuggingFaceTB/SmolVLM-500M-Instruct \
  --train-jsonl data/vlm_seatcontact/train.jsonl \
  --val-jsonl data/vlm_seatcontact/val.jsonl \
  --output-dir runs/vlm_resume/gate \
  --resume-from-checkpoint runs/previous/gate/checkpoint-45
```

## 6. 이전 후 확인

1. 모든 JSONL 이미지 경로가 현재 장치에서 존재하는지 확인합니다.
2. base model 이름과 adapter의 `base_model_name_or_path`가 일치해야 합니다.
3. 학습과 추론의 crop, resize, target shape, coordinate mode를 동일하게 유지합니다.
4. 먼저 dry-run과 단일 이미지 추론을 실행합니다.
5. validation IoU뿐 아니라 고정 hardcase 결과를 함께 확인합니다.
