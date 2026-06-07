# VLM Fine-Tuning on Apple Silicon

이 문서는 LabelMe `seat_contact` annotation으로 SmolVLM LoRA 학습을 준비하고
실행하는 최소 절차를 설명합니다.

## 1. 환경 구성

```bash
cd /path/to/chair_pose
bash scripts/setup_mps_vlm_env.sh
source .venv-mps-vlm/bin/activate
python scripts/check_mps_vlm_env.py
```

## 2. 데이터 준비

LabelMe 이미지와 JSON 파일을 같은 디렉터리에 배치하고 동일한 stem을 사용합니다.

```bash
export SRC_A=/path/to/chair_dataset
bash scripts/prepare_vlm_training_assets.sh
```

현재 권장 실험 설정을 직접 생성하려면:

```bash
python scripts/build_vlm_dataset_from_labelme.py \
  --src-dir "$SRC_A" \
  --out-dir data/vlm_seatcontact_v18 \
  --label seat_contact \
  --target-shape bbox \
  --coord-mode grid \
  --grid-size 16 \
  --crop-mode chair \
  --chair-detector yolov8x-seg.pt \
  --chair-device mps \
  --target-width 224 \
  --target-height 224 \
  --write-resized-images \
  --train-ratio 0.90 \
  --val-ratio 0.10
```

검증:

```bash
python scripts/validate_vlm_jsonl.py \
  --jsonl data/vlm_seatcontact_v18/train.jsonl \
  --jsonl data/vlm_seatcontact_v18/val.jsonl \
  --target-shape bbox \
  --coord-mode grid
```

## 3. Dry Run

최적화를 시작하지 않고 모델과 데이터 로딩만 확인합니다.

```bash
python scripts/train_vlm_lora_mps.py \
  --model-name HuggingFaceTB/SmolVLM-500M-Instruct \
  --train-jsonl data/vlm_seatcontact_v18/train.jsonl \
  --val-jsonl data/vlm_seatcontact_v18/val.jsonl \
  --output-dir runs/vlm_dry_run \
  --dry-run \
  --max-train-samples 8 \
  --max-eval-samples 4
```

## 4. 학습

```bash
python scripts/train_vlm_lora_mps.py \
  --model-name HuggingFaceTB/SmolVLM-500M-Instruct \
  --train-jsonl data/vlm_seatcontact_v18/train.jsonl \
  --val-jsonl data/vlm_seatcontact_v18/val.jsonl \
  --output-dir runs/vlm_v18/gate \
  --epochs 3 \
  --lr 2e-5 \
  --train-batch-size 1 \
  --eval-batch-size 1 \
  --grad-accum 8 \
  --eval-steps 5 \
  --save-steps 5 \
  --max-steps 45 \
  --gradient-checkpointing \
  --no-dataloader-pin-memory
```

## MPS 메모리 팁

- batch size는 1로 유지합니다.
- effective batch는 gradient accumulation으로 조절합니다.
- gradient checkpointing을 활성화합니다.
- 메모리가 부족하면 이미지 해상도, LoRA rank, sequence 길이를 줄입니다.
- 여러 MPS 평가 프로세스를 동시에 실행하지 않습니다.
