# VLM Fine-Tuning Prep (MPS)

This guide prepares everything up to the point just before full training.

## 1) Activate environment

```bash
cd /Users/ganghyeongyu/Documents/chairpose
source .venv-mps-vlm/bin/activate
python scripts/check_mps_vlm_env.py
```

## 2) Build/validate VLM training dataset

```bash
bash scripts/prepare_vlm_training_assets.sh
```

Output directory:

- `/Users/ganghyeongyu/Documents/chairpose/data/vlm_seatcontact/train.jsonl`
- `/Users/ganghyeongyu/Documents/chairpose/data/vlm_seatcontact/val.jsonl`
- `/Users/ganghyeongyu/Documents/chairpose/data/vlm_seatcontact/test.jsonl`
- `/Users/ganghyeongyu/Documents/chairpose/data/vlm_seatcontact/build_report.json`

## 3) Pre-training dry run (no optimization)

```bash
python scripts/train_vlm_lora_mps.py \
  --model-name HuggingFaceTB/SmolVLM-256M-Instruct \
  --dry-run \
  --max-train-samples 8 \
  --max-eval-samples 4
```

If this succeeds, the pipeline is ready for full training.

## 4) Full training command (when you start)

```bash
bash scripts/run_vlm_train_mps.sh
```

## Notes for M1 16GB

- Keep `--train-batch-size 1`
- Increase `--grad-accum` instead of batch size
- Keep `--gradient-checkpointing` enabled
- If OOM occurs: reduce `--max-train-samples` for tests first and lower sequence/image load
