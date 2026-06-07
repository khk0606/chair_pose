# MPS VLM Fine-Tuning Setup (Mac)

This project includes a reusable setup flow for VLM LoRA fine-tuning on Apple Silicon (MPS).

## 1) Run setup

```bash
cd /path/to/chair_pose
bash scripts/setup_mps_vlm_env.sh
```

## 2) Activate environment

```bash
source .venv-mps-vlm/bin/activate
```

## 3) Validate

```bash
python scripts/check_mps_vlm_env.py
```

Expected:
- `torch`, `transformers`, `accelerate`, `datasets`, `trl`, `peft` all `OK`
- `mps available=True`
- `READY: True`

## Recommended runtime flags for 16GB unified memory

- `batch_size=1`
- `gradient_accumulation_steps` increase to keep effective batch size
- short `max_seq_length`
- enable `gradient_checkpointing`

If OOM appears:
- lower image resolution
- lower sequence length
- reduce LoRA rank
- keep only trainable LoRA layers (freeze base model)
