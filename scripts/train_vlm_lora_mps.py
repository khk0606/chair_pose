#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoProcessor
from trl import SFTConfig, SFTTrainer

try:
    from transformers import AutoModelForImageTextToText as AutoVLMModel
except Exception:  # pragma: no cover
    from transformers import AutoModelForVision2Seq as AutoVLMModel  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train VLM LoRA on Apple Silicon MPS using chat-format JSONL dataset."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="HuggingFaceTB/SmolVLM-256M-Instruct",
        help="HF model id for image-text generation.",
    )
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        default=Path(
            "/Users/ganghyeongyu/Documents/chairpose/data/vlm_seatcontact/train.jsonl"
        ),
    )
    parser.add_argument(
        "--val-jsonl",
        type=Path,
        default=Path(
            "/Users/ganghyeongyu/Documents/chairpose/data/vlm_seatcontact/val.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/ganghyeongyu/Documents/chairpose/runs/vlm_lora_mps"),
    )
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=40)
    parser.add_argument("--save-steps", type=int, default=40)
    parser.add_argument("--save-total-limit", type=int, default=20)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Optional cap on total update steps. -1 means disabled.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="For smoke test. 0 means use all samples.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=0,
        help="For smoke test. 0 means use all samples.",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="all-linear",
        help='Either "all-linear" or comma-separated module names.',
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16", "auto"],
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dataloader-pin-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--load-best-model-at-end",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--assistant-only-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train/eval loss only on assistant response tokens for chat data.",
    )
    parser.add_argument(
        "--completion-only-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train/eval loss only on completion tokens (for prompt-completion VLM format).",
    )
    parser.add_argument(
        "--convert-messages-to-prompt-completion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Convert chat-format `messages` rows into prompt/completion rows for completion-only loss.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default="",
        help="Path to checkpoint dir. Empty means disabled.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load model/dataset and print a sample, but do not train.",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


def resolve_lora_targets(spec: str) -> str | list[str]:
    spec = spec.strip()
    if not spec:
        return "all-linear"
    if spec == "all-linear":
        return spec
    return [s.strip() for s in spec.split(",") if s.strip()]


def dataset_summary(ds: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"rows": len(ds)}
    if len(ds) > 0:
        one = ds[0]
        out["keys"] = list(one.keys())
        out["first_id"] = one.get("id", "")
        out["first_image"] = (one.get("images") or [""])[0]
    return out


def split_prompt_completion_from_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not messages:
        return [], []

    assistant_idx = None
    for i, m in enumerate(messages):
        if isinstance(m, dict) and m.get("role") == "assistant":
            assistant_idx = i
            break

    if assistant_idx is None:
        assistant_idx = max(1, len(messages) - 1)
    elif assistant_idx <= 0:
        assistant_idx = 1

    prompt = messages[:assistant_idx]
    completion = messages[assistant_idx:]
    if not completion:
        completion = [messages[-1]]
        prompt = messages[:-1]
    if not prompt and len(messages) >= 1:
        prompt = [messages[0]]
    return prompt, completion


def convert_dataset_to_prompt_completion(ds: Any) -> Any:
    if "messages" not in ds.column_names:
        return ds

    def _map_row(example: dict[str, Any]) -> dict[str, Any]:
        messages = example.get("messages")
        if not isinstance(messages, list):
            return {"prompt": [], "completion": []}
        prompt, completion = split_prompt_completion_from_messages(messages)
        return {"prompt": prompt, "completion": completion}

    ds = ds.map(_map_row, desc="convert-messages-to-prompt-completion")
    if "messages" in ds.column_names:
        ds = ds.remove_columns(["messages"])
    return ds


def main() -> int:
    args = parse_args()
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    if not args.train_jsonl.exists():
        raise FileNotFoundError(f"train jsonl not found: {args.train_jsonl}")
    if not args.val_jsonl.exists():
        raise FileNotFoundError(f"val jsonl not found: {args.val_jsonl}")

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available on this machine.")

    dtype = resolve_dtype(args.torch_dtype)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_files = {"train": str(args.train_jsonl), "validation": str(args.val_jsonl)}
    ds_dict = load_dataset("json", data_files=data_files)
    train_ds = ds_dict["train"]
    val_ds = ds_dict["validation"]

    if args.convert_messages_to_prompt_completion:
        train_ds = convert_dataset_to_prompt_completion(train_ds)
        val_ds = convert_dataset_to_prompt_completion(val_ds)

    if args.max_train_samples > 0:
        train_ds = train_ds.select(range(min(len(train_ds), args.max_train_samples)))
    if args.max_eval_samples > 0:
        val_ds = val_ds.select(range(min(len(val_ds), args.max_eval_samples)))

    print(
        json.dumps(
            {
                "model_name": args.model_name,
                "dtype": str(dtype),
                "train": dataset_summary(train_ds),
                "validation": dataset_summary(val_ds),
                "completion_only_loss": bool(args.completion_only_loss),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print("[setup] loading processor")
    processor = AutoProcessor.from_pretrained(
        args.model_name, trust_remote_code=bool(args.trust_remote_code)
    )
    print("[setup] loading model")
    model = AutoVLMModel.from_pretrained(
        args.model_name,
        dtype=dtype,
        trust_remote_code=bool(args.trust_remote_code),
    )

    if args.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    peft_config = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=resolve_lora_targets(args.lora_target_modules),
    )

    assistant_only_loss = bool(args.assistant_only_loss)
    if assistant_only_loss:
        # TRL currently does not support assistant-only loss for VLM training.
        print(
            "[warn] assistant_only_loss=True is not supported for vision-language models. "
            "Forcing assistant_only_loss=False."
        )
        assistant_only_loss = False

    use_fp16 = bool(dtype == torch.float16)
    use_bf16 = bool(dtype == torch.bfloat16)
    if torch.backends.mps.is_available() and (use_fp16 or use_bf16):
        # Newer accelerate versions reject fp16/bf16 mixed precision on MPS.
        print(
            "[warn] disabling fp16/bf16 mixed_precision flags on MPS "
            "(model dtype remains unchanged)."
        )
        use_fp16 = False
        use_bf16 = False

    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        do_train=True,
        do_eval=True,
        num_train_epochs=float(args.epochs),
        learning_rate=float(args.lr),
        weight_decay=float(args.weight_decay),
        warmup_ratio=float(args.warmup_ratio),
        per_device_train_batch_size=int(args.train_batch_size),
        per_device_eval_batch_size=int(args.eval_batch_size),
        gradient_accumulation_steps=int(args.grad_accum),
        logging_steps=int(args.logging_steps),
        eval_strategy="steps",
        eval_steps=int(args.eval_steps),
        save_strategy="steps",
        save_steps=int(args.save_steps),
        save_total_limit=int(args.save_total_limit),
        max_steps=int(args.max_steps),
        gradient_checkpointing=bool(args.gradient_checkpointing),
        load_best_model_at_end=bool(args.load_best_model_at_end),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        assistant_only_loss=assistant_only_loss,
        completion_only_loss=bool(args.completion_only_loss),
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=bool(args.dataloader_pin_memory),
        fp16=use_fp16,
        bf16=use_bf16,
        max_length=None,
        use_cpu=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=processor,
        peft_config=peft_config,
    )

    if getattr(trainer.accelerator, "scaler", None) is None:
        # Keep checkpoint load/save codepaths compatible on MPS when mixed precision is disabled.
        print(
            "[warn] accelerator scaler is None; attaching disabled GradScaler "
            "for checkpoint compatibility."
        )
        try:
            trainer.accelerator.scaler = torch.amp.GradScaler("cuda", enabled=False)
        except Exception:
            trainer.accelerator.scaler = torch.cuda.amp.GradScaler(enabled=False)

    if args.dry_run:
        print("[dry-run] trainer init success. no training executed.")
        return 0

    resume = args.resume_from_checkpoint.strip() or None
    train_result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))

    metrics = dict(train_result.metrics)
    metrics_path = args.output_dir / "train_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[done] metrics saved: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
