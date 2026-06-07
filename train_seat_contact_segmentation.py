#!/usr/bin/env python3
"""
Train a single-class seat-contact segmentation model with Ultralytics YOLO.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train seat-contact segmentation model (YOLO-seg)."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data" / "seat_contact_yolo" / "dataset.yaml",
        help="Path to Ultralytics dataset.yaml",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s-seg.pt",
        help="Base checkpoint to fine-tune (e.g., yolov8n-seg.pt, yolov8s-seg.pt)",
    )
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Dataloader workers (0 recommended on macOS for stability)",
    )
    parser.add_argument(
        "--patience", type=int, default=20, help="Early stopping patience"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--cache",
        type=str,
        default="ram",
        help="Ultralytics cache option: False|True|ram|disk",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable per-iteration verbose logs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device: auto|mps|cpu|cuda:0",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=REPO_ROOT / "runs" / "yolo",
        help="Output project dir (Ultralytics)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="seat_contact_yolov8s_seg_v1",
        help="Run name (Ultralytics)",
    )
    parser.add_argument(
        "--copy-best-to",
        type=Path,
        default=REPO_ROOT / "models" / "seat_contact_yolov8s_seg_v1_best.pt",
        help="Copy trained best.pt to this path",
    )
    parser.add_argument("--optimizer", type=str, default="auto", help="Optimizer: auto|AdamW|SGD")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final LR fraction")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic probability")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Disable mosaic in final N epochs")
    parser.add_argument("--translate", type=float, default=0.1, help="Translate augmentation strength")
    parser.add_argument("--scale", type=float, default=0.5, help="Scale augmentation strength")
    parser.add_argument("--degrees", type=float, default=0.0, help="Rotation augmentation (degrees)")
    parser.add_argument("--perspective", type=float, default=0.0, help="Perspective augmentation strength")
    parser.add_argument("--erasing", type=float, default=0.4, help="Random erasing probability")
    parser.add_argument("--hsv-s", type=float, default=0.7, help="HSV saturation augmentation")
    parser.add_argument("--hsv-v", type=float, default=0.4, help="HSV value augmentation")
    parser.add_argument(
        "--mask-ratio",
        type=int,
        default=4,
        help=(
            "Mask downsample ratio for segmentation head. "
            "Smaller values (e.g., 1-2) preserve finer boundaries."
        ),
    )
    parser.add_argument(
        "--overlap-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use overlapping masks during training.",
    )
    return parser.parse_args()


def resolve_device(user_device: str) -> str:
    if user_device and user_device.lower() != "auto":
        return user_device

    import torch  # type: ignore

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return str(obj)


def main() -> int:
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"dataset yaml not found: {args.data}")

    args.project.mkdir(parents=True, exist_ok=True)
    args.copy_best_to.parent.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO  # type: ignore

    device = resolve_device(args.device)
    print(f"[train] data={args.data}")
    print(f"[train] model={args.model}")
    print(f"[train] device={device}")
    print(
        f"[train] epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} patience={args.patience}"
    )
    print(
        "[train] aug "
        f"mosaic={args.mosaic} close_mosaic={args.close_mosaic} "
        f"translate={args.translate} scale={args.scale} degrees={args.degrees} "
        f"perspective={args.perspective} erasing={args.erasing} hsv_s={args.hsv_s} hsv_v={args.hsv_v} "
        f"mask_ratio={args.mask_ratio} overlap_mask={args.overlap_mask}"
    )

    model = YOLO(args.model)
    cache_opt: Any
    cache_raw = str(args.cache).strip().lower()
    if cache_raw in {"false", "0", "none", ""}:
        cache_opt = False
    elif cache_raw in {"true", "1"}:
        cache_opt = True
    else:
        cache_opt = cache_raw

    train_results = model.train(
        data=str(args.data),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        workers=int(args.workers),
        patience=int(args.patience),
        seed=int(args.seed),
        device=device,
        project=str(args.project),
        name=args.name,
        pretrained=True,
        cache=cache_opt,
        verbose=bool(args.verbose),
        optimizer=str(args.optimizer),
        lr0=float(args.lr0),
        lrf=float(args.lrf),
        mosaic=float(args.mosaic),
        close_mosaic=int(args.close_mosaic),
        translate=float(args.translate),
        scale=float(args.scale),
        degrees=float(args.degrees),
        perspective=float(args.perspective),
        erasing=float(args.erasing),
        hsv_s=float(args.hsv_s),
        hsv_v=float(args.hsv_v),
        mask_ratio=int(args.mask_ratio),
        overlap_mask=bool(args.overlap_mask),
    )

    save_dir = Path(str(getattr(train_results, "save_dir", "")))
    if not save_dir.exists():
        trainer = getattr(model, "trainer", None)
        if trainer is not None:
            save_dir = Path(str(getattr(trainer, "save_dir", "")))

    best_pt = save_dir / "weights" / "best.pt"
    last_pt = save_dir / "weights" / "last.pt"

    if best_pt.exists():
        shutil.copy2(best_pt, args.copy_best_to)
        print(f"[train] copied best weights to: {args.copy_best_to}")
    else:
        print(f"[warn] best.pt not found at: {best_pt}")

    summary = {
        "run_name": args.name,
        "save_dir": str(save_dir),
        "best_pt": str(best_pt),
        "best_pt_exists": best_pt.exists(),
        "last_pt": str(last_pt),
        "last_pt_exists": last_pt.exists(),
        "copied_best_to": str(args.copy_best_to),
        "train_metrics": to_serializable(getattr(train_results, "results_dict", {})),
    }

    # Optional post-train validation snapshot.
    try:
        val_results = model.val(
            data=str(args.data),
            split="val",
            imgsz=int(args.imgsz),
            device=device,
        )
        summary["val_metrics"] = to_serializable(
            getattr(val_results, "results_dict", {})
        )
    except Exception as exc:  # pragma: no cover - best effort
        summary["val_error"] = str(exc)

    summary_path = save_dir / "train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[train] summary saved: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
