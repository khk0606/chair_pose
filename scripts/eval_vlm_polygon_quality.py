#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from peft import PeftModel
from transformers import AutoProcessor

try:
    from transformers import AutoModelForImageTextToText as AutoVLMModel
except Exception:  # pragma: no cover
    from transformers import AutoModelForVision2Seq as AutoVLMModel  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate VLM checkpoint quality on seat-contact polygon generation."
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="",
        help=(
            "HF base model id. If empty, auto-detect from adapter_config.json "
            "inside --adapter-dir."
        ),
    )
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu"])
    return parser.parse_args()


def resolve_base_model_name(explicit_base_model: str, adapter_dir: Path) -> str:
    if explicit_base_model.strip():
        return explicit_base_model.strip()

    adapter_cfg = adapter_dir / "adapter_config.json"
    if adapter_cfg.exists():
        try:
            obj = json.loads(adapter_cfg.read_text(encoding="utf-8"))
            name = obj.get("base_model_name_or_path")
            if isinstance(name, str) and name.strip():
                return name.strip()
        except Exception:
            pass

    return "HuggingFaceTB/SmolVLM-256M-Instruct"


def resolve_device(spec: str) -> str:
    if spec == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return "mps"
    if spec == "cpu":
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(text[start:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        start = text.find("{", start + 1)
    return None


def parse_numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        s = s.replace(",", "")
        if s.lower().endswith("px"):
            s = s[:-2].strip()
        try:
            v = float(s)
        except Exception:
            return None
        return v if math.isfinite(v) else None
    return None


def parse_positive_int(value: Any) -> int | None:
    v = parse_numeric(value)
    if v is None:
        return None
    iv = int(round(v))
    if iv <= 0:
        return None
    return iv


def normalize_points(poly: Any, width: int, height: int) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    if not isinstance(poly, list):
        return pts
    for p in poly:
        if not isinstance(p, dict):
            return []
        xf = parse_numeric(p.get("x"))
        yf = parse_numeric(p.get("y"))
        if xf is None or yf is None:
            return []
        xf = min(max(xf, 0.0), float(max(0, width - 1)))
        yf = min(max(yf, 0.0), float(max(0, height - 1)))
        pts.append((xf, yf))
    return pts


def has_placeholder_pattern(text: str) -> bool:
    low = text.lower()
    tokens = ["...", "100%", "\"int\"", "\"float\"", "<x>", "<y>"]
    return any(t in low for t in tokens)


def is_schema_valid(obj: dict[str, Any]) -> bool:
    required = {"task", "image_size", "seat_contact_polygon"}
    if set(obj.keys()) != required:
        return False
    if obj.get("task") != "seat_contact_segmentation_points":
        return False
    image_size = obj.get("image_size")
    if not isinstance(image_size, dict):
        return False
    if set(image_size.keys()) != {"width", "height"}:
        return False
    w = parse_positive_int(image_size.get("width"))
    h = parse_positive_int(image_size.get("height"))
    if w is None or h is None:
        return False
    poly = obj.get("seat_contact_polygon")
    if not isinstance(poly, list):
        return False
    pts = normalize_points(poly, int(w), int(h))
    if len(pts) < 3:
        return False
    return True


def scale_points(
    points: list[tuple[float, float]],
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> list[tuple[float, float]]:
    if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
        return points
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    out: list[tuple[float, float]] = []
    for x, y in points:
        nx = min(max(x * sx, 0.0), float(max(0, dst_w - 1)))
        ny = min(max(y * sy, 0.0), float(max(0, dst_h - 1)))
        out.append((nx, ny))
    return out


def polygon_iou(
    pred_pts: list[tuple[float, float]],
    gt_pts: list[tuple[float, float]],
    width: int,
    height: int,
) -> float:
    if len(pred_pts) < 3 or len(gt_pts) < 3:
        return 0.0
    pred_mask = Image.new("1", (width, height), 0)
    gt_mask = Image.new("1", (width, height), 0)
    ImageDraw.Draw(pred_mask).polygon(pred_pts, fill=1)
    ImageDraw.Draw(gt_mask).polygon(gt_pts, fill=1)
    a = np.array(pred_mask, dtype=np.uint8)
    b = np.array(gt_mask, dtype=np.uint8)
    inter = np.logical_and(a == 1, b == 1).sum()
    union = np.logical_or(a == 1, b == 1).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def main() -> int:
    args = parse_args()
    if not args.adapter_dir.exists():
        raise FileNotFoundError(f"adapter-dir not found: {args.adapter_dir}")
    if not args.jsonl.exists():
        raise FileNotFoundError(f"jsonl not found: {args.jsonl}")
    base_model_name = resolve_base_model_name(args.base_model, args.adapter_dir)

    device = resolve_device(args.device)
    dtype = torch.float16 if device == "mps" else torch.float32

    rows = [
        json.loads(line)
        for line in args.jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise RuntimeError("No rows in jsonl to evaluate.")

    processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
    base_model = AutoVLMModel.from_pretrained(
        base_model_name,
        dtype=dtype,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(args.adapter_dir))
    model.to(device)
    model.eval()

    samples: list[dict[str, Any]] = []
    valid_json_count = 0
    schema_valid_count = 0
    placeholder_count = 0

    for row in rows:
        image_path = str(row["images"][0])
        gt_obj = json.loads(row["messages"][-1]["content"][0]["text"])
        gt_w = int(gt_obj["image_size"]["width"])
        gt_h = int(gt_obj["image_size"]["height"])
        gt_pts = normalize_points(gt_obj["seat_contact_polygon"], gt_w, gt_h)

        messages = [
            {"role": "system", "content": row["messages"][0]["content"]},
            {"role": "user", "content": row["messages"][1]["content"]},
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()
        }

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=False,
            )
        input_len = inputs["input_ids"].shape[1]
        pred_text = processor.batch_decode(
            generated[:, input_len:],
            skip_special_tokens=True,
        )[0].strip()

        placeholder = has_placeholder_pattern(pred_text)
        pred_obj = extract_json_object(pred_text)
        valid_json = pred_obj is not None
        schema_valid = valid_json and is_schema_valid(pred_obj)
        pred_pts: list[tuple[float, float]] = []
        iou = 0.0
        if valid_json:
            image_size = pred_obj.get("image_size")
            if isinstance(image_size, dict):
                pred_w_raw = image_size.get("width", gt_w)
                pred_h_raw = image_size.get("height", gt_h)
            else:
                # Some malformed outputs return non-dict image_size; treat as schema invalid.
                schema_valid = False
                pred_w_raw = gt_w
                pred_h_raw = gt_h

            pred_w = parse_positive_int(pred_w_raw)
            pred_h = parse_positive_int(pred_h_raw)
            if pred_w is None or pred_h is None:
                schema_valid = False
                pred_w = int(gt_w)
                pred_h = int(gt_h)

            pred_pts_src = normalize_points(
                pred_obj.get("seat_contact_polygon", []),
                int(pred_w),
                int(pred_h),
            )
            pred_pts = scale_points(
                pred_pts_src,
                int(pred_w),
                int(pred_h),
                int(gt_w),
                int(gt_h),
            )
            if len(pred_pts) < 3:
                schema_valid = False
            iou = polygon_iou(pred_pts, gt_pts, gt_w, gt_h)

        valid_json_count += int(valid_json)
        schema_valid_count += int(schema_valid)
        placeholder_count += int(placeholder)

        samples.append(
            {
                "id": row.get("id"),
                "image": image_path,
                "valid_json": bool(valid_json),
                "schema_valid": bool(schema_valid),
                "has_placeholder_pattern": bool(placeholder),
                "pred_points": len(pred_pts),
                "pred_points_xy": [
                    {"x": float(x), "y": float(y)} for x, y in pred_pts
                ],
                "gt_points": len(gt_pts),
                "iou": iou,
                "pred_preview": pred_text[:350],
                "pred_text": pred_text,
                "pred_json": pred_obj,
            }
        )

    ious_all = [s["iou"] for s in samples]
    ious_schema_valid = [s["iou"] for s in samples if s["schema_valid"]]
    summary = {
        "adapter_dir": str(args.adapter_dir),
        "jsonl": str(args.jsonl),
        "num_samples": len(samples),
        "valid_json_rate": valid_json_count / len(samples),
        "schema_valid_rate": schema_valid_count / len(samples),
        "placeholder_rate": placeholder_count / len(samples),
        "mean_iou_all": float(np.mean(ious_all)) if ious_all else 0.0,
        "mean_iou_schema_valid_only": (
            float(np.mean(ious_schema_valid)) if ious_schema_valid else 0.0
        ),
        "median_iou_schema_valid_only": (
            float(np.median(ious_schema_valid)) if ious_schema_valid else 0.0
        ),
    }

    output = {"summary": summary, "samples": samples}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[saved] {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
