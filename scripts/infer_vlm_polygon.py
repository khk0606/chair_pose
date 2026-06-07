#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from peft import PeftModel
from transformers import AutoProcessor

from build_vlm_dataset_from_labelme import (
    build_system_prompt,
    build_user_prompt,
    detect_chair_box,
    expand_crop_box,
    require_ultralytics,
    resolve_target_key,
    resolve_task_name,
)
from eval_vlm_polygon_quality import (
    box_to_polygon,
    extract_json_object,
    is_schema_valid,
    normalize_box,
    normalize_points,
    parse_positive_int,
    resolve_base_model_name,
    resolve_device,
    scale_points,
)

try:
    from transformers import AutoModelForImageTextToText as AutoVLMModel
except Exception:  # pragma: no cover
    from transformers import AutoModelForVision2Seq as AutoVLMModel  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-image VLM polygon inference with optional reject rules."
    )
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-overlay", type=Path, required=True)
    parser.add_argument("--base-model", type=str, default="")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu"])
    parser.add_argument("--resize-width", type=int, default=224)
    parser.add_argument("--resize-height", type=int, default=224)
    parser.add_argument(
        "--target-shape",
        type=str,
        default="polygon",
        choices=["polygon", "quad", "bbox"],
    )
    parser.add_argument(
        "--coord-mode",
        type=str,
        default="pixel",
        choices=["pixel", "grid"],
    )
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument(
        "--crop-mode",
        type=str,
        default="none",
        choices=["none", "chair"],
    )
    parser.add_argument(
        "--chair-detector",
        type=str,
        default="yolov8x-seg.pt",
    )
    parser.add_argument("--chair-conf", type=float, default=0.2)
    parser.add_argument("--chair-device", type=str, default="mps")
    parser.add_argument("--crop-margin-ratio", type=float, default=0.10)
    parser.add_argument(
        "--task-name",
        type=str,
        default="seat_contact_segmentation_points",
    )
    parser.add_argument(
        "--target-key",
        type=str,
        default="seat_contact_polygon",
    )
    parser.add_argument(
        "--expected-points",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--reject-bad-polygon",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    acc = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        acc += x1 * y2 - x2 * y1
    return abs(acc) * 0.5


def reject_polygon(
    points: list[tuple[float, float]],
    width: int,
    height: int,
) -> tuple[bool, list[str], dict[str, float | int]]:
    if width <= 0 or height <= 0:
        return True, ["invalid_image_size"], {}

    reasons: list[str] = []
    unique_points = {(int(round(x)), int(round(y))) for x, y in points}
    xs = [p[0] for p in points] if points else []
    ys = [p[1] for p in points] if points else []

    area = polygon_area(points)
    area_ratio = area / float(width * height)
    bbox_w = (max(xs) - min(xs)) if xs else 0.0
    bbox_h = (max(ys) - min(ys)) if ys else 0.0
    bbox_h_ratio = bbox_h / float(height)
    bbox_top = min(ys) if ys else 0.0
    bbox_bottom = max(ys) if ys else 0.0

    if len(points) < 3:
        reasons.append("too_few_points")
    if len(unique_points) < 3:
        reasons.append("too_few_unique_points")
    if area_ratio > 0.35:
        reasons.append("area_too_large")
    if area_ratio < 0.003:
        reasons.append("area_too_small")
    if bbox_h_ratio > 0.45:
        reasons.append("bbox_too_tall")
    if bbox_bottom < 0.45 * float(height):
        reasons.append("bbox_too_high")
    if bbox_top > 0.80 * float(height):
        reasons.append("bbox_too_low")
    if bbox_w < 0.08 * float(width) or bbox_h < 0.08 * float(height):
        reasons.append("collapsed_extent")

    corner_margin = 0.03 * float(min(width, height))
    corners = [
        (0.0, 0.0),
        (float(width - 1), 0.0),
        (0.0, float(height - 1)),
        (float(width - 1), float(height - 1)),
    ]
    corner_hits = 0
    for cx, cy in corners:
        hit = any(abs(x - cx) <= corner_margin and abs(y - cy) <= corner_margin for x, y in points)
        corner_hits += int(hit)
    if corner_hits >= 2:
        reasons.append("touches_multiple_corners")

    metrics: dict[str, float | int] = {
        "points": len(points),
        "unique_points": len(unique_points),
        "polygon_area": float(area),
        "polygon_area_ratio": float(area_ratio),
        "bbox_width": float(bbox_w),
        "bbox_height": float(bbox_h),
        "bbox_height_ratio": float(bbox_h_ratio),
        "bbox_top": float(bbox_top),
        "bbox_bottom": float(bbox_bottom),
        "corner_hits": int(corner_hits),
    }
    return bool(reasons), reasons, metrics


def draw_overlay(
    image: Image.Image,
    points: list[tuple[float, float]],
    rejected: bool,
    output_path: Path,
) -> None:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    fill = (255, 0, 0, 80) if rejected else (0, 180, 0, 80)
    outline = (255, 0, 0, 220) if rejected else (0, 160, 0, 220)
    if len(points) >= 3:
        draw.polygon(points, fill=fill, outline=outline, width=3)
        for x, y in points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=outline)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(base, overlay).convert("RGB").save(output_path)


def main() -> int:
    args = parse_args()
    if not args.adapter_dir.exists():
        raise FileNotFoundError(f"adapter-dir not found: {args.adapter_dir}")
    if not args.image.exists():
        raise FileNotFoundError(f"image not found: {args.image}")

    base_model_name = resolve_base_model_name(args.base_model, args.adapter_dir)
    task_name = str(args.task_name)
    if task_name == "seat_contact_segmentation_points":
        task_name = resolve_task_name(str(args.target_shape), str(args.coord_mode))
    target_key = str(args.target_key)
    if target_key == "seat_contact_polygon":
        target_key = resolve_target_key(str(args.target_shape))
    device = resolve_device(args.device)
    dtype = torch.float16 if device == "mps" else torch.float32

    processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
    base_model = AutoVLMModel.from_pretrained(
        base_model_name,
        dtype=dtype,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(args.adapter_dir))
    model.to(device)
    model.eval()

    orig_image = Image.open(args.image).convert("RGB")
    orig_w, orig_h = orig_image.size
    crop_box = (0, 0, int(orig_w), int(orig_h))
    chair_bbox: tuple[float, float, float, float] | None = None
    chair_detected = False
    input_image = orig_image
    if str(args.crop_mode) == "chair":
        YOLO = require_ultralytics()
        chair_model = YOLO(str(args.chair_detector))
        chair_bbox = detect_chair_box(
            chair_model,
            args.image,
            conf=float(args.chair_conf),
            device=str(args.chair_device),
        )
        if chair_bbox is not None:
            chair_detected = True
            crop_box = expand_crop_box(
                chair_bbox,
                image_width=int(orig_w),
                image_height=int(orig_h),
                margin_ratio=float(args.crop_margin_ratio),
            )
            input_image = orig_image.crop(crop_box)
    crop_w, crop_h = input_image.size
    infer_w = int(args.resize_width) if int(args.resize_width) > 0 else int(crop_w)
    infer_h = int(args.resize_height) if int(args.resize_height) > 0 else int(crop_h)
    infer_image = input_image.resize((infer_w, infer_h), Image.Resampling.BICUBIC)
    system_prompt = build_system_prompt(
        str(args.target_shape),
        coord_mode=str(args.coord_mode),
        grid_size=int(args.grid_size),
        crop_mode=str(args.crop_mode),
    )
    user_prompt = build_user_prompt(
        str(args.target_shape),
        coord_mode=str(args.coord_mode),
        grid_size=int(args.grid_size),
        crop_mode=str(args.crop_mode),
    )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=infer_image, return_tensors="pt")
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

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

    pred_obj = extract_json_object(pred_text, target_key=target_key)
    valid_json = pred_obj is not None
    schema_valid = valid_json and is_schema_valid(
        pred_obj,
        task_name=task_name,
        expected_points=int(args.expected_points),
        target_shape=str(args.target_shape),
        target_key=target_key,
    )
    pred_points_infer: list[tuple[float, float]] = []
    pred_points_crop: list[tuple[float, float]] = []
    pred_points_orig: list[tuple[float, float]] = []

    if valid_json:
        image_size = pred_obj.get("image_size")
        if isinstance(image_size, dict):
            pred_w = parse_positive_int(image_size.get("width", infer_w))
            pred_h = parse_positive_int(image_size.get("height", infer_h))
        else:
            pred_w = None
            pred_h = None
            schema_valid = False
        if pred_w is None or pred_h is None:
            pred_w = int(infer_w)
            pred_h = int(infer_h)
            schema_valid = False

        if str(args.target_shape) == "bbox":
            pred_box = normalize_box(
                pred_obj.get(target_key),
                int(pred_w),
                int(pred_h),
            )
            pred_pts_src = box_to_polygon(pred_box) if pred_box is not None else []
        else:
            pred_pts_src = normalize_points(
                pred_obj.get(target_key, []),
                int(pred_w),
                int(pred_h),
            )
        pred_points_infer = scale_points(
            pred_pts_src,
            int(pred_w),
            int(pred_h),
            int(infer_w),
            int(infer_h),
        )
        pred_points_crop = scale_points(
            pred_points_infer,
            int(infer_w),
            int(infer_h),
            int(crop_w),
            int(crop_h),
        )
        pred_points_orig = [
            (float(x) + float(crop_box[0]), float(y) + float(crop_box[1]))
            for x, y in pred_points_crop
        ]
        if len(pred_points_orig) < 3:
            schema_valid = False

    rejected = False
    reject_reasons: list[str] = []
    reject_metrics: dict[str, float | int] = {}
    if args.reject_bad_polygon:
        if not valid_json:
            rejected = True
            reject_reasons.append("invalid_json")
        elif not schema_valid:
            rejected = True
            reject_reasons.append("schema_invalid")
        elif not pred_points_orig:
            rejected = True
            reject_reasons.append("empty_polygon")
        else:
            rejected, reject_reasons, reject_metrics = reject_polygon(
                pred_points_orig,
                orig_w,
                orig_h,
            )

    draw_overlay(
        image=orig_image,
        points=pred_points_orig,
        rejected=bool(rejected),
        output_path=args.output_overlay,
    )

    output = {
        "adapter_dir": str(args.adapter_dir),
        "base_model": base_model_name,
        "image": str(args.image),
        "target_shape": str(args.target_shape),
        "task_name": task_name,
        "expected_points": int(args.expected_points),
        "target_key": target_key,
        "original_image_size": {"width": int(orig_w), "height": int(orig_h)},
        "crop_box_xyxy": {
            "x_min": int(crop_box[0]),
            "y_min": int(crop_box[1]),
            "x_max": int(crop_box[2]),
            "y_max": int(crop_box[3]),
        },
        "chair_bbox_xyxy": (
            {
                "x_min": float(chair_bbox[0]),
                "y_min": float(chair_bbox[1]),
                "x_max": float(chair_bbox[2]),
                "y_max": float(chair_bbox[3]),
            }
            if chair_bbox is not None
            else None
        ),
        "chair_detected": bool(chair_detected),
        "coord_mode": str(args.coord_mode),
        "grid_size": int(args.grid_size),
        "crop_mode": str(args.crop_mode),
        "inference_image_size": {"width": int(infer_w), "height": int(infer_h)},
        "crop_image_size": {"width": int(crop_w), "height": int(crop_h)},
        "valid_json": bool(valid_json),
        "schema_valid": bool(schema_valid),
        "rejected": bool(rejected),
        "reject_reasons": reject_reasons,
        "reject_metrics": reject_metrics,
        "pred_text": pred_text,
        "pred_json": pred_obj,
        "pred_points_inference_xy": [
            {"x": float(x), "y": float(y)} for x, y in pred_points_infer
        ],
        "pred_points_crop_xy": [
            {"x": float(x), "y": float(y)} for x, y in pred_points_crop
        ],
        "pred_points_original_xy": [
            {"x": float(x), "y": float(y)} for x, y in pred_points_orig
        ],
        "output_overlay": str(args.output_overlay),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
