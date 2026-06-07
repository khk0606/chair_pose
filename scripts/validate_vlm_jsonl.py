#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from build_vlm_dataset_from_labelme import resolve_target_key, resolve_task_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate VLM chat-format JSONL generated for SFT."
    )
    parser.add_argument("--jsonl", type=Path, action="append", required=True)
    parser.add_argument(
        "--task-name",
        type=str,
        default="seat_contact_segmentation_points",
        help="Exact task string expected inside assistant JSON.",
    )
    parser.add_argument(
        "--expected-points",
        type=int,
        default=0,
        help="If >0, require exactly this number of points in seat_contact_polygon.",
    )
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
    parser.add_argument(
        "--target-key",
        type=str,
        default="seat_contact_polygon",
    )
    return parser.parse_args()


def extract_assistant_json(messages: list[dict[str, Any]]) -> dict[str, Any]:
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                return json.loads(block["text"])
    raise ValueError("assistant JSON text block not found")


def is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return False


def validate_target_schema(
    target: dict[str, Any],
    assistant_text: str,
    task_name: str,
    expected_points: int,
    target_shape: str,
    target_key: str,
) -> None:
    if not isinstance(target, dict):
        raise ValueError("assistant target must be a JSON object")

    required = {"task", "image_size", target_key}
    missing = required - set(target.keys())
    if missing:
        raise ValueError(f"target missing keys: {sorted(missing)}")

    if target.get("task") != task_name:
        raise ValueError(f"task must be {task_name}")

    image_size = target.get("image_size")
    if not isinstance(image_size, dict):
        raise ValueError("image_size must be object")

    width = image_size.get("width")
    height = image_size.get("height")
    if not isinstance(width, int) or not isinstance(height, int):
        raise ValueError("image_size.width/height must be integers")
    if width <= 0 or height <= 0:
        raise ValueError("image_size.width/height must be positive")

    if target_shape == "bbox":
        box = target.get(target_key)
        if not isinstance(box, dict):
            raise ValueError(f"{target_key} must be object")
        required_box = {"x_min", "y_min", "x_max", "y_max"}
        if set(box.keys()) != required_box:
            raise ValueError(f"{target_key} must contain {sorted(required_box)}")
        for k in sorted(required_box):
            if not is_number(box[k]):
                raise ValueError(f"{target_key}.{k} must be a finite number")
        x_min = float(box["x_min"])
        y_min = float(box["y_min"])
        x_max = float(box["x_max"])
        y_max = float(box["y_max"])
        if x_min < 0 or x_max > width - 1 or y_min < 0 or y_max > height - 1:
            raise ValueError(f"{target_key} out of image boundary")
        if x_max <= x_min or y_max <= y_min:
            raise ValueError(f"{target_key} must satisfy x_max>x_min and y_max>y_min")
    else:
        polygon = target.get(target_key)
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise ValueError(f"{target_key} must contain >=3 points")
        if expected_points > 0 and len(polygon) != expected_points:
            raise ValueError(f"{target_key} must contain exactly {expected_points} points")

        for idx, p in enumerate(polygon):
            if not isinstance(p, dict):
                raise ValueError(f"point[{idx}] must be object")
            if "x" not in p or "y" not in p:
                raise ValueError(f"point[{idx}] missing x/y")
            if not is_number(p["x"]) or not is_number(p["y"]):
                raise ValueError(f"point[{idx}] x/y must be finite numbers")
            x = float(p["x"])
            y = float(p["y"])
            if x < 0 or x > width - 1 or y < 0 or y > height - 1:
                raise ValueError(f"point[{idx}] out of image boundary")

    low = assistant_text.lower()
    banned_tokens = ["...", "<x>", "<y>", "100%", "\"int\"", "\"float\"", ":int", ":float"]
    if any(tok in low for tok in banned_tokens):
        raise ValueError("assistant text contains placeholder-like tokens")


def validate_file(
    path: Path,
    task_name: str,
    expected_points: int,
    target_shape: str,
    target_key: str,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    total = 0
    ok = 0
    bad: list[str] = []

    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        total += 1
        try:
            row = json.loads(line)
            images = row["images"]
            messages = row["messages"]
            if not isinstance(images, list) or not images:
                raise ValueError("images missing")
            if not isinstance(messages, list) or not messages:
                raise ValueError("messages missing")
            image_path = Path(str(images[0]))
            if not image_path.exists():
                raise FileNotFoundError(f"image does not exist: {image_path}")
            target = extract_assistant_json(messages)
            assistant_text = ""
            for msg in messages:
                if msg.get("role") == "assistant":
                    content = msg.get("content")
                    if isinstance(content, list):
                        for block in content:
                            if (
                                isinstance(block, dict)
                                and block.get("type") == "text"
                                and isinstance(block.get("text"), str)
                            ):
                                assistant_text = block["text"]
                                break
            validate_target_schema(
                target,
                assistant_text,
                task_name,
                expected_points,
                target_shape,
                target_key,
            )
            ok += 1
        except Exception as exc:
            bad.append(f"line {lineno}: {exc}")

    return {
        "file": str(path),
        "total_lines": total,
        "valid_lines": ok,
        "invalid_lines": len(bad),
        "errors_preview": bad[:20],
    }


def main() -> int:
    args = parse_args()
    task_name = str(args.task_name)
    if task_name == "seat_contact_segmentation_points":
        task_name = resolve_task_name(str(args.target_shape), str(args.coord_mode))
    target_key = str(args.target_key)
    if target_key == "seat_contact_polygon":
        target_key = resolve_target_key(str(args.target_shape))
    reports = [
        validate_file(
            p,
            task_name=task_name,
            expected_points=int(args.expected_points),
            target_shape=str(args.target_shape),
            target_key=target_key,
        )
        for p in args.jsonl
    ]
    print(json.dumps({"reports": reports}, ensure_ascii=False, indent=2))
    any_bad = any(r["invalid_lines"] > 0 for r in reports)
    return 1 if any_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
