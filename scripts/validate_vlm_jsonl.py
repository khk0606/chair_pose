#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate VLM chat-format JSONL generated for SFT."
    )
    parser.add_argument("--jsonl", type=Path, action="append", required=True)
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


def validate_target_schema(target: dict[str, Any], assistant_text: str) -> None:
    if not isinstance(target, dict):
        raise ValueError("assistant target must be a JSON object")

    required = {"task", "image_size", "seat_contact_polygon"}
    missing = required - set(target.keys())
    if missing:
        raise ValueError(f"target missing keys: {sorted(missing)}")

    if target.get("task") != "seat_contact_segmentation_points":
        raise ValueError("task must be seat_contact_segmentation_points")

    image_size = target.get("image_size")
    if not isinstance(image_size, dict):
        raise ValueError("image_size must be object")

    width = image_size.get("width")
    height = image_size.get("height")
    if not isinstance(width, int) or not isinstance(height, int):
        raise ValueError("image_size.width/height must be integers")
    if width <= 0 or height <= 0:
        raise ValueError("image_size.width/height must be positive")

    polygon = target.get("seat_contact_polygon")
    if not isinstance(polygon, list) or len(polygon) < 3:
        raise ValueError("seat_contact_polygon must contain >=3 points")

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


def validate_file(path: Path) -> dict[str, Any]:
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
            validate_target_schema(target, assistant_text)
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
    reports = [validate_file(p) for p in args.jsonl]
    print(json.dumps({"reports": reports}, ensure_ascii=False, indent=2))
    any_bad = any(r["invalid_lines"] > 0 for r in reports)
    return 1 if any_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
