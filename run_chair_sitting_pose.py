#!/usr/bin/env python3
"""Chair-only image -> plausible sitting 2D keypoints (VLM pipeline).

Pipeline:
1) Stage 0: optional chair-part segmentation anchors
2) Stage A: chair geometry extraction
3) Stage B: keypoint prediction (persona optional)
4) Stage C: strict checker
5) Failure handling: retry + repair loop
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import mimetypes
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable
from urllib import error, request

KEYPOINT_NAMES = [
    "hip_center",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]

CHECK_NAMES = [
    "hip_on_seat",
    "knees_in_front_of_hip",
    "ankles_below_seat_near_floor",
    "limb_ordering_consistent",
]

_YOLO_SEG_MODEL_CACHE: dict[str, Any] = {}


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def load_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return data


def guess_image_mime(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type and mime_type.startswith("image/"):
        return mime_type

    ext_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".gif": "image/gif",
    }
    guessed = ext_map.get(image_path.suffix.lower())
    if guessed:
        return guessed
    raise ValueError("Unsupported image extension. Use jpg/jpeg/png/webp/bmp/gif.")


def prepare_model_image(
    image_path: Path, *, max_side: int, jpeg_quality: int
) -> tuple[str, str, int, int, bool]:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    src_w, src_h = infer_image_size(image_path)
    mime_type = guess_image_mime(image_path)

    if max_side <= 0 or max(src_w, src_h) <= max_side:
        payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return mime_type, payload, src_w, src_h, False

    try:
        from PIL import Image  # type: ignore
    except Exception:
        if sys.platform == "darwin":
            try:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
                    subprocess.run(
                        [
                            "sips",
                            "-Z",
                            str(max_side),
                            str(image_path),
                            "--out",
                            tmp.name,
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    resized_path = Path(tmp.name)
                    resized_w, resized_h = infer_image_size(resized_path)
                    payload = base64.b64encode(resized_path.read_bytes()).decode("ascii")
                    return "image/jpeg", payload, resized_w, resized_h, True
            except Exception:
                pass
        payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return mime_type, payload, src_w, src_h, False

    scale = max_side / float(max(src_w, src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    jpeg_quality = int(_clamp(float(jpeg_quality), 30.0, 95.0))

    with Image.open(image_path) as image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        resample = getattr(Image, "Resampling", Image).LANCZOS
        resized = image.resize((new_w, new_h), resample=resample)
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        payload = base64.b64encode(buf.getvalue()).decode("ascii")

    return "image/jpeg", payload, new_w, new_h, True


def infer_image_size(image_path: Path) -> tuple[int, int]:
    with image_path.open("rb") as fh:
        head = fh.read(32)

    if head.startswith(b"\x89PNG\r\n\x1a\n") and len(head) >= 24:
        width = int.from_bytes(head[16:20], "big")
        height = int.from_bytes(head[20:24], "big")
        if width > 0 and height > 0:
            return width, height

    if head.startswith((b"GIF87a", b"GIF89a")) and len(head) >= 10:
        width = int.from_bytes(head[6:8], "little")
        height = int.from_bytes(head[8:10], "little")
        if width > 0 and height > 0:
            return width, height

    if head.startswith(b"BM") and len(head) >= 26:
        width = int.from_bytes(head[18:22], "little", signed=True)
        height = int.from_bytes(head[22:26], "little", signed=True)
        if width > 0 and height != 0:
            return width, abs(height)

    # JPEG dimensions are in SOF segments.
    with image_path.open("rb") as fh:
        if fh.read(2) == b"\xff\xd8":
            sof_markers = {
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            }
            while True:
                marker_start = fh.read(1)
                while marker_start and marker_start != b"\xff":
                    marker_start = fh.read(1)
                if not marker_start:
                    break

                marker = fh.read(1)
                while marker == b"\xff":
                    marker = fh.read(1)
                if not marker:
                    break

                marker_val = marker[0]
                if marker_val in (0xD8, 0xD9):
                    continue

                seg_len_raw = fh.read(2)
                if len(seg_len_raw) != 2:
                    break
                seg_len = int.from_bytes(seg_len_raw, "big")
                if seg_len < 2:
                    break

                if marker_val in sof_markers:
                    sof_data = fh.read(5)
                    if len(sof_data) != 5:
                        break
                    height = int.from_bytes(sof_data[1:3], "big")
                    width = int.from_bytes(sof_data[3:5], "big")
                    if width > 0 and height > 0:
                        return width, height
                    break

                fh.seek(seg_len - 2, 1)

    try:
        from PIL import Image  # type: ignore

        with Image.open(image_path) as image:
            width, height = image.size
            if width > 0 and height > 0:
                return int(width), int(height)
    except Exception:
        pass

    raise RuntimeError(f"Could not infer image dimensions: {image_path}")


def _round_point(x: float, y: float) -> dict[str, float]:
    return {"x": round(float(x), 2), "y": round(float(y), 2)}


def build_fallback_geometry(image_width: int, image_height: int) -> dict[str, Any]:
    w = float(image_width)
    h = float(image_height)
    x_max = max(0.0, w - 1.0)
    y_max = max(0.0, h - 1.0)

    seat_left = _clamp(0.28 * w, 0.0, x_max)
    seat_right = _clamp(0.72 * w, 0.0, x_max)
    seat_top = _clamp(0.52 * h, 0.0, y_max)
    seat_bottom = _clamp(0.62 * h, 0.0, y_max)

    floor_y = _clamp(0.88 * h, 0.0, y_max)
    back_top_y = _clamp(0.30 * h, 0.0, y_max)

    geometry = {
        "image_width": image_width,
        "image_height": image_height,
        "seat_region": [
            _round_point(seat_left, seat_top),
            _round_point(seat_right, seat_top),
            _round_point(seat_right, seat_bottom),
            _round_point(seat_left, seat_bottom),
        ],
        "floor_line": [
            _round_point(_clamp(0.10 * w, 0.0, x_max), floor_y),
            _round_point(_clamp(0.90 * w, 0.0, x_max), floor_y),
        ],
        "backrest_region": [
            _round_point(_clamp(0.28 * w, 0.0, x_max), back_top_y),
            _round_point(_clamp(0.72 * w, 0.0, x_max), back_top_y),
        ],
    }
    return geometry


def build_fallback_pose(geometry: dict[str, Any], persona: str = "") -> dict[str, Any]:
    width = int(geometry["image_width"])
    height = int(geometry["image_height"])
    w = float(width)
    h = float(height)
    x_max = max(0.0, w - 1.0)
    y_max = max(0.0, h - 1.0)

    seat = geometry["seat_region"]
    floor = geometry["floor_line"]
    seat_left = min(float(p["x"]) for p in seat)
    seat_right = max(float(p["x"]) for p in seat)
    seat_top = min(float(p["y"]) for p in seat)
    seat_bottom = max(float(p["y"]) for p in seat)
    floor_ax = float(floor[0]["x"])
    floor_ay = float(floor[0]["y"])
    floor_bx = float(floor[1]["x"])
    floor_by = float(floor[1]["y"])
    floor_y = _clamp((floor_ay + floor_by) * 0.5, 0.0, y_max)

    hip_x = (seat_left + seat_right) * 0.5
    hip_y = _clamp(seat_top + 0.55 * (seat_bottom - seat_top), 0.0, y_max)

    persona_lower = persona.lower().strip()
    if "tired" in persona_lower:
        hip_y = _clamp(hip_y + 0.02 * h, 0.0, y_max)

    leg_spread = max(0.06 * w, 20.0)
    shoulder_spread = max(0.10 * w, 24.0)
    arm_offset = max(0.05 * w, 16.0)

    knee_y = _clamp(hip_y + 0.20 * h, seat_bottom + 0.04 * h, floor_y - 0.12 * h)
    shoulder_y = _clamp(hip_y - 0.22 * h, 0.0, y_max)
    elbow_y = _clamp(shoulder_y + 0.11 * h, 0.0, y_max)
    wrist_y = _clamp(elbow_y + 0.10 * h, 0.0, y_max)

    left_knee_x = _clamp(hip_x - leg_spread, 0.0, x_max)
    right_knee_x = _clamp(hip_x + leg_spread, 0.0, x_max)
    left_ankle_pref_x = _clamp(hip_x - 0.9 * leg_spread, 0.0, x_max)
    right_ankle_pref_x = _clamp(hip_x + 0.9 * leg_spread, 0.0, x_max)

    def foot_on_floor(pref_x: float) -> tuple[float, float]:
        dx = floor_bx - floor_ax
        dy = floor_by - floor_ay
        denom = dx * dx + dy * dy
        if denom < 1e-8:
            x = _clamp(pref_x, 0.0, x_max)
            y = floor_y
        else:
            t = ((pref_x - floor_ax) * dx + (floor_y - floor_ay) * dy) / denom
            t = _clamp(t, 0.08, 0.92)
            x = _clamp(floor_ax + t * dx, 0.0, x_max)
            y = _clamp(floor_ay + t * dy, 0.0, y_max)
        y = _clamp(max(y, seat_bottom + 0.04 * h), 0.0, y_max)
        return x, y

    left_ankle_x, left_ankle_y = foot_on_floor(left_ankle_pref_x)
    right_ankle_x, right_ankle_y = foot_on_floor(right_ankle_pref_x)

    left_shoulder_x = _clamp(hip_x - shoulder_spread, 0.0, x_max)
    right_shoulder_x = _clamp(hip_x + shoulder_spread, 0.0, x_max)
    left_elbow_x = _clamp(left_shoulder_x - arm_offset * 0.5, 0.0, x_max)
    right_elbow_x = _clamp(right_shoulder_x + arm_offset * 0.5, 0.0, x_max)
    left_wrist_x = _clamp(left_elbow_x - arm_offset * 0.5, 0.0, x_max)
    right_wrist_x = _clamp(right_elbow_x + arm_offset * 0.5, 0.0, x_max)

    def kp(x: float, y: float, conf: float = 0.25, visible: bool = False) -> dict[str, Any]:
        return {
            "x": round(x, 2),
            "y": round(y, 2),
            "confidence": conf,
            "visible": visible,
        }

    return {
        "image_width": width,
        "image_height": height,
        "keypoints": {
            "hip_center": kp(_clamp(hip_x, 0.0, x_max), hip_y),
            "left_knee": kp(left_knee_x, knee_y),
            "right_knee": kp(right_knee_x, knee_y),
            "left_ankle": kp(left_ankle_x, left_ankle_y),
            "right_ankle": kp(right_ankle_x, right_ankle_y),
            "left_shoulder": kp(left_shoulder_x, shoulder_y),
            "right_shoulder": kp(right_shoulder_x, shoulder_y),
            "left_elbow": kp(left_elbow_x, elbow_y),
            "right_elbow": kp(right_elbow_x, elbow_y),
            "left_wrist": kp(left_wrist_x, wrist_y),
            "right_wrist": kp(right_wrist_x, wrist_y),
        },
    }


def extract_chair_geometry_yolo(
    image_path: Path,
    *,
    model_path: str = "yolov8n-seg.pt",
    conf: float = 0.2,
    device: str = "",
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
    log: dict[str, Any] = {
        "ok": False,
        "used": False,
        "model_path": model_path,
    }

    try:
        import numpy as np  # type: ignore
        import torch  # type: ignore
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        log["error"] = f"yolo unavailable: {exc}"
        return None, None, log

    resolved_model_path = model_path.strip() if model_path.strip() else "yolov8n-seg.pt"
    log["model_path"] = resolved_model_path

    model = _YOLO_SEG_MODEL_CACHE.get(resolved_model_path)
    if model is None:
        try:
            model = YOLO(resolved_model_path)
            _YOLO_SEG_MODEL_CACHE[resolved_model_path] = model
        except Exception as exc:
            log["error"] = f"failed to load YOLO model: {exc}"
            return None, None, log

    chair_ids = [
        int(cls_id)
        for cls_id, name in model.names.items()
        if str(name).strip().lower() == "chair"
    ]
    if not chair_ids:
        log["error"] = "chair class id not found in model names"
        return None, None, log

    infer_device = device.strip()
    if not infer_device:
        infer_device = "mps" if torch.backends.mps.is_available() else "cpu"
    log["device"] = infer_device

    try:
        pred = model.predict(
            source=str(image_path),
            conf=float(_clamp(conf, 0.01, 0.95)),
            classes=chair_ids,
            retina_masks=True,
            verbose=False,
            device=infer_device,
        )
    except Exception as exc:
        log["error"] = f"YOLO predict failed: {exc}"
        return None, None, log

    if not pred:
        log["error"] = "YOLO returned no predictions"
        return None, None, log
    res = pred[0]
    if res.masks is None or res.boxes is None or len(res.boxes) == 0:
        log["error"] = "YOLO found no chair masks"
        return None, None, log

    masks = res.masks.data.detach().cpu().numpy()
    boxes = res.boxes
    n = min(int(masks.shape[0]), int(len(boxes)))
    if n <= 0:
        log["error"] = "YOLO mask/box outputs are empty"
        return None, None, log

    image_h = int(masks.shape[1])
    image_w = int(masks.shape[2])

    best_idx = -1
    best_score = -1e9
    best_area = 0
    for i in range(n):
        mask_bin = masks[i] > 0.5
        area = int(mask_bin.sum())
        if area < int(0.002 * image_w * image_h):
            continue

        ys, xs = np.where(mask_bin)
        if len(xs) == 0:
            continue
        cx = float(xs.mean())
        cy = float(ys.mean())
        center_dist = ((cx - 0.5 * image_w) / image_w) ** 2 + (
            (cy - 0.5 * image_h) / image_h
        ) ** 2
        area_ratio = float(area) / float(image_w * image_h)
        conf_i = float(boxes.conf[i].item())

        score = conf_i + 0.5 * min(area_ratio / 0.25, 1.0) - 0.45 * center_dist
        if score > best_score:
            best_score = score
            best_idx = i
            best_area = area

    if best_idx < 0:
        log["error"] = "YOLO found masks but none passed area/center filters"
        return None, None, log

    mask = (masks[best_idx] > 0.5).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        log["error"] = "selected YOLO mask is empty"
        return None, None, log

    x0 = int(xs.min())
    x1 = int(xs.max())
    y0 = int(ys.min())
    y1 = int(ys.max())
    box_h = max(1, y1 - y0 + 1)

    row_w = np.zeros((box_h,), dtype=np.int32)
    row_min = np.full((box_h,), -1, dtype=np.int32)
    row_max = np.full((box_h,), -1, dtype=np.int32)
    for iy, y in enumerate(range(y0, y1 + 1)):
        cols = np.where(mask[y, :] > 0)[0]
        if len(cols) == 0:
            continue
        row_w[iy] = int(len(cols))
        row_min[iy] = int(cols.min())
        row_max[iy] = int(cols.max())

    mid_start = int(0.30 * box_h)
    mid_end = int(0.82 * box_h)
    if mid_end <= mid_start:
        mid_start = 0
        mid_end = box_h
    mid_slice = row_w[mid_start:mid_end]
    if mid_slice.size == 0:
        log["error"] = "failed to compute row width profile from mask"
        return None, None, log

    peak_rel = int(mid_start + int(np.argmax(mid_slice)))
    peak_w = max(1, int(row_w[peak_rel]))
    drop_threshold = max(5, int(0.55 * peak_w))
    drop_candidates = np.where(row_w[peak_rel:] < drop_threshold)[0]
    if drop_candidates.size > 0:
        drop_rel = int(peak_rel + int(drop_candidates[0]))
    else:
        drop_rel = int(_clamp(0.78 * box_h, 0.0, float(box_h - 1)))

    seat_bottom_rel = int(
        _clamp(
            float(drop_rel - int(0.02 * box_h)),
            float(int(0.46 * box_h)),
            float(int(0.90 * box_h)),
        )
    )
    seat_h = int(
        _clamp(
            float(int(0.15 * box_h)),
            float(max(6, int(0.07 * box_h))),
            float(max(10, int(0.24 * box_h))),
        )
    )
    seat_top_rel = max(int(0.18 * box_h), seat_bottom_rel - seat_h)
    if seat_top_rel >= seat_bottom_rel:
        seat_top_rel = max(0, seat_bottom_rel - max(6, int(0.07 * box_h)))

    def span_at(rel_idx: int) -> tuple[int, int]:
        idx = int(_clamp(float(rel_idx), 0.0, float(box_h - 1)))
        if row_min[idx] >= 0:
            return int(row_min[idx]), int(row_max[idx])
        radius = max(8, box_h // 6)
        for r in range(1, radius + 1):
            j0 = idx - r
            j1 = idx + r
            if j0 >= 0 and row_min[j0] >= 0:
                return int(row_min[j0]), int(row_max[j0])
            if j1 < box_h and row_min[j1] >= 0:
                return int(row_min[j1]), int(row_max[j1])
        return x0, x1

    seat_mid_rel = int((seat_top_rel + seat_bottom_rel) * 0.5)
    sx0, sx1 = span_at(seat_mid_rel)
    if sx1 <= sx0:
        sx0, sx1 = x0, x1
    seat_w = max(12, sx1 - sx0)
    inner_margin = max(2, int(0.11 * seat_w))
    sx0 = int(_clamp(float(sx0 + inner_margin), 0.0, float(image_w - 1)))
    sx1 = int(_clamp(float(sx1 - inner_margin), float(sx0 + 2), float(image_w - 1)))

    back_rel = max(int(0.08 * box_h), seat_top_rel - max(8, int(0.18 * box_h)))
    bx0, bx1 = span_at(back_rel)
    if bx1 <= bx0:
        bx0, bx1 = sx0, sx1
    back_margin = max(2, int(0.08 * max(1, bx1 - bx0)))
    bx0 = int(_clamp(float(bx0 + back_margin), 0.0, float(image_w - 1)))
    bx1 = int(_clamp(float(bx1 - back_margin), float(bx0 + 2), float(image_w - 1)))

    seat_top = int(y0 + seat_top_rel)
    seat_bottom = int(y0 + seat_bottom_rel)
    back_y = int(y0 + back_rel)

    seat_rect_xy = [
        (float(sx0), float(seat_top)),
        (float(sx1), float(seat_top)),
        (float(sx1), float(seat_bottom)),
        (float(sx0), float(seat_bottom)),
    ]
    back_cx = (float(bx0) + float(bx1)) * 0.5
    back_cy = float(back_y)
    best_idx = 0
    best_dist = float("inf")
    for i in range(4):
        ax, ay = seat_rect_xy[i]
        bx, by = seat_rect_xy[(i + 1) % 4]
        mx = (ax + bx) * 0.5
        my = (ay + by) * 0.5
        d = math.hypot(mx - back_cx, my - back_cy)
        if d < best_dist:
            best_dist = d
            best_idx = i
    back_edge_a = seat_rect_xy[best_idx]
    back_edge_b = seat_rect_xy[(best_idx + 1) % 4]
    front_edge_a = seat_rect_xy[(best_idx + 2) % 4]
    front_edge_b = seat_rect_xy[(best_idx + 3) % 4]
    back_mid = (
        (back_edge_a[0] + back_edge_b[0]) * 0.5,
        (back_edge_a[1] + back_edge_b[1]) * 0.5,
    )
    front_mid = (
        (front_edge_a[0] + front_edge_b[0]) * 0.5,
        (front_edge_a[1] + front_edge_b[1]) * 0.5,
    )
    depth_vx = front_mid[0] - back_mid[0]
    depth_vy = front_mid[1] - back_mid[1]
    depth_len2 = depth_vx * depth_vx + depth_vy * depth_vy
    depth_len = math.sqrt(max(depth_len2, 1e-8))
    depth_vertical_ratio = abs(depth_vy) / depth_len
    depth_len = math.sqrt(max(depth_len2, 1e-8))
    depth_vertical_ratio = abs(depth_vy) / depth_len

    floor_y = int(
        _clamp(
            float(y1 + int(0.06 * image_h)),
            float(seat_bottom + int(0.12 * image_h)),
            float(image_h - 1),
        )
    )
    floor_x0 = int(_clamp(float(sx0 - 0.65 * (sx1 - sx0)), 0.0, float(image_w - 1)))
    floor_x1 = int(
        _clamp(float(sx1 + 0.65 * (sx1 - sx0)), float(floor_x0 + 2), float(image_w - 1))
    )

    seat_contact_mask = np.zeros_like(mask, dtype=np.uint8)
    row_spans: list[tuple[int, int, int]] = []
    for y in range(seat_top, seat_bottom + 1):
        cols = np.where(mask[y, :] > 0)[0]
        if len(cols) < 4:
            continue
        left = int(np.percentile(cols, 1.5))
        right = int(np.percentile(cols, 98.5))
        if right <= left:
            continue
        row_wi = right - left + 1
        in_margin = max(1, int(0.025 * row_wi))
        left = int(_clamp(float(left + in_margin), 0.0, float(image_w - 1)))
        right = int(_clamp(float(right - in_margin), float(left + 2), float(image_w - 1)))
        seat_contact_mask[y, left : right + 1] = 1
        row_spans.append((y, left, right))

    if len(row_spans) < 8:
        seat_contact_mask[seat_top : seat_bottom + 1, sx0 : sx1 + 1] = 1
        row_spans = [(y, sx0, sx1) for y in range(seat_top, seat_bottom + 1)]

    cv2_mod = None
    try:
        import cv2 as cv2_mod  # type: ignore
    except Exception:
        cv2_mod = None

    if cv2_mod is not None:
        k = max(3, int(0.012 * box_h))
        if (k % 2) == 0:
            k += 1
        kernel = np.ones((k, k), dtype=np.uint8)
        seat_contact_mask = cv2_mod.morphologyEx(
            (seat_contact_mask * 255).astype(np.uint8),
            cv2_mod.MORPH_CLOSE,
            kernel,
            iterations=1,
        )
        seat_contact_mask = cv2_mod.morphologyEx(
            seat_contact_mask,
            cv2_mod.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )
        seat_contact_mask = (seat_contact_mask > 0).astype(np.uint8)

        num_labels, labels, stats, _ = cv2_mod.connectedComponentsWithStats(
            (seat_contact_mask * 255).astype(np.uint8), 8
        )
        if num_labels > 1:
            best_cc = 1
            best_cc_area = int(stats[1, cv2_mod.CC_STAT_AREA])
            for i in range(2, num_labels):
                area_i = int(stats[i, cv2_mod.CC_STAT_AREA])
                if area_i > best_cc_area:
                    best_cc_area = area_i
                    best_cc = i
            seat_contact_mask = (labels == best_cc).astype(np.uint8)

    # Remove the backrest-adjacent strip: this region is often not a true sit-contact area.
    depth_trim_applied = False
    if depth_vertical_ratio >= 0.75:
        trim_candidates = ((0.32, 0.86), (0.28, 0.9), (0.24, 0.92))
    elif depth_vertical_ratio >= 0.55:
        trim_candidates = ((0.26, 0.9), (0.22, 0.93), (0.18, 0.95))
    else:
        trim_candidates = ((0.2, 0.95), (0.16, 0.96))
    depth_trim_range = {
        "min_depth_t": round(float(trim_candidates[0][0]), 3),
        "max_depth_t": round(float(trim_candidates[0][1]), 3),
    }
    original_mask_area = int(np.sum(seat_contact_mask > 0))
    if original_mask_area > 0 and depth_len2 > 1e-8:
        ys, xs = np.where(seat_contact_mask > 0)
        if len(xs) > 0:
            xs_f = xs.astype(np.float32)
            ys_f = ys.astype(np.float32)
            depth_t = ((xs_f - back_mid[0]) * depth_vx + (ys_f - back_mid[1]) * depth_vy) / depth_len2

            min_keep_area = max(64, int(0.36 * original_mask_area))
            for min_t, max_t in trim_candidates:
                keep = (depth_t >= float(min_t)) & (depth_t <= float(max_t))
                kept = int(np.count_nonzero(keep))
                if kept < min_keep_area:
                    continue
                trimmed_mask = np.zeros_like(seat_contact_mask, dtype=np.uint8)
                trimmed_mask[ys[keep], xs[keep]] = 1
                seat_contact_mask = trimmed_mask
                depth_trim_applied = True
                depth_trim_range = {
                    "min_depth_t": round(float(min_t), 3),
                    "max_depth_t": round(float(max_t), 3),
                }
                break

    if depth_trim_applied and cv2_mod is not None:
        seat_contact_mask = cv2_mod.morphologyEx(
            (seat_contact_mask * 255).astype(np.uint8),
            cv2_mod.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )
        seat_contact_mask = (seat_contact_mask > 0).astype(np.uint8)

    poly_xy: list[tuple[float, float]] = []
    if cv2_mod is not None:
        contours, _ = cv2_mod.findContours(
            (seat_contact_mask * 255).astype(np.uint8),
            cv2_mod.RETR_EXTERNAL,
            cv2_mod.CHAIN_APPROX_NONE,
        )
        if contours:
            cnt = max(contours, key=cv2_mod.contourArea)
            peri = float(cv2_mod.arcLength(cnt, True))
            eps = max(1.0, 0.0045 * peri)
            approx = cv2_mod.approxPolyDP(cnt, eps, True)
            raw = approx if len(approx) >= 6 else cnt
            step_keep = max(1, len(raw) // 140)
            for i in range(0, len(raw), step_keep):
                px = float(raw[i][0][0])
                py = float(raw[i][0][1])
                poly_xy.append((px, py))

    if len(poly_xy) < 6:
        rows_l = [(float(y), float(l), float(r)) for y, l, r in row_spans]
        if rows_l:
            left_chain = [(l, y) for y, l, _ in rows_l]
            right_chain = [(r, y) for y, _, r in reversed(rows_l)]
            stride = max(1, len(left_chain) // 70)
            poly_xy = left_chain[::stride] + right_chain[::stride]

    if len(poly_xy) < 6:
        poly_xy = [
            (float(sx0), float(seat_top)),
            (float(sx1), float(seat_top)),
            (float(sx1), float(seat_bottom)),
            (float(sx0), float(seat_bottom)),
        ]

    cleaned_poly: list[tuple[float, float]] = []
    for px, py in poly_xy:
        cx = _clamp(float(px), 0.0, float(image_w - 1))
        cy = _clamp(float(py), 0.0, float(image_h - 1))
        if not cleaned_poly:
            cleaned_poly.append((cx, cy))
            continue
        if math.hypot(cleaned_poly[-1][0] - cx, cleaned_poly[-1][1] - cy) >= 1.25:
            cleaned_poly.append((cx, cy))
    if len(cleaned_poly) >= 3 and math.hypot(
        cleaned_poly[0][0] - cleaned_poly[-1][0],
        cleaned_poly[0][1] - cleaned_poly[-1][1],
    ) < 1.25:
        cleaned_poly = cleaned_poly[:-1]

    if len(cleaned_poly) < 3:
        cleaned_poly = [
            (float(sx0), float(seat_top)),
            (float(sx1), float(seat_top)),
            (float(sx1), float(seat_bottom)),
            (float(sx0), float(seat_bottom)),
        ]

    seat_poly_area = polygon_area(cleaned_poly)
    if seat_poly_area < max(20.0, 0.0012 * image_w * image_h):
        cleaned_poly = [
            (float(sx0), float(seat_top)),
            (float(sx1), float(seat_top)),
            (float(sx1), float(seat_bottom)),
            (float(sx0), float(seat_bottom)),
        ]
        seat_poly_area = polygon_area(cleaned_poly)

    geometry = {
        "image_width": image_w,
        "image_height": image_h,
        "seat_region": [
            {"x": float(seat_rect_xy[0][0]), "y": float(seat_rect_xy[0][1])},
            {"x": float(seat_rect_xy[1][0]), "y": float(seat_rect_xy[1][1])},
            {"x": float(seat_rect_xy[2][0]), "y": float(seat_rect_xy[2][1])},
            {"x": float(seat_rect_xy[3][0]), "y": float(seat_rect_xy[3][1])},
        ],
        "floor_line": [
            {"x": float(floor_x0), "y": float(floor_y)},
            {"x": float(floor_x1), "y": float(floor_y)},
        ],
        "backrest_region": [
            {"x": float(bx0), "y": float(back_y)},
            {"x": float(bx1), "y": float(back_y)},
        ],
    }
    geometry_s, sanitize_notes = sanitize_geometry_candidate(geometry)
    if geometry_s is None:
        log["error"] = "YOLO geometry sanitize failed"
        log["sanitize_notes"] = sanitize_notes
        return None, None, log

    prior_issues = geometry_sitting_prior_issues(geometry_s)
    if prior_issues:
        repaired, repair_notes = repair_geometry_with_sitting_priors(geometry_s)
        if repaired is not None:
            geometry_s = repaired
            sanitize_notes.extend(repair_notes)
            prior_issues = []
        else:
            log["error"] = "YOLO geometry failed sitting priors"
            log["prior_issues"] = prior_issues
            return None, None, log

    seat_contact_region = [
        {"x": round(float(px), 2), "y": round(float(py), 2)} for px, py in cleaned_poly
    ]
    geometry_s["seat_contact_region"] = seat_contact_region

    seat_segmentation: dict[str, Any] = {
        "image_width": image_w,
        "image_height": image_h,
        "source": "yolo_chair_mask_band_depthtrim_v2",
        "seat_band": {"y_top": int(seat_top), "y_bottom": int(seat_bottom)},
        "seat_contact_polygon": seat_contact_region,
        "polygon_area": round(float(seat_poly_area), 2),
        "depth_trim": depth_trim_range,
        "depth_vertical_ratio": round(float(depth_vertical_ratio), 4),
        "confidence": round(float(boxes.conf[best_idx].item()), 4),
    }

    log.update(
        {
            "ok": True,
            "notes": sanitize_notes,
            "selected_mask_index": best_idx,
            "selected_mask_area": int(best_area),
            "selected_score": round(float(best_score), 5),
            "selected_bbox": {"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1},
            "seat_polygon_points": len(seat_contact_region),
        }
    )
    return geometry_s, seat_segmentation, log


def call_gemini_json(
    *,
    api_key: str,
    model: str,
    prompt_text: str,
    image_mime: str | None,
    image_b64: str | None,
    temperature: float,
    max_output_tokens: int,
    timeout_sec: float,
) -> str:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
        f":generateContent?key={api_key}"
    )
    parts: list[dict[str, Any]] = [{"text": prompt_text}]
    if image_mime and image_b64:
        parts.append(
            {
                "inline_data": {
                    "mime_type": image_mime,
                    "data": image_b64,
                }
            }
        )

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }

    req = request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout_sec) as resp:
            raw_body = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        if len(detail) > 600:
            detail = detail[:600] + "..."
        raise RuntimeError(f"Gemini API error {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Gemini API connection error: {exc}") from exc

    body = json.loads(raw_body)
    candidates = body.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini response has no candidates")

    parts = candidates[0].get("content", {}).get("parts", [])
    texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
    text = "\n".join(t for t in texts if t).strip()
    if not text:
        raise RuntimeError("Gemini response does not contain text")

    return text


def call_ollama_json(
    *,
    ollama_host: str,
    model: str,
    prompt_text: str,
    image_b64: str | None,
    response_schema: dict[str, Any] | None,
    temperature: float,
    max_output_tokens: int,
    timeout_sec: float,
) -> str:
    url = ollama_host.rstrip("/") + "/api/chat"
    message: dict[str, Any] = {
        "role": "user",
        "content": prompt_text,
    }
    if image_b64:
        message["images"] = [image_b64]

    payload = {
        "model": model,
        "stream": False,
        "messages": [message],
        "options": {
            "temperature": temperature,
            # Ollama's equivalent of max output tokens
            "num_predict": max_output_tokens,
        },
    }
    if response_schema is not None:
        payload["format"] = response_schema
    else:
        payload["format"] = "json"

    req = request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout_sec) as resp:
            raw_body = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        if len(detail) > 600:
            detail = detail[:600] + "..."
        raise RuntimeError(f"Ollama API error {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Ollama API connection error: {exc}") from exc

    body = json.loads(raw_body)
    message = body.get("message") or {}
    text = message.get("content", "")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Ollama response does not contain message.content")

    return text.strip()


def strip_code_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\\s*```$", "", value)
    return value.strip()


def extract_json_object(raw_text: str) -> dict[str, Any]:
    text = strip_code_fence(raw_text)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    candidates: list[str] = []
    depth = 0
    start = None
    in_string = False
    escape = False

    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
            continue

        if ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start : idx + 1])
                start = None

    for candidate in reversed(candidates):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    raise ValueError("Could not parse valid JSON object from model response")


def validate_point(
    point: Any,
    *,
    label: str,
    width: int,
    height: int,
    errors: list[str],
) -> None:
    if not isinstance(point, dict):
        errors.append(f"{label} must be an object")
        return

    x = point.get("x")
    y = point.get("y")

    if not is_number(x) or not is_number(y):
        errors.append(f"{label}.x and {label}.y must be numeric")
        return

    if x < 0 or x > (width - 1):
        errors.append(f"{label}.x out of bounds: {x}")
    if y < 0 or y > (height - 1):
        errors.append(f"{label}.y out of bounds: {y}")


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def validate_geometry(data: Any) -> list[str]:
    errors: list[str] = []

    if not isinstance(data, dict):
        return ["Geometry output must be an object"]

    width = data.get("image_width")
    height = data.get("image_height")

    if not isinstance(width, int) or width <= 0:
        errors.append("image_width must be a positive integer")
    if not isinstance(height, int) or height <= 0:
        errors.append("image_height must be a positive integer")

    if errors:
        return errors

    seat = data.get("seat_region")
    floor = data.get("floor_line")
    backrest = data.get("backrest_region")

    if not isinstance(seat, list) or len(seat) != 4:
        errors.append("seat_region must be a list of 4 points")
    else:
        for idx, point in enumerate(seat):
            validate_point(
                point,
                label=f"seat_region[{idx}]",
                width=width,
                height=height,
                errors=errors,
            )

    if not isinstance(floor, list) or len(floor) != 2:
        errors.append("floor_line must be a list of 2 points")
    else:
        for idx, point in enumerate(floor):
            validate_point(
                point,
                label=f"floor_line[{idx}]",
                width=width,
                height=height,
                errors=errors,
            )

    if not isinstance(backrest, list) or len(backrest) != 2:
        errors.append("backrest_region must be a list of 2 points")
    else:
        for idx, point in enumerate(backrest):
            validate_point(
                point,
                label=f"backrest_region[{idx}]",
                width=width,
                height=height,
                errors=errors,
            )

    # Reject degenerate geometry that often breaks downstream Stage B.
    if isinstance(seat, list) and len(seat) == 4 and not errors:
        seat_poly = [(float(p["x"]), float(p["y"])) for p in seat]
        seat_area = polygon_area(seat_poly)
        min_area = max(20.0, 0.002 * width * height)
        if seat_area < min_area:
            errors.append(
                f"seat_region is degenerate (area={seat_area:.2f} < {min_area:.2f})"
            )

    if isinstance(floor, list) and len(floor) == 2 and not errors:
        dx = float(floor[0]["x"]) - float(floor[1]["x"])
        dy = float(floor[0]["y"]) - float(floor[1]["y"])
        floor_len = math.hypot(dx, dy)
        min_len = max(10.0, 0.05 * width)
        if floor_len < min_len:
            errors.append(
                f"floor_line too short (len={floor_len:.2f} < {min_len:.2f})"
            )

    if isinstance(backrest, list) and len(backrest) == 2 and not errors:
        dx = float(backrest[0]["x"]) - float(backrest[1]["x"])
        dy = float(backrest[0]["y"]) - float(backrest[1]["y"])
        back_len = math.hypot(dx, dy)
        min_len = max(10.0, 0.03 * width)
        if back_len < min_len:
            errors.append(
                f"backrest_region too short (len={back_len:.2f} < {min_len:.2f})"
            )

    return errors


def validate_part_segmentation(data: Any) -> list[str]:
    errors: list[str] = []

    if not isinstance(data, dict):
        return ["Part segmentation output must be an object"]

    width = data.get("image_width")
    height = data.get("image_height")
    if not isinstance(width, int) or width <= 0:
        errors.append("image_width must be a positive integer")
    if not isinstance(height, int) or height <= 0:
        errors.append("image_height must be a positive integer")

    seat = data.get("seat_region")
    floor = data.get("floor_line")
    backrest = data.get("backrest_region")

    if not isinstance(seat, list) or len(seat) != 4:
        errors.append("seat_region must be a list of 4 points")
    if not isinstance(floor, list) or len(floor) != 2:
        errors.append("floor_line must be a list of 2 points")
    if not isinstance(backrest, list) or len(backrest) != 2:
        errors.append("backrest_region must be a list of 2 points")

    if isinstance(width, int) and width > 0 and isinstance(height, int) and height > 0:
        if isinstance(seat, list) and len(seat) == 4:
            for idx, point in enumerate(seat):
                validate_point(
                    point,
                    label=f"seat_region[{idx}]",
                    width=width,
                    height=height,
                    errors=errors,
                )
        if isinstance(floor, list) and len(floor) == 2:
            for idx, point in enumerate(floor):
                validate_point(
                    point,
                    label=f"floor_line[{idx}]",
                    width=width,
                    height=height,
                    errors=errors,
                )
        if isinstance(backrest, list) and len(backrest) == 2:
            for idx, point in enumerate(backrest):
                validate_point(
                    point,
                    label=f"backrest_region[{idx}]",
                    width=width,
                    height=height,
                    errors=errors,
                )

    geom_errors = validate_geometry(
        {
            "image_width": width,
            "image_height": height,
            "seat_region": seat,
            "floor_line": floor,
            "backrest_region": backrest,
        }
    )
    for err in geom_errors:
        if err not in errors:
            errors.append(err)

    parts = data.get("parts")
    if not isinstance(parts, dict):
        errors.append("parts must be an object")
        return errors

    for key in (
        "seat_confidence",
        "backrest_confidence",
        "floor_confidence",
    ):
        value = parts.get(key)
        if not is_number(value):
            errors.append(f"parts.{key} must be numeric")
        elif value < 0 or value > 1:
            errors.append(f"parts.{key} must be in [0,1]")

    for key in ("seat_visible", "backrest_visible", "floor_visible"):
        value = parts.get(key)
        if not isinstance(value, bool):
            errors.append(f"parts.{key} must be boolean")

    def validate_line_items(name: str, max_items: int) -> None:
        raw_items = parts.get(name, [])
        if raw_items is None:
            return
        if not isinstance(raw_items, list):
            errors.append(f"parts.{name} must be a list")
            return
        if len(raw_items) > max_items:
            errors.append(f"parts.{name} has too many items (>{max_items})")
        for idx, item in enumerate(raw_items):
            if not isinstance(item, dict):
                errors.append(f"parts.{name}[{idx}] must be an object")
                continue
            line = item.get("line")
            if not isinstance(line, list) or len(line) != 2:
                errors.append(f"parts.{name}[{idx}].line must be a list of 2 points")
            elif isinstance(width, int) and width > 0 and isinstance(height, int) and height > 0:
                for p_idx, point in enumerate(line):
                    validate_point(
                        point,
                        label=f"parts.{name}[{idx}].line[{p_idx}]",
                        width=width,
                        height=height,
                        errors=errors,
                    )

            conf = item.get("confidence")
            if not is_number(conf):
                errors.append(f"parts.{name}[{idx}].confidence must be numeric")
            elif conf < 0 or conf > 1:
                errors.append(f"parts.{name}[{idx}].confidence must be in [0,1]")

            vis = item.get("visible")
            if not isinstance(vis, bool):
                errors.append(f"parts.{name}[{idx}].visible must be boolean")

    validate_line_items("armrests", max_items=2)
    validate_line_items("legs", max_items=8)

    return errors


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _scale_xy(
    x: float,
    y: float,
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> tuple[float, float]:
    if src_w <= 1 or src_h <= 1 or dst_w <= 1 or dst_h <= 1:
        return float(x), float(y)

    sx = (dst_w - 1) / float(src_w - 1)
    sy = (dst_h - 1) / float(src_h - 1)
    nx = _clamp(float(x) * sx, 0.0, float(dst_w - 1))
    ny = _clamp(float(y) * sy, 0.0, float(dst_h - 1))
    return nx, ny


def rescale_geometry(
    geometry: dict[str, Any], *, target_width: int, target_height: int
) -> dict[str, Any]:
    src_w = int(geometry.get("image_width", 0))
    src_h = int(geometry.get("image_height", 0))
    if src_w <= 0 or src_h <= 0:
        return geometry

    if src_w == target_width and src_h == target_height:
        return geometry

    out = {
        "image_width": int(target_width),
        "image_height": int(target_height),
    }
    for key, expected_len in (
        ("seat_region", 4),
        ("floor_line", 2),
        ("backrest_region", 2),
    ):
        pts = geometry.get(key)
        if not isinstance(pts, list) or len(pts) != expected_len:
            out[key] = pts
            continue
        out_pts: list[dict[str, float]] = []
        for p in pts:
            x = float(p.get("x", 0.0))
            y = float(p.get("y", 0.0))
            nx, ny = _scale_xy(x, y, src_w, src_h, target_width, target_height)
            out_pts.append({"x": round(nx, 2), "y": round(ny, 2)})
        out[key] = out_pts

    contact = geometry.get("seat_contact_region")
    if isinstance(contact, list) and len(contact) >= 3:
        out_contact: list[dict[str, float]] = []
        for p in contact:
            if not isinstance(p, dict):
                continue
            x = p.get("x")
            y = p.get("y")
            if not is_number(x) or not is_number(y):
                continue
            nx, ny = _scale_xy(
                float(x), float(y), src_w, src_h, target_width, target_height
            )
            out_contact.append({"x": round(nx, 2), "y": round(ny, 2)})
        if len(out_contact) >= 3:
            out["seat_contact_region"] = out_contact
    return out


def rescale_part_segmentation(
    part_segmentation: dict[str, Any], *, target_width: int, target_height: int
) -> dict[str, Any]:
    src_w = int(part_segmentation.get("image_width", 0))
    src_h = int(part_segmentation.get("image_height", 0))
    if src_w <= 0 or src_h <= 0:
        return part_segmentation

    if src_w == target_width and src_h == target_height:
        return part_segmentation

    out = {
        "image_width": int(target_width),
        "image_height": int(target_height),
    }
    for key, expected_len in (
        ("seat_region", 4),
        ("floor_line", 2),
        ("backrest_region", 2),
    ):
        pts = part_segmentation.get(key)
        if not isinstance(pts, list) or len(pts) != expected_len:
            out[key] = pts
            continue
        out_pts: list[dict[str, float]] = []
        for p in pts:
            x = float(p.get("x", 0.0))
            y = float(p.get("y", 0.0))
            nx, ny = _scale_xy(x, y, src_w, src_h, target_width, target_height)
            out_pts.append({"x": round(nx, 2), "y": round(ny, 2)})
        out[key] = out_pts

    parts_raw = part_segmentation.get("parts")
    if not isinstance(parts_raw, dict):
        out["parts"] = parts_raw
        return out

    parts_out: dict[str, Any] = {}
    for key in ("seat_confidence", "backrest_confidence", "floor_confidence"):
        val = parts_raw.get(key)
        parts_out[key] = float(_clamp(float(val), 0.0, 1.0)) if is_number(val) else 0.0
    for key in ("seat_visible", "backrest_visible", "floor_visible"):
        parts_out[key] = bool(parts_raw.get(key, False))

    def scale_line_items(name: str, max_items: int) -> list[dict[str, Any]]:
        items = parts_raw.get(name, [])
        if not isinstance(items, list):
            return []
        scaled_items: list[dict[str, Any]] = []
        for item in items[:max_items]:
            if not isinstance(item, dict):
                continue
            line = item.get("line")
            if not isinstance(line, list) or len(line) != 2:
                continue
            scaled_line: list[dict[str, float]] = []
            ok = True
            for p in line:
                if not isinstance(p, dict):
                    ok = False
                    break
                x = p.get("x")
                y = p.get("y")
                if not is_number(x) or not is_number(y):
                    ok = False
                    break
                nx, ny = _scale_xy(
                    float(x), float(y), src_w, src_h, target_width, target_height
                )
                scaled_line.append({"x": round(nx, 2), "y": round(ny, 2)})
            if not ok:
                continue
            conf = item.get("confidence")
            scaled_items.append(
                {
                    "line": scaled_line,
                    "confidence": float(_clamp(float(conf), 0.0, 1.0))
                    if is_number(conf)
                    else 0.0,
                    "visible": bool(item.get("visible", False)),
                }
            )
        return scaled_items

    parts_out["armrests"] = scale_line_items("armrests", max_items=2)
    parts_out["legs"] = scale_line_items("legs", max_items=8)
    out["parts"] = parts_out
    return out


def denormalize_part_segmentation_candidate(
    candidate: Any, *, target_width: int, target_height: int
) -> tuple[Any, bool]:
    if not isinstance(candidate, dict):
        return candidate, False

    src_w = candidate.get("image_width")
    src_h = candidate.get("image_height")
    if not isinstance(src_w, int) or not isinstance(src_h, int):
        return candidate, False
    if src_w <= 0 or src_h <= 0:
        return candidate, False

    seat = candidate.get("seat_region")
    floor = candidate.get("floor_line")
    back = candidate.get("backrest_region")
    if not isinstance(seat, list) or len(seat) != 4:
        return candidate, False
    if not isinstance(floor, list) or len(floor) != 2:
        return candidate, False
    if not isinstance(back, list) or len(back) != 2:
        return candidate, False

    pts: list[tuple[float, float]] = []
    for arr in (seat, floor, back):
        for p in arr:
            if not isinstance(p, dict):
                return candidate, False
            x = p.get("x")
            y = p.get("y")
            if not is_number(x) or not is_number(y):
                return candidate, False
            pts.append((float(x), float(y)))

    parts = candidate.get("parts")
    if isinstance(parts, dict):
        for key, limit in (("armrests", 2), ("legs", 8)):
            arr = parts.get(key)
            if not isinstance(arr, list):
                continue
            for item in arr[:limit]:
                if not isinstance(item, dict):
                    continue
                line = item.get("line")
                if not isinstance(line, list) or len(line) != 2:
                    continue
                for p in line:
                    if not isinstance(p, dict):
                        continue
                    x = p.get("x")
                    y = p.get("y")
                    if is_number(x) and is_number(y):
                        pts.append((float(x), float(y)))

    if not pts:
        return candidate, False

    max_x = max(p[0] for p in pts)
    max_y = max(p[1] for p in pts)
    eff_w = max(int(src_w), int(math.ceil(max_x)) + 1)
    eff_h = max(int(src_h), int(math.ceil(max_y)) + 1)

    # Already in target pixel frame.
    if abs(eff_w - int(target_width)) <= 2 and abs(eff_h - int(target_height)) <= 2:
        return candidate, False

    # Heuristic: only denormalize likely compact grids (0..100/200 style).
    if eff_w > 500 or eff_h > 500:
        return candidate, False

    scaled = {
        "image_width": int(target_width),
        "image_height": int(target_height),
    }
    for key in ("seat_region", "floor_line", "backrest_region"):
        out_pts: list[dict[str, float]] = []
        for p in candidate[key]:
            nx, ny = _scale_xy(
                float(p["x"]),
                float(p["y"]),
                eff_w,
                eff_h,
                int(target_width),
                int(target_height),
            )
            out_pts.append({"x": round(nx, 2), "y": round(ny, 2)})
        scaled[key] = out_pts

    parts_out: dict[str, Any] = {}
    if isinstance(parts, dict):
        for conf_key in ("seat_confidence", "backrest_confidence", "floor_confidence"):
            value = parts.get(conf_key)
            parts_out[conf_key] = (
                float(_clamp(float(value), 0.0, 1.0)) if is_number(value) else 0.0
            )
        for vis_key in ("seat_visible", "backrest_visible", "floor_visible"):
            parts_out[vis_key] = bool(parts.get(vis_key, False))

        def scale_lines(name: str, max_items: int) -> list[dict[str, Any]]:
            raw = parts.get(name, [])
            if not isinstance(raw, list):
                return []
            out: list[dict[str, Any]] = []
            for item in raw[:max_items]:
                if not isinstance(item, dict):
                    continue
                line = item.get("line")
                if not isinstance(line, list) or len(line) != 2:
                    continue
                p0 = line[0] if isinstance(line[0], dict) else None
                p1 = line[1] if isinstance(line[1], dict) else None
                if p0 is None or p1 is None:
                    continue
                x0 = p0.get("x")
                y0 = p0.get("y")
                x1 = p1.get("x")
                y1 = p1.get("y")
                if not (
                    is_number(x0)
                    and is_number(y0)
                    and is_number(x1)
                    and is_number(y1)
                ):
                    continue
                n0x, n0y = _scale_xy(
                    float(x0), float(y0), eff_w, eff_h, int(target_width), int(target_height)
                )
                n1x, n1y = _scale_xy(
                    float(x1), float(y1), eff_w, eff_h, int(target_width), int(target_height)
                )
                conf = item.get("confidence")
                out.append(
                    {
                        "line": [
                            {"x": round(n0x, 2), "y": round(n0y, 2)},
                            {"x": round(n1x, 2), "y": round(n1y, 2)},
                        ],
                        "confidence": float(_clamp(float(conf), 0.0, 1.0))
                        if is_number(conf)
                        else 0.0,
                        "visible": bool(item.get("visible", False)),
                    }
                )
            return out

        parts_out["armrests"] = scale_lines("armrests", max_items=2)
        parts_out["legs"] = scale_lines("legs", max_items=8)
    else:
        parts_out = {
            "seat_confidence": 0.0,
            "backrest_confidence": 0.0,
            "floor_confidence": 0.0,
            "seat_visible": False,
            "backrest_visible": False,
            "floor_visible": False,
            "armrests": [],
            "legs": [],
        }

    scaled["parts"] = parts_out
    return scaled, True


def denormalize_geometry_candidate(
    candidate: Any, *, target_width: int, target_height: int
) -> tuple[Any, bool]:
    if not isinstance(candidate, dict):
        return candidate, False

    src_w = candidate.get("image_width")
    src_h = candidate.get("image_height")
    if not isinstance(src_w, int) or not isinstance(src_h, int):
        return candidate, False
    if src_w <= 0 or src_h <= 0 or src_w > 200 or src_h > 200:
        return candidate, False

    keys = ("seat_region", "floor_line", "backrest_region")
    pts: list[tuple[float, float]] = []
    for key, expected_len in (("seat_region", 4), ("floor_line", 2), ("backrest_region", 2)):
        arr = candidate.get(key)
        if not isinstance(arr, list) or len(arr) != expected_len:
            return candidate, False
        for p in arr:
            if not isinstance(p, dict):
                return candidate, False
            x = p.get("x")
            y = p.get("y")
            if not is_number(x) or not is_number(y):
                return candidate, False
            pts.append((float(x), float(y)))

    if not pts:
        return candidate, False

    max_x = max(p[0] for p in pts)
    max_y = max(p[1] for p in pts)
    if max_x > src_w + 1 or max_y > src_h + 1:
        return candidate, False

    scaled = {
        "image_width": int(target_width),
        "image_height": int(target_height),
    }
    for key in keys:
        scaled_pts: list[dict[str, float]] = []
        for p in candidate[key]:
            nx, ny = _scale_xy(
                float(p["x"]),
                float(p["y"]),
                int(src_w),
                int(src_h),
                int(target_width),
                int(target_height),
            )
            scaled_pts.append({"x": round(nx, 2), "y": round(ny, 2)})
        scaled[key] = scaled_pts

    return scaled, True


def geometry_sitting_prior_issues(geometry: Any) -> list[str]:
    if not isinstance(geometry, dict):
        return ["geometry is not an object"]
    width = geometry.get("image_width")
    height = geometry.get("image_height")
    seat = geometry.get("seat_region")
    floor = geometry.get("floor_line")
    back = geometry.get("backrest_region")
    if (
        not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
        or not isinstance(seat, list)
        or len(seat) != 4
        or not isinstance(floor, list)
        or len(floor) != 2
        or not isinstance(back, list)
        or len(back) != 2
    ):
        return ["geometry fields are incomplete"]

    h = float(height)
    seat_top = min(float(p["y"]) for p in seat)
    seat_bottom = max(float(p["y"]) for p in seat)
    floor_y = (float(floor[0]["y"]) + float(floor[1]["y"])) * 0.5
    back_y = (float(back[0]["y"]) + float(back[1]["y"])) * 0.5
    seat_h = max(1.0, seat_bottom - seat_top)

    issues: list[str] = []
    if seat_h < 0.04 * h:
        issues.append("seat height too small for a usable seat")
    if seat_h > 0.28 * h:
        issues.append("seat height too large for a realistic seat slab")
    if seat_top > 0.74 * h:
        issues.append("seat top is unrealistically close to bottom of image")
    if seat_bottom > 0.90 * h:
        issues.append("seat bottom is unrealistically low")
    if floor_y <= seat_bottom + 0.08 * h:
        issues.append("floor line is not sufficiently below seat")
    if floor_y < 0.62 * h:
        issues.append("floor line is unrealistically high")
    if back_y > seat_top + 0.06 * h:
        issues.append("backrest is not above seat region")
    return issues


def sanitize_geometry_candidate(data: Any) -> tuple[dict[str, Any] | None, list[str]]:
    notes: list[str] = []
    if not isinstance(data, dict):
        return None, notes

    width = data.get("image_width")
    height = data.get("image_height")
    if not isinstance(width, int) or width <= 0 or not isinstance(height, int) or height <= 0:
        return None, notes

    x_max = float(width - 1)
    y_max = float(height - 1)

    seat = data.get("seat_region")
    floor = data.get("floor_line")
    back = data.get("backrest_region")
    if not isinstance(seat, list) or len(seat) != 4:
        return None, notes
    if not isinstance(floor, list) or len(floor) != 2:
        return None, notes
    if not isinstance(back, list) or len(back) != 2:
        return None, notes

    def clamp_point(p: dict[str, Any]) -> dict[str, float]:
        x = float(p.get("x", 0.0))
        y = float(p.get("y", 0.0))
        cx = _clamp(x, 0.0, x_max)
        cy = _clamp(y, 0.0, y_max)
        if cx != x or cy != y:
            notes.append("clamped out-of-bounds geometry point(s)")
        return {"x": cx, "y": cy}

    seat_c = [clamp_point(p) for p in seat if isinstance(p, dict)]
    floor_c = [clamp_point(p) for p in floor if isinstance(p, dict)]
    back_c = [clamp_point(p) for p in back if isinstance(p, dict)]
    if len(seat_c) != 4 or len(floor_c) != 2 or len(back_c) != 2:
        return None, notes

    seat_poly = [(p["x"], p["y"]) for p in seat_c]
    seat_area = polygon_area(seat_poly)
    min_area = max(20.0, 0.002 * width * height)
    if seat_area < min_area:
        cx = sum(p["x"] for p in seat_c) / 4.0
        cy = sum(p["y"] for p in seat_c) / 4.0
        half_w = max(0.08 * width, 24.0)
        half_h = max(0.05 * height, 16.0)
        seat_c = [
            {"x": _clamp(cx - half_w, 0.0, x_max), "y": _clamp(cy - half_h, 0.0, y_max)},
            {"x": _clamp(cx + half_w, 0.0, x_max), "y": _clamp(cy - half_h, 0.0, y_max)},
            {"x": _clamp(cx + half_w, 0.0, x_max), "y": _clamp(cy + half_h, 0.0, y_max)},
            {"x": _clamp(cx - half_w, 0.0, x_max), "y": _clamp(cy + half_h, 0.0, y_max)},
        ]
        notes.append("repaired degenerate seat_region")

    dx = floor_c[0]["x"] - floor_c[1]["x"]
    dy = floor_c[0]["y"] - floor_c[1]["y"]
    if math.hypot(dx, dy) < max(10.0, 0.05 * width):
        y = _clamp(max(floor_c[0]["y"], floor_c[1]["y"]), 0.75 * height, y_max)
        floor_c = [
            {"x": _clamp(0.15 * width, 0.0, x_max), "y": y},
            {"x": _clamp(0.85 * width, 0.0, x_max), "y": y},
        ]
        notes.append("repaired short floor_line")

    dx = back_c[0]["x"] - back_c[1]["x"]
    dy = back_c[0]["y"] - back_c[1]["y"]
    if math.hypot(dx, dy) < max(10.0, 0.03 * width):
        y = _clamp(min(back_c[0]["y"], back_c[1]["y"]), 0.15 * height, 0.6 * height)
        back_c = [
            {"x": _clamp(0.25 * width, 0.0, x_max), "y": y},
            {"x": _clamp(0.75 * width, 0.0, x_max), "y": y},
        ]
        notes.append("repaired short backrest_region")

    sanitized = {
        "image_width": width,
        "image_height": height,
        "seat_region": seat_c,
        "floor_line": floor_c,
        "backrest_region": back_c,
    }
    if validate_geometry(sanitized):
        return None, notes
    return sanitized, notes


def sanitize_part_segmentation_candidate(
    data: Any,
) -> tuple[dict[str, Any] | None, list[str]]:
    notes: list[str] = []
    if not isinstance(data, dict):
        return None, notes

    geometry, geometry_notes = sanitize_geometry_candidate(data)
    if geometry is None:
        return None, notes
    notes.extend(geometry_notes)

    width = int(geometry["image_width"])
    height = int(geometry["image_height"])
    x_max = float(width - 1)
    y_max = float(height - 1)

    parts_raw = data.get("parts")
    if not isinstance(parts_raw, dict):
        parts_raw = {}
        notes.append("parts metadata missing, filled defaults")

    def conf_value(name: str, default: float) -> float:
        raw = parts_raw.get(name, default)
        if not is_number(raw):
            notes.append(f"defaulted parts.{name}")
            return default
        clamped = float(_clamp(float(raw), 0.0, 1.0))
        if clamped != float(raw):
            notes.append(f"clamped parts.{name} into [0,1]")
        return clamped

    def bool_value(name: str, default: bool) -> bool:
        raw = parts_raw.get(name, default)
        if isinstance(raw, bool):
            return raw
        notes.append(f"defaulted parts.{name}")
        return default

    def clamp_point_dict(point: Any, *, label: str) -> dict[str, float] | None:
        if not isinstance(point, dict):
            notes.append(f"dropped invalid point in {label}")
            return None
        x = point.get("x")
        y = point.get("y")
        if not is_number(x) or not is_number(y):
            notes.append(f"dropped non-numeric point in {label}")
            return None
        cx = _clamp(float(x), 0.0, x_max)
        cy = _clamp(float(y), 0.0, y_max)
        if cx != float(x) or cy != float(y):
            notes.append(f"clamped out-of-bounds point in {label}")
        return {"x": round(cx, 2), "y": round(cy, 2)}

    def sanitize_line_items(name: str, max_items: int) -> list[dict[str, Any]]:
        raw_items = parts_raw.get(name, [])
        if raw_items is None:
            return []
        if not isinstance(raw_items, list):
            notes.append(f"parts.{name} must be a list; dropped")
            return []
        out_items: list[dict[str, Any]] = []
        for idx, item in enumerate(raw_items[:max_items]):
            if not isinstance(item, dict):
                notes.append(f"dropped invalid parts.{name}[{idx}]")
                continue
            line = item.get("line")
            if not isinstance(line, list) or len(line) != 2:
                notes.append(f"dropped parts.{name}[{idx}] with invalid line")
                continue
            p0 = clamp_point_dict(line[0], label=f"parts.{name}[{idx}].line[0]")
            p1 = clamp_point_dict(line[1], label=f"parts.{name}[{idx}].line[1]")
            if p0 is None or p1 is None:
                notes.append(f"dropped parts.{name}[{idx}] due to point parse failure")
                continue
            raw_conf = item.get("confidence", 0.35)
            if not is_number(raw_conf):
                raw_conf = 0.35
                notes.append(f"defaulted parts.{name}[{idx}].confidence")
            conf = float(_clamp(float(raw_conf), 0.0, 1.0))
            if conf != float(raw_conf):
                notes.append(f"clamped parts.{name}[{idx}].confidence into [0,1]")

            raw_visible = item.get("visible", False)
            if not isinstance(raw_visible, bool):
                notes.append(f"defaulted parts.{name}[{idx}].visible")
                raw_visible = False
            out_items.append(
                {
                    "line": [p0, p1],
                    "confidence": round(conf, 4),
                    "visible": raw_visible,
                }
            )
        if len(raw_items) > max_items:
            notes.append(f"truncated parts.{name} to {max_items} items")
        return out_items

    parts = {
        "seat_confidence": round(conf_value("seat_confidence", 0.5), 4),
        "backrest_confidence": round(conf_value("backrest_confidence", 0.5), 4),
        "floor_confidence": round(conf_value("floor_confidence", 0.5), 4),
        "seat_visible": bool_value("seat_visible", True),
        "backrest_visible": bool_value("backrest_visible", True),
        "floor_visible": bool_value("floor_visible", True),
        "armrests": sanitize_line_items("armrests", max_items=2),
        "legs": sanitize_line_items("legs", max_items=8),
    }

    sanitized = {
        "image_width": width,
        "image_height": height,
        "seat_region": geometry["seat_region"],
        "floor_line": geometry["floor_line"],
        "backrest_region": geometry["backrest_region"],
        "parts": parts,
    }
    if validate_part_segmentation(sanitized):
        return None, notes
    return sanitized, notes


def part_segmentation_quality_issues(
    part_segmentation: Any, *, min_confidence: float
) -> list[str]:
    if not isinstance(part_segmentation, dict):
        return ["part segmentation is missing"]

    parts = part_segmentation.get("parts")
    if not isinstance(parts, dict):
        return ["part segmentation parts metadata is missing"]

    issues: list[str] = []
    required_conf = ("seat_confidence", "floor_confidence")
    optional_conf = ("backrest_confidence",)
    for key in required_conf:
        value = parts.get(key)
        if not is_number(value):
            issues.append(f"{key} missing")
            continue
        if float(value) < min_confidence:
            issues.append(
                f"{key}={float(value):.3f} below threshold {float(min_confidence):.3f}"
            )

    for key in ("seat_visible", "floor_visible"):
        value = parts.get(key)
        if value is not True:
            issues.append(f"{key} is false")

    for key in optional_conf:
        value = parts.get(key)
        if is_number(value) and float(value) < max(0.05, min_confidence * 0.5):
            issues.append(f"{key} too low for stable torso anchoring ({float(value):.3f})")

    return issues


def repair_geometry_with_sitting_priors(
    geometry: Any,
) -> tuple[dict[str, Any] | None, list[str]]:
    notes: list[str] = []
    sanitized, sanitize_notes = sanitize_geometry_candidate(geometry)
    notes.extend(sanitize_notes)
    if sanitized is None:
        return None, notes

    issues = geometry_sitting_prior_issues(sanitized)
    if not issues:
        return sanitized, notes

    width = float(sanitized["image_width"])
    height = float(sanitized["image_height"])
    x_max = max(0.0, width - 1.0)
    y_max = max(0.0, height - 1.0)

    seat = [
        {"x": float(p["x"]), "y": float(p["y"])}
        for p in sanitized["seat_region"]
    ]
    floor = [
        {"x": float(p["x"]), "y": float(p["y"])}
        for p in sanitized["floor_line"]
    ]
    back = [
        {"x": float(p["x"]), "y": float(p["y"])}
        for p in sanitized["backrest_region"]
    ]

    seat_top = min(p["y"] for p in seat)
    seat_bottom = max(p["y"] for p in seat)
    seat_h = max(1.0, seat_bottom - seat_top)
    min_seat_h = 0.06 * height
    max_seat_h = 0.22 * height

    if seat_h < min_seat_h:
        center_y = 0.5 * (seat_top + seat_bottom)
        half = 0.5 * min_seat_h
        new_top = _clamp(center_y - half, 0.18 * height, y_max - min_seat_h)
        new_bottom = _clamp(new_top + min_seat_h, 0.0, y_max)
        notes.append("expanded seat vertical thickness to meet sitting prior")
        for p in seat:
            p["y"] = new_top if p["y"] <= center_y else new_bottom
        seat_top, seat_bottom = new_top, new_bottom
    elif seat_h > max_seat_h:
        center_y = 0.5 * (seat_top + seat_bottom)
        half = 0.5 * max_seat_h
        new_top = _clamp(center_y - half, 0.18 * height, y_max - max_seat_h)
        new_bottom = _clamp(new_top + max_seat_h, 0.0, y_max)
        notes.append("compressed seat vertical thickness to meet sitting prior")
        for p in seat:
            p["y"] = new_top if p["y"] <= center_y else new_bottom
        seat_top, seat_bottom = new_top, new_bottom

    if seat_top > 0.70 * height:
        shift = seat_top - 0.66 * height
        for p in seat:
            p["y"] = _clamp(p["y"] - shift, 0.0, y_max)
        seat_top = min(p["y"] for p in seat)
        seat_bottom = max(p["y"] for p in seat)
        notes.append("shifted seat upward to plausible height")
    if seat_bottom > 0.86 * height:
        shift = seat_bottom - 0.84 * height
        for p in seat:
            p["y"] = _clamp(p["y"] - shift, 0.0, y_max)
        seat_top = min(p["y"] for p in seat)
        seat_bottom = max(p["y"] for p in seat)
        notes.append("raised low seat bottom to plausible range")

    floor_y = 0.5 * (floor[0]["y"] + floor[1]["y"])
    floor_min = max(seat_bottom + 0.14 * height, 0.72 * height)
    if floor_y < floor_min:
        target_floor = _clamp(floor_min, 0.0, y_max)
        for p in floor:
            p["y"] = target_floor
            p["x"] = _clamp(p["x"], 0.0, x_max)
        notes.append("moved floor_line below seat to satisfy sitting prior")

    back_y = 0.5 * (back[0]["y"] + back[1]["y"])
    back_max_y = seat_top - 0.02 * height
    if back_y > seat_top + 0.06 * height:
        target_back = _clamp(seat_top - 0.12 * height, 0.12 * height, back_max_y)
        if target_back >= seat_top:
            target_back = _clamp(seat_top - 0.05 * height, 0.0, y_max)
        for p in back:
            p["y"] = target_back
            p["x"] = _clamp(p["x"], 0.0, x_max)
        notes.append("moved backrest_region above seat to satisfy sitting prior")

    repaired = {
        "image_width": int(sanitized["image_width"]),
        "image_height": int(sanitized["image_height"]),
        "seat_region": [{"x": round(p["x"], 2), "y": round(p["y"], 2)} for p in seat],
        "floor_line": [{"x": round(p["x"], 2), "y": round(p["y"], 2)} for p in floor],
        "backrest_region": [{"x": round(p["x"], 2), "y": round(p["y"], 2)} for p in back],
    }
    repaired, sanitize_notes_2 = sanitize_geometry_candidate(repaired)
    notes.extend(sanitize_notes_2)
    if repaired is None:
        return None, notes

    remaining = geometry_sitting_prior_issues(repaired)
    if remaining:
        notes.extend(remaining)
        return None, notes
    return repaired, notes


def geometry_from_part_segmentation(
    part_segmentation: Any, *, min_confidence: float
) -> tuple[dict[str, Any] | None, list[str]]:
    notes: list[str] = []
    if not isinstance(part_segmentation, dict):
        return None, ["part segmentation is not an object"]

    quality_issues = part_segmentation_quality_issues(
        part_segmentation, min_confidence=min_confidence
    )
    if quality_issues:
        return None, quality_issues

    candidate = {
        "image_width": part_segmentation.get("image_width"),
        "image_height": part_segmentation.get("image_height"),
        "seat_region": part_segmentation.get("seat_region"),
        "floor_line": part_segmentation.get("floor_line"),
        "backrest_region": part_segmentation.get("backrest_region"),
    }
    geometry, sanitize_notes = sanitize_geometry_candidate(candidate)
    notes.extend(sanitize_notes)
    if geometry is None:
        notes.append("failed to sanitize geometry derived from part segmentation")
        return None, notes

    prior_issues = geometry_sitting_prior_issues(geometry)
    if prior_issues:
        repaired, repair_notes = repair_geometry_with_sitting_priors(geometry)
        notes.extend(repair_notes)
        if repaired is not None:
            notes.append("repaired geometry derived from part segmentation priors")
            return repaired, notes
        notes.extend(prior_issues)
        return None, notes

    return geometry, notes


def validate_pose(data: Any, expected_width: int, expected_height: int) -> list[str]:
    errors: list[str] = []

    if not isinstance(data, dict):
        return ["Pose output must be an object"]

    width = data.get("image_width")
    height = data.get("image_height")
    if width != expected_width:
        errors.append(
            f"image_width mismatch: expected {expected_width}, got {width}"
        )
    if height != expected_height:
        errors.append(
            f"image_height mismatch: expected {expected_height}, got {height}"
        )

    keypoints = data.get("keypoints")
    if not isinstance(keypoints, dict):
        errors.append("keypoints must be an object")
        return errors

    for name in KEYPOINT_NAMES:
        if name not in keypoints:
            errors.append(f"missing keypoint: {name}")

    for name, value in keypoints.items():
        if name not in KEYPOINT_NAMES:
            errors.append(f"unknown keypoint: {name}")
            continue

        if not isinstance(value, dict):
            errors.append(f"{name} must be an object")
            continue

        validate_point(
            value,
            label=name,
            width=expected_width,
            height=expected_height,
            errors=errors,
        )

        conf = value.get("confidence")
        vis = value.get("visible")
        if not is_number(conf):
            errors.append(f"{name}.confidence must be numeric")
        elif conf < 0 or conf > 1:
            errors.append(f"{name}.confidence out of range: {conf}")

        if not isinstance(vis, bool):
            errors.append(f"{name}.visible must be boolean")

    return errors


def validate_checker(data: Any) -> list[str]:
    errors: list[str] = []

    if not isinstance(data, dict):
        return ["Checker output must be an object"]

    score = data.get("score")
    if not isinstance(score, int) or score < 0 or score > 4:
        errors.append("score must be an integer in [0, 4]")

    checks = data.get("checks")
    if not isinstance(checks, dict):
        errors.append("checks must be an object")
    else:
        for name in CHECK_NAMES:
            val = checks.get(name)
            if val not in (0, 1):
                errors.append(f"checks.{name} must be 0 or 1")

    failures = data.get("failures")
    if not isinstance(failures, list) or not all(
        isinstance(item, str) for item in failures
    ):
        errors.append("failures must be a list of strings")

    if isinstance(score, int) and isinstance(checks, dict):
        if all(checks.get(k) in (0, 1) for k in CHECK_NAMES):
            total = sum(int(checks[k]) for k in CHECK_NAMES)
            if total != score:
                errors.append(f"score mismatch: expected {total} from checks, got {score}")

    if score == 4 and isinstance(failures, list) and failures:
        errors.append("failures must be empty when score is 4")

    return errors


def point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-8) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def point_to_segment_distance(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    denom = abx * abx + aby * aby
    if denom < 1e-8:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
    cx = ax + t * abx
    cy = ay + t * aby
    return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5


def local_checker(
    geometry: dict[str, Any], pose: dict[str, Any]
) -> dict[str, Any]:
    seat = [(float(p["x"]), float(p["y"])) for p in geometry["seat_region"]]
    floor_a = geometry["floor_line"][0]
    floor_b = geometry["floor_line"][1]
    kps = pose["keypoints"]

    w = int(pose["image_width"])
    h = int(pose["image_height"])
    margin_y = 0.05 * h

    hip = kps["hip_center"]
    lk = kps["left_knee"]
    rk = kps["right_knee"]
    la = kps["left_ankle"]
    ra = kps["right_ankle"]
    ls = kps["left_shoulder"]
    rs = kps["right_shoulder"]

    seat_bottom = max(p[1] for p in seat)

    check_hip = int(point_in_polygon(float(hip["x"]), float(hip["y"]), seat))

    check_knees_front = int(
        float(lk["y"]) >= float(hip["y"]) - margin_y
        and float(rk["y"]) >= float(hip["y"]) - margin_y
    )

    dist_la = point_to_segment_distance(
        float(la["x"]),
        float(la["y"]),
        float(floor_a["x"]),
        float(floor_a["y"]),
        float(floor_b["x"]),
        float(floor_b["y"]),
    )
    dist_ra = point_to_segment_distance(
        float(ra["x"]),
        float(ra["y"]),
        float(floor_a["x"]),
        float(floor_a["y"]),
        float(floor_b["x"]),
        float(floor_b["y"]),
    )
    floor_tol = 0.15 * h
    check_ankles = int(
        float(la["y"]) > seat_bottom + 0.02 * h
        and float(ra["y"]) > seat_bottom + 0.02 * h
        and dist_la <= floor_tol
        and dist_ra <= floor_tol
    )

    knee_between = (
        min(float(hip["y"]), float(la["y"])) - margin_y
        <= float(lk["y"])
        <= max(float(hip["y"]), float(la["y"])) + margin_y
        and min(float(hip["y"]), float(ra["y"])) - margin_y
        <= float(rk["y"])
        <= max(float(hip["y"]), float(ra["y"])) + margin_y
    )
    shoulders_above_hip = (
        float(ls["y"]) <= float(hip["y"]) + margin_y
        and float(rs["y"]) <= float(hip["y"]) + margin_y
    )
    check_order = int(knee_between and shoulders_above_hip)

    checks = {
        "hip_on_seat": check_hip,
        "knees_in_front_of_hip": check_knees_front,
        "ankles_below_seat_near_floor": check_ankles,
        "limb_ordering_consistent": check_order,
    }

    failures: list[str] = []
    if not check_hip:
        failures.append("hip_center is outside seat_region")
    if not check_knees_front:
        failures.append("knees are not in front/lower image direction of hip")
    if not check_ankles:
        failures.append("ankles are not below seat or too far from floor_line")
    if not check_order:
        failures.append("limb ordering is inconsistent with sitting")

    return {
        "score": sum(checks.values()),
        "checks": checks,
        "failures": failures,
    }


def compute_hip_contact_candidates(
    geometry: dict[str, Any],
    *,
    grid_step_px: int = 12,
    max_points: int = 1500,
) -> dict[str, Any]:
    seat = geometry["seat_region"]
    back = geometry["backrest_region"]
    width = int(geometry["image_width"])
    height = int(geometry["image_height"])

    seat_rect_xy = [(float(p["x"]), float(p["y"])) for p in seat]
    contact_poly = seat_rect_xy
    contact_raw = geometry.get("seat_contact_region")
    if isinstance(contact_raw, list) and len(contact_raw) >= 3:
        contact_candidate: list[tuple[float, float]] = []
        for p in contact_raw:
            if not isinstance(p, dict):
                continue
            x = p.get("x")
            y = p.get("y")
            if not is_number(x) or not is_number(y):
                continue
            contact_candidate.append((float(x), float(y)))
        if len(contact_candidate) >= 3 and polygon_area(contact_candidate) > 1.0:
            contact_poly = contact_candidate

    use_contact_poly = contact_poly is not seat_rect_xy
    area = polygon_area(contact_poly)
    min_x = min(x for x, _ in contact_poly)
    max_x = max(x for x, _ in contact_poly)
    min_y = min(y for _, y in contact_poly)
    max_y = max(y for _, y in contact_poly)

    # Pick the edge nearest to backrest center as the seat-back edge.
    back_cx = (float(back[0]["x"]) + float(back[1]["x"])) * 0.5
    back_cy = (float(back[0]["y"]) + float(back[1]["y"])) * 0.5
    best_idx = 0
    best_dist = float("inf")
    for i in range(4):
        ax, ay = seat_rect_xy[i]
        bx, by = seat_rect_xy[(i + 1) % 4]
        mx, my = (ax + bx) * 0.5, (ay + by) * 0.5
        d = math.hypot(mx - back_cx, my - back_cy)
        if d < best_dist:
            best_dist = d
            best_idx = i

    back_a = seat_rect_xy[best_idx]
    back_b = seat_rect_xy[(best_idx + 1) % 4]
    front_a = seat_rect_xy[(best_idx + 2) % 4]
    front_b = seat_rect_xy[(best_idx + 3) % 4]

    back_mid = ((back_a[0] + back_b[0]) * 0.5, (back_a[1] + back_b[1]) * 0.5)
    front_mid = ((front_a[0] + front_b[0]) * 0.5, (front_a[1] + front_b[1]) * 0.5)

    depth_vx = front_mid[0] - back_mid[0]
    depth_vy = front_mid[1] - back_mid[1]
    depth_len2 = depth_vx * depth_vx + depth_vy * depth_vy
    depth_len = math.sqrt(max(depth_len2, 1e-8))
    depth_vertical_ratio = abs(depth_vy) / depth_len

    lat_vx = back_b[0] - back_a[0]
    lat_vy = back_b[1] - back_a[1]
    lat_len2 = lat_vx * lat_vx + lat_vy * lat_vy

    step = max(2, int(grid_step_px))
    margin_px = max(4.0, min(30.0, 0.04 * math.sqrt(max(area, 1.0))))

    candidates: list[dict[str, float]] = []
    y0 = max(0, int(math.floor(min_y)))
    y1 = min(height - 1, int(math.ceil(max_y)))
    x0 = max(0, int(math.floor(min_x)))
    x1 = min(width - 1, int(math.ceil(max_x)))

    edges = [
        (contact_poly[i], contact_poly[(i + 1) % len(contact_poly)])
        for i in range(len(contact_poly))
    ]

    for y in range(y0, y1 + 1, step):
        for x in range(x0, x1 + 1, step):
            xf = float(x)
            yf = float(y)
            if not point_in_polygon(xf, yf, contact_poly):
                continue

            edge_dist = min(
                point_to_segment_distance(xf, yf, a[0], a[1], b[0], b[1])
                for a, b in edges
            )
            if edge_dist < margin_px:
                continue

            if depth_len2 > 1e-8:
                depth_t = (
                    (xf - back_mid[0]) * depth_vx + (yf - back_mid[1]) * depth_vy
                ) / depth_len2
            else:
                depth_t = 0.5
            if use_contact_poly:
                if depth_vertical_ratio >= 0.75:
                    min_depth_t = 0.34
                    max_depth_t = 0.84
                    target_depth_t = 0.54
                    depth_sigma = 0.16
                elif depth_vertical_ratio >= 0.55:
                    min_depth_t = 0.27
                    max_depth_t = 0.88
                    target_depth_t = 0.49
                    depth_sigma = 0.18
                else:
                    min_depth_t = 0.2
                    max_depth_t = 0.9
                    target_depth_t = 0.46
                    depth_sigma = 0.2
            else:
                min_depth_t = 0.08
                max_depth_t = 0.92
                target_depth_t = 0.35
                depth_sigma = 0.23
            if depth_t < min_depth_t or depth_t > max_depth_t:
                continue

            if lat_len2 > 1e-8:
                lat_u = ((xf - back_a[0]) * lat_vx + (yf - back_a[1]) * lat_vy) / lat_len2
            else:
                lat_u = 0.5

            # Prefer the middle sit-contact zone, not the backrest-adjacent strip.
            w_depth = math.exp(-0.5 * ((depth_t - target_depth_t) / depth_sigma) ** 2)
            w_lat = math.exp(-0.5 * ((lat_u - 0.5) / 0.33) ** 2)
            weight = _clamp(w_depth * w_lat, 0.0, 1.0)

            candidates.append(
                {
                    "x": round(xf, 2),
                    "y": round(yf, 2),
                    "weight": round(weight, 4),
                    "depth_t": round(depth_t, 4),
                }
            )

    if not candidates:
        cx = sum(p[0] for p in contact_poly) / float(len(contact_poly))
        cy = sum(p[1] for p in contact_poly) / float(len(contact_poly))
        candidates = [
            {
                "x": round(_clamp(cx, 0.0, float(width - 1)), 2),
                "y": round(_clamp(cy, 0.0, float(height - 1)), 2),
                "weight": 1.0,
                "depth_t": 0.5,
            }
        ]

    candidates.sort(key=lambda p: p["weight"], reverse=True)
    total_count = len(candidates)
    if max_points > 0 and len(candidates) > max_points:
        stride = len(candidates) / float(max_points)
        sampled = []
        for i in range(max_points):
            idx = int(round(i * stride))
            idx = min(idx, len(candidates) - 1)
            sampled.append(candidates[idx])
        candidates = sampled

    recommended = candidates[0]

    return {
        "method": "seat_contact_region_dense_sampling_v2"
        if use_contact_poly
        else "seat_region_dense_sampling_v1",
        "sampling_step_px": step,
        "edge_margin_px": round(margin_px, 2),
        "num_total_candidates": total_count,
        "num_returned_candidates": len(candidates),
        "recommended": recommended,
        "back_edge": [
            {"x": round(back_a[0], 2), "y": round(back_a[1], 2)},
            {"x": round(back_b[0], 2), "y": round(back_b[1], 2)},
        ],
        "front_edge": [
            {"x": round(front_a[0], 2), "y": round(front_a[1], 2)},
            {"x": round(front_b[0], 2), "y": round(front_b[1], 2)},
        ],
        "contact_region_polygon": [
            {"x": round(float(p[0]), 2), "y": round(float(p[1]), 2)}
            for p in contact_poly
        ],
        "candidate_points": candidates,
    }


def run_stage(
    *,
    stage_name: str,
    backend: str,
    api_key: str,
    ollama_host: str,
    model: str,
    system_prompt: str,
    image_mime: str,
    image_b64: str,
    include_image: bool,
    input_payload: dict[str, Any] | None,
    response_schema: dict[str, Any] | None,
    validator: Callable[[Any], list[str]],
    max_retries: int,
    temperature: float,
    timeout_sec: float,
    max_output_tokens: int,
) -> dict[str, Any]:
    errors: list[str] = []
    last_raw = ""
    issue_note = ""
    last_parsed: dict[str, Any] | None = None

    for attempt in range(1, max_retries + 1):
        pieces = [system_prompt]

        if input_payload is not None:
            pieces.append(
                "Input JSON:\n" + json.dumps(input_payload, ensure_ascii=False, indent=2)
            )

        if issue_note:
            pieces.append(
                "Previous output issues:\n"
                f"{issue_note}\n"
                "Return ONLY valid JSON object."
            )

        prompt_text = "\n\n".join(pieces)

        try:
            if backend == "gemini":
                raw = call_gemini_json(
                    api_key=api_key,
                    model=model,
                    prompt_text=prompt_text,
                    image_mime=image_mime if include_image else None,
                    image_b64=image_b64 if include_image else None,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    timeout_sec=timeout_sec,
                )
            elif backend == "ollama":
                raw = call_ollama_json(
                    ollama_host=ollama_host,
                    model=model,
                    prompt_text=prompt_text,
                    image_b64=image_b64 if include_image else None,
                    response_schema=response_schema,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    timeout_sec=timeout_sec,
                )
            else:
                raise ValueError(f"Unsupported backend: {backend}")
            last_raw = raw
        except Exception as exc:
            message = f"{stage_name} attempt {attempt}: API error: {exc}"
            errors.append(message)
            issue_note = message
            continue

        try:
            parsed = extract_json_object(raw)
        except Exception as exc:
            preview = strip_code_fence(raw).replace("\n", " ")[:240]
            message = (
                f"{stage_name} attempt {attempt}: JSON parse failed: {exc}; "
                f"raw_preview={preview!r}"
            )
            errors.append(message)
            issue_note = message
            continue

        val_errors = validator(parsed)
        if val_errors:
            last_parsed = parsed
            joined = "; ".join(val_errors)
            message = f"{stage_name} attempt {attempt}: validation failed: {joined}"
            errors.append(message)
            issue_note = joined
            continue

        return {
            "ok": True,
            "attempts": attempt,
            "data": parsed,
            "errors": errors,
            "last_raw": last_raw,
            "last_parsed": parsed,
        }

    return {
        "ok": False,
        "attempts": max_retries,
        "data": None,
        "errors": errors,
        "last_raw": last_raw,
        "last_parsed": last_parsed,
    }


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    api_key = ""
    if args.backend == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")

    prompt_dir = Path(args.prompt_dir).resolve()
    schema_dir = Path(args.schema_dir).resolve()
    stage_0_prompt = ""
    part_schema: dict[str, Any] | None = None
    if args.part_segmentation:
        stage_0_prompt = load_text(prompt_dir / "chair_pose_stage_0_parts_system.txt")
        part_schema = load_json(schema_dir / "chair_parts_v1.schema.json")
    stage_a_prompt = load_text(prompt_dir / "chair_pose_stage_a_system.txt")
    stage_b_prompt = load_text(prompt_dir / "chair_pose_stage_b_system.txt")
    checker_prompt = load_text(prompt_dir / "chair_pose_checker_system.txt")
    repair_prompt = load_text(prompt_dir / "chair_pose_repair_system.txt")
    geometry_schema = load_json(schema_dir / "chair_geometry_v1.schema.json")
    pose_schema = load_json(schema_dir / "sitting_keypoints_v1.schema.json")
    checker_schema = load_json(schema_dir / "sitting_pose_check_v1.schema.json")

    image_path = Path(args.image).resolve()
    image_width, image_height = infer_image_size(image_path)
    image_mime, image_b64, model_image_w, model_image_h, was_resized = prepare_model_image(
        image_path,
        max_side=args.vision_max_side,
        jpeg_quality=args.vision_jpeg_quality,
    )

    stage_0_parts: dict[str, Any] = {
        "ok": not args.part_segmentation,
        "attempts": 0,
        "errors": [],
        "data": None,
        "last_parsed": None,
    }
    stage_0_recovery_notes: list[str] = []
    part_segmentation_model: dict[str, Any] | None = None
    part_quality_issues: list[str] = []
    use_parts_for_stage_a = False
    stage_0_trustworthy = False

    if args.part_segmentation:
        stage_0_parts = run_stage(
            stage_name="stage_0_part_segmentation",
            backend=args.backend,
            api_key=api_key,
            ollama_host=args.ollama_host,
            model=args.model,
            system_prompt=stage_0_prompt,
            image_mime=image_mime,
            image_b64=image_b64,
            include_image=True,
            input_payload=None,
            response_schema=part_schema,
            validator=validate_part_segmentation,
            max_retries=args.max_retries,
            temperature=min(args.temperature, 0.15),
            timeout_sec=args.timeout,
            max_output_tokens=args.max_output_tokens,
        )
        if stage_0_parts["ok"]:
            part_segmentation_model = stage_0_parts["data"]
            stage_0_trustworthy = True
        else:
            stage_0_candidate = stage_0_parts.get("last_parsed")
            denormed_parts = False
            stage_0_candidate, denormed_parts = denormalize_part_segmentation_candidate(
                stage_0_candidate,
                target_width=model_image_w,
                target_height=model_image_h,
            )
            recovered_parts, notes = sanitize_part_segmentation_candidate(stage_0_candidate)
            if denormed_parts:
                notes = ["denormalized part segmentation from compact coordinate grid"] + notes
            if recovered_parts is not None:
                part_segmentation_model = recovered_parts
                stage_0_recovery_notes = notes

    if part_segmentation_model is not None and (
        int(part_segmentation_model.get("image_width", 0)) != model_image_w
        or int(part_segmentation_model.get("image_height", 0)) != model_image_h
    ):
        part_segmentation_model = rescale_part_segmentation(
            part_segmentation_model,
            target_width=model_image_w,
            target_height=model_image_h,
        )

    if part_segmentation_model is not None:
        part_quality_issues = part_segmentation_quality_issues(
            part_segmentation_model,
            min_confidence=args.parts_min_confidence,
        )
        if part_quality_issues:
            stage_0_trustworthy = False

    use_parts_for_stage_a = bool(stage_0_trustworthy and part_segmentation_model is not None)

    stage_a_input: dict[str, Any] | None = None
    if use_parts_for_stage_a and part_segmentation_model is not None:
        stage_a_input = {"part_segmentation": part_segmentation_model}

    stage_a_part_fallback: dict[str, Any] | None = None
    stage_a_part_fallback_notes: list[str] = []
    if use_parts_for_stage_a and part_segmentation_model is not None:
        stage_a_part_fallback, stage_a_part_fallback_notes = (
            geometry_from_part_segmentation(
                part_segmentation_model,
                min_confidence=args.parts_min_confidence,
            )
        )

    stage_a = run_stage(
        stage_name="stage_a_geometry",
        backend=args.backend,
        api_key=api_key,
        ollama_host=args.ollama_host,
        model=args.model,
        system_prompt=stage_a_prompt,
        image_mime=image_mime,
        image_b64=image_b64,
        include_image=True,
        input_payload=stage_a_input,
        response_schema=geometry_schema,
        validator=validate_geometry,
        max_retries=args.max_retries,
        temperature=args.temperature,
        timeout_sec=args.timeout,
        max_output_tokens=args.max_output_tokens,
    )

    part_segmentation_output = part_segmentation_model
    if part_segmentation_output is not None and (
        int(part_segmentation_output.get("image_width", 0)) != image_width
        or int(part_segmentation_output.get("image_height", 0)) != image_height
    ):
        part_segmentation_output = rescale_part_segmentation(
            part_segmentation_output,
            target_width=image_width,
            target_height=image_height,
        )

    stage_0_log: dict[str, Any] = {
        "enabled": args.part_segmentation,
        "ok": stage_0_parts["ok"],
        "trustworthy": stage_0_trustworthy,
        "attempts": stage_0_parts["attempts"],
        "errors": stage_0_parts["errors"],
        "used_for_stage_a": use_parts_for_stage_a,
        "min_confidence_threshold": args.parts_min_confidence,
    }
    if stage_0_recovery_notes:
        stage_0_log["recovered"] = True
        stage_0_log["recovery_notes"] = stage_0_recovery_notes
    if part_quality_issues:
        stage_0_log["quality_issues"] = part_quality_issues
    if stage_a_part_fallback is not None:
        stage_0_log["geometry_fallback_available"] = True
    elif stage_a_part_fallback_notes:
        stage_0_log["geometry_fallback_issues"] = stage_a_part_fallback_notes

    result: dict[str, Any] = {
        "status": "failed",
        "input_image": str(image_path),
        "backend": args.backend,
        "model": args.model,
        "persona": args.persona,
        "image_preprocess": {
            "original_width": image_width,
            "original_height": image_height,
            "model_input_width": model_image_w,
            "model_input_height": model_image_h,
            "resized_for_vlm": was_resized,
            "vision_max_side": args.vision_max_side,
        },
        "stage_logs": {
            "stage_0_parts": stage_0_log,
            "stage_a": {
                "ok": stage_a["ok"],
                "attempts": stage_a["attempts"],
                "errors": stage_a["errors"],
                "used_part_segmentation_input": use_parts_for_stage_a,
            },
            "geometry_refine": {},
            "stage_b": {},
            "checker": {},
            "postprocess": [],
            "repairs": [],
        },
        "part_segmentation": part_segmentation_output,
        "seat_segmentation": None,
        "geometry": None,
        "hip_contact": None,
        "pose": None,
        "checker": None,
        "local_checker": None,
        "effective_score": None,
        "min_score_threshold": args.min_score,
        "failure_reason": None,
    }

    if not stage_a["ok"]:
        stage_a_candidate = stage_a.get("last_parsed")
        denormed = False
        stage_a_candidate, denormed = denormalize_geometry_candidate(
            stage_a_candidate,
            target_width=model_image_w,
            target_height=model_image_h,
        )
        recovered, notes = sanitize_geometry_candidate(stage_a_candidate)
        if denormed:
            notes = ["denormalized geometry from small coordinate grid"] + notes
        if recovered is not None:
            geometry = recovered
            result["stage_logs"]["stage_a"]["recovered"] = True
            result["stage_logs"]["stage_a"]["recovery_notes"] = notes
            result["stage_logs"]["stage_a"]["recovered_source"] = "stage_a_output"
        elif stage_a_part_fallback is not None:
            geometry = stage_a_part_fallback
            result["stage_logs"]["stage_a"]["recovered"] = True
            result["stage_logs"]["stage_a"]["recovery_notes"] = [
                "used part segmentation geometry due to stage A failure"
            ] + stage_a_part_fallback_notes
            result["stage_logs"]["stage_a"]["recovered_source"] = "stage_0_part_segmentation"
        else:
            geometry = build_fallback_geometry(image_width, image_height)
            result["stage_logs"]["stage_a"]["recovered"] = True
            fallback_notes = [
                "used deterministic fallback geometry due to stage failure"
            ]
            if stage_a_part_fallback_notes:
                fallback_notes.append(
                    "part segmentation fallback unavailable: "
                    + "; ".join(stage_a_part_fallback_notes)
                )
            result["stage_logs"]["stage_a"]["recovery_notes"] = fallback_notes
            result["stage_logs"]["stage_a"]["recovered_source"] = "deterministic_fallback"
    else:
        geometry = stage_a["data"]

    if (
        isinstance(geometry.get("image_width"), int)
        and isinstance(geometry.get("image_height"), int)
        and (
            int(geometry["image_width"]) != image_width
            or int(geometry["image_height"]) != image_height
        )
    ):
        geometry = rescale_geometry(
            geometry, target_width=image_width, target_height=image_height
        )
        result["stage_logs"]["stage_a"]["rescaled_to_original_image"] = True

    prior_issues = geometry_sitting_prior_issues(geometry)
    if prior_issues:
        result["stage_logs"]["stage_a"]["prior_fallback"] = True
        result["stage_logs"]["stage_a"]["prior_issues"] = prior_issues
        repaired_geometry, repaired_notes = repair_geometry_with_sitting_priors(geometry)
        if repaired_geometry is not None:
            geometry = repaired_geometry
            result["stage_logs"]["stage_a"]["prior_fallback_source"] = "stage_a_prior_repair"
            result["stage_logs"]["stage_a"]["prior_fallback_notes"] = [
                "repaired stage A geometry to satisfy sitting priors"
            ] + repaired_notes
        elif stage_a_part_fallback is not None:
            part_geometry = stage_a_part_fallback
            if (
                int(part_geometry.get("image_width", 0)) != image_width
                or int(part_geometry.get("image_height", 0)) != image_height
            ):
                part_geometry = rescale_geometry(
                    part_geometry,
                    target_width=image_width,
                    target_height=image_height,
                )
            geometry = part_geometry
            result["stage_logs"]["stage_a"]["prior_fallback_source"] = (
                "stage_0_part_segmentation"
            )
            result["stage_logs"]["stage_a"]["prior_fallback_notes"] = [
                "replaced geometry with part segmentation geometry due to prior issues"
            ] + stage_a_part_fallback_notes
        else:
            geometry = build_fallback_geometry(image_width, image_height)
            result["stage_logs"]["stage_a"]["prior_fallback_source"] = (
                "deterministic_fallback"
            )
            prior_notes = ["replaced geometry with canonical sitting-friendly fallback"]
            if stage_a_part_fallback_notes:
                prior_notes.append(
                    "part segmentation fallback unavailable: "
                    + "; ".join(stage_a_part_fallback_notes)
                )
            result["stage_logs"]["stage_a"]["prior_fallback_notes"] = prior_notes

    geometry_refine_log: dict[str, Any] = {
        "enabled": bool(args.geometry_refine_yolo),
        "used": False,
    }
    if args.geometry_refine_yolo:
        yolo_geometry, yolo_seat_segmentation, yolo_log = extract_chair_geometry_yolo(
            image_path,
            model_path=args.yolo_model,
            conf=args.yolo_chair_conf,
            device=args.yolo_device,
        )
        geometry_refine_log.update(yolo_log)
        if yolo_geometry is not None:
            if (
                int(yolo_geometry.get("image_width", 0)) != image_width
                or int(yolo_geometry.get("image_height", 0)) != image_height
            ):
                yolo_geometry = rescale_geometry(
                    yolo_geometry,
                    target_width=image_width,
                    target_height=image_height,
                )
            geometry = yolo_geometry
            geometry_refine_log["used"] = True
            geometry_refine_log["reason"] = "replaced geometry with yolo chair mask anchors"
            if yolo_seat_segmentation is not None:
                if (
                    int(yolo_seat_segmentation.get("image_width", 0)) != image_width
                    or int(yolo_seat_segmentation.get("image_height", 0)) != image_height
                ):
                    poly = yolo_seat_segmentation.get("seat_contact_polygon")
                    if isinstance(poly, list):
                        scaled_poly: list[dict[str, float]] = []
                        src_w = int(yolo_seat_segmentation.get("image_width", image_width))
                        src_h = int(yolo_seat_segmentation.get("image_height", image_height))
                        for p in poly:
                            if not isinstance(p, dict):
                                continue
                            x = p.get("x")
                            y = p.get("y")
                            if not is_number(x) or not is_number(y):
                                continue
                            nx, ny = _scale_xy(
                                float(x),
                                float(y),
                                src_w,
                                src_h,
                                image_width,
                                image_height,
                            )
                            scaled_poly.append({"x": round(nx, 2), "y": round(ny, 2)})
                        yolo_seat_segmentation["seat_contact_polygon"] = scaled_poly
                        yolo_seat_segmentation["image_width"] = image_width
                        yolo_seat_segmentation["image_height"] = image_height
                result["seat_segmentation"] = yolo_seat_segmentation

    result["stage_logs"]["geometry_refine"] = geometry_refine_log

    result["geometry"] = geometry
    result["hip_contact"] = compute_hip_contact_candidates(
        geometry,
        grid_step_px=args.hip_grid_step,
        max_points=args.hip_max_points,
    )

    pose_input: dict[str, Any] = {"chair_geometry": geometry}
    if args.persona:
        pose_input["persona_constraints"] = args.persona

    stage_b = run_stage(
        stage_name="stage_b_pose",
        backend=args.backend,
        api_key=api_key,
        ollama_host=args.ollama_host,
        model=args.model,
        system_prompt=stage_b_prompt,
        image_mime=image_mime,
        image_b64=image_b64,
        include_image=args.stage_b_use_image,
        input_payload=pose_input,
        response_schema=pose_schema,
        validator=lambda data: validate_pose(
            data,
            expected_width=geometry["image_width"],
            expected_height=geometry["image_height"],
        ),
        max_retries=args.max_retries,
        temperature=args.temperature,
        timeout_sec=args.timeout,
        max_output_tokens=args.max_output_tokens,
    )

    result["stage_logs"]["stage_b"] = {
        "ok": stage_b["ok"],
        "attempts": stage_b["attempts"],
        "errors": stage_b["errors"],
        "used_image": args.stage_b_use_image,
    }

    if not stage_b["ok"]:
        pose = build_fallback_pose(geometry, args.persona)
        pose_errors = validate_pose(
            pose,
            expected_width=geometry["image_width"],
            expected_height=geometry["image_height"],
        )
        if pose_errors:
            result["failure_reason"] = (
                "Stage B failed and fallback pose was invalid: "
                + "; ".join(pose_errors)
            )
            return result
        result["pose"] = pose
        result["stage_logs"]["stage_b"]["recovered"] = True
        result["stage_logs"]["stage_b"]["recovery_notes"] = [
            "used deterministic fallback pose due to stage failure"
        ]
    else:
        pose = stage_b["data"]
        result["pose"] = pose

    checker_input = {
        "chair_geometry": geometry,
        "proposed_keypoints": pose,
    }

    checker_stage = run_stage(
        stage_name="stage_c_checker",
        backend=args.backend,
        api_key=api_key,
        ollama_host=args.ollama_host,
        model=args.model,
        system_prompt=checker_prompt,
        image_mime=image_mime,
        image_b64=image_b64,
        include_image=args.checker_use_image,
        input_payload=checker_input,
        response_schema=checker_schema,
        validator=validate_checker,
        max_retries=args.max_retries,
        temperature=min(args.temperature, 0.1),
        timeout_sec=args.timeout,
        max_output_tokens=args.max_output_tokens,
    )

    result["stage_logs"]["checker"] = {
        "ok": checker_stage["ok"],
        "attempts": checker_stage["attempts"],
        "errors": checker_stage["errors"],
        "used_image": args.checker_use_image,
    }

    if not checker_stage["ok"]:
        local = local_checker(geometry, pose)
        checker = local
        result["stage_logs"]["checker"]["recovered"] = True
        result["stage_logs"]["checker"]["recovery_notes"] = [
            "used local checker because stage checker failed"
        ]
    else:
        checker = checker_stage["data"]
        local = local_checker(geometry, pose)

    result["checker"] = checker
    result["local_checker"] = local
    result["effective_score"] = min(checker["score"], local["score"])

    # If model pose misses hard geometric checks, compare against
    # a deterministic geometry-constrained fallback pose.
    must_project = (
        local["score"] < args.min_score
        or int(local["checks"].get("hip_on_seat", 1)) == 0
        or int(local["checks"].get("ankles_below_seat_near_floor", 1)) == 0
    )
    if must_project:
        fallback_pose = build_fallback_pose(geometry, args.persona)
        fallback_local = local_checker(geometry, fallback_pose)
        fallback_effective = min(checker["score"], fallback_local["score"])
        fallback_log: dict[str, Any] = {
            "type": "fallback_pose_projection",
            "base_local_score": local["score"],
            "candidate_local_score": fallback_local["score"],
            "accepted": False,
        }

        if fallback_local["score"] > local["score"]:
            if args.checker_use_image:
                checker_input = {
                    "chair_geometry": geometry,
                    "proposed_keypoints": fallback_pose,
                }
                checker_fallback_stage = run_stage(
                    stage_name="checker_after_fallback_pose",
                    backend=args.backend,
                    api_key=api_key,
                    ollama_host=args.ollama_host,
                    model=args.model,
                    system_prompt=checker_prompt,
                    image_mime=image_mime,
                    image_b64=image_b64,
                    include_image=args.checker_use_image,
                    input_payload=checker_input,
                    response_schema=checker_schema,
                    validator=validate_checker,
                    max_retries=args.max_retries,
                    temperature=min(args.temperature, 0.1),
                    timeout_sec=args.timeout,
                    max_output_tokens=args.max_output_tokens,
                )

                if checker_fallback_stage["ok"]:
                    fallback_checker = checker_fallback_stage["data"]
                    fallback_log["checker_recomputed"] = True
                else:
                    fallback_checker = checker
                    fallback_log["checker_recomputed"] = False
                    fallback_log["checker_reused"] = True
                    fallback_log["checker_errors"] = checker_fallback_stage["errors"]
            else:
                fallback_checker = checker
                fallback_log["checker_recomputed"] = False
                fallback_log["checker_reused"] = True

            fallback_effective = min(
                fallback_checker["score"], fallback_local["score"]
            )
            fallback_log["candidate_checker_score"] = fallback_checker["score"]
            fallback_log["candidate_effective_score"] = fallback_effective

            if fallback_effective >= result["effective_score"]:
                pose = fallback_pose
                checker = fallback_checker
                local = fallback_local
                result["pose"] = pose
                result["checker"] = checker
                result["local_checker"] = local
                result["effective_score"] = fallback_effective
                fallback_log["accepted"] = True

        result["stage_logs"]["postprocess"].append(fallback_log)

    repair_idx = 0
    while (
        result["effective_score"] < args.min_score
        and repair_idx < args.max_repairs
    ):
        repair_idx += 1

        repair_input = {
            "chair_geometry": geometry,
            "previous_keypoints": pose,
            "vlm_checker": checker,
            "local_checker": local,
            "target": {
                "minimum_effective_score": args.min_score,
                "effective_score_definition": "min(vlm_checker.score, local_checker.score)",
            },
        }

        repair_stage = run_stage(
            stage_name=f"repair_{repair_idx}",
            backend=args.backend,
            api_key=api_key,
            ollama_host=args.ollama_host,
            model=args.model,
            system_prompt=repair_prompt,
            image_mime=image_mime,
            image_b64=image_b64,
            include_image=args.stage_b_use_image,
            input_payload=repair_input,
            response_schema=pose_schema,
            validator=lambda data: validate_pose(
                data,
                expected_width=geometry["image_width"],
                expected_height=geometry["image_height"],
            ),
            max_retries=args.max_retries,
            temperature=args.temperature,
            timeout_sec=args.timeout,
            max_output_tokens=args.max_output_tokens,
        )

        repair_log: dict[str, Any] = {
            "repair_round": repair_idx,
            "attempts": repair_stage["attempts"],
            "errors": repair_stage["errors"],
            "accepted": False,
        }

        if not repair_stage["ok"]:
            repair_log["reason"] = "repair stage could not produce valid keypoints"
            result["stage_logs"]["repairs"].append(repair_log)
            break

        candidate_pose = repair_stage["data"]

        checker_input = {
            "chair_geometry": geometry,
            "proposed_keypoints": candidate_pose,
        }
        checker_stage = run_stage(
            stage_name=f"checker_after_repair_{repair_idx}",
            backend=args.backend,
            api_key=api_key,
            ollama_host=args.ollama_host,
            model=args.model,
            system_prompt=checker_prompt,
            image_mime=image_mime,
            image_b64=image_b64,
            include_image=args.checker_use_image,
            input_payload=checker_input,
            response_schema=checker_schema,
            validator=validate_checker,
            max_retries=args.max_retries,
            temperature=min(args.temperature, 0.1),
            timeout_sec=args.timeout,
            max_output_tokens=args.max_output_tokens,
        )

        if not checker_stage["ok"]:
            repair_log["reason"] = "checker failed after repair"
            result["stage_logs"]["repairs"].append(repair_log)
            break

        candidate_checker = checker_stage["data"]
        candidate_local = local_checker(geometry, candidate_pose)
        candidate_effective = min(candidate_checker["score"], candidate_local["score"])

        repair_log["candidate_vlm_score"] = candidate_checker["score"]
        repair_log["candidate_local_score"] = candidate_local["score"]
        repair_log["candidate_effective_score"] = candidate_effective

        if candidate_effective >= result["effective_score"]:
            pose = candidate_pose
            checker = candidate_checker
            local = candidate_local
            result["pose"] = pose
            result["checker"] = checker
            result["local_checker"] = local
            result["effective_score"] = candidate_effective
            repair_log["accepted"] = True

        result["stage_logs"]["repairs"].append(repair_log)

    if result["effective_score"] is None:
        result["failure_reason"] = "No score available"
        return result

    if result["effective_score"] < args.min_score:
        result["status"] = "low_score"
        result["failure_reason"] = (
            f"effective_score={result['effective_score']} below threshold {args.min_score}"
        )
    else:
        result["status"] = "ok"

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict plausible sitting 2D keypoints from a chair-only image"
    )
    default_backend = os.getenv("POSE_VLM_BACKEND", "gemini").strip().lower()
    if default_backend not in {"gemini", "ollama"}:
        default_backend = "gemini"

    default_model = os.getenv("POSE_VLM_MODEL", "").strip()
    if not default_model:
        default_model = (
            "qwen2.5vl:3b" if default_backend == "ollama" else "gemini-2.5-pro"
        )

    parser.add_argument("--image", required=True, help="Path to chair image")
    parser.add_argument(
        "--output",
        default="runs/chair_sitting_pose_result.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--backend",
        choices=["gemini", "ollama"],
        default=default_backend,
        help="VLM backend",
    )
    parser.add_argument(
        "--model",
        default=default_model,
        help="VLM model name (e.g., gemini-2.5-pro, qwen2.5vl:3b)",
    )
    parser.add_argument(
        "--ollama-host",
        default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
        help="Ollama host URL (used when --backend ollama)",
    )
    parser.add_argument(
        "--prompt-dir",
        default=str(Path(__file__).resolve().parent / "prompts"),
        help="Directory containing prompt files",
    )
    parser.add_argument(
        "--schema-dir",
        default=str(Path(__file__).resolve().parent / "schemas"),
        help="Directory containing JSON schema files",
    )
    parser.add_argument(
        "--persona",
        default="",
        help=(
            "Optional geometric persona constraints, e.g. "
            "'tired: hips posterior, shoulders lower, feet forward'"
        ),
    )
    parser.add_argument(
        "--vision-max-side",
        type=int,
        default=-1,
        help=(
            "Resize image before VLM; 0 disables resize. "
            "Default: 256 for ollama, 0 for gemini."
        ),
    )
    parser.add_argument(
        "--vision-jpeg-quality",
        type=int,
        default=75,
        help="JPEG quality when resized for VLM input (30..95)",
    )
    parser.add_argument(
        "--stage-b-use-image",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Attach image in Stage B pose prediction. "
            "Default: false for ollama, true for gemini."
        ),
    )
    parser.add_argument(
        "--checker-use-image",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Attach image in checker stage. "
            "Default: false for ollama, true for gemini."
        ),
    )
    parser.add_argument(
        "--geometry-refine-yolo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Refine chair geometry with YOLO instance segmentation "
            "(recommended for more accurate seat anchors)."
        ),
    )
    parser.add_argument(
        "--yolo-model",
        default="yolov8n-seg.pt",
        help="YOLO segmentation model path/name for geometry refinement.",
    )
    parser.add_argument(
        "--yolo-chair-conf",
        type=float,
        default=0.2,
        help="Confidence threshold for YOLO chair instance detection (0..1).",
    )
    parser.add_argument(
        "--yolo-device",
        default="",
        help="YOLO device override (e.g., mps, cpu). Empty = auto.",
    )
    parser.add_argument(
        "--part-segmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable Stage 0 chair part segmentation anchors "
            "before Stage A geometry extraction."
        ),
    )
    parser.add_argument(
        "--parts-min-confidence",
        type=float,
        default=0.2,
        help=(
            "Minimum confidence threshold for using part segmentation "
            "as Stage A input/fallback (0..1)."
        ),
    )
    parser.add_argument(
        "--hip-grid-step",
        type=int,
        default=12,
        help="Pixel step for dense hip-contact sampling inside seat polygon.",
    )
    parser.add_argument(
        "--hip-max-points",
        type=int,
        default=1500,
        help="Maximum number of hip-contact candidate points to return.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retries per stage for parse/schema failure",
    )
    parser.add_argument(
        "--max-repairs",
        type=int,
        default=2,
        help="Maximum repair rounds after checker",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=3,
        help="Minimum accepted effective score in [0,4]",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Generation temperature",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=2048,
        help="Max output tokens (Gemini maxOutputTokens / Ollama num_predict)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout seconds",
    )
    args = parser.parse_args()

    if args.max_retries < 1:
        parser.error("--max-retries must be >= 1")
    if args.max_repairs < 0:
        parser.error("--max-repairs must be >= 0")
    if args.min_score < 0 or args.min_score > 4:
        parser.error("--min-score must be between 0 and 4")
    if args.hip_grid_step < 2:
        parser.error("--hip-grid-step must be >= 2")
    if args.hip_max_points < 1:
        parser.error("--hip-max-points must be >= 1")
    if args.yolo_chair_conf < 0 or args.yolo_chair_conf > 1:
        parser.error("--yolo-chair-conf must be between 0 and 1")
    if args.parts_min_confidence < 0 or args.parts_min_confidence > 1:
        parser.error("--parts-min-confidence must be between 0 and 1")
    if args.vision_max_side < -1:
        parser.error("--vision-max-side must be -1, 0, or >= 128")
    if args.vision_jpeg_quality < 30 or args.vision_jpeg_quality > 95:
        parser.error("--vision-jpeg-quality must be between 30 and 95")

    if args.vision_max_side == -1:
        args.vision_max_side = 256 if args.backend == "ollama" else 0
    if args.vision_max_side not in (0,) and args.vision_max_side < 128:
        parser.error("--vision-max-side must be 0 or >= 128")

    if args.stage_b_use_image is None:
        args.stage_b_use_image = args.backend != "ollama"
    if args.checker_use_image is None:
        args.checker_use_image = args.backend != "ollama"

    return args


def main() -> int:
    args = parse_args()
    if args.backend == "ollama" and args.max_output_tokens < 400:
        print(
            "Warning: --max-output-tokens < 400 can truncate JSON on Ollama. "
            "Consider 600+."
        )
    if args.backend == "ollama" and args.stage_b_use_image:
        print(
            "Warning: --stage-b-use-image is enabled on Ollama; this may slow "
            "inference significantly."
        )
    if args.backend == "ollama" and args.part_segmentation:
        print(
            "Info: --part-segmentation is enabled on Ollama; "
            "use --no-part-segmentation for faster runtime."
        )
    result = run_pipeline(args)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Saved result: {output_path}")
    print(f"status={result['status']} effective_score={result['effective_score']}")

    if result["status"] == "ok":
        return 0
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
