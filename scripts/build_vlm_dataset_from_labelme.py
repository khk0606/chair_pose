#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHAIR_DETECTOR = "yolov8x-seg.pt"


def resolve_task_name(target_shape: str, coord_mode: str = "pixel") -> str:
    if target_shape == "bbox" and coord_mode == "grid":
        return "seat_contact_bounding_box_grid"
    if target_shape == "bbox":
        return "seat_contact_bounding_box"
    if target_shape == "quad":
        return "seat_contact_quad_points"
    return "seat_contact_segmentation_points"


def resolve_target_key(target_shape: str) -> str:
    if target_shape == "bbox":
        return "seat_contact_box"
    return "seat_contact_polygon"


def build_system_prompt(
    target_shape: str,
    coord_mode: str = "pixel",
    grid_size: int = 16,
    crop_mode: str = "none",
) -> str:
    crop_hint = (
        "The input image is already cropped around a single chair. "
        if crop_mode == "chair"
        else ""
    )
    if target_shape == "bbox" and coord_mode == "grid":
        return (
            "You are a chair seat-contact annotator. "
            f"{crop_hint}"
            "Given one chair image, output exactly one JSON object and nothing else. "
            "Return an object with exactly these top-level keys only: "
            "task, image_size, seat_contact_box. "
            "task must be the exact string seat_contact_bounding_box_grid. "
            f"image_size must be an object with integer fields width={int(grid_size)} and height={int(grid_size)}. "
            f"Divide the image into a {int(grid_size)} by {int(grid_size)} grid. "
            "seat_contact_box must be an object with integer fields x_min, y_min, x_max, y_max "
            "using grid indices, not pixel coordinates. "
            "The box must tightly cover only the visible seat-contact area. "
            "Do not emit placeholder tokens or quoted numbers for numeric fields."
        )
    if target_shape == "bbox":
        return (
            "You are a chair seat-contact annotator. "
            f"{crop_hint}"
            "Given one chair image, output exactly one JSON object and nothing else. "
            "Return an object with exactly these top-level keys only: "
            "task, image_size, seat_contact_box. "
            "task must be the exact string seat_contact_bounding_box. "
            "image_size must be an object with integer fields width and height matching the input image size. "
            "seat_contact_box must be an object with integer fields x_min, y_min, x_max, y_max. "
            "The box must tightly cover only the visible seat-contact area. "
            "Do not emit placeholder tokens or quoted numbers for numeric fields."
        )
    if target_shape == "quad":
        return (
            "You are a chair seat-contact annotator. "
            f"{crop_hint}"
            "Given one chair image, output exactly one JSON object and nothing else. "
            "Return an object with exactly these top-level keys only: "
            "task, image_size, seat_contact_polygon. "
            "task must be the exact string seat_contact_quad_points. "
            "image_size must be an object with integer fields width and height matching the input image size. "
            "seat_contact_polygon must contain exactly 4 points describing the seat-contact quadrilateral corners in clockwise order. "
            "Each polygon point must be an object with numeric pixel coordinates x and y. "
            "Do not emit placeholder tokens or quoted numbers for numeric fields."
        )
    return (
        "You are a chair seat-contact annotator. "
        f"{crop_hint}"
        "Given one chair image, output exactly one JSON object and nothing else. "
        "Return an object with exactly these top-level keys only: "
        "task, image_size, seat_contact_polygon. "
        "task must be the exact string seat_contact_segmentation_points. "
        "image_size must be an object with integer fields width and height matching the input image size. "
        "seat_contact_polygon must contain at least 6 points. "
        "Each polygon point must be an object with numeric pixel coordinates x and y. "
        "Do not emit placeholder tokens or quoted numbers for numeric fields."
    )


def build_user_prompt(
    target_shape: str,
    coord_mode: str = "pixel",
    grid_size: int = 16,
    crop_mode: str = "none",
) -> str:
    crop_hint = "이미지는 의자 주변으로 crop되어 있어. " if crop_mode == "chair" else ""
    if target_shape == "bbox" and coord_mode == "grid":
        return (
            f"{crop_hint}"
            "의자 이미지에서 사람이 실제로 앉는 좌판 면만 감싸는 bounding box를 반환해. "
            f"이미지를 {int(grid_size)}x{int(grid_size)} grid로 본다고 생각하고 "
            "seat_contact_box에 x_min, y_min, x_max, y_max 정수 grid index만 넣어. "
            "JSON 하나만 출력하고 설명/마크다운/코드블록은 절대 쓰지 마."
        )
    if target_shape == "bbox":
        return (
            f"{crop_hint}"
            "의자 이미지에서 사람이 실제로 앉는 좌판 면만 감싸는 bounding box를 반환해. "
            "JSON 하나만 출력하고 seat_contact_box에 x_min, y_min, x_max, y_max 정수만 넣어. "
            "설명/마크다운/코드블록은 절대 쓰지 마."
        )
    if target_shape == "quad":
        return (
            f"{crop_hint}"
            "의자 이미지에서 사람이 실제로 앉는 좌판 면의 네 꼭짓점만 quadrilateral 점 좌표로 반환해. "
            "점은 시계 방향으로 4개만 주고 JSON 하나만 출력해. 설명/마크다운/코드블록은 절대 쓰지 마."
        )
    return (
        f"{crop_hint}"
        "의자 이미지에서 사람이 실제로 앉는 쿠션 경계만 polygon 점 좌표로 반환해. "
        "JSON 하나만 출력하고 설명/마크다운/코드블록은 절대 쓰지 마."
    )


@dataclass
class Sample:
    stem: str
    image_path: Path
    json_path: Path
    width: int
    height: int
    render_width: int
    render_height: int
    orig_width: int
    orig_height: int
    target: dict[str, Any] | list[dict[str, float]]
    target_count: int
    source_dir: Path
    crop_box: tuple[int, int, int, int] | None
    chair_bbox: tuple[float, float, float, float] | None
    chair_detected: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build VLM SFT dataset from LabelMe chair seat-contact annotations."
    )
    parser.add_argument(
        "--src-dir",
        action="append",
        type=Path,
        required=True,
        help="Source directory containing .json/.jpg pairs. Repeatable.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "vlm_seatcontact",
        help="Output directory for train/val/test jsonl.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="seat_contact",
        help="LabelMe label name to extract.",
    )
    parser.add_argument(
        "--target-shape",
        type=str,
        default="polygon",
        choices=["polygon", "quad", "bbox"],
        help="Target geometry emitted into assistant JSON.",
    )
    parser.add_argument(
        "--coord-mode",
        type=str,
        default="pixel",
        choices=["pixel", "grid"],
        help="Coordinate frame emitted into assistant JSON.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=16,
        help="Grid width/height used when --coord-mode=grid.",
    )
    parser.add_argument(
        "--crop-mode",
        type=str,
        default="none",
        choices=["none", "chair"],
        help="Optional pre-crop applied before building VLM samples.",
    )
    parser.add_argument(
        "--chair-detector",
        type=str,
        default=str(DEFAULT_CHAIR_DETECTOR),
        help="YOLO chair detector used when --crop-mode=chair.",
    )
    parser.add_argument(
        "--chair-conf",
        type=float,
        default=0.2,
        help="Confidence threshold for chair detection when --crop-mode=chair.",
    )
    parser.add_argument(
        "--chair-device",
        type=str,
        default="mps",
        help="Inference device for chair detector when --crop-mode=chair.",
    )
    parser.add_argument(
        "--crop-margin-ratio",
        type=float,
        default=0.10,
        help="Margin ratio around detected chair bbox when --crop-mode=chair.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument(
        "--train-ratio", type=float, default=0.85, help="Train split ratio."
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.10, help="Validation split ratio."
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=8.0,
        help="Minimum polygon area (pixels) to keep.",
    )
    parser.add_argument(
        "--fixed-points",
        type=int,
        default=12,
        help="Resample polygon to a fixed number of points.",
    )
    parser.add_argument(
        "--integer-coords",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit polygon coordinates as integers.",
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=0,
        help="If >0, scale polygon coordinates and image_size.width to this value.",
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=0,
        help="If >0, scale polygon coordinates and image_size.height to this value.",
    )
    parser.add_argument(
        "--write-resized-images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If enabled, write resized image copies under out-dir and use those "
            "paths in jsonl rows."
        ),
    )
    parser.add_argument(
        "--resized-images-subdir",
        type=str,
        default="images_resized",
        help="Subdirectory name under out-dir to store resized images.",
    )
    parser.add_argument(
        "--prefer-last-source",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If duplicated stem exists across sources, keep sample from later src-dir.",
    )
    return parser.parse_args()


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    acc = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        acc += x1 * y2 - x2 * y1
    return abs(acc) * 0.5


def find_image_for_stem(src_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        p = src_dir / f"{stem}{ext}"
        if p.exists():
            return p
    # Case-insensitive fallback.
    for p in src_dir.glob(f"{stem}.*"):
        if p.suffix.lower() in IMAGE_EXTS:
            return p
    return None


def clamp_point(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    cx = min(max(float(x), 0.0), float(max(0, width - 1)))
    cy = min(max(float(y), 0.0), float(max(0, height - 1)))
    return cx, cy


def require_ultralytics() -> Any:
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "ultralytics is required for --crop-mode=chair. "
            "Install it in the active environment first."
        ) from exc
    return YOLO


def detect_chair_box(
    model: Any,
    image_path: Path,
    *,
    conf: float,
    device: str,
) -> tuple[float, float, float, float] | None:
    pred = model.predict(
        source=str(image_path),
        classes=[56],
        conf=float(conf),
        retina_masks=True,
        device=device,
        verbose=False,
    )
    if not pred:
        return None
    result = pred[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    best_i = max(
        range(len(boxes)),
        key=lambda i: float(confs[i])
        * max(float((boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])), 1.0),
    )
    x0, y0, x1, y1 = boxes[best_i].tolist()
    return float(x0), float(y0), float(x1), float(y1)


def expand_crop_box(
    chair_box: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
    margin_ratio: float,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = chair_box
    bw = max(2.0, x1 - x0)
    bh = max(2.0, y1 - y0)
    mx = bw * float(margin_ratio)
    my = bh * float(margin_ratio)
    cx0 = max(0, int(round(x0 - mx)))
    cy0 = max(0, int(round(y0 - my)))
    cx1 = min(int(image_width), int(round(x1 + mx)))
    cy1 = min(int(image_height), int(round(y1 + my)))
    if cx1 <= cx0 + 1 or cy1 <= cy0 + 1:
        return 0, 0, int(image_width), int(image_height)
    return cx0, cy0, cx1, cy1


def _dedupe_polygon_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not points:
        return []
    out: list[tuple[float, float]] = [points[0]]
    for x, y in points[1:]:
        px, py = out[-1]
        if abs(x - px) > 1e-6 or abs(y - py) > 1e-6:
            out.append((x, y))
    if len(out) > 1:
        x0, y0 = out[0]
        x1, y1 = out[-1]
        if abs(x0 - x1) <= 1e-6 and abs(y0 - y1) <= 1e-6:
            out.pop()
    return out


def resample_polygon_uniform(
    points: list[tuple[float, float]], n_points: int
) -> list[tuple[float, float]]:
    pts = _dedupe_polygon_points(points)
    if len(pts) < 3:
        return []
    if n_points <= 0:
        return pts

    ring = pts + [pts[0]]
    seg_lens: list[float] = []
    total_len = 0.0
    for i in range(len(pts)):
        x1, y1 = ring[i]
        x2, y2 = ring[i + 1]
        seg_len = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        seg_lens.append(seg_len)
        total_len += seg_len

    if total_len <= 1e-6:
        return pts[:n_points]

    targets = [(total_len * i) / float(n_points) for i in range(n_points)]
    sampled: list[tuple[float, float]] = []
    seg_idx = 0
    seg_start = 0.0

    for t in targets:
        while seg_idx < len(seg_lens) - 1 and seg_start + seg_lens[seg_idx] < t:
            seg_start += seg_lens[seg_idx]
            seg_idx += 1

        x1, y1 = ring[seg_idx]
        x2, y2 = ring[seg_idx + 1]
        seg_len = seg_lens[seg_idx]
        if seg_len <= 1e-6:
            sampled.append((x1, y1))
            continue
        ratio = (t - seg_start) / seg_len
        sampled.append((x1 + (x2 - x1) * ratio, y1 + (y2 - y1) * ratio))

    return sampled


def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = sorted({(round(x, 6), round(y, 6)) for x, y in points})
    if len(pts) <= 1:
        return list(pts)

    def cross(
        o: tuple[float, float],
        a: tuple[float, float],
        b: tuple[float, float],
    ) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def order_clockwise_start_top_left(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    if len(points) <= 1:
        return points[:]
    cx = sum(x for x, _ in points) / float(len(points))
    cy = sum(y for _, y in points) / float(len(points))
    ordered = sorted(
        points,
        key=lambda p: (-math.atan2(p[1] - cy, p[0] - cx)),
    )
    start_idx = min(range(len(ordered)), key=lambda i: ordered[i][0] + ordered[i][1])
    return ordered[start_idx:] + ordered[:start_idx]


def quad_from_bbox(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [
        (min(xs), min(ys)),
        (max(xs), min(ys)),
        (max(xs), max(ys)),
        (min(xs), max(ys)),
    ]


def polygon_to_quad(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = _dedupe_polygon_points(points)
    if len(pts) < 4:
        return []
    hull = convex_hull(pts)
    src = hull if len(hull) >= 4 else pts

    tl = min(src, key=lambda p: p[0] + p[1])
    tr = max(src, key=lambda p: p[0] - p[1])
    br = max(src, key=lambda p: p[0] + p[1])
    bl = max(src, key=lambda p: p[1] - p[0])

    quad = [tl, tr, br, bl]
    deduped = _dedupe_polygon_points(quad)
    unique_count = len({(round(x, 3), round(y, 3)) for x, y in deduped})
    if unique_count < 4:
        deduped = quad_from_bbox(src)

    ordered = order_clockwise_start_top_left(deduped)
    if len({(round(x, 3), round(y, 3)) for x, y in ordered}) < 4:
        return []
    if polygon_area(ordered) <= 1e-3:
        return []
    return ordered[:4]


def extract_largest_polygon_points(
    labelme_obj: dict[str, Any],
    label_name: str,
    width: int,
    height: int,
    min_area: float,
) -> list[tuple[float, float]]:
    best_poly: list[tuple[float, float]] = []
    best_area = 0.0
    for shape in labelme_obj.get("shapes", []):
        if not isinstance(shape, dict):
            continue
        if shape.get("label") != label_name:
            continue
        points_raw = shape.get("points")
        if not isinstance(points_raw, list):
            continue
        pts: list[tuple[float, float]] = []
        for p in points_raw:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                continue
            x, y = clamp_point(float(p[0]), float(p[1]), width, height)
            pts.append((x, y))
        if len(pts) < 3:
            continue
        area = polygon_area(pts)
        if area >= min_area and area > best_area:
            best_area = area
            best_poly = pts

    return best_poly


def format_points(
    points: list[tuple[float, float]],
    width: int,
    height: int,
    integer_coords: bool,
) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for x, y in points:
        cx, cy = clamp_point(x, y, width, height)
        if integer_coords:
            out.append({"x": int(round(cx)), "y": int(round(cy))})
        else:
            out.append({"x": round(cx, 2), "y": round(cy, 2)})
    return out


def polygon_to_bbox(points: list[tuple[float, float]]) -> dict[str, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return {
        "x_min": min(xs),
        "y_min": min(ys),
        "x_max": max(xs),
        "y_max": max(ys),
    }


def format_box(
    box: dict[str, float],
    width: int,
    height: int,
    integer_coords: bool,
) -> dict[str, float]:
    x_min, y_min = clamp_point(float(box["x_min"]), float(box["y_min"]), width, height)
    x_max, y_max = clamp_point(float(box["x_max"]), float(box["y_max"]), width, height)
    if x_max < x_min:
        x_min, x_max = x_max, x_min
    if y_max < y_min:
        y_min, y_max = y_max, y_min
    if integer_coords:
        x_min_i = int(math.floor(x_min))
        y_min_i = int(math.floor(y_min))
        x_max_i = int(math.ceil(x_max))
        y_max_i = int(math.ceil(y_max))
        x_min_i = min(max(x_min_i, 0), max(0, width - 1))
        y_min_i = min(max(y_min_i, 0), max(0, height - 1))
        x_max_i = min(max(x_max_i, 0), max(0, width - 1))
        y_max_i = min(max(y_max_i, 0), max(0, height - 1))
        if width >= 2 and x_max_i <= x_min_i:
            if x_min_i < width - 1:
                x_max_i = x_min_i + 1
            else:
                x_min_i = max(0, x_min_i - 1)
        if height >= 2 and y_max_i <= y_min_i:
            if y_min_i < height - 1:
                y_max_i = y_min_i + 1
            else:
                y_min_i = max(0, y_min_i - 1)
        return {
            "x_min": x_min_i,
            "y_min": y_min_i,
            "x_max": x_max_i,
            "y_max": y_max_i,
        }
    return {
        "x_min": round(x_min, 2),
        "y_min": round(y_min, 2),
        "x_max": round(x_max, 2),
        "y_max": round(y_max, 2),
    }


def extract_target_annotation(
    labelme_obj: dict[str, Any],
    label_name: str,
    width: int,
    height: int,
    min_area: float,
    fixed_points: int,
    integer_coords: bool,
    target_shape: str,
) -> tuple[dict[str, Any] | list[dict[str, float]], int]:
    best_poly = extract_largest_polygon_points(
        labelme_obj=labelme_obj,
        label_name=label_name,
        width=width,
        height=height,
        min_area=min_area,
    )
    if len(best_poly) < 3:
        return [], 0

    if target_shape == "bbox":
        box = polygon_to_bbox(best_poly)
        return format_box(box, width, height, integer_coords), 4

    if target_shape == "quad":
        poly = polygon_to_quad(best_poly)
        if len(poly) != 4:
            return [], 0
    else:
        poly = resample_polygon_uniform(best_poly, int(fixed_points))
        if len(poly) < 3:
            return [], 0

    return format_points(poly, width, height, integer_coords), len(poly)


def scale_target(
    target: dict[str, Any] | list[dict[str, float]],
    target_shape: str,
    src_width: int,
    src_height: int,
    dst_width: int,
    dst_height: int,
    integer_coords: bool,
) -> dict[str, Any] | list[dict[str, float]]:
    if src_width <= 0 or src_height <= 0 or dst_width <= 0 or dst_height <= 0:
        return target
    sx = float(dst_width) / float(src_width)
    sy = float(dst_height) / float(src_height)
    if target_shape == "bbox":
        if not isinstance(target, dict):
            return target
        scaled = {
            "x_min": float(target["x_min"]) * sx,
            "y_min": float(target["y_min"]) * sy,
            "x_max": float(target["x_max"]) * sx,
            "y_max": float(target["y_max"]) * sy,
        }
        return format_box(scaled, dst_width, dst_height, integer_coords)

    out: list[dict[str, float]] = []
    if not isinstance(target, list):
        return out
    for p in target:
        x = float(p["x"]) * sx
        y = float(p["y"]) * sy
        cx, cy = clamp_point(x, y, dst_width, dst_height)
        if integer_coords:
            out.append({"x": int(round(cx)), "y": int(round(cy))})
        else:
            out.append({"x": round(cx, 2), "y": round(cy, 2)})
    return out


def shift_target_into_crop(
    target: dict[str, Any] | list[dict[str, float]],
    target_shape: str,
    crop_box: tuple[int, int, int, int],
    crop_width: int,
    crop_height: int,
    integer_coords: bool,
) -> dict[str, Any] | list[dict[str, float]]:
    cx0, cy0, _, _ = crop_box
    if target_shape == "bbox":
        if not isinstance(target, dict):
            return target
        shifted = {
            "x_min": float(target["x_min"]) - float(cx0),
            "y_min": float(target["y_min"]) - float(cy0),
            "x_max": float(target["x_max"]) - float(cx0),
            "y_max": float(target["y_max"]) - float(cy0),
        }
        return format_box(shifted, crop_width, crop_height, integer_coords)

    shifted_poly: list[dict[str, float]] = []
    if not isinstance(target, list):
        return shifted_poly
    for p in target:
        x = float(p["x"]) - float(cx0)
        y = float(p["y"]) - float(cy0)
        sx, sy = clamp_point(x, y, crop_width, crop_height)
        if integer_coords:
            shifted_poly.append({"x": int(round(sx)), "y": int(round(sy))})
        else:
            shifted_poly.append({"x": round(sx, 2), "y": round(sy, 2)})
    return shifted_poly


def build_messages(
    image_path: Path,
    width: int,
    height: int,
    target_value: dict[str, Any] | list[dict[str, float]],
    rng: random.Random,
    task_name: str,
    target_key: str,
    system_prompt: str,
    user_prompt: str,
) -> list[dict[str, Any]]:
    del image_path
    del rng
    target: dict[str, Any] = {
        "task": task_name,
        "image_size": {"width": int(width), "height": int(height)},
    }
    target[target_key] = target_value
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        target,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ],
        },
    ]


def collect_samples(args: argparse.Namespace) -> tuple[list[Sample], dict[str, Any]]:
    sample_by_stem: dict[str, Sample] = {}
    stats: dict[str, Any] = {
        "source_dirs": [str(d) for d in args.src_dir],
        "skipped_no_image": 0,
        "skipped_bad_json": 0,
        "skipped_empty_label": 0,
        "duplicates_overwritten": 0,
        "chair_detect_success": 0,
        "chair_detect_fail": 0,
    }

    chair_model = None
    if str(args.crop_mode) == "chair":
        YOLO = require_ultralytics()
        chair_model = YOLO(str(args.chair_detector))

    for src in args.src_dir:
        if not src.exists():
            raise FileNotFoundError(f"src-dir not found: {src}")
        for jf in sorted(src.glob("*.json")):
            stem = jf.stem
            image_path = find_image_for_stem(src, stem)
            if image_path is None:
                stats["skipped_no_image"] += 1
                continue
            try:
                labelme_obj = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                stats["skipped_bad_json"] += 1
                continue

            with Image.open(image_path) as im:
                width, height = im.size

            target_value, target_count = extract_target_annotation(
                labelme_obj=labelme_obj,
                label_name=args.label,
                width=width,
                height=height,
                min_area=float(args.min_area),
                fixed_points=int(args.fixed_points),
                integer_coords=bool(args.integer_coords),
                target_shape=str(args.target_shape),
            )
            if target_count < 3:
                stats["skipped_empty_label"] += 1
                continue

            crop_box: tuple[int, int, int, int] | None = None
            chair_bbox: tuple[float, float, float, float] | None = None
            chair_detected = False
            crop_width = int(width)
            crop_height = int(height)
            if str(args.crop_mode) == "chair":
                if chair_model is None:
                    raise RuntimeError("chair detector failed to initialize")
                chair_bbox = detect_chair_box(
                    chair_model,
                    image_path,
                    conf=float(args.chair_conf),
                    device=str(args.chair_device),
                )
                if chair_bbox is None:
                    stats["chair_detect_fail"] += 1
                    crop_box = (0, 0, int(width), int(height))
                else:
                    chair_detected = True
                    stats["chair_detect_success"] += 1
                    crop_box = expand_crop_box(
                        chair_bbox,
                        image_width=int(width),
                        image_height=int(height),
                        margin_ratio=float(args.crop_margin_ratio),
                    )
                assert crop_box is not None
                crop_width = int(crop_box[2] - crop_box[0])
                crop_height = int(crop_box[3] - crop_box[1])
                target_value = shift_target_into_crop(
                    target=target_value,
                    target_shape=str(args.target_shape),
                    crop_box=crop_box,
                    crop_width=crop_width,
                    crop_height=crop_height,
                    integer_coords=bool(args.integer_coords),
                )

            render_width = (
                int(args.target_width) if int(args.target_width) > 0 else int(crop_width)
            )
            render_height = (
                int(args.target_height) if int(args.target_height) > 0 else int(crop_height)
            )
            if render_width != int(crop_width) or render_height != int(crop_height):
                target_value = scale_target(
                    target=target_value,
                    target_shape=str(args.target_shape),
                    src_width=int(crop_width),
                    src_height=int(crop_height),
                    dst_width=render_width,
                    dst_height=render_height,
                    integer_coords=bool(args.integer_coords),
                )

            coord_width = int(render_width)
            coord_height = int(render_height)
            if str(args.coord_mode) == "grid":
                if str(args.target_shape) != "bbox":
                    raise ValueError("--coord-mode=grid currently supports only --target-shape=bbox")
                if int(args.grid_size) <= 1:
                    raise ValueError("--grid-size must be > 1 when --coord-mode=grid")
                target_value = scale_target(
                    target=target_value,
                    target_shape=str(args.target_shape),
                    src_width=int(render_width),
                    src_height=int(render_height),
                    dst_width=int(args.grid_size),
                    dst_height=int(args.grid_size),
                    integer_coords=True,
                )
                coord_width = int(args.grid_size)
                coord_height = int(args.grid_size)

            sample = Sample(
                stem=stem,
                image_path=image_path.resolve(),
                json_path=jf.resolve(),
                width=coord_width,
                height=coord_height,
                render_width=render_width,
                render_height=render_height,
                orig_width=int(width),
                orig_height=int(height),
                target=target_value,
                target_count=int(target_count),
                source_dir=src.resolve(),
                crop_box=crop_box,
                chair_bbox=chair_bbox,
                chair_detected=chair_detected,
            )

            if stem in sample_by_stem:
                if args.prefer_last_source:
                    sample_by_stem[stem] = sample
                    stats["duplicates_overwritten"] += 1
                continue
            sample_by_stem[stem] = sample

    return list(sample_by_stem.values()), stats


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [json.dumps(r, ensure_ascii=False) for r in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_resized_image_map(
    samples: list[Sample],
    out_dir: Path,
    enabled: bool,
    subdir: str,
) -> dict[str, Path]:
    out_map: dict[str, Path] = {}
    img_dir = out_dir / subdir

    try:
        resample = Image.Resampling.BICUBIC  # type: ignore[attr-defined]
    except Exception:
        resample = Image.BICUBIC

    for s in samples:
        needs_render = (
            bool(enabled)
            or s.crop_box is not None
            or int(s.render_width) != int(s.orig_width)
            or int(s.render_height) != int(s.orig_height)
        )
        if not needs_render:
            continue
        img_dir.mkdir(parents=True, exist_ok=True)
        dst = img_dir / f"{s.stem}.jpg"
        with Image.open(s.image_path) as im:
            rendered = im.convert("RGB")
            if s.crop_box is not None:
                rendered = rendered.crop(s.crop_box)
            if rendered.size != (int(s.render_width), int(s.render_height)):
                rendered = rendered.resize(
                    (int(s.render_width), int(s.render_height)),
                    resample=resample,
                )
            rendered.save(dst, format="JPEG", quality=95, optimize=True)
        out_map[s.stem] = dst.resolve()
    return out_map


def main() -> int:
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    task_name = resolve_task_name(str(args.target_shape), str(args.coord_mode))
    target_key = resolve_target_key(str(args.target_shape))
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

    samples, stats = collect_samples(args)
    if not samples:
        raise RuntimeError("No valid samples found.")
    if args.write_resized_images and (
        int(args.target_width) <= 0 or int(args.target_height) <= 0
    ):
        raise ValueError(
            "--write-resized-images requires both --target-width and --target-height to be > 0."
        )

    rng = random.Random(args.seed)
    rng.shuffle(samples)
    resized_image_map = build_resized_image_map(
        samples=samples,
        out_dir=out_dir,
        enabled=bool(args.write_resized_images),
        subdir=str(args.resized_images_subdir),
    )

    n = len(samples)
    n_train = int(round(n * float(args.train_ratio)))
    n_val = int(round(n * float(args.val_ratio)))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    n_test = n - n_train - n_val

    train_samples = samples[:n_train]
    val_samples = samples[n_train : n_train + n_val]
    test_samples = samples[n_train + n_val :]

    def to_rows(chunk: list[Sample]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for s in chunk:
            used_image_path = resized_image_map.get(s.stem, s.image_path)
            rows.append(
                {
                    "id": s.stem,
                    "images": [str(used_image_path)],
                    "messages": build_messages(
                        image_path=used_image_path,
                        width=s.width,
                        height=s.height,
                        target_value=s.target,
                        rng=rng,
                        task_name=task_name,
                        target_key=target_key,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                    ),
                    "meta": {
                        "stem": s.stem,
                        "source_dir": str(s.source_dir),
                        "labelme_json": str(s.json_path),
                        "image_width": s.width,
                        "image_height": s.height,
                        "rendered_image_width": s.render_width,
                        "rendered_image_height": s.render_height,
                        "orig_image_width": s.orig_width,
                        "orig_image_height": s.orig_height,
                        "resized_image_path": str(used_image_path),
                        "uses_resized_image": bool(s.stem in resized_image_map),
                        "target_count": int(s.target_count),
                        "target_shape": str(args.target_shape),
                        "coord_mode": str(args.coord_mode),
                        "grid_size": int(args.grid_size),
                        "crop_mode": str(args.crop_mode),
                        "crop_box": (
                            list(s.crop_box) if s.crop_box is not None else None
                        ),
                        "chair_bbox": (
                            [float(v) for v in s.chair_bbox]
                            if s.chair_bbox is not None
                            else None
                        ),
                        "chair_detected": bool(s.chair_detected),
                        "task_name": task_name,
                        "target_key": target_key,
                    },
                }
            )
        return rows

    train_rows = to_rows(train_samples)
    val_rows = to_rows(val_samples)
    test_rows = to_rows(test_samples)

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    test_path = out_dir / "test.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    write_jsonl(test_path, test_rows)

    manifest_path = out_dir / "manifest.csv"
    manifest_lines = [
        "id,split,image_path,coord_width,coord_height,render_width,render_height,orig_width,orig_height,target_count,target_shape,coord_mode,crop_mode,chair_detected,labelme_json,source_dir"
    ]
    for split_name, rows in [
        ("train", train_rows),
        ("val", val_rows),
        ("test", test_rows),
    ]:
        for r in rows:
            m = r["meta"]
            manifest_lines.append(
                ",".join(
                    [
                        str(m["stem"]),
                        split_name,
                        str(r["images"][0]),
                        str(m["image_width"]),
                        str(m["image_height"]),
                        str(m["rendered_image_width"]),
                        str(m["rendered_image_height"]),
                        str(m["orig_image_width"]),
                        str(m["orig_image_height"]),
                        str(m["target_count"]),
                        str(m["target_shape"]),
                        str(m["coord_mode"]),
                        str(m["crop_mode"]),
                        str(int(bool(m["chair_detected"]))),
                        str(m["labelme_json"]),
                        str(m["source_dir"]),
                    ]
                )
            )
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    report = {
        "out_dir": str(out_dir),
        "label": str(args.label),
        "target_shape": str(args.target_shape),
        "coord_mode": str(args.coord_mode),
        "grid_size": int(args.grid_size),
        "crop_mode": str(args.crop_mode),
        "task_name": task_name,
        "target_key": target_key,
        "seed": int(args.seed),
        "write_resized_images": bool(args.write_resized_images),
        "resized_images_subdir": str(args.resized_images_subdir),
        "resized_images_count": int(len(resized_image_map)),
        "counts": {
            "total": int(n),
            "train": int(n_train),
            "val": int(n_val),
            "test": int(n_test),
        },
        "stats": stats,
        "files": {
            "train_jsonl": str(train_path),
            "val_jsonl": str(val_path),
            "test_jsonl": str(test_path),
            "manifest_csv": str(manifest_path),
        },
        "target_size": {
            "width": int(args.target_width),
            "height": int(args.target_height),
        },
    }
    report_path = out_dir / "build_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
