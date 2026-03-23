#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]


SYSTEM_PROMPT = (
    "You are a chair seat-contact annotator. "
    "Given one chair image, output exactly one JSON object and nothing else. "
    "Return an object with exactly these top-level keys only: "
    "task, image_size, seat_contact_polygon. "
    "task must be the exact string seat_contact_segmentation_points. "
    "image_size must be an object with integer fields width and height matching the input image size. "
    "seat_contact_polygon must contain at least 6 points. "
    "Each polygon point must be an object with numeric pixel coordinates x and y. "
    "Do not emit placeholder tokens or quoted numbers for numeric fields."
)

USER_PROMPT = (
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
    orig_width: int
    orig_height: int
    polygon: list[dict[str, float]]
    source_dir: Path


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
        default=Path("/Users/ganghyeongyu/Documents/chairpose/data/vlm_seatcontact"),
        help="Output directory for train/val/test jsonl.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="seat_contact",
        help="LabelMe label name to extract.",
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


def extract_largest_polygon(
    labelme_obj: dict[str, Any],
    label_name: str,
    width: int,
    height: int,
    min_area: float,
    fixed_points: int,
    integer_coords: bool,
) -> list[dict[str, float]]:
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

    if len(best_poly) < 3:
        return []

    poly = resample_polygon_uniform(best_poly, int(fixed_points))
    if len(poly) < 3:
        return []

    out: list[dict[str, float]] = []
    for x, y in poly:
        cx, cy = clamp_point(x, y, width, height)
        if integer_coords:
            out.append({"x": int(round(cx)), "y": int(round(cy))})
        else:
            out.append({"x": round(cx, 2), "y": round(cy, 2)})
    return out


def scale_polygon(
    polygon: list[dict[str, float]],
    src_width: int,
    src_height: int,
    dst_width: int,
    dst_height: int,
    integer_coords: bool,
) -> list[dict[str, float]]:
    if src_width <= 0 or src_height <= 0 or dst_width <= 0 or dst_height <= 0:
        return polygon
    sx = float(dst_width) / float(src_width)
    sy = float(dst_height) / float(src_height)
    out: list[dict[str, float]] = []
    for p in polygon:
        x = float(p["x"]) * sx
        y = float(p["y"]) * sy
        cx, cy = clamp_point(x, y, dst_width, dst_height)
        if integer_coords:
            out.append({"x": int(round(cx)), "y": int(round(cy))})
        else:
            out.append({"x": round(cx, 2), "y": round(cy, 2)})
    return out


def build_messages(
    image_path: Path,
    width: int,
    height: int,
    polygon: list[dict[str, float]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    del image_path
    del rng
    target = {
        "task": "seat_contact_segmentation_points",
        "image_size": {"width": int(width), "height": int(height)},
        "seat_contact_polygon": polygon,
    }
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": USER_PROMPT},
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
    }

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

            polygon = extract_largest_polygon(
                labelme_obj=labelme_obj,
                label_name=args.label,
                width=width,
                height=height,
                min_area=float(args.min_area),
                fixed_points=int(args.fixed_points),
                integer_coords=bool(args.integer_coords),
            )
            if len(polygon) < 3:
                stats["skipped_empty_label"] += 1
                continue

            out_width = int(args.target_width) if int(args.target_width) > 0 else int(width)
            out_height = int(args.target_height) if int(args.target_height) > 0 else int(height)
            if out_width != int(width) or out_height != int(height):
                polygon = scale_polygon(
                    polygon=polygon,
                    src_width=int(width),
                    src_height=int(height),
                    dst_width=out_width,
                    dst_height=out_height,
                    integer_coords=bool(args.integer_coords),
                )

            sample = Sample(
                stem=stem,
                image_path=image_path.resolve(),
                json_path=jf.resolve(),
                width=out_width,
                height=out_height,
                orig_width=int(width),
                orig_height=int(height),
                polygon=polygon,
                source_dir=src.resolve(),
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
    if not enabled:
        return {}

    out_map: dict[str, Path] = {}
    img_dir = out_dir / subdir
    img_dir.mkdir(parents=True, exist_ok=True)

    try:
        resample = Image.Resampling.BICUBIC  # type: ignore[attr-defined]
    except Exception:
        resample = Image.BICUBIC

    for s in samples:
        dst = img_dir / f"{s.stem}.jpg"
        with Image.open(s.image_path) as im:
            resized = im.convert("RGB").resize((int(s.width), int(s.height)), resample=resample)
            resized.save(dst, format="JPEG", quality=95, optimize=True)
        out_map[s.stem] = dst.resolve()
    return out_map


def main() -> int:
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

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
                        polygon=s.polygon,
                        rng=rng,
                    ),
                    "meta": {
                        "stem": s.stem,
                        "source_dir": str(s.source_dir),
                        "labelme_json": str(s.json_path),
                        "image_width": s.width,
                        "image_height": s.height,
                        "orig_image_width": s.orig_width,
                        "orig_image_height": s.orig_height,
                        "resized_image_path": str(used_image_path),
                        "uses_resized_image": bool(s.stem in resized_image_map),
                        "polygon_points": len(s.polygon),
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
        "id,split,image_path,labelme_json,width,height,orig_width,orig_height,polygon_points,source_dir"
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
                        str(m["labelme_json"]),
                        str(m["image_width"]),
                        str(m["image_height"]),
                        str(m["orig_image_width"]),
                        str(m["orig_image_height"]),
                        str(m["polygon_points"]),
                        str(m["source_dir"]),
                    ]
                )
            )
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    report = {
        "out_dir": str(out_dir),
        "label": str(args.label),
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
