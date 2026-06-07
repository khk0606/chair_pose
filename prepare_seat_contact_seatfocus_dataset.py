#!/usr/bin/env python3
"""
Prepare a seat-focused YOLO segmentation dataset from LabelMe JSON.

This builder crops around the annotated seat_contact region (with context margins)
to reduce floor/background dominance and improve seat localization stability.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build seat-focused YOLO-seg dataset from LabelMe JSON."
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=REPO_ROOT / "data" / "labelme",
        help="Directory containing LabelMe JSON and source images.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "seat_contact_yolo_seatfocus",
        help="Output YOLO dataset root.",
    )
    parser.add_argument(
        "--x-margin-ratio",
        type=float,
        default=0.55,
        help="Horizontal margin relative to seat bbox width.",
    )
    parser.add_argument(
        "--top-margin-ratio",
        type=float,
        default=1.35,
        help="Top margin relative to seat bbox height (keeps backrest context).",
    )
    parser.add_argument(
        "--bottom-margin-ratio",
        type=float,
        default=0.70,
        help="Bottom margin relative to seat bbox height.",
    )
    parser.add_argument(
        "--min-seat-area",
        type=float,
        default=40.0,
        help="Skip samples whose labeled seat polygon area is below this threshold.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def clip_polygon_to_rect(
    points: list[tuple[float, float]],
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> list[tuple[float, float]]:
    w = int(max(2, round(x_max - x_min)))
    h = int(max(2, round(y_max - y_min)))
    src = np.zeros((h, w), dtype=np.uint8)

    shifted = np.array([[p[0] - x_min, p[1] - y_min] for p in points], dtype=np.float32)
    if shifted.shape[0] < 3:
        return []

    cv2.fillPoly(src, [shifted.astype(np.int32)], 255)
    contours, _ = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < 4:
        return []

    peri = cv2.arcLength(best, True)
    approx = cv2.approxPolyDP(best, max(1.0, 0.003 * peri), True)
    out: list[tuple[float, float]] = []
    for p in approx:
        out.append((float(p[0][0]) + x_min, float(p[0][1]) + y_min))
    return out


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def main() -> int:
    args = parse_args()

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    for split in ["train", "val", "test"]:
        (args.out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (args.out_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    json_files = sorted(args.src_dir.glob("*.json"))
    samples: list[dict[str, Any]] = []
    skipped_missing = 0
    skipped_no_label = 0
    skipped_tiny = 0

    for jf in json_files:
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            skipped_missing += 1
            continue

        image_path_name = d.get("imagePath")
        if not image_path_name:
            skipped_missing += 1
            continue
        img_path = args.src_dir / image_path_name
        if not img_path.exists():
            skipped_missing += 1
            continue

        shapes = d.get("shapes") or []
        polys: list[list[tuple[float, float]]] = []
        total_area = 0.0
        for sh in shapes:
            if sh.get("label") != "seat_contact":
                continue
            pts = sh.get("points") or []
            if len(pts) < 3:
                continue
            poly: list[tuple[float, float]] = []
            for p in pts:
                if not isinstance(p, (list, tuple)) or len(p) < 2:
                    continue
                poly.append((float(p[0]), float(p[1])))
            if len(poly) >= 3:
                polys.append(poly)
                total_area += polygon_area(poly)

        if not polys:
            skipped_no_label += 1
            continue
        if total_area < float(args.min_seat_area):
            skipped_tiny += 1
            continue

        samples.append(
            {
                "sample_id": img_path.stem,
                "json_path": jf,
                "image_path": img_path,
                "polygons": polys,
                "seat_area": total_area,
            }
        )

    uniq: dict[str, dict[str, Any]] = {}
    for s in samples:
        sid = str(s["sample_id"])
        if sid not in uniq:
            uniq[sid] = s
    samples = [uniq[k] for k in sorted(uniq.keys())]

    rng = random.Random(args.seed)
    rng.shuffle(samples)

    n = len(samples)
    n_train = int(round(n * 0.8))
    n_val = int(round(n * 0.1))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)

    split_map: dict[str, str] = {}
    for i, s in enumerate(samples):
        sid = str(s["sample_id"])
        if i < n_train:
            split_map[sid] = "train"
        elif i < n_train + n_val:
            split_map[sid] = "val"
        else:
            split_map[sid] = "test"

    manifest = [
        "sample_id,split,image_path,json_path,seat_bbox,crop_box,num_polygons,seat_area"
    ]
    kept = 0

    for s in samples:
        sid = str(s["sample_id"])
        split = split_map[sid]
        img_path: Path = s["image_path"]
        polygons: list[list[tuple[float, float]]] = s["polygons"]

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        all_x = [p[0] for poly in polygons for p in poly]
        all_y = [p[1] for poly in polygons for p in poly]
        if not all_x or not all_y:
            continue

        sx0 = max(0.0, min(all_x))
        sy0 = max(0.0, min(all_y))
        sx1 = min(float(w - 1), max(all_x))
        sy1 = min(float(h - 1), max(all_y))
        sbw = max(2.0, sx1 - sx0)
        sbh = max(2.0, sy1 - sy0)

        mx = float(args.x_margin_ratio) * sbw
        mt = float(args.top_margin_ratio) * sbh
        mb = float(args.bottom_margin_ratio) * sbh

        cx0 = max(0, int(round(sx0 - mx)))
        cy0 = max(0, int(round(sy0 - mt)))
        cx1 = min(w, int(round(sx1 + mx)))
        cy1 = min(h, int(round(sy1 + mb)))
        if cx1 <= cx0 + 1 or cy1 <= cy0 + 1:
            continue

        crop = img.crop((cx0, cy0, cx1, cy1))
        cw, ch = crop.size

        out_img = args.out_dir / split / "images" / f"{sid}.jpg"
        out_lbl = args.out_dir / split / "labels" / f"{sid}.txt"
        crop.save(out_img, quality=95)

        label_lines: list[str] = []
        for poly in polygons:
            clipped = clip_polygon_to_rect(poly, float(cx0), float(cy0), float(cx1), float(cy1))
            if len(clipped) < 3:
                continue
            coords: list[str] = []
            for x, y in clipped:
                xn = min(max((x - cx0) / float(cw), 0.0), 1.0)
                yn = min(max((y - cy0) / float(ch), 0.0), 1.0)
                coords.append(f"{xn:.6f}")
                coords.append(f"{yn:.6f}")
            if len(coords) >= 6:
                label_lines.append("0 " + " ".join(coords))

        if not label_lines:
            out_lbl.write_text("", encoding="utf-8")
        else:
            out_lbl.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

        manifest.append(
            (
                f"{sid},{split},{out_img},{s['json_path']},"
                f"{sx0:.2f}|{sy0:.2f}|{sx1:.2f}|{sy1:.2f},"
                f"{cx0}|{cy0}|{cx1}|{cy1},{len(label_lines)},{float(s['seat_area']):.2f}"
            )
        )
        kept += 1

    (args.out_dir / "manifest.csv").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    (args.out_dir / "dataset.yaml").write_text(
        "\n".join(
            [
                f"path: {args.out_dir}",
                "train: train/images",
                "val: val/images",
                "test: test/images",
                "",
                "names:",
                "  0: seat_contact",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"samples_total={len(samples)}")
    print(f"samples_written={kept}")
    print(f"skipped_missing={skipped_missing}")
    print(f"skipped_no_label={skipped_no_label}")
    print(f"skipped_tiny={skipped_tiny}")
    print(f"split_train={n_train} split_val={n_val} split_test={max(0, n - n_train - n_val)}")
    print(f"out_dir={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
