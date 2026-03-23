#!/usr/bin/env python3
"""
Prepare a fresh YOLO-seg dataset from LabelMe annotations.

This converter uses only image/json pairs with the same stem and expects
`seat_contact` polygons that represent the full sittable cushion region.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build full-seat-contact YOLO dataset from LabelMe pairs."
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        required=True,
        help="Directory containing .json and image files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output YOLO dataset directory.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="seat_contact",
        help="LabelMe polygon label name.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--preview-count",
        type=int,
        default=12,
        help="Number of random overlay previews to save.",
    )
    return parser.parse_args()


def normalize_polygon(points: list[Any], w: int, h: int) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    max_x = float(max(1, w - 1))
    max_y = float(max(1, h - 1))
    for p in points:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        x = min(max(float(p[0]), 0.0), max_x)
        y = min(max(float(p[1]), 0.0), max_y)
        out.append((x, y))
    return out


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    acc = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        acc += x1 * y2 - x2 * y1
    return abs(acc) * 0.5


def image_candidates(src_dir: Path, stem: str) -> list[Path]:
    cands = []
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        p = src_dir / f"{stem}{ext}"
        if p.exists():
            cands.append(p)
    return cands


def save_preview(
    image_path: Path,
    polygons: list[list[tuple[float, float]]],
    out_path: Path,
) -> None:
    img = Image.open(image_path).convert("RGBA")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov, "RGBA")
    for poly in polygons:
        if len(poly) >= 3:
            dr.polygon(poly, fill=(255, 70, 170, 95), outline=(255, 70, 220, 230), width=3)
    out = Image.alpha_composite(img, ov).convert("RGB")
    out.save(out_path, quality=90)


def main() -> int:
    args = parse_args()
    src_dir = args.src_dir
    out_dir = args.out_dir
    label_name = args.label

    if not src_dir.exists():
        raise FileNotFoundError(f"src dir not found: {src_dir}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ["train", "val", "test"]:
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    (out_dir / "previews").mkdir(parents=True, exist_ok=True)

    json_files = sorted(src_dir.glob("*.json"))
    matched: list[dict[str, Any]] = []
    json_without_image: list[str] = []
    invalid_json: list[str] = []
    dropped_empty: list[str] = []

    for jf in json_files:
        stem = jf.stem
        cands = image_candidates(src_dir, stem)
        if not cands:
            json_without_image.append(stem)
            continue
        image_path = cands[0]
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            invalid_json.append(stem)
            continue

        with Image.open(image_path) as im:
            w, h = im.size

        polys: list[list[tuple[float, float]]] = []
        shapes = data.get("shapes") or []
        for sh in shapes:
            if sh.get("label") != label_name:
                continue
            pts = normalize_polygon(sh.get("points") or [], w, h)
            if len(pts) >= 3 and polygon_area(pts) >= 6.0:
                polys.append(pts)

        if not polys:
            dropped_empty.append(stem)
            continue

        matched.append(
            {
                "stem": stem,
                "image_path": image_path,
                "json_path": jf,
                "width": w,
                "height": h,
                "polygons": polys,
                "area_sum": float(sum(polygon_area(p) for p in polys)),
            }
        )

    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        image_files.extend(src_dir.glob(ext))
    image_stems = {p.stem for p in image_files}
    json_stems = {p.stem for p in json_files}
    image_without_json = sorted(image_stems - json_stems)

    rng = random.Random(args.seed)
    rng.shuffle(matched)

    n = len(matched)
    n_train = int(round(n * float(args.train_ratio)))
    n_val = int(round(n * float(args.val_ratio)))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)

    split_of: dict[str, str] = {}
    for i, s in enumerate(matched):
        stem = str(s["stem"])
        if i < n_train:
            split_of[stem] = "train"
        elif i < n_train + n_val:
            split_of[stem] = "val"
        else:
            split_of[stem] = "test"

    manifest = [
        "stem,split,image_path,json_path,width,height,num_polygons,area_sum"
    ]
    polygon_counts: list[int] = []
    area_sums: list[float] = []
    written = 0

    for s in matched:
        stem = str(s["stem"])
        split = split_of[stem]
        image_path: Path = s["image_path"]
        w = int(s["width"])
        h = int(s["height"])
        polygons: list[list[tuple[float, float]]] = s["polygons"]

        out_img = out_dir / split / "images" / f"{stem}{image_path.suffix.lower()}"
        out_lbl = out_dir / split / "labels" / f"{stem}.txt"
        shutil.copy2(image_path, out_img)

        lines: list[str] = []
        for poly in polygons:
            coords: list[str] = []
            for x, y in poly:
                xn = min(max(x / max(1.0, float(w)), 0.0), 1.0)
                yn = min(max(y / max(1.0, float(h)), 0.0), 1.0)
                coords.append(f"{xn:.6f}")
                coords.append(f"{yn:.6f}")
            if len(coords) >= 6:
                lines.append("0 " + " ".join(coords))
        out_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")

        polygon_counts.append(len(lines))
        area_sums.append(float(s["area_sum"]))
        manifest.append(
            f"{stem},{split},{image_path},{s['json_path']},{w},{h},{len(lines)},{float(s['area_sum']):.2f}"
        )
        written += 1

    (out_dir / "manifest.csv").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    (out_dir / "dataset.yaml").write_text(
        "\n".join(
            [
                f"path: {out_dir}",
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

    preview_count = min(int(args.preview_count), len(matched))
    preview_samples = matched[:]
    rng.shuffle(preview_samples)
    for s in preview_samples[:preview_count]:
        stem = str(s["stem"])
        save_preview(
            s["image_path"],
            s["polygons"],
            out_dir / "previews" / f"{stem}_overlay.jpg",
        )

    report = {
        "src_dir": str(src_dir),
        "out_dir": str(out_dir),
        "label": label_name,
        "json_files": len(json_files),
        "image_files": len(image_files),
        "matched_pairs_written": written,
        "json_without_image": json_without_image,
        "image_without_json": image_without_json,
        "invalid_json": invalid_json,
        "dropped_empty_or_invalid_label": dropped_empty,
        "split": {
            "train": n_train,
            "val": n_val,
            "test": max(0, n - n_train - n_val),
        },
        "polygon_count_min": min(polygon_counts) if polygon_counts else 0,
        "polygon_count_max": max(polygon_counts) if polygon_counts else 0,
        "area_sum_min": min(area_sums) if area_sums else 0.0,
        "area_sum_max": max(area_sums) if area_sums else 0.0,
    }
    (out_dir / "prepare_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"matched_pairs_written={written}")
    print(f"split_train={n_train} split_val={n_val} split_test={max(0, n - n_train - n_val)}")
    print(f"image_without_json={len(image_without_json)}")
    print(f"json_without_image={len(json_without_image)}")
    print(f"dropped_empty_or_invalid_label={len(dropped_empty)}")
    print(f"out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
