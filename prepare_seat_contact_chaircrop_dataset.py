#!/usr/bin/env python3
"""
Prepare chair-cropped YOLO-seg dataset for seat_contact.

Pipeline:
1) Parse LabelMe JSON seat_contact polygons.
2) Detect chair bbox with a COCO chair detector.
3) Crop image around detected chair + margin.
4) Transform polygons to crop coordinates and export YOLO-seg labels.
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
from ultralytics import YOLO  # type: ignore


REPO_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build chair-cropped YOLO-seg dataset from LabelMe JSON."
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
        default=REPO_ROOT / "data" / "seat_contact_yolo_crop",
        help="Output YOLO dataset root.",
    )
    parser.add_argument(
        "--chair-detector",
        type=str,
        default="yolov8x-seg.pt",
        help="YOLO model for chair detection.",
    )
    parser.add_argument(
        "--chair-conf",
        type=float,
        default=0.2,
        help="Confidence threshold for chair detection.",
    )
    parser.add_argument(
        "--crop-margin-ratio",
        type=float,
        default=0.1,
        help="Margin ratio around detected chair bbox.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Inference device for chair detector.",
    )
    parser.add_argument(
        "--mask-outside-chair",
        action="store_true",
        help="If set, zero out non-chair pixels inside each crop using detector mask.",
    )
    parser.add_argument(
        "--outside-fill",
        type=int,
        default=114,
        help="Fill value (0..255) for non-chair pixels when --mask-outside-chair is set.",
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
    # Use mask clipping for robustness with arbitrary polygons.
    w = int(max(2, round(x_max - x_min)))
    h = int(max(2, round(y_max - y_min)))
    src = np.zeros((h, w), dtype=np.uint8)

    shifted = np.array(
        [[p[0] - x_min, p[1] - y_min] for p in points],
        dtype=np.float32,
    )
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
        x = float(p[0][0]) + x_min
        y = float(p[0][1]) + y_min
        out.append((x, y))
    return out


def detect_chair_box(
    model: YOLO,
    image_path: Path,
    *,
    conf: float,
    device: str,
) -> tuple[tuple[float, float, float, float], np.ndarray] | None:
    # COCO chair class id = 56
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
    r = pred[0]
    if r.boxes is None or len(r.boxes) == 0 or r.masks is None:
        return None
    boxes = r.boxes.xyxy.detach().cpu().numpy()
    confs = r.boxes.conf.detach().cpu().numpy()
    masks = r.masks.data.detach().cpu().numpy()
    best_i = max(
        range(min(len(boxes), len(masks))),
        key=lambda i: float(confs[i])
        * max(float((boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])), 1.0),
    )
    x0, y0, x1, y1 = boxes[best_i].tolist()
    mask = (masks[best_i] > 0.5).astype(np.uint8)
    return (float(x0), float(y0), float(x1), float(y1)), mask


def main() -> int:
    args = parse_args()

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    for split in ["train", "val", "test"]:
        (args.out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (args.out_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    model = YOLO(args.chair_detector)

    json_files = sorted(args.src_dir.glob("*.json"))
    samples: list[dict[str, Any]] = []
    for jf in json_files:
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue

        image_path_name = d.get("imagePath")
        if not image_path_name:
            continue
        img_path = args.src_dir / image_path_name
        if not img_path.exists():
            continue

        shapes = d.get("shapes") or []
        polys: list[list[tuple[float, float]]] = []
        for sh in shapes:
            if sh.get("label") != "seat_contact":
                continue
            pts = sh.get("points") or []
            if len(pts) < 3:
                continue
            poly = []
            for p in pts:
                if not isinstance(p, (list, tuple)) or len(p) < 2:
                    continue
                poly.append((float(p[0]), float(p[1])))
            if len(poly) >= 3:
                polys.append(poly)
        if not polys:
            continue

        samples.append(
            {
                "sample_id": img_path.stem,
                "json_path": jf,
                "image_path": img_path,
                "polygons": polys,
            }
        )

    # Deduplicate by sample id
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
        "sample_id,split,image_path,json_path,chair_box,crop_box,chair_detected,num_polygons"
    ]
    chair_detect_fail = 0
    kept = 0

    for s in samples:
        sid = str(s["sample_id"])
        split = split_map[sid]
        img_path: Path = s["image_path"]
        polygons: list[list[tuple[float, float]]] = s["polygons"]

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        detected = detect_chair_box(
            model,
            img_path,
            conf=args.chair_conf,
            device=args.device,
        )

        chair_mask_full: np.ndarray | None = None
        if detected is None:
            chair_detect_fail += 1
            # fallback: full image
            cx0, cy0, cx1, cy1 = 0, 0, w, h
            chair_box_txt = ""
            chair_detected = 0
        else:
            chair_box, chair_mask_full = detected
            x0, y0, x1, y1 = chair_box
            bw = max(2.0, x1 - x0)
            bh = max(2.0, y1 - y0)
            mx = bw * float(args.crop_margin_ratio)
            my = bh * float(args.crop_margin_ratio)
            cx0 = max(0, int(round(x0 - mx)))
            cy0 = max(0, int(round(y0 - my)))
            cx1 = min(w, int(round(x1 + mx)))
            cy1 = min(h, int(round(y1 + my)))
            if cx1 <= cx0 + 1 or cy1 <= cy0 + 1:
                cx0, cy0, cx1, cy1 = 0, 0, w, h
            chair_box_txt = f"{x0:.2f}|{y0:.2f}|{x1:.2f}|{y1:.2f}"
            chair_detected = 1

        crop = img.crop((cx0, cy0, cx1, cy1))
        if args.mask_outside_chair and chair_mask_full is not None:
            crop_arr = np.array(crop, dtype=np.uint8)
            cm = chair_mask_full[cy0:cy1, cx0:cx1]
            if cm.shape[0] != crop_arr.shape[0] or cm.shape[1] != crop_arr.shape[1]:
                cm = cv2.resize(
                    cm.astype(np.uint8),
                    (crop_arr.shape[1], crop_arr.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            fill = int(max(0, min(255, args.outside_fill)))
            masked = np.full_like(crop_arr, fill, dtype=np.uint8)
            keep = cm > 0
            masked[keep] = crop_arr[keep]
            crop = Image.fromarray(masked)
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
            # Keep file empty for consistency.
            out_lbl.write_text("", encoding="utf-8")
        else:
            out_lbl.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

        manifest.append(
            f"{sid},{split},{out_img},{s['json_path']},{chair_box_txt},{cx0}|{cy0}|{cx1}|{cy1},{chair_detected},{len(label_lines)}"
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
    print(f"chair_detect_fail={chair_detect_fail}")
    print(f"out_dir={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
