#!/usr/bin/env python3
"""
Build a hard-case-only retraining dataset with dual seat-contact styles:
- front-contact style (lower band)
- deep-contact style (expanded upper seat region)

Source is LabelMe JSON (seat_contact). Hard cases are mined by low IoU
against the current model prediction.
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
from PIL import Image, ImageDraw
from ultralytics import YOLO  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare hard-case dual-style dataset for seat_contact retraining."
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=Path("/Users/ganghyeongyu/Desktop/chair_dataset/images"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/ganghyeongyu/Desktop/chair_dataset/seat_contact_hardcases_dualstyle_v1"),
    )
    parser.add_argument(
        "--mine-model",
        type=Path,
        default=Path(
            "/Users/ganghyeongyu/Desktop/chair_dataset/models/seat_contact_yolov8s_seg_seatfocus_v1_e35_best.pt"
        ),
        help="Current seat model used to mine hard samples.",
    )
    parser.add_argument("--chair-detector", type=str, default="yolov8x-seg.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seat-conf", type=float, default=0.05)
    parser.add_argument(
        "--hard-iou-thr",
        type=float,
        default=0.75,
        help="Samples with IoU < threshold are treated as hard cases.",
    )
    parser.add_argument(
        "--deep-expand-ratio",
        type=float,
        default=0.42,
        help="Upward expansion ratio (relative to GT seat height) for deep style.",
    )
    parser.add_argument(
        "--front-band-ratio",
        type=float,
        default=0.50,
        help="Keep lower ratio of deep mask for front style.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap on hard sample count (0 = no cap).",
    )
    parser.add_argument(
        "--include-orig",
        dest="include_orig",
        action="store_true",
        help="Include original GT variant.",
    )
    parser.add_argument(
        "--no-include-orig",
        dest="include_orig",
        action="store_false",
        help="Disable original GT variant.",
    )
    parser.add_argument(
        "--include-front",
        dest="include_front",
        action="store_true",
        help="Include front-band variant.",
    )
    parser.add_argument(
        "--no-include-front",
        dest="include_front",
        action="store_false",
        help="Disable front-band variant.",
    )
    parser.add_argument(
        "--include-deep",
        dest="include_deep",
        action="store_true",
        help="Include deep-contact variant.",
    )
    parser.add_argument(
        "--no-include-deep",
        dest="include_deep",
        action="store_false",
        help="Disable deep-contact variant.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(include_orig=True, include_front=True, include_deep=True)
    return parser.parse_args()


def largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), 8
    )
    if num_labels <= 1:
        return (mask > 0).astype(np.uint8)
    best_idx = 1
    best_area = int(stats[1, cv2.CC_STAT_AREA])
    for i in range(2, num_labels):
        area_i = int(stats[i, cv2.CC_STAT_AREA])
        if area_i > best_area:
            best_area = area_i
            best_idx = i
    return (labels == best_idx).astype(np.uint8)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def shape_points_to_mask(
    shapes: list[dict[str, Any]],
    w: int,
    h: int,
    label: str = "seat_contact",
) -> np.ndarray:
    m = Image.fromarray(np.zeros((h, w), dtype=np.uint8))
    dr = ImageDraw.Draw(m)
    for sh in shapes:
        if sh.get("label") != label:
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
            dr.polygon(poly, fill=1)
    return np.array(m, dtype=np.uint8)


def detect_chair_mask(model: YOLO, image_path: Path, conf: float, device: str) -> np.ndarray | None:
    pred = model.predict(
        source=str(image_path),
        classes=[56],  # COCO chair
        conf=float(conf),
        retina_masks=True,
        device=device,
        verbose=False,
    )
    if (
        not pred
        or pred[0].boxes is None
        or pred[0].masks is None
        or len(pred[0].boxes) == 0
    ):
        return None

    boxes = pred[0].boxes.xyxy.detach().cpu().numpy()
    confs = pred[0].boxes.conf.detach().cpu().numpy()
    masks = pred[0].masks.data.detach().cpu().numpy()
    best_i = max(
        range(min(len(boxes), len(masks))),
        key=lambda i: float(confs[i])
        * max(float((boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])), 1.0),
    )
    return (masks[best_i] > 0.5).astype(np.uint8)


def predict_mask(seat_model: YOLO, image_path: Path, conf: float, device: str) -> np.ndarray:
    pred = seat_model.predict(
        source=str(image_path),
        conf=float(conf),
        iou=0.5,
        retina_masks=True,
        device=device,
        verbose=False,
    )
    if (
        not pred
        or pred[0].boxes is None
        or pred[0].masks is None
        or len(pred[0].boxes) == 0
    ):
        return np.zeros(Image.open(image_path).size[::-1], dtype=np.uint8)

    s_confs = pred[0].boxes.conf.detach().cpu().numpy()
    s_masks = pred[0].masks.data.detach().cpu().numpy()
    best_s = max(
        range(min(len(s_confs), len(s_masks))),
        key=lambda i: float(s_confs[i]) * max(float((s_masks[i] > 0.5).sum()), 1.0),
    )
    return (s_masks[best_s] > 0.5).astype(np.uint8)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = (a > 0).astype(np.uint8)
    bb = (b > 0).astype(np.uint8)
    inter = float((aa & bb).sum())
    union = float((aa | bb).sum())
    if union <= 0:
        return 0.0
    return inter / union


def deep_style_mask(gt_mask: np.ndarray, chair_mask: np.ndarray | None, deep_expand_ratio: float) -> np.ndarray:
    m = largest_component(gt_mask)
    bb = mask_bbox(m)
    if bb is None:
        return m
    _, y0, _, y1 = bb
    h = max(1, y1 - y0 + 1)
    k = max(3, int(round(float(deep_expand_ratio) * h)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, k))
    dm = cv2.dilate(m, kernel, iterations=1)

    if chair_mask is not None:
        dm = (dm & (chair_mask > 0).astype(np.uint8)).astype(np.uint8)

    bb2 = mask_bbox(m)
    if bb2 is not None:
        _, gy0, _, gy1 = bb2
        gh = max(1, gy1 - gy0 + 1)
        y_top = max(0, int(round(gy0 - 1.1 * gh)))
        y_bot = min(dm.shape[0] - 1, int(round(gy1 + 0.35 * gh)))
        gate = np.zeros_like(dm, dtype=np.uint8)
        gate[y_top : y_bot + 1, :] = 1
        dm = (dm & gate).astype(np.uint8)

    dm = cv2.morphologyEx(dm, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return largest_component(dm)


def front_style_mask(deep_mask: np.ndarray, front_band_ratio: float) -> np.ndarray:
    m = largest_component(deep_mask)
    bb = mask_bbox(m)
    if bb is None:
        return m
    _, y0, _, y1 = bb
    h = max(1, y1 - y0 + 1)
    keep_h = max(1, int(round(float(front_band_ratio) * h)))
    y_cut = y1 - keep_h + 1
    fm = np.zeros_like(m, dtype=np.uint8)
    fm[y_cut : y1 + 1, :] = m[y_cut : y1 + 1, :]
    fm = cv2.morphologyEx(fm, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return largest_component(fm)


def yolo_line_from_mask(mask: np.ndarray) -> str | None:
    cnts, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 8:
        return None
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, max(1.0, 0.003 * peri), True)
    if len(approx) < 3:
        return None
    h, w = mask.shape
    coords: list[str] = []
    for p in approx:
        x = float(p[0][0]) / max(1.0, float(w))
        y = float(p[0][1]) / max(1.0, float(h))
        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)
        coords.append(f"{x:.6f}")
        coords.append(f"{y:.6f}")
    if len(coords) < 6:
        return None
    return "0 " + " ".join(coords)


def save_variant(
    image: Image.Image,
    mask: np.ndarray,
    out_img: Path,
    out_lbl: Path,
) -> bool:
    line = yolo_line_from_mask(mask)
    if line is None:
        return False
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_lbl.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_img, quality=95)
    out_lbl.write_text(line + "\n", encoding="utf-8")
    return True


def main() -> int:
    args = parse_args()
    if not args.src_dir.exists():
        raise FileNotFoundError(f"src dir not found: {args.src_dir}")
    if not args.mine_model.exists():
        raise FileNotFoundError(f"mine model not found: {args.mine_model}")

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    for split in ["train", "val", "test"]:
        (args.out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (args.out_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    seat_model = YOLO(str(args.mine_model))
    chair_model = YOLO(args.chair_detector)

    candidates: list[dict[str, Any]] = []
    json_files = sorted(args.src_dir.glob("*.json"))
    for jf in json_files:
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        image_name = d.get("imagePath")
        if not image_name:
            continue
        image_path = args.src_dir / image_name
        if not image_path.exists():
            continue

        with Image.open(image_path) as im:
            w, h = im.size
        gt = shape_points_to_mask(d.get("shapes") or [], w, h, "seat_contact")
        gt = largest_component(gt)
        if gt.sum() < 10:
            continue

        pred = predict_mask(seat_model, image_path, args.seat_conf, args.device)
        if pred.shape != gt.shape:
            pred = cv2.resize(
                pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            )
        score = iou(pred, gt)
        if score >= float(args.hard_iou_thr):
            continue

        chair = detect_chair_mask(chair_model, image_path, conf=0.2, device=args.device)
        candidates.append(
            {
                "sample_id": image_path.stem,
                "image_path": image_path,
                "json_path": jf,
                "gt": gt,
                "chair": chair,
                "iou": score,
            }
        )

    candidates.sort(key=lambda x: float(x["iou"]))
    if args.max_samples > 0:
        candidates = candidates[: int(args.max_samples)]

    rng = random.Random(args.seed)
    rng.shuffle(candidates)

    n = len(candidates)
    n_train = int(round(n * 0.8))
    n_val = int(round(n * 0.1))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)

    split_of: dict[str, str] = {}
    for i, c in enumerate(candidates):
        sid = str(c["sample_id"])
        if i < n_train:
            split_of[sid] = "train"
        elif i < n_train + n_val:
            split_of[sid] = "val"
        else:
            split_of[sid] = "test"

    manifest = [
        "sample_id,split,variant,image_path,json_path,mining_iou,mask_area_px"
    ]
    written = 0

    for c in candidates:
        sid = str(c["sample_id"])
        split = split_of[sid]
        img = Image.open(c["image_path"]).convert("RGB")
        gt = (c["gt"] > 0).astype(np.uint8)
        chair = c["chair"]

        deep = deep_style_mask(gt, chair, args.deep_expand_ratio)
        front = front_style_mask(deep, args.front_band_ratio)

        variants: list[tuple[str, np.ndarray]] = []
        if args.include_orig:
            variants.append(("orig", gt))
        if args.include_front:
            variants.append(("front", front))
        if args.include_deep:
            variants.append(("deep", deep))
        if not variants:
            variants = [("orig", gt)]

        for v_name, v_mask in variants:
            out_img = args.out_dir / split / "images" / f"{sid}_{v_name}.jpg"
            out_lbl = args.out_dir / split / "labels" / f"{sid}_{v_name}.txt"
            ok = save_variant(img, v_mask, out_img, out_lbl)
            if not ok:
                continue
            manifest.append(
                f"{sid},{split},{v_name},{c['image_path']},{c['json_path']},{float(c['iou']):.6f},{int(v_mask.sum())}"
            )
            written += 1

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

    report = {
        "src_dir": str(args.src_dir),
        "out_dir": str(args.out_dir),
        "mine_model": str(args.mine_model),
        "hard_iou_thr": float(args.hard_iou_thr),
        "include_orig": bool(args.include_orig),
        "include_front": bool(args.include_front),
        "include_deep": bool(args.include_deep),
        "hard_samples": len(candidates),
        "variants_written": written,
        "split_train_samples": n_train,
        "split_val_samples": n_val,
        "split_test_samples": max(0, n - n_train - n_val),
    }
    (args.out_dir / "prepare_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"hard_samples={len(candidates)}")
    print(f"variants_written={written}")
    print(f"train_samples={n_train} val_samples={n_val} test_samples={max(0, n - n_train - n_val)}")
    print(f"dataset={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
