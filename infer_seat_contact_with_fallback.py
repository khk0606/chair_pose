#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer seat_contact mask with seat-ROI gating and fallback logic."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-overlay", type=Path, required=True)
    parser.add_argument(
        "--seat-model",
        type=Path,
        default=Path(
            "/Users/ganghyeongyu/Desktop/chair_dataset_fresh_models/seat_contact_fullseat_v2_plus_hard_v2_finetune1_best.pt"
        ),
    )
    parser.add_argument("--chair-detector", type=str, default="yolov8x-seg.pt")
    parser.add_argument(
        "--chair-gate-mode",
        type=str,
        default="bbox",
        choices=["bbox", "mask"],
        help="Gate predicted seat mask by chair bbox or chair segmentation mask.",
    )
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--chair-conf", type=float, default=0.2)
    parser.add_argument("--seat-conf", type=float, default=0.05)
    parser.add_argument("--seat-conf-low", type=float, default=0.001)
    parser.add_argument(
        "--fallback-seat-conf-thr",
        type=float,
        default=0.0,
        help="Use fallback when seat confidence is below this threshold.",
    )
    parser.add_argument(
        "--fallback-overlap-thr",
        type=float,
        default=0.0,
        help="Use fallback when overlap(pred, chair_mask) is below this threshold.",
    )
    parser.add_argument(
        "--fallback-prior-overlap-thr",
        type=float,
        default=0.0,
        help="Use fallback when overlap(pred, seat_prior) is below this threshold.",
    )
    parser.add_argument(
        "--fallback-min-area-ratio",
        type=float,
        default=0.0,
        help="Use fallback when pred_area / prior_area is below this threshold.",
    )
    parser.add_argument("--crop-margin-ratio", type=float, default=0.12)
    parser.add_argument(
        "--seat-roi-expand-x",
        type=float,
        default=0.55,
        help="Horizontal expansion ratio for seat ROI around seat prior bbox.",
    )
    parser.add_argument(
        "--seat-roi-expand-top",
        type=float,
        default=1.25,
        help="Top expansion ratio for seat ROI around seat prior bbox.",
    )
    parser.add_argument(
        "--seat-roi-expand-bottom",
        type=float,
        default=0.65,
        help="Bottom expansion ratio for seat ROI around seat prior bbox.",
    )
    parser.add_argument(
        "--prior-band-top-expand",
        type=float,
        default=0.22,
        help="Predictions are gated to a vertical band above seat prior.",
    )
    parser.add_argument(
        "--prior-band-bottom-expand",
        type=float,
        default=0.35,
        help="Predictions are gated to a vertical band below seat prior.",
    )
    parser.add_argument(
        "--prior-side-margin-ratio",
        type=float,
        default=0.005,
        help="Side trim ratio applied while building seat prior from chair mask.",
    )
    parser.add_argument(
        "--morph-kernel",
        type=int,
        default=5,
        help="Morphology kernel for cleanup (odd number).",
    )
    parser.add_argument(
        "--prior-dilate-kernel",
        type=int,
        default=301,
        help="Dilation kernel for seat prior before intersecting with model mask.",
    )
    parser.add_argument(
        "--prior-max-down-ratio",
        type=float,
        default=0.32,
        help="Clamp final model mask to this ratio below prior bbox bottom.",
    )
    parser.add_argument(
        "--use-prior-band-gate",
        action="store_true",
        help="Enable strict vertical gating by seat prior (off by default).",
    )
    parser.add_argument(
        "--use-prior-intersection",
        dest="use_prior_intersection",
        action="store_true",
        help="Intersect model mask with dilated prior.",
    )
    parser.add_argument(
        "--no-prior-intersection",
        dest="use_prior_intersection",
        action="store_false",
        help="Disable prior intersection.",
    )
    parser.add_argument(
        "--use-prior-y-cap",
        action="store_true",
        help="Clamp mask bottom by prior-derived y cap (off by default).",
    )
    parser.add_argument(
        "--mask-fusion-topk",
        type=int,
        default=4,
        help="Fuse up to K seat candidates from model prediction.",
    )
    parser.add_argument(
        "--mask-fusion-min-conf-ratio",
        type=float,
        default=0.25,
        help="Candidate conf must be >= best_conf * ratio to be fused.",
    )
    parser.add_argument(
        "--mask-fusion-min-area-ratio",
        type=float,
        default=0.30,
        help="Candidate area must be >= best_area * ratio to be fused.",
    )
    parser.add_argument(
        "--mask-fusion-max-area-ratio",
        type=float,
        default=10.0,
        help="Candidate area must be <= best_area * ratio to be fused.",
    )
    parser.add_argument(
        "--mask-fusion-iou-min",
        type=float,
        default=0.02,
        help="Candidate IoU w.r.t. best mask must be >= this value to be fused.",
    )
    parser.add_argument(
        "--mask-fusion-iou-max",
        type=float,
        default=0.70,
        help="Candidate IoU w.r.t. best mask must be <= this value to be fused.",
    )
    parser.add_argument(
        "--disable-row-trim",
        action="store_true",
        help="Disable row-width trim that removes narrow downward spikes.",
    )
    parser.add_argument(
        "--enable-row-trim",
        dest="disable_row_trim",
        action="store_false",
        help="Enable row-width trim.",
    )
    parser.add_argument(
        "--row-trim-min-width-ratio",
        type=float,
        default=0.68,
        help="Keep rows whose width is >= peak_width * ratio around peak row.",
    )
    parser.add_argument(
        "--row-trim-pad-rows",
        type=int,
        default=2,
        help="Extra rows to keep above/below trimmed core band.",
    )
    parser.add_argument(
        "--side-trim-ratio",
        type=float,
        default=0.0,
        help=(
            "Trim this ratio from both left and right sides per occupied row "
            "(0 disables)."
        ),
    )
    parser.add_argument(
        "--side-trim-min-px",
        type=int,
        default=0,
        help="Minimum pixels to trim on each side when --side-trim-ratio is used.",
    )
    parser.add_argument(
        "--disable-hull-smooth",
        dest="disable_hull_smooth",
        action="store_true",
        help="Disable convex-hull based mask smoothing.",
    )
    parser.add_argument(
        "--enable-hull-smooth",
        dest="disable_hull_smooth",
        action="store_false",
        help="Enable convex-hull based mask smoothing.",
    )
    parser.add_argument(
        "--disable-flip-tta",
        action="store_true",
        help="Disable horizontal-flip test-time augmentation for seat prediction.",
    )
    parser.add_argument(
        "--disable-prior-backfill",
        dest="disable_prior_backfill",
        action="store_true",
        help="Disable prior-guided backfill when model mask is too thin.",
    )
    parser.add_argument(
        "--enable-prior-backfill",
        dest="disable_prior_backfill",
        action="store_false",
        help="Enable prior-guided backfill when model mask is too thin.",
    )
    parser.add_argument(
        "--backfill-min-area-ratio",
        type=float,
        default=0.55,
        help="Run prior backfill when pred_area / prior_area is below this threshold.",
    )
    parser.add_argument(
        "--backfill-grow-x-ratio",
        type=float,
        default=0.08,
        help="Horizontal dilation ratio (relative to prior width) during backfill.",
    )
    parser.add_argument(
        "--backfill-grow-y-ratio",
        type=float,
        default=0.36,
        help="Upward expansion ratio (relative to prior height) during backfill.",
    )
    parser.add_argument(
        "--backfill-grow-down-ratio",
        type=float,
        default=0.06,
        help="Downward expansion ratio (relative to prior height) during backfill.",
    )
    parser.add_argument(
        "--backfill-max-steps",
        type=int,
        default=64,
        help="Maximum geodesic dilation steps used during prior backfill.",
    )
    parser.add_argument(
        "--final-close-kernel",
        type=int,
        default=1,
        help="Final morphology close kernel to remove small cutouts/holes.",
    )
    parser.add_argument(
        "--chair-mask-close-kernel",
        type=int,
        default=1,
        help="Close kernel for chair mask gate smoothing.",
    )
    parser.add_argument(
        "--chair-mask-dilate-kernel",
        type=int,
        default=1,
        help="Dilation kernel for chair mask gate smoothing.",
    )
    parser.add_argument(
        "--polygon-eps-ratio",
        type=float,
        default=0.0016,
        help="Douglas-Peucker epsilon ratio for contour-to-polygon simplification.",
    )
    parser.add_argument(
        "--polygon-min-eps",
        type=float,
        default=0.6,
        help="Minimum epsilon in pixels for polygon simplification.",
    )
    parser.add_argument(
        "--polygon-max-points",
        type=int,
        default=120,
        help="Maximum polygon points after simplification/subsampling.",
    )
    parser.set_defaults(
        use_prior_intersection=False,
        disable_row_trim=False,
        disable_hull_smooth=True,
        disable_prior_backfill=True,
    )
    return parser.parse_args()


def seat_from_chair_mask_full(
    chair_mask: np.ndarray,
    *,
    side_margin_ratio: float,
) -> np.ndarray:
    h, w = chair_mask.shape
    ys, xs = np.where(chair_mask > 0)
    if len(xs) < 20:
        return np.zeros_like(chair_mask, dtype=np.uint8)

    y0, y1 = int(ys.min()), int(ys.max())
    bh = max(1, y1 - y0 + 1)

    left = np.full(h, -1, dtype=np.int32)
    right = np.full(h, -1, dtype=np.int32)
    for y in range(y0, y1 + 1):
        cols = np.where(chair_mask[y] > 0)[0]
        if len(cols) == 0:
            continue
        left[y] = int(cols.min())
        right[y] = int(cols.max())
    # Full-cushion prior: use lower-mid chair band rather than front-edge strip.
    band_top = max(y0, int(round(y0 + 0.34 * bh)))
    band_bot = min(y1, int(round(y0 + 0.70 * bh)))
    if band_bot <= band_top:
        return np.zeros_like(chair_mask, dtype=np.uint8)

    out = np.zeros_like(chair_mask, dtype=np.uint8)
    for y in range(band_top, band_bot + 1):
        if left[y] < 0:
            continue
        row_w = max(1, right[y] - left[y] + 1)
        margin = max(0, int(round(float(side_margin_ratio) * row_w)))
        xl = min(max(0, left[y] + margin), w - 1)
        xr = max(min(w - 1, right[y] - margin), xl)
        out[y, xl : xr + 1] = 1

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (out * 255).astype(np.uint8), 8
    )
    if num_labels > 1:
        best = 1
        best_area = int(stats[1, cv2.CC_STAT_AREA])
        for i in range(2, num_labels):
            area_i = int(stats[i, cv2.CC_STAT_AREA])
            if area_i > best_area:
                best_area = area_i
                best = i
        out = (labels == best).astype(np.uint8)

    return out


def polygon_from_mask(
    mask: np.ndarray,
    *,
    eps_ratio: float,
    min_eps: float,
    max_points: int,
) -> list[dict[str, int]]:
    contours, _ = cv2.findContours(
        (mask * 255).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return []
    cnt = max(contours, key=cv2.contourArea)
    # Keep concavity so armrest cutouts are not filled by convex hull.
    peri = float(cv2.arcLength(cnt, True))
    approx = cv2.approxPolyDP(cnt, max(float(min_eps), float(eps_ratio) * peri), True)
    if len(approx) < 6:
        approx = cnt

    cap = max(12, int(max_points))
    if len(approx) > cap:
        step = max(1, len(approx) // cap)
        approx = approx[::step]
    out: list[dict[str, int]] = []
    for p in approx:
        out.append({"x": int(p[0][0]), "y": int(p[0][1])})
    return out


def largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask * 255).astype(np.uint8), 8
    )
    if num_labels <= 1:
        return mask.astype(np.uint8)
    best_idx = 1
    best_area = int(stats[1, cv2.CC_STAT_AREA])
    for i in range(2, num_labels):
        area_i = int(stats[i, cv2.CC_STAT_AREA])
        if area_i > best_area:
            best_idx = i
            best_area = area_i
    return (labels == best_idx).astype(np.uint8)


def smooth_chair_gate(mask: np.ndarray, close_k: int, dilate_k: int) -> np.ndarray:
    mm = largest_component((mask > 0).astype(np.uint8))
    ck = int(max(1, close_k))
    if ck % 2 == 0:
        ck += 1
    if ck > 1 and mm.sum() > 0:
        mm = cv2.morphologyEx(
            mm,
            cv2.MORPH_CLOSE,
            np.ones((ck, ck), dtype=np.uint8),
        )
        mm = largest_component(mm)
    dk = int(max(1, dilate_k))
    if dk % 2 == 0:
        dk += 1
    if dk > 1 and mm.sum() > 0:
        mm = cv2.dilate(mm, np.ones((dk, dk), dtype=np.uint8), iterations=1)
        mm = largest_component(mm)
    return mm


def smooth_mask_with_hull(mask: np.ndarray, chair_mask: np.ndarray) -> np.ndarray:
    mm = largest_component((mask > 0).astype(np.uint8))
    if mm.sum() < 20:
        return mm
    cnts, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mm
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    hull_mask = np.zeros_like(mm, dtype=np.uint8)
    cv2.fillPoly(hull_mask, [hull], 1)
    hull_mask = (hull_mask & (chair_mask > 0).astype(np.uint8)).astype(np.uint8)
    base_area = float(mm.sum())
    hull_area = float(hull_mask.sum())
    if hull_area <= 0:
        return mm
    # Avoid over-smoothing that expands too much outside the original prediction.
    if hull_area <= 1.35 * max(1.0, base_area):
        return largest_component(hull_mask)
    return mm


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def trim_rows_around_peak(
    mask: np.ndarray,
    *,
    min_width_ratio: float,
    pad_rows: int,
) -> np.ndarray:
    mm = (mask > 0).astype(np.uint8)
    bb = bbox_from_mask(mm)
    if bb is None:
        return mm
    _, y0, _, y1 = bb
    rows: list[tuple[int, int]] = []
    peak_y = y0
    peak_w = 0
    for y in range(y0, y1 + 1):
        cols = np.where(mm[y] > 0)[0]
        if len(cols) == 0:
            rows.append((y, 0))
            continue
        w = int(cols.max() - cols.min() + 1)
        rows.append((y, w))
        if w > peak_w:
            peak_w = w
            peak_y = y
    if peak_w <= 1:
        return mm

    thr = max(1, int(round(float(min_width_ratio) * float(peak_w))))
    width_of = {y: w for y, w in rows}
    top = peak_y
    while top - 1 >= y0 and width_of.get(top - 1, 0) >= thr:
        top -= 1
    bot = peak_y
    while bot + 1 <= y1 and width_of.get(bot + 1, 0) >= thr:
        bot += 1

    pad = max(0, int(pad_rows))
    keep_top = max(y0, top - pad)
    keep_bot = min(y1, bot + pad)
    out = mm.copy()
    if keep_top > y0:
        out[y0:keep_top, :] = 0
    if keep_bot < y1:
        out[keep_bot + 1 : y1 + 1, :] = 0
    return largest_component(out)


def trim_side_margins(
    mask: np.ndarray,
    *,
    side_trim_ratio: float,
    min_trim_px: int,
) -> np.ndarray:
    mm = (mask > 0).astype(np.uint8)
    ratio = float(max(0.0, min(0.45, side_trim_ratio)))
    if ratio <= 0.0:
        return mm

    bb = bbox_from_mask(mm)
    if bb is None:
        return mm
    _, y0, _, y1 = bb
    out = np.zeros_like(mm, dtype=np.uint8)
    min_px = max(0, int(min_trim_px))

    for y in range(y0, y1 + 1):
        cols = np.where(mm[y] > 0)[0]
        if len(cols) == 0:
            continue
        xl = int(cols.min())
        xr = int(cols.max())
        w = max(1, xr - xl + 1)
        trim = max(min_px, int(round(ratio * float(w))))
        if trim * 2 >= w - 2:
            trim = max(0, (w - 3) // 2)
        nx0 = xl + trim
        nx1 = xr - trim
        if nx1 <= nx0:
            nx0, nx1 = xl, xr
        out[y, nx0 : nx1 + 1] = 1

    if out.sum() == 0:
        return mm
    return largest_component(out)


def prior_backfill(
    pred_mask: np.ndarray,
    seat_prior: np.ndarray,
    *,
    grow_x_ratio: float,
    grow_y_ratio: float,
    grow_down_ratio: float,
    max_steps: int,
) -> np.ndarray:
    pm = largest_component((pred_mask > 0).astype(np.uint8))
    sp = largest_component((seat_prior > 0).astype(np.uint8))
    if pm.sum() == 0 or sp.sum() == 0:
        return pm
    pbb = bbox_from_mask(pm)
    sbb = bbox_from_mask(sp)
    if pbb is None or sbb is None:
        return pm
    px0, py0, px1, py1 = pbb
    sx0, sy0, sx1, sy1 = sbb
    pw = max(1, sx1 - sx0 + 1)
    ph = max(1, sy1 - sy0 + 1)

    expand_x = max(2, int(round(float(grow_x_ratio) * float(pw))))
    expand_up = max(2, int(round(float(grow_y_ratio) * float(ph))))
    expand_down = max(1, int(round(float(grow_down_ratio) * float(ph))))
    ax0 = max(sx0, px0 - expand_x)
    ax1 = min(sx1, px1 + expand_x)
    ay0 = max(sy0, py0 - expand_up)
    ay1 = min(sy1, py1 + expand_down)
    if ax1 <= ax0 or ay1 <= ay0:
        return pm

    allowed = np.zeros_like(sp, dtype=np.uint8)
    allowed[ay0 : ay1 + 1, ax0 : ax1 + 1] = 1
    target = (sp & allowed).astype(np.uint8)
    seed = (pm & target).astype(np.uint8)
    if seed.sum() == 0:
        return pm

    # Geodesic region growing inside target avoids spilling to unrelated chair parts.
    kernel = np.ones((3, 3), dtype=np.uint8)
    grown = seed.copy()
    steps = max(1, int(max_steps))
    for _ in range(steps):
        nxt = cv2.dilate(grown, kernel, iterations=1)
        nxt = (nxt & target).astype(np.uint8)
        if np.array_equal(nxt, grown):
            break
        grown = nxt
    return largest_component(grown)


def expand_box(
    bbox: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    expand_x: float,
    expand_top: float,
    expand_bottom: float,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    nx0 = max(0, int(round(x0 - float(expand_x) * bw)))
    nx1 = min(img_w, int(round(x1 + 1 + float(expand_x) * bw)))
    ny0 = max(0, int(round(y0 - float(expand_top) * bh)))
    ny1 = min(img_h, int(round(y1 + 1 + float(expand_bottom) * bh)))
    if nx1 <= nx0 + 1:
        nx0, nx1 = max(0, x0), min(img_w, x1 + 1)
    if ny1 <= ny0 + 1:
        ny0, ny1 = max(0, y0), min(img_h, y1 + 1)
    return nx0, ny0, nx1, ny1


def infer_seat_mask_on_crop(
    seat_model: YOLO,
    image: Image.Image,
    crop_box: tuple[int, int, int, int],
    *,
    seat_conf: float,
    seat_conf_low: float,
    device: str,
    fusion_topk: int,
    fusion_min_conf_ratio: float,
    fusion_min_area_ratio: float,
    fusion_max_area_ratio: float,
    fusion_iou_min: float,
    fusion_iou_max: float,
    use_flip_tta: bool,
) -> tuple[np.ndarray, float, bool, int, int]:
    W, H = image.size
    cx0, cy0, cx1, cy1 = crop_box
    crop = image.crop((cx0, cy0, cx1, cy1))
    crop_np = np.array(crop)
    variants: list[tuple[np.ndarray, bool]] = [(crop_np, False)]
    if use_flip_tta:
        variants.append((np.ascontiguousarray(crop_np[:, ::-1, :]), True))

    full_mask = np.zeros((H, W), dtype=np.uint8)
    candidates: list[tuple[float, float, np.ndarray]] = []
    model_raw_count = 0

    for source_arr, is_flip in variants:
        pred = seat_model.predict(
            source=source_arr,
            conf=float(seat_conf),
            iou=0.5,
            device=device,
            retina_masks=True,
            verbose=False,
        )
        if (
            not pred
            or pred[0].boxes is None
            or pred[0].masks is None
            or len(pred[0].boxes) == 0
        ):
            pred = seat_model.predict(
                source=source_arr,
                conf=float(seat_conf_low),
                iou=0.5,
                device=device,
                retina_masks=True,
                verbose=False,
            )
        if (
            not pred
            or pred[0].boxes is None
            or pred[0].masks is None
            or len(pred[0].boxes) == 0
        ):
            continue

        s_confs = pred[0].boxes.conf.detach().cpu().numpy()
        s_masks = pred[0].masks.data.detach().cpu().numpy()
        model_raw_count += int(len(s_confs))
        for i in range(min(len(s_confs), len(s_masks))):
            m = (s_masks[i] > 0.5).astype(np.uint8)
            if is_flip:
                m = np.ascontiguousarray(m[:, ::-1])
            area = float(m.sum())
            if area <= 1:
                continue
            conf_i = float(s_confs[i])
            score = conf_i * area
            candidates.append((conf_i, score, m))

    if not candidates:
        return full_mask, 0.0, False, int(model_raw_count), 0

    candidates.sort(key=lambda t: t[1], reverse=True)
    best_conf, _, best_mask_crop = candidates[0]
    fused_mask_crop = best_mask_crop.copy()
    best_area = max(float(best_mask_crop.sum()), 1.0)
    added = 1
    max_k = max(1, int(fusion_topk))
    for cand_conf, _, cand_mask in candidates[1:]:
        if added >= max_k:
            break
        cand_area = float(cand_mask.sum())
        if cand_conf < float(fusion_min_conf_ratio) * best_conf:
            continue
        if cand_area < float(fusion_min_area_ratio) * best_area:
            continue
        if cand_area > float(fusion_max_area_ratio) * best_area:
            continue
        inter = float((cand_mask & best_mask_crop).sum())
        union = float((cand_mask | best_mask_crop).sum())
        iou = inter / max(1.0, union)
        if iou < float(fusion_iou_min) or iou > float(fusion_iou_max):
            continue
        fused_mask_crop = ((fused_mask_crop > 0) | (cand_mask > 0)).astype(np.uint8)
        added += 1

    model_conf = float(best_conf)
    seat_mask_crop = fused_mask_crop
    crop_w = cx1 - cx0
    crop_h = cy1 - cy0
    if seat_mask_crop.shape[1] != crop_w or seat_mask_crop.shape[0] != crop_h:
        seat_mask_crop = cv2.resize(
            seat_mask_crop.astype(np.uint8),
            (crop_w, crop_h),
            interpolation=cv2.INTER_NEAREST,
        )
    full_mask[cy0:cy1, cx0:cx1] = seat_mask_crop
    return full_mask, model_conf, True, int(model_raw_count), int(added)


def main() -> int:
    args = parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"image not found: {args.image}")
    if not args.seat_model.exists():
        raise FileNotFoundError(f"seat model not found: {args.seat_model}")

    chair_model = YOLO(args.chair_detector)
    seat_model = YOLO(str(args.seat_model))

    image = Image.open(args.image).convert("RGB")
    W, H = image.size

    chair_pred = chair_model.predict(
        source=str(args.image),
        classes=[56],  # COCO chair
        conf=float(args.chair_conf),
        device=args.device,
        retina_masks=True,
        verbose=False,
    )
    if (
        not chair_pred
        or chair_pred[0].boxes is None
        or chair_pred[0].masks is None
        or len(chair_pred[0].boxes) == 0
    ):
        raise RuntimeError("No chair detection found")

    boxes = chair_pred[0].boxes.xyxy.detach().cpu().numpy()
    cconfs = chair_pred[0].boxes.conf.detach().cpu().numpy()
    cmasks = chair_pred[0].masks.data.detach().cpu().numpy()
    best_i = max(
        range(min(len(boxes), len(cmasks))),
        key=lambda i: float(cconfs[i])
        * max(float((boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])), 1.0),
    )
    x0, y0, x1, y1 = [float(v) for v in boxes[best_i]]
    chair_mask_raw = (cmasks[best_i] > 0.5).astype(np.uint8)
    chair_mask = smooth_chair_gate(
        chair_mask_raw,
        close_k=int(args.chair_mask_close_kernel),
        dilate_k=int(args.chair_mask_dilate_kernel),
    )
    if str(args.chair_gate_mode).lower() == "mask":
        chair_gate = chair_mask
    else:
        chair_gate = np.zeros((H, W), dtype=np.uint8)
        gx0 = max(0, int(round(x0)))
        gy0 = max(0, int(round(y0)))
        gx1 = min(W - 1, int(round(x1)))
        gy1 = min(H - 1, int(round(y1)))
        chair_gate[gy0 : gy1 + 1, gx0 : gx1 + 1] = 1

    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    mx = float(args.crop_margin_ratio) * bw
    my = float(args.crop_margin_ratio) * bh
    chair_crop = (
        max(0, int(round(x0 - mx))),
        max(0, int(round(y0 - my))),
        min(W, int(round(x1 + mx))),
        min(H, int(round(y1 + my))),
    )
    if chair_crop[2] <= chair_crop[0] + 1 or chair_crop[3] <= chair_crop[1] + 1:
        raise RuntimeError("Invalid chair crop")

    seat_prior = seat_from_chair_mask_full(
        chair_mask,
        side_margin_ratio=float(args.prior_side_margin_ratio),
    )
    seat_prior = largest_component(seat_prior)
    seat_prior_bbox = bbox_from_mask(seat_prior)

    if seat_prior_bbox is None:
        seat_roi = chair_crop
    else:
        seat_roi = expand_box(
            seat_prior_bbox,
            W,
            H,
            float(args.seat_roi_expand_x),
            float(args.seat_roi_expand_top),
            float(args.seat_roi_expand_bottom),
        )

    pred_mask_full, model_conf, has_model_pred, model_raw_count, model_fused_count = infer_seat_mask_on_crop(
        seat_model,
        image,
        seat_roi,
        seat_conf=float(args.seat_conf),
        seat_conf_low=float(args.seat_conf_low),
        device=args.device,
        fusion_topk=int(args.mask_fusion_topk),
        fusion_min_conf_ratio=float(args.mask_fusion_min_conf_ratio),
        fusion_min_area_ratio=float(args.mask_fusion_min_area_ratio),
        fusion_max_area_ratio=float(args.mask_fusion_max_area_ratio),
        fusion_iou_min=float(args.mask_fusion_iou_min),
        fusion_iou_max=float(args.mask_fusion_iou_max),
        use_flip_tta=not bool(args.disable_flip_tta),
    )

    pred_mask_full = (pred_mask_full & chair_gate).astype(np.uint8)

    if (
        args.use_prior_band_gate
        and seat_prior_bbox is not None
        and pred_mask_full.sum() > 0
    ):
        _, py0, _, py1 = seat_prior_bbox
        ph = max(1, py1 - py0 + 1)
        gate_top = max(0, int(round(py0 - float(args.prior_band_top_expand) * ph)))
        gate_bottom = min(
            H - 1, int(round(py1 + float(args.prior_band_bottom_expand) * ph))
        )
        if gate_bottom > gate_top:
            gate = np.zeros((H, W), dtype=np.uint8)
            gate[gate_top : gate_bottom + 1, :] = 1
            pred_mask_full = (pred_mask_full & gate).astype(np.uint8)

    if pred_mask_full.sum() > 0:
        k = int(max(3, args.morph_kernel))
        if k % 2 == 0:
            k += 1
        kernel = np.ones((k, k), dtype=np.uint8)
        pred_mask_full = cv2.morphologyEx(pred_mask_full, cv2.MORPH_OPEN, kernel)
        pred_mask_full = cv2.morphologyEx(pred_mask_full, cv2.MORPH_CLOSE, kernel)

    if args.use_prior_intersection and seat_prior.sum() > 0 and pred_mask_full.sum() > 0:
        dk = int(max(1, args.prior_dilate_kernel))
        if dk % 2 == 0:
            dk += 1
        prior_kernel = np.ones((dk, dk), dtype=np.uint8)
        prior_gate = cv2.dilate(seat_prior, prior_kernel, iterations=1)
        pred_mask_full = (pred_mask_full & prior_gate).astype(np.uint8)

    if (
        args.use_prior_y_cap
        and seat_prior_bbox is not None
        and pred_mask_full.sum() > 0
    ):
        _, py0, _, py1 = seat_prior_bbox
        ph = max(1, py1 - py0 + 1)
        y_cap = min(H - 1, int(round(py1 + float(args.prior_max_down_ratio) * ph)))
        if y_cap + 1 < H:
            pred_mask_full[y_cap + 1 :, :] = 0

    if pred_mask_full.sum() > 0:
        pred_mask_full = largest_component(pred_mask_full)
        if not args.disable_row_trim:
            pred_mask_full = trim_rows_around_peak(
                pred_mask_full,
                min_width_ratio=float(args.row_trim_min_width_ratio),
                pad_rows=int(args.row_trim_pad_rows),
            )

    pred_area_ratio = 0.0
    if seat_prior.sum() > 0 and pred_mask_full.sum() > 0:
        pred_area_ratio = float(pred_mask_full.sum()) / max(1.0, float(seat_prior.sum()))

    used_backfill = False
    if (
        not args.disable_prior_backfill
        and seat_prior.sum() > 0
        and pred_mask_full.sum() > 0
        and pred_area_ratio < float(args.backfill_min_area_ratio)
    ):
        repaired = prior_backfill(
            pred_mask_full,
            seat_prior,
            grow_x_ratio=float(args.backfill_grow_x_ratio),
            grow_y_ratio=float(args.backfill_grow_y_ratio),
            grow_down_ratio=float(args.backfill_grow_down_ratio),
            max_steps=int(args.backfill_max_steps),
        )
        if repaired.sum() > pred_mask_full.sum():
            pred_mask_full = repaired
            used_backfill = True
            pred_area_ratio = float(pred_mask_full.sum()) / max(
                1.0, float(seat_prior.sum())
            )

    overlap = float((pred_mask_full & chair_gate).sum()) / max(1.0, float(pred_mask_full.sum()))
    prior_overlap = float((pred_mask_full & seat_prior).sum()) / max(
        1.0, float(pred_mask_full.sum())
    )
    inter_prior = float((pred_mask_full & seat_prior).sum())
    union_prior = float((pred_mask_full | seat_prior).sum())
    prior_iou = inter_prior / max(1.0, union_prior)

    use_fallback = (
        (not has_model_pred)
        or model_conf < float(args.fallback_seat_conf_thr)
        or overlap < float(args.fallback_overlap_thr)
        or prior_overlap < float(args.fallback_prior_overlap_thr)
        or (
            seat_prior.sum() > 0
            and pred_mask_full.sum() > 0
            and pred_area_ratio < float(args.fallback_min_area_ratio)
        )
    )
    if use_fallback:
        final_mask = seat_prior
        source = "chair_mask_fallback_full"
    else:
        final_mask = pred_mask_full
        source = (
            "seat_model_prediction_backfilled"
            if used_backfill
            else "seat_model_prediction_gated"
        )

    final_mask = largest_component(final_mask)
    if not args.disable_row_trim:
        final_mask = trim_rows_around_peak(
            final_mask,
            min_width_ratio=float(args.row_trim_min_width_ratio),
            pad_rows=int(args.row_trim_pad_rows),
        )
    if float(args.side_trim_ratio) > 0.0:
        final_mask = trim_side_margins(
            final_mask,
            side_trim_ratio=float(args.side_trim_ratio),
            min_trim_px=int(args.side_trim_min_px),
        )
    if not args.disable_hull_smooth:
        final_mask = smooth_mask_with_hull(final_mask, chair_mask)
    fk = int(max(1, args.final_close_kernel))
    if fk % 2 == 0:
        fk += 1
    if fk > 1 and final_mask.sum() > 0:
        final_mask = cv2.morphologyEx(
            final_mask.astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((fk, fk), dtype=np.uint8),
        )
        final_mask = largest_component(final_mask)
    polygon = polygon_from_mask(
        final_mask,
        eps_ratio=float(args.polygon_eps_ratio),
        min_eps=float(args.polygon_min_eps),
        max_points=int(args.polygon_max_points),
    )
    area = int(final_mask.sum())

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_overlay.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "image": str(args.image),
        "seat_model": str(args.seat_model),
        "chair_detector": args.chair_detector,
        "device": args.device,
        "chair_confidence": float(cconfs[best_i]),
        "seat_confidence": float(model_conf),
        "model_overlap_with_chair": round(overlap, 6),
        "model_overlap_with_prior": round(prior_overlap, 6),
        "model_iou_with_prior": round(prior_iou, 6),
        "model_area_ratio_to_prior": round(pred_area_ratio, 6),
        "used_fallback": bool(use_fallback),
        "used_prior_backfill": bool(used_backfill),
        "source": source,
        "chair_bbox": {
            "x_min": round(x0, 2),
            "y_min": round(y0, 2),
            "x_max": round(x1, 2),
            "y_max": round(y1, 2),
        },
        "chair_crop": {
            "x0": int(chair_crop[0]),
            "y0": int(chair_crop[1]),
            "x1": int(chair_crop[2]),
            "y1": int(chair_crop[3]),
        },
        "seat_roi": {
            "x0": int(seat_roi[0]),
            "y0": int(seat_roi[1]),
            "x1": int(seat_roi[2]),
            "y1": int(seat_roi[3]),
        },
        "model_raw_count": int(model_raw_count),
        "model_fused_count": int(model_fused_count),
        "flip_tta_enabled": bool(not args.disable_flip_tta),
        "seat_prior_area_px": int(seat_prior.sum()),
        "seat_mask_area_px": area,
        "seat_contact_polygon": polygon,
    }
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    if len(polygon) >= 3:
        pts = [(p["x"], p["y"]) for p in polygon]
        draw.polygon(pts, fill=(255, 70, 170, 95), outline=(255, 70, 220, 235), width=4)

    prior_poly = polygon_from_mask(
        seat_prior,
        eps_ratio=float(args.polygon_eps_ratio),
        min_eps=float(args.polygon_min_eps),
        max_points=int(args.polygon_max_points),
    )
    if len(prior_poly) >= 3:
        prior_pts = [(p["x"], p["y"]) for p in prior_poly]
        draw.polygon(
            prior_pts,
            fill=(120, 240, 120, 35),
            outline=(120, 240, 120, 170),
            width=2,
        )

    draw.rectangle((int(x0), int(y0), int(x1), int(y1)), outline=(80, 220, 255, 180), width=3)
    draw.rectangle(
        (
            int(seat_roi[0]),
            int(seat_roi[1]),
            int(seat_roi[2]),
            int(seat_roi[3]),
        ),
        outline=(255, 210, 80, 170),
        width=2,
    )
    out = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    out.save(args.output_overlay)

    print(f"saved json: {args.output_json}")
    print(f"saved overlay: {args.output_overlay}")
    print(f"source={source} used_fallback={use_fallback} area={area}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
