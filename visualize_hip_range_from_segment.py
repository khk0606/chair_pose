#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and visualize plausible hip-contact 2D range from seat-contact "
            "segmentation polygons."
        )
    )
    parser.add_argument(
        "--segment-json",
        type=Path,
        action="append",
        required=True,
        help="Path to seat-contact inference JSON. Repeat for multiple files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/ganghyeongyu/Documents/chairpose/runs"),
        help="Directory to write hip-range JSON and overlay images.",
    )
    parser.add_argument(
        "--grid-step",
        type=int,
        default=14,
        help="Grid step (pixels) for candidate point sampling.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=1200,
        help="Maximum candidate points to keep in output.",
    )
    return parser.parse_args()


def largest_component(mask: np.ndarray) -> np.ndarray:
    mm = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mm, 8)
    if n <= 1:
        return mm
    best_i = 1
    best_a = int(stats[1, cv2.CC_STAT_AREA])
    for i in range(2, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a > best_a:
            best_i = i
            best_a = a
    return (labels == best_i).astype(np.uint8)


def polygon_from_mask(mask: np.ndarray) -> list[dict[str, int]]:
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return []
    cnt = max(contours, key=cv2.contourArea)
    peri = float(cv2.arcLength(cnt, True))
    approx = cv2.approxPolyDP(cnt, max(0.8, 0.002 * peri), True)
    if len(approx) < 8:
        approx = cnt
    max_points = 180
    if len(approx) > max_points:
        step = max(1, len(approx) // max_points)
        approx = approx[::step]
    out: list[dict[str, int]] = []
    for p in approx:
        out.append({"x": int(p[0][0]), "y": int(p[0][1])})
    return out


def build_hip_mask(
    seat_mask: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float], float, np.ndarray]:
    ys, xs = np.where(seat_mask > 0)
    if len(xs) < 20:
        raise RuntimeError("seat mask too small")

    area = float(len(xs))
    y_min, y_max = float(ys.min()), float(ys.max())
    band_h = max(4.0, y_max - y_min)

    back_thr = y_min + 0.18 * band_h
    front_thr = y_min + 0.82 * band_h

    back_sel = ys <= back_thr
    front_sel = ys >= front_thr
    if int(back_sel.sum()) < 20:
        back_sel = ys <= (y_min + 0.25 * band_h)
    if int(front_sel.sum()) < 20:
        front_sel = ys >= (y_min + 0.75 * band_h)

    back_pt = (float(xs[back_sel].mean()), float(ys[back_sel].mean()))
    front_pt = (float(xs[front_sel].mean()), float(ys[front_sel].mean()))

    vx = front_pt[0] - back_pt[0]
    vy = front_pt[1] - back_pt[1]
    v_norm2 = vx * vx + vy * vy
    if v_norm2 < 1e-6:
        vx, vy = 0.0, 1.0
        v_norm2 = 1.0

    h, w = seat_mask.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    t_map = ((xx - back_pt[0]) * vx + (yy - back_pt[1]) * vy) / float(v_norm2)

    # Exclude edges and use middle-to-front depth zone.
    dist = cv2.distanceTransform((seat_mask * 255).astype(np.uint8), cv2.DIST_L2, 5)
    margin = max(4.0, min(30.0, 0.045 * math.sqrt(max(area, 1.0))))
    core = (seat_mask > 0) & (dist >= margin)
    depth = (t_map >= 0.30) & (t_map <= 0.86)

    hip_mask = (core & depth).astype(np.uint8)
    hip_mask = largest_component(hip_mask)

    # Relax constraints if mask is empty/tiny.
    if int(hip_mask.sum()) < 80:
        core2 = (seat_mask > 0) & (dist >= max(2.0, 0.5 * margin))
        depth2 = (t_map >= 0.22) & (t_map <= 0.92)
        hip_mask = largest_component((core2 & depth2).astype(np.uint8))

    return hip_mask, back_pt, front_pt, float(margin), t_map


def sample_candidates(
    hip_mask: np.ndarray,
    back_pt: tuple[float, float],
    front_pt: tuple[float, float],
    t_map: np.ndarray,
    grid_step: int,
    max_points: int,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    ys, xs = np.where(hip_mask > 0)
    if len(xs) == 0:
        cx = float(hip_mask.shape[1] * 0.5)
        cy = float(hip_mask.shape[0] * 0.5)
        rec = {"x": round(cx, 2), "y": round(cy, 2), "weight": 1.0}
        return [rec], rec

    vx = front_pt[0] - back_pt[0]
    vy = front_pt[1] - back_pt[1]
    lat_x, lat_y = -vy, vx
    lat_norm = math.hypot(lat_x, lat_y)
    if lat_norm < 1e-6:
        lat_x, lat_y = 1.0, 0.0
        lat_norm = 1.0
    lat_x /= lat_norm
    lat_y /= lat_norm

    # Normalize lateral coordinate to [0, 1].
    us = ((xs.astype(np.float32) - back_pt[0]) * lat_x + (ys.astype(np.float32) - back_pt[1]) * lat_y)
    u_min = float(us.min())
    u_max = float(us.max())
    u_span = max(1e-6, u_max - u_min)

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    step = max(2, int(grid_step))

    candidates: list[dict[str, float]] = []
    for y in range(y0, y1 + 1, step):
        for x in range(x0, x1 + 1, step):
            if hip_mask[y, x] == 0:
                continue
            t = float(t_map[y, x])
            u = ((float(x) - back_pt[0]) * lat_x + (float(y) - back_pt[1]) * lat_y)
            u_norm = (u - u_min) / u_span

            w_depth = math.exp(-0.5 * ((t - 0.52) / 0.20) ** 2)
            w_lat = math.exp(-0.5 * ((u_norm - 0.50) / 0.33) ** 2)
            w = max(0.0, min(1.0, w_depth * w_lat))
            candidates.append(
                {
                    "x": round(float(x), 2),
                    "y": round(float(y), 2),
                    "weight": round(float(w), 4),
                    "depth_t": round(float(t), 4),
                }
            )

    if not candidates:
        cx = float(xs.mean())
        cy = float(ys.mean())
        rec = {"x": round(cx, 2), "y": round(cy, 2), "weight": 1.0}
        return [rec], rec

    candidates.sort(key=lambda p: p["weight"], reverse=True)
    if max_points > 0 and len(candidates) > max_points:
        stride = len(candidates) / float(max_points)
        sampled = []
        for i in range(max_points):
            idx = min(len(candidates) - 1, int(round(i * stride)))
            sampled.append(candidates[idx])
        candidates = sampled

    return candidates, candidates[0]


def render_overlay(
    image_bgr: np.ndarray,
    seat_poly: np.ndarray,
    hip_poly: np.ndarray,
    candidates: list[dict[str, float]],
    recommended: dict[str, float],
) -> np.ndarray:
    out = image_bgr.copy()
    overlay = out.copy()

    cv2.fillPoly(overlay, [seat_poly], (170, 70, 150))
    cv2.fillPoly(overlay, [hip_poly], (60, 220, 235))
    out = cv2.addWeighted(overlay, 0.32, out, 0.68, 0)

    cv2.polylines(out, [seat_poly], True, (255, 0, 180), 4, cv2.LINE_AA)
    cv2.polylines(out, [hip_poly], True, (0, 255, 255), 4, cv2.LINE_AA)

    for p in candidates[:300]:
        cv2.circle(out, (int(round(p["x"])), int(round(p["y"]))), 1, (80, 255, 255), -1)

    rx = int(round(float(recommended["x"])))
    ry = int(round(float(recommended["y"])))
    cv2.circle(out, (rx, ry), 7, (0, 80, 255), -1)
    cv2.circle(out, (rx, ry), 12, (255, 255, 255), 2)
    cv2.putText(
        out,
        "hip-center recommendation",
        (max(8, rx - 160), max(22, ry - 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def run_one(seg_json: Path, output_dir: Path, grid_step: int, max_points: int) -> tuple[Path, Path]:
    data = json.loads(seg_json.read_text(encoding="utf-8"))
    image_path = Path(str(data.get("image", "")))
    if not image_path.exists():
        raise FileNotFoundError(f"image not found in json: {image_path}")

    poly_raw = data.get("seat_contact_polygon")
    if not isinstance(poly_raw, list) or len(poly_raw) < 3:
        raise RuntimeError(f"seat_contact_polygon missing in {seg_json}")

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"failed to read image: {image_path}")
    h, w = image_bgr.shape[:2]

    seat_poly = np.array(
        [[int(round(float(p["x"]))), int(round(float(p["y"])))] for p in poly_raw],
        dtype=np.int32,
    )
    seat_poly[:, 0] = np.clip(seat_poly[:, 0], 0, w - 1)
    seat_poly[:, 1] = np.clip(seat_poly[:, 1], 0, h - 1)

    seat_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(seat_mask, [seat_poly], 1)
    seat_mask = largest_component(seat_mask)

    hip_mask, back_pt, front_pt, margin_px, t_map = build_hip_mask(seat_mask)
    hip_poly_json = polygon_from_mask(hip_mask)
    if len(hip_poly_json) < 3:
        raise RuntimeError(f"failed to build hip polygon for {seg_json}")
    hip_poly = np.array([[p["x"], p["y"]] for p in hip_poly_json], dtype=np.int32)

    candidates, rec = sample_candidates(
        hip_mask,
        back_pt,
        front_pt,
        t_map,
        grid_step=grid_step,
        max_points=max_points,
    )

    out_obj = {
        "segment_json": str(seg_json),
        "image": str(image_path),
        "source_seat_polygon_points": len(poly_raw),
        "hip_range_method": "seat_contact_inner_depth_band_v1",
        "edge_margin_px": round(float(margin_px), 2),
        "hip_range_polygon": hip_poly_json,
        "recommended_hip_center": rec,
        "candidate_points_count": len(candidates),
        "candidate_points": candidates,
        "depth_axis": {
            "back_point": {"x": round(back_pt[0], 2), "y": round(back_pt[1], 2)},
            "front_point": {"x": round(front_pt[0], 2), "y": round(front_pt[1], 2)},
        },
    }

    stem = seg_json.stem
    out_json = output_dir / f"{stem}_hip_range_2d.json"
    out_overlay = output_dir / f"{stem}_hip_range_2d_overlay.png"
    out_json.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    vis = render_overlay(
        image_bgr=image_bgr,
        seat_poly=seat_poly,
        hip_poly=hip_poly,
        candidates=candidates,
        recommended=rec,
    )
    cv2.imwrite(str(out_overlay), vis)
    return out_json, out_overlay


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for p in args.segment_json:
        out_json, out_overlay = run_one(
            p,
            output_dir=output_dir,
            grid_step=int(args.grid_step),
            max_points=int(args.max_points),
        )
        print(f"saved hip-range json: {out_json}")
        print(f"saved hip-range overlay: {out_overlay}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
