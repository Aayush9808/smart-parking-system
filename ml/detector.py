"""
Car Detector — YOLOv8 + SAHI tiling, car-class-only pipeline.

Pipeline:
  1. Full-image YOLOv8 pass @ 1280px (catches large/medium cars)
  2. SAHI tiled pass @ 640px tiles with 25% overlap (catches small/distant cars)
  3. NMS merge across both passes (removes duplicates)
  4. Geometric filtering (area ratio, aspect ratio)

Returns:
  (detections, diagnostics)
  - detections: list of {x1, y1, x2, y2, confidence, class_name, cx, cy}
  - diagnostics: dict with counts at each pipeline stage
"""

import numpy as np
from ultralytics import YOLO
from ml.config import (
    YOLO_MODEL, YOLO_CONF, YOLO_IOU_NMS, YOLO_IMG_SIZE,
    YOLO_CAR_CLASS_ID,
    SAHI_TILE_SIZE, SAHI_OVERLAP_RATIO, SAHI_MERGE_IOU,
    MIN_BOX_AREA_RATIO, MAX_BOX_AREA_RATIO, MAX_ASPECT_RATIO,
)

# ─── Load model once at import time ─────────────────────────────
_model = YOLO(YOLO_MODEL)
print(f"[Detector] {YOLO_MODEL} loaded — car-only mode (class {YOLO_CAR_CLASS_ID})")


def detect_cars(image: np.ndarray) -> tuple[list[dict], dict]:
    """
    Run the full detection pipeline on a BGR numpy image.
    Returns (detections, diagnostics).
    """
    h, w = image.shape[:2]
    img_area = h * w

    # Stage 1: Full-image pass
    full_boxes = _run_yolo(image, img_size=YOLO_IMG_SIZE)

    # Stage 2: SAHI tiled pass
    tile_boxes = _run_sahi_tiles(image, h, w)

    # Stage 3: Merge + NMS
    all_boxes = full_boxes + tile_boxes
    merged = _nms(all_boxes, SAHI_MERGE_IOU)

    # Stage 4: Geometric filtering
    filtered = _filter_boxes(merged, img_area)

    # Build output
    detections = []
    for box in filtered:
        x1, y1, x2, y2, conf = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        detections.append({
            "x1": int(x1), "y1": int(y1),
            "x2": int(x2), "y2": int(y2),
            "confidence": round(float(conf), 3),
            "class_name": "car",
            "cx": round(float(cx), 1),
            "cy": round(float(cy), 1),
        })

    diagnostics = {
        "image_size": f"{w}x{h}",
        "full_pass_raw": len(full_boxes),
        "sahi_tiles_raw": len(tile_boxes),
        "after_nms": len(merged),
        "after_filter": len(filtered),
        "cars_detected": len(detections),
    }

    return detections, diagnostics


# ─── Internal helpers ────────────────────────────────────────────

def _run_yolo(img: np.ndarray, img_size: int) -> list:
    """Run YOLOv8 on a single image, return car-only boxes."""
    results = _model(
        img,
        conf=YOLO_CONF,
        iou=YOLO_IOU_NMS,
        imgsz=img_size,
        classes=[YOLO_CAR_CLASS_ID],
        verbose=False,
    )
    boxes = []
    for r in results:
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
            conf = float(b.conf[0].cpu().numpy())
            boxes.append([x1, y1, x2, y2, conf])
    return boxes


def _run_sahi_tiles(image: np.ndarray, h: int, w: int) -> list:
    """Sliding-window tiled detection for small cars."""
    tile = SAHI_TILE_SIZE
    stride = int(tile * (1 - SAHI_OVERLAP_RATIO))
    boxes = []

    for y0 in range(0, h, stride):
        for x0 in range(0, w, stride):
            y1 = min(y0 + tile, h)
            x1 = min(x0 + tile, w)
            crop = image[y0:y1, x0:x1]

            # Skip tiny edge crops
            if crop.shape[0] < tile // 2 or crop.shape[1] < tile // 2:
                continue

            tile_results = _run_yolo(crop, img_size=tile)
            for b in tile_results:
                # Shift coordinates back to full-image space
                b[0] += x0
                b[1] += y0
                b[2] += x0
                b[3] += y0
                boxes.append(b)

    return boxes


def _nms(boxes: list, iou_thresh: float) -> list:
    """Non-Maximum Suppression to remove duplicate detections."""
    if not boxes:
        return []

    arr = np.array(boxes)
    x1 = arr[:, 0]
    y1 = arr[:, 1]
    x2 = arr[:, 2]
    y2 = arr[:, 3]
    scores = arr[:, 4]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        remaining = np.where(iou <= iou_thresh)[0]
        order = order[remaining + 1]

    return [boxes[i] for i in keep]


def _filter_boxes(boxes: list, img_area: float) -> list:
    """Remove noise: too small, too large, or elongated boxes."""
    good = []
    for b in boxes:
        x1, y1, x2, y2, conf = b
        bw = x2 - x1
        bh = y2 - y1
        area = bw * bh
        ratio = area / img_area
        aspect = max(bw, bh) / (min(bw, bh) + 1e-6)

        if ratio < MIN_BOX_AREA_RATIO:
            continue
        if ratio > MAX_BOX_AREA_RATIO:
            continue
        if aspect > MAX_ASPECT_RATIO:
            continue

        good.append(b)
    return good
