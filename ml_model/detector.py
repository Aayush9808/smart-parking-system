"""
Vehicle detection using YOLOv8 with SAHI-style tiling for parking lots.

Pipeline
--------
1. YOLOv8s (small) — full image at imgsz=1280.
2. SAHI-style sliding-window tiles (640px, 25 % overlap) for small vehicles.
3. NMS merge across passes (IoU 0.45).
4. Area filter — remove boxes that are implausibly small or large.
5. Aspect-ratio filter — remove extremely elongated false positives.
"""

import cv2
import numpy as np

# COCO class IDs that represent vehicles
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Size filters (fraction of total image area)
MIN_AREA_RATIO = 0.0005   # box must be ≥ 0.05 % of the image
MAX_AREA_RATIO = 0.25     # box must be ≤ 25 % of the image
MAX_ASPECT_RATIO = 5.0    # w/h or h/w must be ≤ 5

# ── Helpers ───────────────────────────────────────────────────────────────────

def _nms_merge(detections: list, iou_threshold: float = 0.5) -> list:
    """Non-maximum suppression to merge overlapping boxes from multiple tiles."""
    if not detections:
        return []

    boxes = np.array([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in detections], dtype=np.float32)
    scores = np.array([d["confidence"] for d in detections], dtype=np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return [detections[i] for i in keep]


def _filter_detections(detections: list, img_area: float) -> tuple:
    """
    Remove implausibly small, large, or distorted boxes.
    Returns (kept, diagnostics_dict).
    """
    min_px = img_area * MIN_AREA_RATIO
    max_px = img_area * MAX_AREA_RATIO
    kept = []
    removed_small = 0
    removed_large = 0
    removed_aspect = 0

    for d in detections:
        w = d["x2"] - d["x1"]
        h = d["y2"] - d["y1"]
        area = w * h
        aspect = max(w, h) / max(min(w, h), 1)

        if area < min_px:
            removed_small += 1
            continue
        if area > max_px:
            removed_large += 1
            continue
        if aspect > MAX_ASPECT_RATIO:
            removed_aspect += 1
            continue
        kept.append(d)

    diag = {
        "removed_small": removed_small,
        "removed_large": removed_large,
        "removed_aspect": removed_aspect,
        "total_removed": removed_small + removed_large + removed_aspect,
    }
    return kept, diag


class VehicleDetector:
    """YOLOv8s + SAHI tiling for robust parking-lot vehicle detection."""

    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load YOLOv8s — downloads weights on first run (~22 MB)."""
        try:
            from ultralytics import YOLO
            self.model = YOLO("yolov8s.pt")
            print("[Detector] YOLOv8s loaded successfully")
        except Exception as e:
            print(f"[Detector] Warning: Could not load YOLO: {e}")
            self.model = None

    # ──────────────────────────────────────────────────────────────────────────
    def _run_yolo(self, image: np.ndarray, conf: float, imgsz: int) -> list:
        """Run YOLO on a single image/crop and return raw vehicle detections."""
        results = self.model(image, conf=conf, imgsz=imgsz, verbose=False)
        dets = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id in VEHICLE_CLASSES:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    dets.append({
                        "x1": int(x1), "y1": int(y1),
                        "x2": int(x2), "y2": int(y2),
                        "confidence": float(box.conf[0]),
                        "class_name": VEHICLE_CLASSES[cls_id],
                    })
        return dets

    # ──────────────────────────────────────────────────────────────────────────
    def _sahi_tile_detect(self, image: np.ndarray, conf: float,
                          tile_size: int = 640, overlap: float = 0.25) -> list:
        """
        Slice the image into overlapping tiles, run YOLO on each,
        then shift coordinates back to original image space.
        """
        h, w = image.shape[:2]
        step = int(tile_size * (1 - overlap))
        all_dets = []

        for y0 in range(0, h, step):
            for x0 in range(0, w, step):
                y1 = min(y0 + tile_size, h)
                x1 = min(x0 + tile_size, w)
                if (y1 - y0) < tile_size * 0.3 or (x1 - x0) < tile_size * 0.3:
                    continue

                crop = image[y0:y1, x0:x1]
                tile_dets = self._run_yolo(crop, conf=conf, imgsz=tile_size)

                for d in tile_dets:
                    d["x1"] += x0
                    d["y1"] += y0
                    d["x2"] += x0
                    d["y2"] += y0
                    all_dets.append(d)

        return all_dets

    # ──────────────────────────────────────────────────────────────────────────
    def detect(self, image: np.ndarray, confidence_threshold: float = 0.15):
        """
        Full detection pipeline → returns (detections_list, diagnostics_dict).

        Diagnostics include counts from each stage so the caller can expose
        exactly what happened to the user.
        """
        diag = {
            "image_w": 0, "image_h": 0,
            "conf_threshold": confidence_threshold,
            "raw_full": 0, "raw_tiled": 0,
            "post_nms": 0, "post_filter": 0,
            "filter_detail": {},
        }

        if self.model is None:
            print("[Detector] Model not loaded — returning empty")
            return [], diag

        h, w = image.shape[:2]
        img_area = h * w
        diag["image_w"] = w
        diag["image_h"] = h
        print(f"[Detector] Image {w}×{h}, conf={confidence_threshold}")

        # Pass 1 — full image at high resolution
        full_dets = self._run_yolo(image, conf=confidence_threshold, imgsz=1280)
        diag["raw_full"] = len(full_dets)
        print(f"[Detector] Full-image pass: {len(full_dets)} vehicles")

        # Pass 2 — SAHI tiling
        tile_dets = []
        if w >= 800 or h >= 800:
            tile_dets = self._sahi_tile_detect(image, conf=confidence_threshold,
                                               tile_size=640, overlap=0.25)
            diag["raw_tiled"] = len(tile_dets)
            print(f"[Detector] Tiled pass: {len(tile_dets)} vehicles")

        # Merge and deduplicate
        all_dets = full_dets + tile_dets
        merged = _nms_merge(all_dets, iou_threshold=0.45)
        diag["post_nms"] = len(merged)
        print(f"[Detector] After NMS: {len(merged)} vehicles")

        # Area / aspect-ratio filter
        filtered, filt_diag = _filter_detections(merged, img_area)
        diag["post_filter"] = len(filtered)
        diag["filter_detail"] = filt_diag
        if filt_diag["total_removed"]:
            print(f"[Detector] Filtered out {filt_diag['total_removed']} "
                  f"(small={filt_diag['removed_small']}, "
                  f"large={filt_diag['removed_large']}, "
                  f"aspect={filt_diag['removed_aspect']})")

        print(f"[Detector] Final: {len(filtered)} vehicles")
        for d in filtered:
            print(f"  {d['class_name']} conf={d['confidence']:.2f} "
                  f"box=({d['x1']},{d['y1']})-({d['x2']},{d['y2']})")

        return filtered, diag
