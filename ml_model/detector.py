"""
Vehicle detection using YOLOv8 with SAHI-style tiling for parking lots.

Key techniques
--------------
1. YOLOv8s (small) — 3.5× more parameters than nano, much better on small objects.
2. High-resolution inference (imgsz=1280) for the full image.
3. SAHI-style sliding-window tiling — slices the image into overlapping crops
   so small/distant cars in aerial views are large enough for the detector.
4. NMS merging across tiles to remove duplicate boxes.
"""

import cv2
import numpy as np

# COCO class IDs that represent vehicles
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

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
                # Skip very small edge tiles
                if (y1 - y0) < tile_size * 0.3 or (x1 - x0) < tile_size * 0.3:
                    continue

                crop = image[y0:y1, x0:x1]
                tile_dets = self._run_yolo(crop, conf=conf, imgsz=tile_size)

                # Shift coords to original image space
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
        Full detection pipeline:
        1. Run YOLO on full image at high resolution (1280px)
        2. Run SAHI tiling (640px tiles, 25% overlap)
        3. Merge all detections via NMS
        """
        if self.model is None:
            print("[Detector] Model not loaded — returning empty")
            return []

        h, w = image.shape[:2]
        print(f"[Detector] Image {w}×{h}, conf={confidence_threshold}")

        # Pass 1 — full image at high resolution
        full_dets = self._run_yolo(image, conf=confidence_threshold, imgsz=1280)
        print(f"[Detector] Full-image pass: {len(full_dets)} vehicles")

        # Pass 2 — SAHI tiling (only if image is large enough to benefit)
        tile_dets = []
        if w >= 800 or h >= 800:
            tile_dets = self._sahi_tile_detect(image, conf=confidence_threshold,
                                               tile_size=640, overlap=0.25)
            print(f"[Detector] Tiled pass: {len(tile_dets)} vehicles")

        # Merge and deduplicate
        all_dets = full_dets + tile_dets
        merged = _nms_merge(all_dets, iou_threshold=0.45)

        print(f"[Detector] After NMS: {len(merged)} vehicles")
        for d in merged:
            print(f"[Detector]   {d['class_name']} conf={d['confidence']:.2f} "
                  f"box=({d['x1']},{d['y1']})-({d['x2']},{d['y2']})")

        return merged
