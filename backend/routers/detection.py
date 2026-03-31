"""
Detection API — upload a real image for YOLO or generate a synthetic sample.
"""

import base64
import random
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from ml_model.detector import VehicleDetector
from ml_model.slot_mapper import SlotMapper
from ml_model.slot_estimator import estimate_parking_capacity, build_slot_list
from ml_model.simulator import generate_parking_image
from ml_model.parking_config import PARKING_SLOTS, IMAGE_WIDTH, IMAGE_HEIGHT
from backend.database import store_snapshot

router = APIRouter()

# Initialise once at import time
detector = VehicleDetector()
mapper = SlotMapper(PARKING_SLOTS)

IMAGES_DIR = Path(__file__).parent.parent.parent / "data" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def _img_to_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode("utf-8")


# ── Visualization helpers ─────────────────────────────────────────────────────

def _draw_vehicle_boxes(img: np.ndarray, detections: list) -> np.ndarray:
    """Draw RED bounding boxes on every detected (occupied) vehicle."""
    out = img.copy()
    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        conf = det["confidence"]
        label = f'{det["class_name"]} {conf:.0%}'

        # Red box for occupied
        color = (0, 0, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def _draw_analytics_overlay(img: np.ndarray, stats: dict) -> np.ndarray:
    """Draw an analytics panel at the bottom of the image."""
    out = img.copy()
    h, w = out.shape[:2]

    # Semi-transparent black bar at bottom
    bar_h = 70
    overlay = out.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)

    y = h - bar_h + 22
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(out, f"Vehicles: {stats['occupied']}", (10, y),
                font, 0.6, (0, 0, 255), 2)
    cv2.putText(out, f"Est. Empty: {stats['empty']}", (200, y),
                font, 0.6, (0, 255, 0), 2)
    cv2.putText(out, f"Est. Capacity: {stats['estimated_capacity']}", (420, y),
                font, 0.6, (255, 200, 50), 2)

    y2 = y + 28
    occ_pct = stats["occupancy_pct"]
    color = (0, 0, 255) if occ_pct > 75 else (0, 200, 255) if occ_pct > 50 else (0, 255, 0)
    cv2.putText(out, f"Occupancy: {occ_pct}%", (10, y2),
                font, 0.7, color, 2)
    cv2.putText(out, stats.get("confidence_note", ""), (250, y2),
                font, 0.45, (180, 180, 180), 1)

    return out


def _draw_slot_overlay(img: np.ndarray, slots: list) -> np.ndarray:
    """Draw coloured status boxes on the synthetic parking-lot image."""
    out = img.copy()
    for s in slots:
        x1, y1, x2, y2 = s["x1"], s["y1"], s["x2"], s["y2"]
        colour = (0, 0, 255) if s["status"] == "occupied" else (0, 200, 0)
        overlay = out.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, -1)
        cv2.addWeighted(overlay, 0.30, out, 0.70, 0, out)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(out, s["name"], (x1 + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    occ = sum(1 for s in slots if s["status"] == "occupied")
    cv2.putText(out, f"Available: {len(slots) - occ}/{len(slots)}",
                (10, img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/detect")
async def detect_from_upload(file: UploadFile = File(...)):
    """
    Upload a real parking-lot image → YOLO detects vehicles →
    dynamic slot estimation → analytics.
    """
    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image file")

    # 1. Detect all vehicles (YOLOv8s + SAHI tiling)
    detections = detector.detect(img, confidence_threshold=0.15)

    # 2. Estimate parking capacity dynamically
    stats = estimate_parking_capacity(
        detections, image_shape=img.shape[:2], min_total_slots=12
    )

    # 3. Build slot list for dashboard grid
    slot_statuses = build_slot_list(detections, stats["estimated_capacity"])

    # 4. Annotate image — red boxes on vehicles + analytics bar
    result_img = _draw_vehicle_boxes(img, detections)
    result_img = _draw_analytics_overlay(result_img, stats)

    ts = datetime.now().isoformat()
    store_snapshot(ts, slot_statuses)
    cv2.imwrite(str(IMAGES_DIR / "latest_detection.jpg"), result_img)

    return {
        "status": "success",
        "timestamp": ts,
        "mode": "yolo_detection",
        # Detection data
        "detections": detections,
        "slots": slot_statuses,
        # Analytics
        "total_vehicles": stats["total_vehicles"],
        "total_occupied": stats["occupied"],
        "total_empty": stats["empty"],
        "total_slots": stats["estimated_capacity"],
        "occupancy_pct": stats["occupancy_pct"],
        "confidence_note": stats["confidence_note"],
        # Image
        "result_image": _img_to_b64(result_img),
    }


@router.get("/detect/sample")
async def detect_sample():
    """Generate a synthetic parking scene (random occupancy)."""
    num_occ = random.randint(3, 9)
    occ_ids = random.sample([s["id"] for s in PARKING_SLOTS], num_occ)

    slot_statuses = []
    for s in PARKING_SLOTS:
        status = "occupied" if s["id"] in occ_ids else "empty"
        slot_statuses.append({
            "id": s["id"], "name": s["name"], "status": status,
            "confidence": round(random.uniform(0.85, 0.99), 2),
            "x1": s["x1"], "y1": s["y1"], "x2": s["x2"], "y2": s["y2"],
        })

    img = generate_parking_image(slot_statuses)
    result_img = _draw_slot_overlay(img, slot_statuses)

    ts = datetime.now().isoformat()
    store_snapshot(ts, slot_statuses)
    cv2.imwrite(str(IMAGES_DIR / "latest_detection.jpg"), result_img)

    return {
        "status": "success",
        "timestamp": ts,
        "mode": "simulation",
        "detections": [],
        "slots": slot_statuses,
        "total_vehicles": num_occ,
        "total_occupied": num_occ,
        "total_empty": len(PARKING_SLOTS) - num_occ,
        "total_slots": len(PARKING_SLOTS),
        "occupancy_pct": round(num_occ / len(PARKING_SLOTS) * 100, 1),
        "confidence_note": "Simulated data — not from real detection.",
        "result_image": _img_to_b64(result_img),
    }
