"""
Detection API — real image detection and synthetic sample generation.

Pipeline (POST /detect):
  1. YOLO + SAHI → raw detections
  2. Area / aspect filter → clean detections
  3. Generate virtual slot grid from vehicle sizes
  4. Map vehicles → slots (center-distance, greedy)
  5. Annotate image: grid overlay + vehicle boxes + stats bar
  6. Return everything to the frontend
"""

import base64
import random
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from ml_model.detector import VehicleDetector
from ml_model.slot_estimator import generate_slot_grid
from ml_model.slot_mapper import map_vehicles_to_slots
from ml_model.simulator import generate_parking_image
from ml_model.parking_config import PARKING_SLOTS, IMAGE_WIDTH, IMAGE_HEIGHT
from backend.database import store_snapshot

router = APIRouter()

# Initialise once
detector = VehicleDetector()

IMAGES_DIR = Path(__file__).parent.parent.parent / "data" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def _img_to_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode("utf-8")


# ── Visualization ─────────────────────────────────────────────────────────────

def _draw_grid_overlay(img: np.ndarray, slots: list) -> np.ndarray:
    """Draw the virtual slot grid on the image (green=empty, red=occupied)."""
    out = img.copy()
    overlay = out.copy()

    for s in slots:
        x1, y1, x2, y2 = s["x1"], s["y1"], s["x2"], s["y2"]
        is_occ = s["status"] == "occupied"
        color = (0, 0, 220) if is_occ else (0, 180, 0)

        # Semi-transparent fill
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    cv2.addWeighted(overlay, 0.20, out, 0.80, 0, out)

    # Draw borders + labels on top
    for s in slots:
        x1, y1, x2, y2 = s["x1"], s["y1"], s["x2"], s["y2"]
        is_occ = s["status"] == "occupied"
        color = (0, 0, 220) if is_occ else (0, 180, 0)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Slot name label
        label = s["name"]
        font_scale = max(0.35, min(0.6, (x2 - x1) / 120))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        lx = x1 + 3
        ly = y2 - 4
        cv2.putText(out, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1,
                    cv2.LINE_AA)

    return out


def _draw_vehicle_boxes(img: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes on detected vehicles."""
    out = img.copy()
    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        conf = det["confidence"]
        label = f'{det["class_name"]} {conf:.0%}'

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 0, 255), -1)
        cv2.putText(out, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _draw_stats_bar(img: np.ndarray, stats: dict, grid_info: dict) -> np.ndarray:
    """Draw a stats panel at the bottom of the image."""
    out = img.copy()
    h, w = out.shape[:2]

    bar_h = 68
    overlay = out.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

    y = h - bar_h + 22
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(out, f"Vehicles: {stats['total_vehicles']}", (10, y),
                font, 0.6, (0, 120, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"Occupied: {stats['occupied']}", (190, y),
                font, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"Empty: {stats['empty']}", (370, y),
                font, 0.6, (0, 220, 0), 2, cv2.LINE_AA)
    cv2.putText(out, f"Slots: {stats['total_slots']}", (520, y),
                font, 0.6, (255, 200, 50), 2, cv2.LINE_AA)

    y2 = y + 26
    occ = stats["occupancy_pct"]
    c = (0, 0, 255) if occ > 75 else (0, 200, 255) if occ > 50 else (0, 220, 0)
    cv2.putText(out, f"Occupancy: {occ}%", (10, y2),
                font, 0.65, c, 2, cv2.LINE_AA)

    grid_label = f"Grid: {grid_info['rows']}x{grid_info['cols']}"
    cv2.putText(out, grid_label, (250, y2),
                font, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    return out


def _draw_slot_overlay(img: np.ndarray, slots: list) -> np.ndarray:
    """Coloured overlay for synthetic parking scenes."""
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
    Upload a real parking-lot image → full pipeline:
    detect → filter → grid → map → annotate → return.
    """
    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image file")

    # ── Step 1: Detect vehicles ───────────────────────────────────────────
    detections, det_diag = detector.detect(img, confidence_threshold=0.15)

    # ── Step 2: Generate virtual slot grid ────────────────────────────────
    slots, grid_info = generate_slot_grid(
        detections, image_shape=img.shape[:2]
    )

    # ── Step 3: Map vehicles → slots ──────────────────────────────────────
    mapped_slots, map_stats = map_vehicles_to_slots(slots, detections)

    # ── Step 4: Annotate image ────────────────────────────────────────────
    result_img = img.copy()
    if mapped_slots:
        result_img = _draw_grid_overlay(result_img, mapped_slots)
    result_img = _draw_vehicle_boxes(result_img, detections)
    result_img = _draw_stats_bar(result_img, map_stats, grid_info)

    ts = datetime.now().isoformat()
    store_snapshot(ts, mapped_slots)
    cv2.imwrite(str(IMAGES_DIR / "latest_detection.jpg"), result_img)

    # Build confidence note
    note = grid_info["confidence_note"]

    return {
        "status": "success",
        "timestamp": ts,
        "mode": "yolo_detection",
        # Core data
        "detections": detections,
        "slots": mapped_slots,
        # Stats (all derived from real mapping, NO fabricated numbers)
        "total_vehicles": map_stats["total_vehicles"],
        "total_occupied": map_stats["occupied"],
        "total_empty": map_stats["empty"],
        "total_slots": map_stats["total_slots"],
        "occupancy_pct": map_stats["occupancy_pct"],
        # Grid info
        "grid_rows": grid_info["rows"],
        "grid_cols": grid_info["cols"],
        # Diagnostics
        "confidence_note": note,
        "diagnostics": {
            "detection": det_diag,
            "grid": {
                "rows": grid_info["rows"],
                "cols": grid_info["cols"],
                "slot_w": grid_info["slot_w"],
                "slot_h": grid_info["slot_h"],
            },
            "mapping": {
                "matched": map_stats["matched_vehicles"],
                "unmatched": map_stats["unmatched_vehicles"],
            },
        },
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
            "class_name": "car" if status == "occupied" else "",
            "x1": s["x1"], "y1": s["y1"], "x2": s["x2"], "y2": s["y2"],
        })

    img = generate_parking_image(slot_statuses)
    result_img = _draw_slot_overlay(img, slot_statuses)

    ts = datetime.now().isoformat()
    store_snapshot(ts, slot_statuses)
    cv2.imwrite(str(IMAGES_DIR / "latest_detection.jpg"), result_img)

    occ = sum(1 for s in slot_statuses if s["status"] == "occupied")
    total = len(slot_statuses)

    return {
        "status": "success",
        "timestamp": ts,
        "mode": "simulation",
        "detections": [],
        "slots": slot_statuses,
        "total_vehicles": occ,
        "total_occupied": occ,
        "total_empty": total - occ,
        "total_slots": total,
        "occupancy_pct": round(occ / total * 100, 1),
        "confidence_note": "Simulated data — not from real detection.",
        "result_image": _img_to_b64(result_img),
    }
