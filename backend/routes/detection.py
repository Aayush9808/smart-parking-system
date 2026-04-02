"""
Detection Routes — upload image → full ML pipeline → annotated result.
"""

import cv2
import numpy as np
import base64
import random
from datetime import datetime
from fastapi import APIRouter, UploadFile, File

from ml.detector import detect_cars
from ml.slot_generator import generate_slots
from ml.mapper import map_cars_to_slots
from services.analytics import compute_analytics
from backend.database import save_occupancy, save_slot_snapshot

router = APIRouter(prefix="/api/detect", tags=["Detection"])


@router.post("")
async def detect_image(file: UploadFile = File(...)):
    """
    Full detection pipeline:
      Upload image → detect cars → generate slots → map → annotate → return
    """
    # Read image
    contents = await file.read()
    arr = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if image is None:
        return {"success": False, "error": "Invalid image file"}

    h, w = image.shape[:2]

    # Step 1: Detect cars
    detections, diagnostics = detect_cars(image)

    # Step 2: Generate slot grid
    slots, grid_info = generate_slots(detections, w, h)

    # Step 3: Map cars to slots
    mapped_slots, map_stats = map_cars_to_slots(detections, slots)

    # Step 4: Compute analytics
    analytics = compute_analytics(mapped_slots, detections)

    # Step 5: Annotate image
    annotated = _annotate_image(image.copy(), detections, mapped_slots, analytics)

    # Step 6: Encode result image
    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    result_b64 = base64.b64encode(buffer).decode("utf-8")

    # Step 7: Save to database
    ts = datetime.now().isoformat()
    save_occupancy({
        "timestamp": ts,
        "total_slots": analytics["total_slots"],
        "occupied": analytics["occupied"],
        "empty": analytics["empty"],
        "occupancy_rate": analytics["occupancy_rate"],
    })
    save_slot_snapshot(ts, mapped_slots)

    return {
        "success": True,
        "detections": detections,
        "slots": mapped_slots,
        "analytics": analytics,
        "diagnostics": diagnostics,
        "grid_info": grid_info,
        "mapping_stats": map_stats,
        "result_image": result_b64,
    }


@router.get("/sample")
async def generate_sample():
    """Generate a synthetic parking scene for demo."""
    width, height = 800, 600
    image = np.full((height, width, 3), (40, 42, 48), dtype=np.uint8)

    # Draw parking lines
    for i in range(5):
        x = 80 + i * 150
        cv2.rectangle(image, (x, 80), (x + 120, 200), (80, 80, 80), 2)
        cv2.rectangle(image, (x, 320), (x + 120, 440), (80, 80, 80), 2)

    # Place synthetic "cars" randomly
    car_positions = []
    num_cars = random.randint(3, 8)
    for _ in range(num_cars):
        row = random.choice([0, 1])
        col = random.randint(0, 4)
        x1 = 85 + col * 150
        y1 = 85 + row * 240
        x2 = x1 + 110
        y2 = y1 + 110

        # Check overlap
        overlap = False
        for cx1, cy1, cx2, cy2 in car_positions:
            if x1 < cx2 and x2 > cx1 and y1 < cy2 and y2 > cy1:
                overlap = True
                break
        if overlap:
            continue

        car_positions.append((x1, y1, x2, y2))

        # Draw car rectangle
        color = (
            random.randint(100, 220),
            random.randint(100, 220),
            random.randint(100, 220),
        )
        cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(image, (x1, y1), (x2, y2), (200, 200, 200), 1)

    # Build mock detections
    detections = []
    for i, (x1, y1, x2, y2) in enumerate(car_positions):
        conf = round(random.uniform(0.65, 0.98), 3)
        detections.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "confidence": conf, "class_name": "car",
            "cx": round((x1 + x2) / 2, 1),
            "cy": round((y1 + y2) / 2, 1),
        })

    # Generate slots and map
    slots, grid_info = generate_slots(detections, width, height)
    mapped_slots, map_stats = map_cars_to_slots(detections, slots)
    analytics = compute_analytics(mapped_slots, detections)

    # Annotate
    annotated = _annotate_image(image.copy(), detections, mapped_slots, analytics)
    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    result_b64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "success": True,
        "detections": detections,
        "slots": mapped_slots,
        "analytics": analytics,
        "grid_info": grid_info,
        "mapping_stats": map_stats,
        "result_image": result_b64,
    }


def _annotate_image(
    img: np.ndarray,
    detections: list[dict],
    slots: list[dict],
    analytics: dict,
) -> np.ndarray:
    """Draw slot grid, car boxes, and stats bar on image."""
    h, w = img.shape[:2]

    # Draw slot grid
    for slot in slots:
        cx, cy = slot["cx"], slot["cy"]
        sw, sh = slot["w"] / 2, slot["h"] / 2
        x1, y1 = int(cx - sw), int(cy - sh)
        x2, y2 = int(cx + sw), int(cy + sh)

        if slot["status"] == "occupied":
            color = (0, 0, 200)     # Red
            thickness = 2
        else:
            color = (0, 180, 0)     # Green
            thickness = 1

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Slot name label
        font_scale = 0.35
        cv2.putText(img, slot["name"], (x1 + 2, y1 + 12),
                     cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)

    # Draw car bounding boxes
    for det in detections:
        cv2.rectangle(img, (det["x1"], det["y1"]), (det["x2"], det["y2"]),
                       (0, 255, 255), 2)
        label = f"car {det['confidence']:.0%}"
        cv2.putText(img, label, (det["x1"], det["y1"] - 6),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Stats bar at bottom
    bar_h = 36
    cv2.rectangle(img, (0, h - bar_h), (w, h), (30, 30, 30), -1)
    stats_text = (
        f"Cars: {analytics['cars_detected']}  |  "
        f"Slots: {analytics['total_slots']}  |  "
        f"Occupied: {analytics['occupied']}  |  "
        f"Empty: {analytics['empty']}  |  "
        f"Rate: {analytics['occupancy_percent']}%"
    )
    cv2.putText(img, stats_text, (10, h - 12),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    return img
