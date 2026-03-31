"""
Dynamic parking-slot estimator.

Instead of hardcoded slot coordinates, this module estimates the total
parking capacity and maps each detected vehicle to a slot region.

Approach (honest for a prototype)
---------------------------------
1. Each detected vehicle bounding box = 1 occupied slot.
2. Estimate the total parking area from the convex hull of all detections
   (with padding to account for visible empty spaces at the edges).
3. Average detected vehicle size → one "standard slot" footprint.
4. Divide parking area by slot footprint → estimated total capacity.
5. Empty slots = estimated capacity − occupied.

This is an ESTIMATION — not ground-truth.  True slot detection requires
either (a) known lot layout or (b) a slot-segmentation model trained on
parking lot images.
"""

import math
from typing import Dict, List, Tuple

import cv2
import numpy as np


def estimate_parking_capacity(
    detections: List[Dict],
    image_shape: Tuple[int, int],   # (h, w)
    min_total_slots: int = 0,
) -> Dict:
    """
    Estimate total parking capacity from detected vehicles.

    Returns
    -------
    dict with keys:
        total_vehicles    – number of detected vehicles
        avg_vehicle_area  – average bounding box area (px²)
        parking_area      – estimated total parking area (px²)
        estimated_capacity – estimated total slots
        occupied           – same as total_vehicles
        empty              – estimated_capacity − occupied
        occupancy_pct      – occupied / estimated_capacity × 100
        confidence_note    – textual note on estimation quality
    """
    n_vehicles = len(detections)
    img_h, img_w = image_shape

    if n_vehicles == 0:
        cap = max(min_total_slots, 12)   # sensible default when nothing detected
        return {
            "total_vehicles": 0,
            "avg_vehicle_area": 0,
            "parking_area": img_h * img_w,
            "estimated_capacity": cap,
            "occupied": 0,
            "empty": cap,
            "occupancy_pct": 0.0,
            "confidence_note": "No vehicles detected — capacity is a rough estimate.",
        }

    # ── Vehicle sizes ─────────────────────────────────────────────────────────
    areas = []
    centers = []
    for d in detections:
        w = d["x2"] - d["x1"]
        h = d["y2"] - d["y1"]
        areas.append(w * h)
        centers.append(((d["x1"] + d["x2"]) / 2, (d["y1"] + d["y2"]) / 2))

    avg_area = float(np.mean(areas))
    median_area = float(np.median(areas))

    # Use median to be robust against outlier-sized detections
    slot_area = median_area

    # ── Parking region estimation ─────────────────────────────────────────────
    # Convex hull of all detection centres, padded outward
    pts = np.array(centers, dtype=np.float32)
    if len(pts) >= 3:
        hull = cv2.convexHull(pts)
        hull_area = float(cv2.contourArea(hull))
    else:
        # Bounding rect of all detections
        all_x1 = min(d["x1"] for d in detections)
        all_y1 = min(d["y1"] for d in detections)
        all_x2 = max(d["x2"] for d in detections)
        all_y2 = max(d["y2"] for d in detections)
        hull_area = float((all_x2 - all_x1) * (all_y2 - all_y1))

    # Add generous padding (×2.0) — the visible parking area is typically
    # much larger than just the hull of occupied spots.
    parking_area = hull_area * 2.0
    # Clamp to image area
    parking_area = min(parking_area, img_h * img_w * 0.85)

    # ── Capacity estimation ───────────────────────────────────────────────────
    # Parking slots are ~1.3× the size of a car bounding box (gap between cars)
    slot_footprint = slot_area * 1.3
    raw_capacity = parking_area / slot_footprint if slot_footprint > 0 else n_vehicles

    # Estimated capacity should be at least the number of parked cars
    estimated_capacity = max(int(round(raw_capacity)), n_vehicles)
    estimated_capacity = max(estimated_capacity, min_total_slots)

    occupied = n_vehicles
    empty = estimated_capacity - occupied
    occ_pct = round(occupied / estimated_capacity * 100, 1) if estimated_capacity else 0

    # ── Confidence note ───────────────────────────────────────────────────────
    if n_vehicles >= 5:
        note = "Good estimate — based on detected vehicle sizes and parking region."
    elif n_vehicles >= 2:
        note = "Approximate — few vehicles detected; capacity may be under/over-estimated."
    else:
        note = "Rough estimate — very few detections; capacity is uncertain."

    return {
        "total_vehicles": n_vehicles,
        "avg_vehicle_area": round(avg_area),
        "parking_area": round(parking_area),
        "estimated_capacity": estimated_capacity,
        "occupied": occupied,
        "empty": empty,
        "occupancy_pct": occ_pct,
        "confidence_note": note,
    }


def build_slot_list(detections: List[Dict], estimated_capacity: int) -> List[Dict]:
    """
    Build a slot-status list from detections + estimated empty count.

    Occupied slots  → named S1, S2, … with the detection bounding box.
    Empty slots     → named E1, E2, … with zeroed coordinates.
    """
    slots = []

    # Occupied slots (one per detection)
    for i, det in enumerate(detections):
        slots.append({
            "id": i + 1,
            "name": f"S{i + 1}",
            "status": "occupied",
            "confidence": round(det["confidence"], 2),
            "class_name": det.get("class_name", "car"),
            "x1": det["x1"], "y1": det["y1"],
            "x2": det["x2"], "y2": det["y2"],
        })

    # Empty slots
    n_empty = max(0, estimated_capacity - len(detections))
    for j in range(n_empty):
        idx = len(detections) + j + 1
        slots.append({
            "id": idx,
            "name": f"E{j + 1}",
            "status": "empty",
            "confidence": 1.0,
            "class_name": "",
            "x1": 0, "y1": 0, "x2": 0, "y2": 0,
        })

    return slots
