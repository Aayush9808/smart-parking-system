"""
Vehicle → Slot spatial mapper.

Given a list of virtual grid slots (from slot_estimator.generate_slot_grid)
and a list of detected vehicles, this module assigns each vehicle to the
nearest slot by center-point distance.

Rules
-----
- Each vehicle occupies at most ONE slot.
- Each slot is occupied by at most ONE vehicle.
- Assignment is greedy by descending confidence (best detections first).
- Slots that get no vehicle remain "empty".
"""

import math
from typing import Dict, List, Tuple


def _center(box: Dict) -> Tuple[float, float]:
    return ((box["x1"] + box["x2"]) / 2.0, (box["y1"] + box["y2"]) / 2.0)


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def map_vehicles_to_slots(
    slots: List[Dict],
    detections: List[Dict],
) -> Tuple[List[Dict], Dict]:
    """
    Assign each detected vehicle to the nearest available slot.

    Parameters
    ----------
    slots : list[dict]
        Grid slots from generate_slot_grid (each has id, name, x1…y2, status="empty").
    detections : list[dict]
        Filtered vehicle detections (each has x1…y2, confidence, class_name).

    Returns
    -------
    mapped_slots : list[dict]
        Updated slot list with status, confidence, class_name filled in.
    stats : dict
        total_slots, occupied, empty, occupancy_pct, mapping diagnostics.
    """
    # Work on copies so we don't mutate the inputs
    mapped = [dict(s) for s in slots]
    taken_slot_ids = set()

    # Sort detections by confidence descending — best matches first
    sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)

    # Pre-compute slot centres
    slot_centers = {s["id"]: _center(s) for s in mapped}

    matched = 0
    unmatched_vehicles = 0

    for det in sorted_dets:
        det_center = _center(det)

        # Find nearest un-taken slot
        best_id = None
        best_dist = float("inf")
        for s in mapped:
            if s["id"] in taken_slot_ids:
                continue
            d = _dist(det_center, slot_centers[s["id"]])
            if d < best_dist:
                best_dist = d
                best_id = s["id"]

        if best_id is not None:
            taken_slot_ids.add(best_id)
            for s in mapped:
                if s["id"] == best_id:
                    s["status"] = "occupied"
                    s["confidence"] = round(det["confidence"], 2)
                    s["class_name"] = det.get("class_name", "car")
                    break
            matched += 1
        else:
            unmatched_vehicles += 1

    total = len(mapped)
    occupied = sum(1 for s in mapped if s["status"] == "occupied")
    empty = total - occupied
    occ_pct = round(occupied / total * 100, 1) if total else 0.0

    stats = {
        "total_slots": total,
        "occupied": occupied,
        "empty": empty,
        "occupancy_pct": occ_pct,
        "matched_vehicles": matched,
        "unmatched_vehicles": unmatched_vehicles,
        "total_vehicles": len(detections),
    }

    return mapped, stats
