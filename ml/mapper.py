"""
Vehicle → Slot Mapper — greedy center-distance assignment.

Algorithm:
  1. Sort cars by confidence (highest first → best detections get priority)
  2. For each car, compute Euclidean distance to every unassigned slot center
  3. Assign car to nearest slot
  4. Mark slot as "occupied" with car's confidence
  5. All unmatched slots remain "empty"

Guarantees:
  - 1:1 mapping (no slot assigned twice, no car mapped to two slots)
  - occupied + empty == total_slots (always)
  - If cars > slots, excess cars are reported but don't create phantom slots
"""

import math


def map_cars_to_slots(
    detections: list[dict],
    slots: list[dict],
) -> tuple[list[dict], dict]:
    """
    Map detected cars to nearest slots.

    Returns:
        (updated_slots, stats)
        - updated_slots: copy of slots with status/confidence updated
        - stats: {total_cars, mapped, unmapped_cars, occupied, empty, total_slots}
    """
    # Deep copy slots so we don't mutate originals
    result_slots = [dict(s) for s in slots]

    if not detections or not result_slots:
        return result_slots, {
            "total_cars": len(detections),
            "mapped": 0,
            "unmapped_cars": len(detections),
            "occupied": 0,
            "empty": len(result_slots),
            "total_slots": len(result_slots),
        }

    # Sort by confidence descending
    sorted_cars = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    assigned_slot_ids = set()
    mapped_count = 0

    for car in sorted_cars:
        car_cx = car["cx"]
        car_cy = car["cy"]

        best_dist = float("inf")
        best_idx = -1

        for i, slot in enumerate(result_slots):
            if slot["id"] in assigned_slot_ids:
                continue
            dist = math.hypot(car_cx - slot["cx"], car_cy - slot["cy"])
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx >= 0:
            result_slots[best_idx]["status"] = "occupied"
            result_slots[best_idx]["confidence"] = car["confidence"]
            result_slots[best_idx]["car_bbox"] = {
                "x1": car["x1"], "y1": car["y1"],
                "x2": car["x2"], "y2": car["y2"],
            }
            assigned_slot_ids.add(result_slots[best_idx]["id"])
            mapped_count += 1

    occupied = sum(1 for s in result_slots if s["status"] == "occupied")
    empty = sum(1 for s in result_slots if s["status"] == "empty")

    stats = {
        "total_cars": len(detections),
        "mapped": mapped_count,
        "unmapped_cars": len(detections) - mapped_count,
        "occupied": occupied,
        "empty": empty,
        "total_slots": len(result_slots),
    }

    return result_slots, stats
