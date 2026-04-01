"""
Virtual parking-slot grid generator.

Given vehicle detections, this module creates a regular grid of parking
slots that covers the detected parking area.  Each slot has real (x, y)
coordinates and a unique row-col name (A1, A2, … B1, B2, …).

Pipeline
--------
1. Compute median vehicle width / height from detections.
2. Find the bounding rectangle of all detections (+ padding).
3. Lay a regular grid of slot-sized cells over that rectangle.
4. Return the list of slot dicts with coordinates, suitable for
   mapping and for drawing on the annotated image.

When no vehicles are detected the module returns an empty grid and a
note explaining why.
"""

import math
import string
from typing import Dict, List, Tuple


# ── Row labels: A–Z, then AA–AZ, BA–BZ … (enough for any realistic lot) ────
def _row_label(index: int) -> str:
    """0→A, 1→B, … 25→Z, 26→AA, 27→AB, …"""
    if index < 26:
        return string.ascii_uppercase[index]
    return string.ascii_uppercase[index // 26 - 1] + string.ascii_uppercase[index % 26]


def generate_slot_grid(
    detections: List[Dict],
    image_shape: Tuple[int, int],   # (h, w)
    pad_factor: float = 0.30,       # expand bbox by 30 % on each side
    gap_ratio: float = 0.20,        # 20 % gap between slots (realistic spacing)
) -> Tuple[List[Dict], Dict]:
    """
    Build a virtual parking-slot grid from detections.

    Returns
    -------
    slots : list[dict]
        Each slot has keys: id, name, x1, y1, x2, y2, status ("unknown"
        at this stage — the mapper will fill in occupied/empty).
    grid_info : dict
        Metadata: rows, cols, slot_w, slot_h, region bounds, vehicle count,
        confidence_note.
    """
    img_h, img_w = image_shape
    n_vehicles = len(detections)

    # ── No detections → return empty grid ─────────────────────────────────
    if n_vehicles == 0:
        return [], {
            "rows": 0, "cols": 0,
            "slot_w": 0, "slot_h": 0,
            "region": (0, 0, img_w, img_h),
            "total_slots": 0,
            "total_vehicles": 0,
            "confidence_note": "No vehicles detected — upload a clearer image or try a different angle.",
        }

    # ── Compute median vehicle size ───────────────────────────────────────
    widths = [d["x2"] - d["x1"] for d in detections]
    heights = [d["y2"] - d["y1"] for d in detections]
    med_w = float(sorted(widths)[len(widths) // 2])
    med_h = float(sorted(heights)[len(heights) // 2])

    # Slot cell = vehicle size + gap
    slot_w = med_w * (1 + gap_ratio)
    slot_h = med_h * (1 + gap_ratio)

    # Clamp to sane minimums (at least 20 px)
    slot_w = max(slot_w, 20)
    slot_h = max(slot_h, 20)

    # ── Parking region (bounding rect of all detections + padding) ────────
    all_x1 = min(d["x1"] for d in detections)
    all_y1 = min(d["y1"] for d in detections)
    all_x2 = max(d["x2"] for d in detections)
    all_y2 = max(d["y2"] for d in detections)

    region_w = all_x2 - all_x1
    region_h = all_y2 - all_y1

    pad_x = region_w * pad_factor
    pad_y = region_h * pad_factor

    rx1 = max(0, int(all_x1 - pad_x))
    ry1 = max(0, int(all_y1 - pad_y))
    rx2 = min(img_w, int(all_x2 + pad_x))
    ry2 = min(img_h, int(all_y2 + pad_y))

    grid_w = rx2 - rx1
    grid_h = ry2 - ry1

    # ── Grid dimensions ───────────────────────────────────────────────────
    cols = max(1, int(round(grid_w / slot_w)))
    rows = max(1, int(round(grid_h / slot_h)))

    # Ensure at least as many slots as vehicles
    while rows * cols < n_vehicles:
        if cols <= rows:
            cols += 1
        else:
            rows += 1

    # Recompute cell size to fill region evenly
    cell_w = grid_w / cols
    cell_h = grid_h / rows

    # ── Build slot list ───────────────────────────────────────────────────
    slots: List[Dict] = []
    slot_id = 0
    for r in range(rows):
        row_label = _row_label(r)
        for c in range(cols):
            slot_id += 1
            sx1 = rx1 + int(c * cell_w)
            sy1 = ry1 + int(r * cell_h)
            sx2 = rx1 + int((c + 1) * cell_w)
            sy2 = ry1 + int((r + 1) * cell_h)
            slots.append({
                "id": slot_id,
                "name": f"{row_label}{c + 1}",
                "x1": sx1, "y1": sy1,
                "x2": sx2, "y2": sy2,
                "status": "empty",          # default; mapper overrides
                "confidence": 0.0,
                "class_name": "",
            })

    # ── Confidence note ───────────────────────────────────────────────────
    if n_vehicles >= 5:
        note = f"Grid: {rows}×{cols} slots derived from {n_vehicles} detected vehicles."
    elif n_vehicles >= 2:
        note = f"Approximate grid ({rows}×{cols}) — few vehicles detected."
    else:
        note = f"Minimal grid ({rows}×{cols}) — only 1 vehicle detected; capacity uncertain."

    grid_info = {
        "rows": rows,
        "cols": cols,
        "slot_w": round(cell_w),
        "slot_h": round(cell_h),
        "region": (rx1, ry1, rx2, ry2),
        "total_slots": len(slots),
        "total_vehicles": n_vehicles,
        "confidence_note": note,
    }

    return slots, grid_info
