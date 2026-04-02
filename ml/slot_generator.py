"""
Slot Generator — builds a dynamic parking grid from detected cars.

Algorithm:
  1. If cars detected → compute median car width/height
  2. Slot size = median × padding factor
  3. Bounding region = convex hull of all car centers + margin
  4. Fill region with rows × cols grid
  5. Each slot gets: id, name (A1, B2...), center (x, y), dimensions

If 0 cars detected → returns empty (no phantom slots).
"""

import numpy as np
from ml.config import SLOT_PAD_FACTOR, GRID_MARGIN, MIN_SLOTS


def generate_slots(
    detections: list[dict],
    image_width: int,
    image_height: int,
) -> tuple[list[dict], dict]:
    """
    Generate a slot grid from car detections.

    Returns:
        (slots, grid_info)
        - slots: list of {id, name, cx, cy, w, h, row, col, status}
        - grid_info: {rows, cols, slot_w, slot_h, total}
    """
    if not detections:
        return [], {"rows": 0, "cols": 0, "slot_w": 0, "slot_h": 0, "total": 0}

    # Step 1: Compute median car size
    widths = [d["x2"] - d["x1"] for d in detections]
    heights = [d["y2"] - d["y1"] for d in detections]
    med_w = float(np.median(widths))
    med_h = float(np.median(heights))

    # Step 2: Slot dimensions (car + padding)
    slot_w = med_w * SLOT_PAD_FACTOR
    slot_h = med_h * SLOT_PAD_FACTOR

    # Step 3: Bounding region of all car centers
    cxs = [d["cx"] for d in detections]
    cys = [d["cy"] for d in detections]

    margin_x = image_width * GRID_MARGIN
    margin_y = image_height * GRID_MARGIN

    region_x1 = max(0, min(cxs) - margin_x)
    region_y1 = max(0, min(cys) - margin_y)
    region_x2 = min(image_width, max(cxs) + margin_x)
    region_y2 = min(image_height, max(cys) + margin_y)

    region_w = region_x2 - region_x1
    region_h = region_y2 - region_y1

    # Step 4: Grid dimensions
    cols = max(2, int(region_w / slot_w))
    rows = max(2, int(region_h / slot_h))

    # Ensure we have at least MIN_SLOTS
    while rows * cols < MIN_SLOTS:
        if cols <= rows:
            cols += 1
        else:
            rows += 1

    # Step 5: Build slot grid
    slots = []
    slot_id = 0
    row_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for r in range(rows):
        for c in range(cols):
            cx = region_x1 + (c + 0.5) * (region_w / cols)
            cy = region_y1 + (r + 0.5) * (region_h / rows)
            label = row_labels[r % 26] if r < 26 else f"R{r}"
            name = f"{label}{c + 1}"

            slots.append({
                "id": slot_id,
                "name": name,
                "cx": round(cx, 1),
                "cy": round(cy, 1),
                "w": round(slot_w, 1),
                "h": round(slot_h, 1),
                "row": r,
                "col": c,
                "status": "empty",
            })
            slot_id += 1

    grid_info = {
        "rows": rows,
        "cols": cols,
        "slot_w": round(slot_w, 1),
        "slot_h": round(slot_h, 1),
        "total": len(slots),
    }

    return slots, grid_info
