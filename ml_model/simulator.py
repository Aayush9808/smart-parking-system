"""
Parking-data simulator.

* Generates realistic historical occupancy patterns (peak-hours, weekday/weekend).
* Draws a synthetic top-down parking-lot image with coloured car shapes so the
  demo works without a real camera.
"""

import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from ml_model.parking_config import (
    PARKING_SLOTS,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    COLOR_EMPTY,
    COLOR_OCCUPIED,
    COLOR_SLOT_LINE,
    COLOR_ASPHALT,
    COLOR_LANE,
    COLOR_LANE_MARK,
    CAR_COLORS,
)


# ── Occupancy probability model ──────────────────────────────────────────────

def _occupancy_probability(hour: int, day_of_week: int) -> float:
    """
    Return a realistic probability that any single slot is occupied,
    given the hour-of-day and day-of-week.

    Patterns modelled:
      • Office parking: peaks 9-11 AM and 2-4 PM
      • Low usage early-morning / late-night
      • Weekends about 50 % quieter
    """
    hourly = {
        0: 0.05, 1: 0.03, 2: 0.02, 3: 0.02, 4: 0.03, 5: 0.05,
        6: 0.15, 7: 0.35, 8: 0.55, 9: 0.75, 10: 0.85, 11: 0.80,
        12: 0.70, 13: 0.75, 14: 0.82, 15: 0.78, 16: 0.65, 17: 0.45,
        18: 0.30, 19: 0.20, 20: 0.15, 21: 0.10, 22: 0.08, 23: 0.06,
    }
    prob = hourly.get(hour, 0.30)
    if day_of_week >= 5:          # Saturday / Sunday
        prob *= 0.50
    prob += random.uniform(-0.10, 0.10)
    return max(0.0, min(1.0, prob))


# ── Historical data generator ────────────────────────────────────────────────

def generate_historical_data(
    days: int = 14,
    interval_minutes: int = 15,
) -> List[Dict]:
    """
    Produce `days` worth of snapshots at the given interval.

    Returns
    -------
    list[dict] – each item:
        {timestamp, slots: [{id, name, status}],
         total_occupied, total_empty, total_slots}
    """
    data: List[Dict] = []
    now = datetime.now()
    start = now - timedelta(days=days)
    current = start
    num_slots = len(PARKING_SLOTS)

    # Per-slot state for continuity (cars don't pop in/out every tick)
    slot_occupied = [False] * num_slots
    slot_since = [current] * num_slots

    while current <= now:
        hour = current.hour
        dow = current.weekday()
        prob = _occupancy_probability(hour, dow)

        slots_status: List[Dict] = []
        for i, slot in enumerate(PARKING_SLOTS):
            if slot_occupied[i]:
                time_parked_h = (current - slot_since[i]).total_seconds() / 3600
                leave_chance = min(0.30, time_parked_h * 0.05 + (1 - prob) * 0.15)
                if random.random() < leave_chance:
                    slot_occupied[i] = False
                    slot_since[i] = current
            else:
                arrive_chance = prob * 0.20
                if random.random() < arrive_chance:
                    slot_occupied[i] = True
                    slot_since[i] = current

            slots_status.append(
                {
                    "id": slot["id"],
                    "name": slot["name"],
                    "status": "occupied" if slot_occupied[i] else "empty",
                }
            )

        total_occ = sum(1 for s in slots_status if s["status"] == "occupied")
        data.append(
            {
                "timestamp": current.isoformat(),
                "slots": slots_status,
                "total_occupied": total_occ,
                "total_empty": num_slots - total_occ,
                "total_slots": num_slots,
            }
        )
        current += timedelta(minutes=interval_minutes)

    return data


# ── Synthetic parking-lot image generator ────────────────────────────────────

def generate_parking_image(
    slot_statuses: Optional[List[Dict]] = None,
    occupied_ids: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Draw a bird's-eye-view parking-lot image.

    If *slot_statuses* is given the status of each slot is taken from it;
    otherwise *occupied_ids* (list of slot IDs) marks those as occupied;
    if neither is given a random selection is used.
    """
    if not HAS_CV2:
        return np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)

    img = np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), COLOR_ASPHALT, dtype=np.uint8)

    # Driving lane
    lane_y1, lane_y2 = 195, 285
    cv2.rectangle(img, (0, lane_y1), (IMAGE_WIDTH, lane_y2), COLOR_LANE, -1)

    # Dashed centre line
    lane_mid = (lane_y1 + lane_y2) // 2
    for x in range(30, IMAGE_WIDTH - 30, 60):
        cv2.line(img, (x, lane_mid), (x + 30, lane_mid), COLOR_LANE_MARK, 2)

    # Arrows
    for x in [200, 500]:
        pts = np.array(
            [[x, lane_mid - 8], [x + 20, lane_mid], [x, lane_mid + 8]], np.int32
        )
        cv2.fillPoly(img, [pts], COLOR_LANE_MARK)

    # Determine which slots are occupied
    occupied_set: set = set()
    if slot_statuses:
        occupied_set = {s["id"] for s in slot_statuses if s["status"] == "occupied"}
    elif occupied_ids:
        occupied_set = set(occupied_ids)
    else:
        occupied_set = set(
            random.sample(
                [s["id"] for s in PARKING_SLOTS], random.randint(3, 9)
            )
        )

    # Seed for reproducible car colours within the same call
    rng = random.Random(42)

    for slot in PARKING_SLOTS:
        x1, y1, x2, y2 = slot["x1"], slot["y1"], slot["x2"], slot["y2"]
        is_occupied = slot["id"] in occupied_set

        # Slot outline
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_SLOT_LINE, 2)

        # Label position
        text_x = x1 + (x2 - x1) // 2 - 12
        text_y = y2 - 10 if slot["row"] == 1 else y1 + 20

        if is_occupied:
            # Draw a car shape
            car_color = rng.choice(CAR_COLORS)
            m = 12  # margin
            cx1, cy1, cx2, cy2 = x1 + m, y1 + m, x2 - m, y2 - m
            cv2.rectangle(img, (cx1, cy1), (cx2, cy2), car_color, -1)
            cv2.rectangle(img, (cx1, cy1), (cx2, cy2), (30, 30, 30), 2)
            ws_y1 = cy1 + 15 if slot["row"] == 1 else cy1 + (cy2 - cy1) - 40
            ws_y2 = ws_y1 + 25
            cv2.rectangle(img, (cx1 + 8, ws_y1), (cx2 - 8, ws_y2), (180, 220, 240), -1)
            # Red tint overlay
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_OCCUPIED, -1)
            cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
            cv2.putText(img, slot["name"], (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 200), 1)
        else:
            # Green tint + 'P'
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_EMPTY, -1)
            cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
            px = x1 + (x2 - x1) // 2 - 10
            py = y1 + (y2 - y1) // 2 + 8
            cv2.putText(img, "P", (px, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_EMPTY, 2)
            cv2.putText(img, slot["name"], (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)

    # Labels
    cv2.putText(img, "SMART PARKING LOT", (IMAGE_WIDTH // 2 - 130, IMAGE_HEIGHT - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(img, "ENTRY >>", (10, lane_mid + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_LANE_MARK, 1)
    cv2.putText(img, "<< EXIT", (IMAGE_WIDTH - 110, lane_mid + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_LANE_MARK, 1)

    return img
