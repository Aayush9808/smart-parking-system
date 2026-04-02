"""
Historical Data Simulator — generates realistic parking occupancy data
for training the prediction model.
"""

import random
import math
from datetime import datetime, timedelta


def simulate_history(days: int = 14, slots_count: int = 20) -> list[dict]:
    """
    Generate `days` worth of hourly parking data with realistic patterns.

    Patterns modeled:
      - Morning rush (8-11 AM): 70-95% occupancy
      - Afternoon peak (12-3 PM): 60-85% occupancy
      - Evening decline (4-7 PM): 40-65% occupancy
      - Night (8 PM - 6 AM): 5-25% occupancy
      - Weekends: ~50% lower than weekdays

    Returns list of {timestamp, total_slots, occupied, empty, occupancy_rate}
    """
    records = []
    now = datetime.now()
    start = now - timedelta(days=days)

    current = start.replace(minute=0, second=0, microsecond=0)

    while current <= now:
        hour = current.hour
        is_weekend = current.weekday() >= 5

        # Base occupancy by time of day
        if 8 <= hour <= 11:
            base = random.uniform(0.70, 0.95)
        elif 12 <= hour <= 15:
            base = random.uniform(0.60, 0.85)
        elif 16 <= hour <= 19:
            base = random.uniform(0.40, 0.65)
        elif 6 <= hour <= 7:
            base = random.uniform(0.25, 0.45)
        else:
            base = random.uniform(0.05, 0.25)

        # Weekend reduction
        if is_weekend:
            base *= random.uniform(0.40, 0.60)

        # Add some noise
        base += random.uniform(-0.05, 0.05)
        base = max(0.0, min(1.0, base))

        occupied = round(base * slots_count)
        occupied = max(0, min(slots_count, occupied))
        empty = slots_count - occupied

        records.append({
            "timestamp": current.isoformat(),
            "total_slots": slots_count,
            "occupied": occupied,
            "empty": empty,
            "occupancy_rate": round(occupied / slots_count, 3),
        })

        current += timedelta(hours=1)

    return records
