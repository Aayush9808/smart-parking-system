"""
Analytics Engine — computes occupancy metrics from slot data.
"""


def compute_analytics(slots: list[dict], detections: list[dict]) -> dict:
    """
    Compute real-time parking analytics.
    All numbers derive directly from slot status — no estimation.
    """
    total = len(slots)
    occupied = sum(1 for s in slots if s["status"] == "occupied")
    empty = total - occupied
    rate = round(occupied / total, 3) if total > 0 else 0.0

    avg_conf = 0.0
    if occupied > 0:
        confs = [s.get("confidence", 0) for s in slots if s["status"] == "occupied"]
        avg_conf = round(sum(confs) / len(confs), 3)

    return {
        "total_slots": total,
        "occupied": occupied,
        "empty": empty,
        "occupancy_rate": rate,
        "occupancy_percent": round(rate * 100, 1),
        "cars_detected": len(detections),
        "avg_confidence": avg_conf,
    }
