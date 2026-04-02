"""
Parking Routes — status, history, heatmap, simulation.
"""

from fastapi import APIRouter
from backend.database import (
    get_history, get_heatmap_data, get_record_count,
    save_occupancy_batch, clear_history,
)
from services.simulator import simulate_history

router = APIRouter(prefix="/api/parking", tags=["Parking"])


@router.get("/status")
async def parking_status():
    """Get the latest parking status from history."""
    history = get_history(limit=1)
    if not history:
        return {
            "has_data": False,
            "total_slots": 0,
            "occupied": 0,
            "empty": 0,
            "occupancy_rate": 0,
            "last_updated": None,
        }

    latest = history[-1]
    return {
        "has_data": True,
        "total_slots": latest["total_slots"],
        "occupied": latest["occupied"],
        "empty": latest["empty"],
        "occupancy_rate": latest["occupancy_rate"],
        "last_updated": latest["timestamp"],
    }


@router.get("/history")
async def parking_history():
    """Get occupancy history for charts."""
    data = get_history(limit=336)
    return {"records": data, "count": len(data)}


@router.get("/heatmap")
async def parking_heatmap():
    """Get 7×24 occupancy heatmap data."""
    data = get_heatmap_data()

    # Build 7×24 matrix (default 0)
    matrix = [[0.0] * 24 for _ in range(7)]
    for row in data:
        d = row["day"]
        h = row["hour"]
        if 0 <= d < 7 and 0 <= h < 24:
            matrix[d][h] = row["avg_rate"]

    return {
        "matrix": matrix,
        "days": ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
        "hours": list(range(24)),
    }


@router.post("/simulate")
async def simulate_data():
    """Generate 14 days of simulated parking history."""
    clear_history()
    records = simulate_history(days=14, slots_count=20)
    save_occupancy_batch(records)
    return {
        "success": True,
        "records_generated": len(records),
        "days": 14,
    }
