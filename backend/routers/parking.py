"""
Parking status & history endpoints.
"""

from fastapi import APIRouter

from backend.database import (
    get_heatmap_data,
    get_latest_snapshot,
    get_occupancy_history,
    store_bulk_data,
)
from ml_model.simulator import generate_historical_data

router = APIRouter()


@router.get("/parking/status")
async def parking_status():
    snapshot = get_latest_snapshot()
    if not snapshot:
        return {
            "status": "no_data",
            "message": "No data yet. Run detection or simulation first.",
            "slots": [], "total_occupied": 0, "total_empty": 0, "total_slots": 0,
        }
    total_occ = sum(1 for s in snapshot if s["status"] == "occupied")
    return {
        "status": "ok",
        "slots": snapshot,
        "total_occupied": total_occ,
        "total_empty": len(snapshot) - total_occ,
        "total_slots": len(snapshot),
    }


@router.get("/parking/history")
async def parking_history(hours: int = 24):
    data = get_occupancy_history(hours)
    return {"status": "ok", "data": data, "count": len(data)}


@router.get("/parking/heatmap")
async def parking_heatmap():
    raw = get_heatmap_data()
    grid = [[0.0] * 24 for _ in range(7)]
    for row in raw:
        d, h = row["day_of_week"], row["hour"]
        if 0 <= d < 7 and 0 <= h < 24:
            grid[d][h] = round(row["avg_rate"], 3)
    days = ["Sunday", "Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday"]
    return {"status": "ok", "heatmap": grid, "days": days, "hours": list(range(24))}


@router.post("/parking/simulate")
async def run_simulation(days: int = 14):
    days = min(days, 30)
    data = generate_historical_data(days=days)
    store_bulk_data(data)
    return {
        "status": "ok",
        "message": f"Generated {len(data)} snapshots over {days} days",
        "data_points": len(data),
    }
