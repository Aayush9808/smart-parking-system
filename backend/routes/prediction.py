"""
Prediction Routes — train model, forecast, peak hours.
"""

from fastapi import APIRouter
from backend.database import get_history
from services.predictor import train, forecast, get_peak_hours

router = APIRouter(prefix="/api/prediction", tags=["Prediction"])


@router.post("/train")
async def train_model():
    """Train RandomForest on historical data."""
    history = get_history(limit=5000)
    if len(history) < 10:
        return {
            "success": False,
            "error": "Not enough data. Simulate history first.",
            "records_available": len(history),
        }

    result = train(history)
    return {"success": result.get("trained", False), **result}


@router.get("/forecast")
async def get_forecast(hours: int = 6):
    """Get occupancy forecast for next N hours."""
    hours = min(max(1, hours), 48)
    predictions = forecast(hours)
    if not predictions:
        return {"success": False, "error": "Model not trained yet"}
    return {"success": True, "predictions": predictions}


@router.get("/peak_hours")
async def peak_hours():
    """Get top 5 busiest hours."""
    peaks = get_peak_hours()
    if not peaks:
        return {"success": False, "error": "Model not trained yet"}
    return {"success": True, "peak_hours": peaks}
