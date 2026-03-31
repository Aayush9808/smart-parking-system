"""
Prediction endpoints — train model, forecast occupancy, estimate slot ETAs.
"""

from datetime import datetime
from fastapi import APIRouter

from backend.database import get_latest_snapshot, get_summary_for_training
from ml_model.parking_config import PARKING_SLOTS
from ml_model.predictor import ParkingPredictor

router = APIRouter()

# Shared instance (lives as long as the server process)
predictor = ParkingPredictor()


@router.post("/prediction/train")
async def train_model():
    data = get_summary_for_training()
    if len(data) < 20:
        return {
            "status": "error",
            "message": f"Only {len(data)} data-points — run simulation first (need >= 20).",
        }
    return predictor.train(data)


@router.get("/prediction/forecast")
async def forecast(hours: int = 6):
    if not predictor.is_trained:
        return {"status": "not_trained",
                "message": "Train the model first.", "predictions": []}
    preds = predictor.predict_next_hours(
        hours=min(hours, 24), total_slots=len(PARKING_SLOTS)
    )
    return {"status": "ok", "predictions": preds,
            "model_accuracy": predictor.test_score}


@router.get("/prediction/slots")
async def slot_predictions():
    if not predictor.is_trained:
        return {"status": "not_trained", "slot_predictions": []}
    snapshot = get_latest_snapshot()
    if not snapshot:
        return {"status": "no_data", "slot_predictions": []}

    results = []
    for s in snapshot:
        est = predictor.estimate_slot_free_time(
            slot_id=s["slot_id"],
            current_status=s["status"],
            total_slots=len(PARKING_SLOTS),
        )
        results.append({
            "slot_id": s["slot_id"],
            "slot_name": s["slot_name"],
            "current_status": s["status"],
            "prediction": est,
        })
    return {"status": "ok", "slot_predictions": results}


@router.get("/prediction/peak_hours")
async def peak_hours():
    if not predictor.is_trained:
        return {"status": "not_trained", "peak_hours": []}
    now = datetime.now()
    hourly = []
    for h in range(24):
        ts = now.replace(hour=h, minute=0, second=0, microsecond=0)
        rate = float(predictor.predict_occupancy(ts))
        hourly.append({
            "hour": h,
            "hour_label": f"{h:02d}:00",
            "predicted_rate": round(rate, 3),
            "is_peak": bool(rate > 0.7),
        })
    peak = [h for h in hourly if h["is_peak"]]
    busiest = max(hourly, key=lambda x: x["predicted_rate"])
    return {
        "status": "ok",
        "hourly": hourly,
        "peak_hours": peak,
        "busiest_hour": {
            "hour": busiest["hour"],
            "hour_label": busiest["hour_label"],
            "predicted_rate": float(busiest["predicted_rate"]),
            "is_peak": bool(busiest["is_peak"]),
        },
    }
