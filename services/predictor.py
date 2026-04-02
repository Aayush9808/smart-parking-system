"""
Occupancy Predictor — RandomForest model for future parking occupancy.

Features (9 total):
  1. hour          — hour of day (0-23)
  2. minute        — minute (0-59)
  3. day_of_week   — 0=Mon, 6=Sun
  4. is_weekend    — 0 or 1
  5. hour_sin      — cyclic encoding sin(2π·hour/24)
  6. hour_cos      — cyclic encoding cos(2π·hour/24)
  7. is_morning    — 1 if 8-11 AM
  8. is_afternoon  — 1 if 12-16
  9. is_night      — 1 if 22-5 AM

Target: occupancy_rate (0.0 → 1.0)
"""

import math
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from config.settings import RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_RANDOM_STATE, FORECAST_HOURS

_model: RandomForestRegressor | None = None
_trained = False


def _extract_features(dt: datetime) -> list[float]:
    """Extract 9 features from a datetime."""
    h = dt.hour
    return [
        h,
        dt.minute,
        dt.weekday(),
        1.0 if dt.weekday() >= 5 else 0.0,
        math.sin(2 * math.pi * h / 24),
        math.cos(2 * math.pi * h / 24),
        1.0 if 8 <= h <= 11 else 0.0,
        1.0 if 12 <= h <= 16 else 0.0,
        1.0 if h >= 22 or h <= 5 else 0.0,
    ]


def train(history: list[dict]) -> dict:
    """
    Train the RandomForest on historical data.
    Returns training metrics.
    """
    global _model, _trained

    if len(history) < 10:
        return {"error": "Need at least 10 data points", "trained": False}

    X = []
    y = []
    for rec in history:
        dt = datetime.fromisoformat(rec["timestamp"])
        X.append(_extract_features(dt))
        y.append(rec["occupancy_rate"])

    X = np.array(X)
    y = np.array(y)

    _model = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RF_RANDOM_STATE,
    )
    _model.fit(X, y)
    _trained = True

    # Training score
    train_score = round(_model.score(X, y), 4)

    # Feature importance
    feature_names = [
        "hour", "minute", "day_of_week", "is_weekend",
        "hour_sin", "hour_cos", "is_morning", "is_afternoon", "is_night"
    ]
    importances = {
        name: round(float(imp), 4)
        for name, imp in zip(feature_names, _model.feature_importances_)
    }

    return {
        "trained": True,
        "samples": len(y),
        "r2_score": train_score,
        "features": 9,
        "feature_importance": importances,
    }


def forecast(hours: int | None = None) -> list[dict]:
    """Predict occupancy for the next N hours."""
    if not _trained or _model is None:
        return []

    if hours is None:
        hours = FORECAST_HOURS

    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    predictions = []

    for i in range(1, hours + 1):
        dt = now + timedelta(hours=i)
        features = np.array([_extract_features(dt)])
        pred = float(np.clip(_model.predict(features)[0], 0.0, 1.0))
        predictions.append({
            "timestamp": dt.isoformat(),
            "hour": dt.hour,
            "predicted_occupancy": round(pred, 3),
            "predicted_percent": round(pred * 100, 1),
        })

    return predictions


def get_peak_hours() -> list[dict]:
    """Identify the busiest hours based on model predictions."""
    if not _trained or _model is None:
        return []

    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    hourly = []

    for h in range(24):
        dt = now.replace(hour=h)
        features = np.array([_extract_features(dt)])
        pred = float(np.clip(_model.predict(features)[0], 0.0, 1.0))
        hourly.append({
            "hour": h,
            "predicted_occupancy": round(pred, 3),
            "label": f"{h:02d}:00",
        })

    hourly.sort(key=lambda x: x["predicted_occupancy"], reverse=True)
    return hourly[:5]  # Top 5 peak hours
