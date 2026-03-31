"""
Parking-occupancy prediction using RandomForestRegressor.

Features are purely time-based (hour, day-of-week, cyclic encodings, peak flags)
so the model trains in < 2 s on simulated data and is easy to explain in a viva.
"""

import math
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class ParkingPredictor:
    """Train once on historical data, then predict occupancy for any timestamp."""

    def __init__(self):
        self.model: Optional[RandomForestRegressor] = None
        self.is_trained: bool = False
        self.train_score: float = 0.0
        self.test_score: float = 0.0
        self.feature_importances: Dict[str, float] = {}

    # ── Feature engineering ───────────────────────────────────────────────────
    @staticmethod
    def _extract_features(ts: datetime) -> List[float]:
        hour = ts.hour
        minute = ts.minute
        dow = ts.weekday()
        return [
            hour,
            minute,
            dow,
            1.0 if dow >= 5 else 0.0,                          # is_weekend
            math.sin(2 * math.pi * hour / 24),                 # cyclic hour sin
            math.cos(2 * math.pi * hour / 24),                 # cyclic hour cos
            1.0 if 8 <= hour <= 11 else 0.0,                   # morning peak
            1.0 if 14 <= hour <= 16 else 0.0,                  # afternoon peak
            1.0 if (hour >= 22 or hour <= 5) else 0.0,         # night
        ]

    _FEATURE_NAMES = [
        "hour", "minute", "day_of_week", "is_weekend",
        "hour_sin", "hour_cos", "is_morning_peak",
        "is_afternoon_peak", "is_night",
    ]

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, historical_data: List[Dict]) -> Dict:
        """
        Train on historical occupancy summaries.

        Each dict must contain:  timestamp (ISO), total_occupied, total_slots.
        """
        if not HAS_SKLEARN:
            return {"status": "error", "message": "scikit-learn is not installed"}

        X, y = [], []
        for rec in historical_data:
            ts = datetime.fromisoformat(rec["timestamp"])
            X.append(self._extract_features(ts))
            y.append(rec["total_occupied"] / max(rec["total_slots"], 1))

        X = np.array(X)
        y = np.array(y)

        if len(X) < 20:
            return {
                "status": "error",
                "message": f"Not enough data ({len(X)} points, need >= 20).",
            }

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        self.model.fit(X_train, y_train)

        self.train_score = round(self.model.score(X_train, y_train), 4)
        self.test_score = round(self.model.score(X_test, y_test), 4)
        self.is_trained = True

        self.feature_importances = {
            name: round(float(imp), 4)
            for name, imp in zip(self._FEATURE_NAMES, self.model.feature_importances_)
        }

        return {
            "status": "trained",
            "train_score": self.train_score,
            "test_score": self.test_score,
            "data_points": len(X),
            "feature_importances": self.feature_importances,
        }

    # ── Prediction helpers ────────────────────────────────────────────────────
    def predict_occupancy(self, ts: datetime) -> float:
        """Return predicted occupancy rate (0–1) at *ts*."""
        if not self.is_trained:
            return 0.5
        features = np.array([self._extract_features(ts)])
        pred = float(self.model.predict(features)[0])
        return max(0.0, min(1.0, pred))

    def predict_next_hours(
        self,
        hours: int = 6,
        interval_minutes: int = 15,
        total_slots: int = 12,
    ) -> List[Dict]:
        """Occupancy forecast for the next *hours*."""
        preds: List[Dict] = []
        now = datetime.now()
        for offset in range(0, hours * 60, interval_minutes):
            ts = now + timedelta(minutes=offset)
            rate = self.predict_occupancy(ts)
            occupied = round(rate * total_slots)
            preds.append(
                {
                    "timestamp": ts.isoformat(),
                    "hour": ts.strftime("%H:%M"),
                    "occupancy_rate": round(float(rate), 3),
                    "predicted_occupied": int(min(occupied, total_slots)),
                    "predicted_empty": int(max(total_slots - occupied, 0)),
                }
            )
        return preds

    def estimate_slot_free_time(
        self,
        slot_id: int,
        current_status: str,
        total_slots: int = 12,
    ) -> Optional[Dict]:
        """Estimate when an *occupied* slot might become free."""
        if current_status != "occupied" or not self.is_trained:
            return None

        now = datetime.now()
        current_rate = self.predict_occupancy(now)

        for minutes_ahead in range(5, 180, 5):
            future = now + timedelta(minutes=minutes_ahead)
            future_rate = self.predict_occupancy(future)
            if future_rate < current_rate - 0.08:
                rng = random.Random(slot_id + minutes_ahead)
                jitter = rng.randint(-3, 5)
                eta = max(5, minutes_ahead + jitter)
                return {
                    "slot_id": slot_id,
                    "estimated_free_in_minutes": eta,
                    "estimated_free_at": (now + timedelta(minutes=eta)).strftime("%H:%M"),
                    "confidence": round(max(0.4, 1.0 - minutes_ahead / 180), 2),
                }

        return {
            "slot_id": slot_id,
            "estimated_free_in_minutes": None,
            "estimated_free_at": None,
            "confidence": 0.0,
        }
