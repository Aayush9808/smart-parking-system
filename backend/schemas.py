"""
Pydantic response schemas — clean, typed API responses.
"""

from pydantic import BaseModel


class Detection(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_name: str
    cx: float
    cy: float


class Slot(BaseModel):
    id: int
    name: str
    cx: float
    cy: float
    w: float
    h: float
    row: int
    col: int
    status: str
    confidence: float = 0.0


class Analytics(BaseModel):
    total_slots: int
    occupied: int
    empty: int
    occupancy_rate: float
    occupancy_percent: float
    cars_detected: int
    avg_confidence: float


class DetectionResponse(BaseModel):
    success: bool
    detections: list[Detection]
    slots: list[Slot]
    analytics: Analytics
    diagnostics: dict
    grid_info: dict
    result_image: str  # base64


class HealthResponse(BaseModel):
    status: str
    version: str
