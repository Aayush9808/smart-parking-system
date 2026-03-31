"""
Maps detected vehicle bounding-boxes to predefined parking-slot regions
using IoU and overlap ratio.
"""

from typing import Dict, List


def compute_iou(box1: Dict, box2: Dict) -> float:
    """Intersection-over-Union between two axis-aligned boxes."""
    x1 = max(box1["x1"], box2["x1"])
    y1 = max(box1["y1"], box2["y1"])
    x2 = min(box1["x2"], box2["x2"])
    y2 = min(box1["y2"], box2["y2"])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
    area2 = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])
    union = area1 + area2 - intersection
    return intersection / union if union else 0.0


def compute_overlap_ratio(detection: Dict, slot: Dict) -> float:
    """What fraction of the *slot* area is covered by the detection box."""
    x1 = max(detection["x1"], slot["x1"])
    y1 = max(detection["y1"], slot["y1"])
    x2 = min(detection["x2"], slot["x2"])
    y2 = min(detection["y2"], slot["y2"])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    slot_area = (slot["x2"] - slot["x1"]) * (slot["y2"] - slot["y1"])
    return intersection / slot_area if slot_area else 0.0


class SlotMapper:
    """Assign occupied / empty status to every parking slot."""

    def __init__(self, parking_slots: List[Dict]):
        self.slots = parking_slots

    def map_detections(
        self,
        detections: List[Dict],
        iou_threshold: float = 0.15,
        overlap_threshold: float = 0.30,
    ) -> List[Dict]:
        """
        Returns a list of slot-status dicts:
        [{id, name, status, confidence, iou, x1, y1, x2, y2}]
        """
        slot_statuses: List[Dict] = []

        for slot in self.slots:
            best_iou = 0.0
            best_overlap = 0.0
            best_conf = 0.0

            for det in detections:
                iou = compute_iou(det, slot)
                overlap = compute_overlap_ratio(det, slot)
                if iou > best_iou:
                    best_iou = iou
                    best_conf = det.get("confidence", 1.0)
                if overlap > best_overlap:
                    best_overlap = overlap

            is_occupied = best_iou >= iou_threshold or best_overlap >= overlap_threshold

            slot_statuses.append(
                {
                    "id": slot["id"],
                    "name": slot["name"],
                    "status": "occupied" if is_occupied else "empty",
                    "confidence": round(best_conf if is_occupied else 1.0 - best_iou, 2),
                    "iou": round(best_iou, 3),
                    "x1": slot["x1"],
                    "y1": slot["y1"],
                    "x2": slot["x2"],
                    "y2": slot["y2"],
                }
            )

        return slot_statuses
