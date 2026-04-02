"""
ML Configuration — all detection hyperparameters in one place.
Imported by detector, slot_generator, and mapper.
"""

from config.settings import (
    YOLO_MODEL, YOLO_CONF, YOLO_IOU_NMS, YOLO_IMG_SIZE,
    YOLO_CAR_CLASS_ID,
    SAHI_TILE_SIZE, SAHI_OVERLAP_RATIO, SAHI_MERGE_IOU,
    MIN_BOX_AREA_RATIO, MAX_BOX_AREA_RATIO, MAX_ASPECT_RATIO,
    SLOT_PAD_FACTOR, GRID_MARGIN, MIN_SLOTS,
)
