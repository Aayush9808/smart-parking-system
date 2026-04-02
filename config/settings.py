"""
Global configuration — single source of truth for all system parameters.
"""

import os

# ─── Server ──────────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 8000

# ─── Paths ───────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
DB_PATH = os.path.join(DATA_DIR, "parking.db")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ─── ML: YOLOv8 ─────────────────────────────────────────────────
YOLO_MODEL = "yolov8s.pt"          # Small model — good accuracy/speed
YOLO_CONF = 0.30                   # Confidence threshold for cars
YOLO_IOU_NMS = 0.45                # NMS IoU threshold
YOLO_IMG_SIZE = 1280               # Inference resolution (full image)
YOLO_CAR_CLASS_ID = 2              # COCO class 2 = car

# ─── ML: SAHI Tiling ────────────────────────────────────────────
SAHI_TILE_SIZE = 640               # Tile dimensions
SAHI_OVERLAP_RATIO = 0.25          # 25% overlap between tiles
SAHI_MERGE_IOU = 0.45              # Merge threshold for tile results

# ─── ML: Detection Filtering ────────────────────────────────────
MIN_BOX_AREA_RATIO = 0.001        # Min 0.1% of image area
MAX_BOX_AREA_RATIO = 0.25          # Max 25% of image area
MAX_ASPECT_RATIO = 4.0             # Max width/height ratio

# ─── Slot Generation ────────────────────────────────────────────
SLOT_PAD_FACTOR = 1.20             # 20% gap around each slot
GRID_MARGIN = 0.15                 # 15% margin around bounding region
MIN_SLOTS = 4                      # Minimum slots to generate

# ─── Prediction: RandomForest ───────────────────────────────────
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
RF_RANDOM_STATE = 42
FORECAST_HOURS = 6
