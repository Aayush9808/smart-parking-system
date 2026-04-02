# ParkSense AI — Technical Documentation

## System Architecture

```
┌────────────────────────────────────────────────────────────┐
│                     FRONTEND (Dashboard)                    │
│           HTML + CSS (Dark Theme) + Chart.js + JS          │
└──────────────────────────┬─────────────────────────────────┘
                           │ HTTP REST
┌──────────────────────────▼─────────────────────────────────┐
│                     FastAPI BACKEND                         │
│                                                            │
│  /api/detect      → Detection pipeline                     │
│  /api/parking/*   → Status, history, heatmap, simulate     │
│  /api/prediction/* → Train, forecast, peak hours           │
└──────┬──────────────────┬──────────────────┬───────────────┘
       │                  │                  │
┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐
│  ML CORE    │    │  SERVICES   │    │  DATABASE   │
│             │    │             │    │             │
│ detector    │    │ analytics   │    │ SQLite      │
│ slot_gen    │    │ predictor   │    │ occupancy   │
│ mapper      │    │ simulator   │    │ snapshots   │
└─────────────┘    └─────────────┘    └─────────────┘
```

---

## ML Pipeline — Detection

### Step-by-Step Processing

```
📸 Input Image (any resolution)
    │
    ├──→ STEP 1: YOLOv8s Full-Image Pass
    │    • Resolution: 1280px
    │    • Classes: [2] (car only)
    │    • Confidence: ≥ 0.30
    │    • Output: raw bounding boxes
    │
    ├──→ STEP 2: SAHI Tiled Pass
    │    • Tile size: 640×640px
    │    • Overlap: 25%
    │    • Each tile → YOLOv8 inference
    │    • Coordinates shifted back to full-image space
    │    • Purpose: catch small/distant cars
    │
    ├──→ STEP 3: NMS Merge
    │    • Combine full-pass + tile-pass detections
    │    • IoU threshold: 0.45
    │    • Remove duplicate detections
    │
    └──→ STEP 4: Geometric Filtering
         • Min area: 0.1% of image
         • Max area: 25% of image
         • Max aspect ratio: 4:1
         • Removes noise (poles, signs, shadows)
```

### Why YOLOv8s?

| Decision | Reasoning |
|----------|-----------|
| **YOLOv8 over v5/v7** | Latest architecture, better accuracy, native Python API |
| **Small (s) variant** | 11.2M params — fast on CPU, accurate enough for cars |
| **Pretrained COCO** | Already knows "car" class — no fine-tuning needed |
| **Car-only filtering** | Single class = higher precision, fewer false positives |
| **Conf 0.30** | Balanced: catches most cars, rejects low-quality detections |

### When Would Fine-Tuning Be Needed?

The pretrained model may fail on:
- **Extreme aerial views** (cars appear as dots < 20px)
- **Night/infrared images** (COCO is mostly daytime)
- **Occluded cars** (partially hidden behind pillars)

**If fine-tuning is needed:**
1. Use **PKLot dataset** (695K annotated parking images) or **CNRPark** dataset
2. Fine-tune YOLOv8s for 50-100 epochs on parking-specific data
3. Command: `yolo train data=pklot.yaml model=yolov8s.pt epochs=50 imgsz=640`
4. Expected improvement: 15-25% mAP increase on parking-specific scenarios

---

## Slot Generation Algorithm

```
Input: List of detected car bounding boxes
Output: Grid of named parking slots

1. Compute median car width & height
2. Slot dimensions = median × 1.20 (padding factor)
3. Bounding region = min/max of all car centers + 15% margin
4. Grid: rows = region_h / slot_h, cols = region_w / slot_w
5. Minimum 4 slots (2×2 grid)
6. Each slot gets:
   - ID (sequential integer)
   - Name (row letter + column number: A1, B3, etc.)
   - Center coordinates (cx, cy)
   - Dimensions (w, h)
   - Status: "empty" (default)
```

### Why Dynamic Over Hardcoded?

| Hardcoded Slots | Dynamic Grid |
|----------------|-------------|
| Works only for one specific lot | Works for any parking image |
| Requires manual configuration | Adapts automatically |
| Breaks on different camera angles | Robust to viewpoint changes |
| Fixed capacity | Capacity scales with detection |

---

## Vehicle → Slot Mapping

```
Algorithm: Greedy Center-Distance Assignment

1. Sort cars by confidence (highest first)
2. For each car:
   a. Compute Euclidean distance from car center to every unassigned slot center
   b. Assign car to nearest available slot
   c. Mark slot as "occupied"
3. Remaining slots stay "empty"

Guarantees:
  ✓ 1:1 mapping (one car per slot, one slot per car)
  ✓ No duplicates
  ✓ occupied + empty = total (always)
  ✓ Deterministic (same input → same output)
```

### Why Greedy Over Hungarian?

- **Greedy (O(n²))**: Simple, fast, good enough when slots >> cars
- **Hungarian (O(n³))**: Optimal assignment, but overkill for this use case
- The confidence-first ordering ensures high-quality detections get first pick

---

## Prediction Model

### Feature Engineering (9 Features)

| # | Feature | Type | Why |
|---|---------|------|-----|
| 1 | `hour` | Numeric | Usage varies dramatically by hour |
| 2 | `minute` | Numeric | Fine-grained timing |
| 3 | `day_of_week` | Numeric | Weekday vs weekend patterns |
| 4 | `is_weekend` | Binary | Weekends have ~50% less traffic |
| 5 | `hour_sin` | Cyclic | sin(2π·h/24) — 11PM is close to 1AM |
| 6 | `hour_cos` | Cyclic | cos(2π·h/24) — captures circular time |
| 7 | `is_morning` | Binary | 8-11 AM rush flag |
| 8 | `is_afternoon` | Binary | 12-4 PM peak flag |
| 9 | `is_night` | Binary | 10PM-5AM low-usage flag |

### Why RandomForest?

| Decision | Reasoning |
|----------|-----------|
| **RF over Linear Regression** | Captures non-linear time patterns |
| **RF over Neural Network** | Works great on small datasets (~337 points) |
| **100 trees, max_depth=10** | Enough complexity, prevents overfitting |
| **R² ≈ 0.97** | Strong fit on simulated data |
| **Training time < 1 sec** | Real-time retraining possible |

---

## API Design

All endpoints return clean JSON. No mixed formats.

### Detection Response Schema

```json
{
  "success": true,
  "detections": [
    {"x1": 120, "y1": 85, "x2": 230, "y2": 195, "confidence": 0.89, "class_name": "car", "cx": 175.0, "cy": 140.0}
  ],
  "slots": [
    {"id": 0, "name": "A1", "cx": 160.0, "cy": 140.0, "w": 132.0, "h": 132.0, "row": 0, "col": 0, "status": "occupied", "confidence": 0.89}
  ],
  "analytics": {
    "total_slots": 15,
    "occupied": 6,
    "empty": 9,
    "occupancy_rate": 0.4,
    "occupancy_percent": 40.0,
    "cars_detected": 6,
    "avg_confidence": 0.82
  },
  "diagnostics": {
    "image_size": "800x600",
    "full_pass_raw": 4,
    "sahi_tiles_raw": 3,
    "after_nms": 6,
    "after_filter": 6,
    "cars_detected": 6
  },
  "grid_info": {"rows": 3, "cols": 5, "slot_w": 132.0, "slot_h": 132.0, "total": 15},
  "result_image": "<base64 JPEG>"
}
```

---

## Design Decisions

| Decision | Alternative Considered | Why This Choice |
|----------|----------------------|-----------------|
| Car-only detection | Multi-class (car, bus, truck) | Higher precision, simpler pipeline, avoid false positives |
| SAHI tiling | Single-pass only | Catches 2-3× more cars in large/aerial images |
| Dynamic grid | Hardcoded slot coordinates | Works on any parking lot without configuration |
| SQLite | PostgreSQL | Zero-config, perfect for prototyping |
| FastAPI | Flask/Django | Auto-docs, async support, Pydantic validation |
| Greedy mapping | Hungarian algorithm | Simpler, fast, good enough for this scale |
| Chart.js | D3.js/Recharts | Lightweight CDN, no build step needed |

---

## Configuration Reference

All tunable parameters in `config/settings.py`:

```python
# Detection
YOLO_MODEL = "yolov8s.pt"     # Model variant
YOLO_CONF = 0.30              # Min confidence
YOLO_IMG_SIZE = 1280          # Inference resolution
YOLO_CAR_CLASS_ID = 2         # COCO car class

# SAHI
SAHI_TILE_SIZE = 640          # Tile dimensions
SAHI_OVERLAP_RATIO = 0.25     # Overlap between tiles

# Filtering
MIN_BOX_AREA_RATIO = 0.001   # Min 0.1% of image
MAX_BOX_AREA_RATIO = 0.25    # Max 25% of image
MAX_ASPECT_RATIO = 4.0        # Max width/height

# Slots
SLOT_PAD_FACTOR = 1.20       # Car size × 1.2
GRID_MARGIN = 0.15           # 15% margin around region

# Prediction
RF_N_ESTIMATORS = 100        # Number of trees
RF_MAX_DEPTH = 10            # Max tree depth
```
