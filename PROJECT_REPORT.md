# ParkSense AI — Project Report

> **Subject:** Machine Learning Techniques (MLT) — Major Project
> **Project Title:** AI-Based Smart Parking System with Slot Detection and Prediction
> **Technology Stack:** Python, YOLOv8, scikit-learn, FastAPI, OpenCV, SQLite, HTML/CSS/JS

---

## 1. Introduction

### 1.1 Background
Urban parking is a significant problem — studies show that up to 30% of urban traffic consists of drivers searching for parking. This wastes fuel, increases emissions, and causes congestion. Existing parking systems either rely on expensive IoT sensors per slot or manual monitoring.

### 1.2 Objective
To build an AI-powered smart parking system that:
- Automatically detects vehicles in parking lot images using deep learning
- Generates a virtual parking grid and determines which slots are occupied/empty
- Predicts future occupancy using machine learning on historical patterns
- Presents all analytics through a real-time web dashboard

### 1.3 Scope
This project demonstrates two core ML techniques:
1. **Object Detection** (Deep Learning) — YOLOv8 convolutional neural network
2. **Regression Prediction** (Classical ML) — RandomForest ensemble method

---

## 2. Literature Review

### 2.1 Object Detection Methods
| Method | Type | Speed | Accuracy | Our Choice |
|--------|------|-------|----------|------------|
| Faster R-CNN | Two-stage | Slow | High | ✗ |
| SSD | Single-stage | Fast | Medium | ✗ |
| YOLOv5 | Single-stage | Fast | High | ✗ |
| **YOLOv8** | **Single-stage** | **Very Fast** | **Very High** | **✓** |

**Why YOLOv8:** Latest in the YOLO family, single-pass architecture for real-time inference, pre-trained on COCO dataset (80 classes including vehicles), easy to deploy via the `ultralytics` Python package.

### 2.2 Prediction Methods
| Method | Interpretability | Training Speed | Accuracy |
|--------|-----------------|----------------|----------|
| Linear Regression | High | Fast | Low |
| Decision Tree | High | Fast | Medium |
| **Random Forest** | **Medium** | **Fast** | **High** |
| Neural Network | Low | Slow | High |
| XGBoost | Low | Medium | Very High |

**Why RandomForest:** Ensemble of 100 decision trees. Handles non-linear patterns (parking usage is cyclic). Provides feature importance scores. Trains in under 2 seconds. R² ≈ 0.72 on test data.

### 2.3 SAHI (Slicing Aided Hyper Inference)
Standard object detectors struggle with small objects in large images. SAHI is a technique that:
1. Slices the input image into overlapping tiles
2. Runs the detector on each tile at higher effective resolution
3. Merges detections across tiles using Non-Maximum Suppression (NMS)

This is critical for aerial/CCTV parking lot images where vehicles appear small.

---

## 3. System Design

### 3.1 Architecture Diagram

```
┌──────────┐     ┌──────────────┐     ┌──────────────┐     ┌───────────┐
│  Image   │────▶│  YOLOv8s +   │────▶│  Virtual     │────▶│  Vehicle  │
│  Input   │     │  SAHI Tiling  │     │  Grid Gen    │     │  Mapper   │
└──────────┘     └──────────────┘     └──────────────┘     └─────┬─────┘
                                                                  │
                                                                  ▼
┌──────────┐     ┌──────────────┐     ┌──────────────┐     ┌───────────┐
│Dashboard │◀────│  FastAPI      │◀────│  SQLite DB   │◀────│  Analytics│
│  (Web)   │     │  Backend      │     │  Storage     │     │  Engine   │
└──────────┘     └──────────────┘     └──────────────┘     └───────────┘
                        │
                        ▼
                 ┌──────────────┐
                 │ RandomForest │
                 │ Predictor    │
                 └──────────────┘
```

### 3.2 Module Description

| Module | File | Responsibility |
|--------|------|---------------|
| Vehicle Detector | `ml_model/detector.py` | YOLOv8s inference + SAHI tiling + NMS merging + area/aspect filtering |
| Slot Grid Generator | `ml_model/slot_estimator.py` | Creates virtual parking grid from detected vehicle sizes |
| Slot Mapper | `ml_model/slot_mapper.py` | Assigns each vehicle to nearest grid slot (center-distance, 1:1) |
| Predictor | `ml_model/predictor.py` | RandomForest training + inference for occupancy prediction |
| Simulator | `ml_model/simulator.py` | Generates realistic historical parking data for training |
| Configuration | `ml_model/parking_config.py` | Slot layout for synthetic parking scenes |
| Detection API | `backend/routers/detection.py` | Image upload endpoint + full detection pipeline |
| Parking API | `backend/routers/parking.py` | Status, history, heatmap endpoints |
| Prediction API | `backend/routers/prediction.py` | Training, forecast, peak hours endpoints |
| Database | `backend/database.py` | SQLite operations for snapshots and summaries |
| Dashboard | `frontend/index.html` | Web UI with stats, charts, slot grid |

### 3.3 Data Flow (Detection Pipeline)

```
1. User uploads parking lot image
2. YOLOv8s runs on full image at 1280px resolution
3. SAHI splits image into 640px tiles (25% overlap), runs YOLO on each
4. All detections merged, NMS removes duplicates (IoU > 0.45)
5. Area filter removes boxes < 0.05% or > 25% of image area
6. Aspect filter removes boxes with width/height ratio > 5:1
7. Median vehicle size computed from remaining detections
8. Virtual grid generated: bounding region + 30% padding, divided into cells
9. Each vehicle assigned to nearest grid cell by center-point distance
10. Annotated image generated with grid overlay + bounding boxes + stats bar
11. Results stored in SQLite and returned to frontend
```

---

## 4. Implementation Details

### 4.1 Vehicle Detection (YOLOv8s + SAHI)

**Model:** YOLOv8s ("small" variant)
- Parameters: 11.2 million
- Pre-trained on: COCO dataset (80 classes)
- Vehicle classes used: Car (2), Motorcycle (3), Bus (5), Truck (7)
- Confidence threshold: 0.15 (deliberately low for better recall)

**SAHI Tiling Parameters:**
- Tile size: 640 × 640 pixels
- Overlap: 25% between adjacent tiles
- Minimum tile fraction: 30% (skip tiny edge tiles)

**Post-processing:**
- NMS IoU threshold: 0.45
- Min area: 0.05% of image area
- Max area: 25% of image area
- Max aspect ratio: 5:1

**Code excerpt (detection pipeline):**
```python
def detect(self, image, confidence_threshold=0.15):
    # Pass 1: Full image at 1280px
    full_dets = self._run_yolo(image, conf=0.15, imgsz=1280)

    # Pass 2: SAHI tiling (640px tiles, 25% overlap)
    tile_dets = self._sahi_tile_detect(image, conf=0.15)

    # Merge + deduplicate
    merged = _nms_merge(full_dets + tile_dets, iou_threshold=0.45)

    # Filter implausible boxes
    filtered, diagnostics = _filter_detections(merged, img_area)

    return filtered, diagnostics
```

### 4.2 Virtual Slot Grid Generation

Instead of hardcoding slot positions, the system dynamically generates a grid:

1. Compute **median width and height** of all detected vehicles
2. Define **slot cell size** = median vehicle × 1.2 (with gap)
3. Find **bounding rectangle** of all detections + 30% padding
4. Divide rectangle into **rows × columns** of equal cells
5. Name slots as **A1, A2, … B1, B2, …** (row letter + column number)

**Key property:** Grid always has ≥ N cells where N = number of detected vehicles.

### 4.3 Vehicle-to-Slot Mapping

Greedy assignment algorithm:
1. Sort detections by confidence (highest first)
2. For each detection, find the nearest unassigned slot by center-point distance
3. Mark that slot as "occupied" with the vehicle's confidence and class
4. Remaining slots stay "empty"

**Guarantees:**
- Each vehicle → exactly 1 slot
- Each slot → at most 1 vehicle
- No duplicates, no phantom data

### 4.4 Occupancy Prediction (RandomForest)

**Features (9 engineered time features):**

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | hour | Numeric | Hour of day (0–23) |
| 2 | minute | Numeric | Minute (0–59) |
| 3 | day_of_week | Numeric | 0=Monday, 6=Sunday |
| 4 | is_weekend | Binary | 1 if Saturday/Sunday |
| 5 | hour_sin | Cyclic | sin(2π × hour/24) |
| 6 | hour_cos | Cyclic | cos(2π × hour/24) |
| 7 | is_morning_peak | Binary | 1 if 8–11 AM |
| 8 | is_afternoon_peak | Binary | 1 if 2–4 PM |
| 9 | is_night | Binary | 1 if 10 PM–5 AM |

**Why cyclic encoding?** Hours 23 and 0 are adjacent in time but far apart numerically. Sin/cos encoding preserves this circular relationship.

**Model parameters:**
- n_estimators = 100 (100 decision trees)
- max_depth = 10 (prevents overfitting)
- random_state = 42 (reproducibility)

**Performance:**
- Training R² ≈ 0.90
- Test R² ≈ 0.72
- Training time: < 2 seconds on 1,345 data points

### 4.5 Database Schema (SQLite)

```sql
CREATE TABLE slot_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL,
    slot_id     INTEGER NOT NULL,
    slot_name   TEXT NOT NULL,
    status      TEXT NOT NULL CHECK(status IN ('occupied','empty')),
    confidence  REAL DEFAULT 1.0
);

CREATE TABLE occupancy_summary (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL UNIQUE,
    total_occupied  INTEGER NOT NULL,
    total_empty     INTEGER NOT NULL,
    total_slots     INTEGER NOT NULL,
    occupancy_rate  REAL NOT NULL
);
```

### 4.6 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/detect` | Upload image → full detection pipeline |
| GET | `/api/detect/sample` | Generate synthetic parking scene |
| GET | `/api/parking/status` | Current slot statuses |
| GET | `/api/parking/history` | Occupancy history (24h default) |
| GET | `/api/parking/heatmap` | 7×24 usage heatmap |
| POST | `/api/parking/simulate` | Generate 14 days of historical data |
| POST | `/api/prediction/train` | Train RandomForest model |
| GET | `/api/prediction/forecast` | Predict next 6–24 hours |
| GET | `/api/prediction/slots` | Per-slot predictions |
| GET | `/api/prediction/peak_hours` | Identify busiest hours |

---

## 5. Results and Testing

### 5.1 Test Results
- **System Tests:** 41/41 passed (server, API, database, ML pipeline)
- **Pipeline Tests:** 23/23 passed (detection invariants, consistency)

### 5.2 Pipeline Invariants Verified
- `occupied + empty == total_slots` — always holds
- `total_vehicles == len(detections)` — always holds
- 0 detections → 0 slots, 0% occupancy (no fabricated data)
- Same image uploaded twice → identical results (deterministic)
- Every slot has real (x, y) coordinates

### 5.3 Prediction Performance
| Metric | Train | Test |
|--------|-------|------|
| R² Score | 0.90 | 0.72 |
| Data Points | ~940 | ~405 |
| Training Time | < 2s | — |

---

## 6. Technologies & Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.10+ | Core language |
| ultralytics | ≥ 8.0 | YOLOv8 model |
| opencv-python-headless | 4.8.1 | Image processing |
| scikit-learn | ≥ 1.3 | RandomForest model |
| FastAPI | 0.104.1 | REST API framework |
| uvicorn | 0.24.0 | ASGI server |
| numpy | ≥ 1.24 | Numerical operations |
| Chart.js | 4.4.1 | Frontend charts |
| Tailwind CSS | 3.x | UI styling |

---

## 7. How to Run

```bash
# Clone
git clone https://github.com/Aayush9808/smart-parking-system.git
cd smart-parking-system

# Setup
python -m venv venv
source venv/bin/activate        # macOS/Linux
pip install -r requirements.txt

# Run
python run.py
# Open http://localhost:8000
```

---

## 8. Conclusion

This project demonstrates the practical application of two ML techniques:

1. **Deep Learning (YOLOv8)** — for real-time object detection in images, enhanced with SAHI tiling for handling varied image resolutions and small objects.

2. **Ensemble Learning (RandomForest)** — for time-series occupancy prediction using engineered features including cyclic encodings.

The system processes real parking lot images through a complete pipeline: detection → filtering → grid generation → spatial mapping → analytics → prediction, with no hardcoded or fabricated data at any stage.

---

## 9. Future Enhancements

- Train YOLOv8 on a parking-specific dataset (e.g., PKLot, CNRPark) for higher accuracy
- Add real-time video feed support (RTSP/webcam)
- Deploy on cloud (AWS/GCP) with containerization
- Mobile app for drivers to check availability
- License plate recognition for entry/exit tracking

---

## 10. References

1. Redmon, J. et al. — "You Only Look Once: Unified, Real-Time Object Detection" (YOLO)
2. Jocher, G. et al. — Ultralytics YOLOv8 Documentation (https://docs.ultralytics.com)
3. Akyon, F.C. et al. — "Slicing Aided Hyper Inference" (SAHI) for small object detection
4. Breiman, L. — "Random Forests" (Machine Learning, 2001)
5. scikit-learn Documentation — RandomForestRegressor
6. FastAPI Documentation — https://fastapi.tiangolo.com

---

*ParkSense AI — MLT Major Project*
