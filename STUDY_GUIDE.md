# ParkSense AI — Complete Study Guide

> Everything about this project. Read this and you can answer ANY question.

---

## Table of Contents

1. [What is this project?](#1-what-is-this-project)
2. [How does it work? (Full Pipeline)](#2-how-does-it-work)
3. [YOLOv8 — Vehicle Detection (Deep Learning)](#3-yolov8--vehicle-detection)
4. [SAHI — Why and How](#4-sahi--why-and-how)
5. [Virtual Slot Grid — How slots are created](#5-virtual-slot-grid)
6. [Vehicle-to-Slot Mapping](#6-vehicle-to-slot-mapping)
7. [RandomForest — Prediction (Classical ML)](#7-randomforest--prediction)
8. [Feature Engineering — The 9 Features](#8-feature-engineering)
9. [Backend & API — How the server works](#9-backend--api)
10. [Frontend — How the dashboard works](#10-frontend--dashboard)
11. [Database — What's stored and how](#11-database)
12. [Every File Explained](#12-every-file-explained)
13. [How to Run & Demo](#13-how-to-run--demo)
14. [Common Viva Questions with Answers](#14-viva-questions)
15. [Algorithms & Math Explained](#15-algorithms--math)
16. [What could go wrong & limitations](#16-limitations)

---

## 1. What is this project?

**ParkSense AI** is an AI-based smart parking system. You upload a photo of a parking lot, and it:

1. **Detects all vehicles** using YOLOv8 (a deep learning model)
2. **Creates a virtual parking grid** based on detected vehicle sizes
3. **Maps each vehicle to a slot** showing which are occupied/empty
4. **Shows real-time stats** — occupied count, empty count, occupancy %
5. **Predicts future occupancy** using RandomForest (classical ML)
6. **Shows analytics** — history charts, heatmaps, peak hours

**Two ML techniques used:**
- **YOLOv8** = Deep Learning (CNN-based object detection)
- **RandomForest** = Classical ML (ensemble of decision trees)

---

## 2. How does it work?

### The Complete Pipeline (step by step):

```
STEP 1: User uploads a parking lot image
         ↓
STEP 2: YOLOv8s scans the FULL image at 1280px resolution
         → Finds vehicles (car, truck, bus, motorcycle)
         → Each vehicle = a bounding box (x1, y1, x2, y2) + confidence + class
         ↓
STEP 3: SAHI tiling (only if image ≥ 800px)
         → Cuts image into 640×640 tiles with 25% overlap
         → Runs YOLOv8 on EACH tile separately
         → Shifts tile coordinates back to full image space
         ↓
STEP 4: NMS (Non-Maximum Suppression)
         → Combines detections from Step 2 + Step 3
         → Removes duplicate boxes (IoU > 0.45 = same vehicle)
         → Keeps the higher-confidence box
         ↓
STEP 5: Filtering
         → Removes boxes too SMALL (< 0.05% of image) = noise
         → Removes boxes too LARGE (> 25% of image) = wrong detection
         → Removes boxes too ELONGATED (aspect ratio > 5:1)
         ↓
STEP 6: Grid Generation
         → Takes median width and height of all detected vehicles
         → Finds bounding rectangle of all vehicles + 30% padding
         → Divides into rows × columns of equal-sized cells
         → Names them: A1, A2, A3... B1, B2, B3...
         ↓
STEP 7: Mapping
         → Sorts vehicles by confidence (best first)
         → Each vehicle → nearest unassigned slot (by center distance)
         → Result: each slot is either "occupied" (with vehicle) or "empty"
         ↓
STEP 8: Visualization
         → Draws grid overlay on image (green=empty, red=occupied)
         → Draws bounding boxes on vehicles with labels
         → Draws stats bar at bottom
         ↓
STEP 9: Storage
         → Saves slot data in SQLite database
         → Returns everything to frontend as JSON
         ↓
STEP 10: Dashboard
         → Shows annotated image
         → Shows stats (vehicles, occupied, empty, slots, occupancy%)
         → Shows slot grid cards
```

---

## 3. YOLOv8 — Vehicle Detection

### What is YOLO?
- **Y**ou **O**nly **L**ook **O**nce
- A neural network that detects objects in images in a **single pass**
- Unlike older methods (R-CNN) that scan the image multiple times

### Why YOLOv8s?
- **s = small** variant (11.2M parameters)
- Good balance of speed and accuracy
- Pre-trained on **COCO dataset** (80 classes, 330K images)
- Vehicle classes already included: car(2), motorcycle(3), bus(5), truck(7)
- We use it **as-is** — no fine-tuning needed

### How it works internally:
1. Image → resized to 1280×1280
2. Passes through a **CSPDarknet backbone** (feature extraction CNN)
3. **FPN neck** creates multi-scale feature maps
4. **Detection head** predicts bounding boxes + classes + confidence
5. **All in one forward pass** (hence "You Only Look Once")

### Our configuration:
```python
model = YOLO("yolov8s.pt")          # Load pretrained weights
results = model(image, conf=0.15, imgsz=1280, verbose=False)
```
- `conf=0.15` → Low threshold = catch more vehicles (even uncertain ones)
- `imgsz=1280` → High resolution inference for better small object detection

### Output format:
```python
{
    "x1": 100, "y1": 200,    # top-left corner
    "x2": 250, "y2": 380,    # bottom-right corner
    "confidence": 0.87,       # how sure (0.0 to 1.0)
    "class_name": "car"       # what type of vehicle
}
```

---

## 4. SAHI — Why and How

### The Problem:
In aerial/CCTV parking lot images, cars appear **very small** (maybe 30×20 pixels in a 3000×2000 image). YOLOv8 resizes images to 1280px max, making these tiny vehicles even smaller → **misses them**.

### The Solution — SAHI (Slicing Aided Hyper Inference):
1. Cut the image into **overlapping tiles** (640×640 pixels each)
2. Run YOLO on **each tile separately** (now each car is bigger in the tile)
3. **Shift coordinates** back to the original image space
4. **Merge all detections** from all tiles + full image pass
5. Run **NMS** to remove duplicates

### Our parameters:
```
Tile size: 640 × 640 pixels
Overlap: 25% (tiles share 160px on each edge)
Skip threshold: tiles smaller than 30% of tile_size
```

### Visual example:
```
Full image (3000 × 2000):
┌────────────────────────────────────────────────┐
│  🚗 tiny    🚗 tiny    🚗 tiny    🚗 tiny     │
│                                                │
│  🚗 tiny    🚗 tiny    🚗 tiny    🚗 tiny     │
└────────────────────────────────────────────────┘

After SAHI tiling (640×640 each):
┌──────────┐┌──────────┐┌──────────┐
│  🚗 BIG  ││  🚗 BIG  ││  🚗 BIG  │  ← Each car is now
│          ││          ││          │     much bigger in
└──────────┘└──────────┘└──────────┘     its tile!
┌──────────┐┌──────────┐┌──────────┐
│  🚗 BIG  ││  🚗 BIG  ││  🚗 BIG  │
└──────────┘└──────────┘└──────────┘
```

---

## 5. Virtual Slot Grid

### Why not hardcode slots?
- Hardcoded slots only work for ONE specific parking lot
- Our system works on ANY parking lot image
- We generate slots **dynamically** from what we detect

### How it works:

**Step 1: Compute median vehicle size**
```
All detected widths:  [80, 85, 90, 75, 88]  → median = 85
All detected heights: [60, 65, 55, 70, 62]  → median = 62
```

**Step 2: Slot cell size = vehicle + gap**
```
slot_width  = 85 × 1.2 = 102 pixels
slot_height = 62 × 1.2 = 74 pixels
```

**Step 3: Find parking region**
```
Bounding box of all detections + 30% padding on each side
```

**Step 4: Create grid**
```
cols = region_width / slot_width   (rounded)
rows = region_height / slot_height (rounded)
```

**Step 5: Name slots**
```
Row 0: A1, A2, A3, A4, A5
Row 1: B1, B2, B3, B4, B5
Row 2: C1, C2, C3, C4, C5
```

### Key guarantee:
The grid always has **at least** as many cells as detected vehicles.

---

## 6. Vehicle-to-Slot Mapping

### Algorithm: Greedy Center-Distance Assignment

```
1. Sort all vehicles by confidence (highest first)
2. For each vehicle:
   a. Calculate center point: cx = (x1+x2)/2, cy = (y1+y2)/2
   b. For each UNASSIGNED slot:
      - Calculate slot center
      - Calculate Euclidean distance between vehicle center and slot center
   c. Assign vehicle to the NEAREST unassigned slot
   d. Mark that slot as "occupied"
3. All remaining slots = "empty"
```

### Why greedy by confidence?
- Most confident detections get first pick → best slots
- Prevents a low-confidence detection from "stealing" a better slot

### Guarantees:
- **1 vehicle → 1 slot** (no vehicle assigned twice)
- **1 slot → 1 vehicle max** (no slot double-occupied)
- **No phantom data** — if 0 vehicles detected, 0 slots created

---

## 7. RandomForest — Prediction

### What is a Random Forest?
- An **ensemble** of many **decision trees**
- Each tree is trained on a **random subset** of data and features
- Final prediction = **average** of all trees (for regression)
- This reduces overfitting compared to a single decision tree

### Our configuration:
```python
RandomForestRegressor(
    n_estimators=100,    # 100 decision trees
    max_depth=10,        # each tree max 10 levels deep
    random_state=42      # reproducible results
)
```

### What it predicts:
- **Input:** 9 time features (see next section)
- **Output:** Occupancy rate (0.0 to 1.0)
  - 0.0 = parking lot empty
  - 1.0 = parking lot completely full
  - 0.65 = 65% of slots occupied

### Training data:
- Generated by simulator: 14 days × ~96 snapshots/day = **1,345 data points**
- Realistic patterns: peaks 9–11 AM and 2–4 PM, low at night, weekends 50% quieter
- Train/test split: 70%/30%

### Performance:
```
Training R² = 0.90   (fits training data well)
Test R²     = 0.72   (generalizes reasonably to unseen times)
```

### What is R²?
- R² = 1.0 → perfect prediction
- R² = 0.0 → predicts just the average
- R² < 0.0 → worse than average
- **Our 0.72 means: the model explains 72% of the variance in parking occupancy**

---

## 8. Feature Engineering

### Why do we need features?
The model doesn't understand "2 PM on a Tuesday" directly. We convert timestamps into **9 numbers** the model can learn from.

### The 9 Features:

| # | Name | Values | Why it matters |
|---|------|--------|---------------|
| 1 | `hour` | 0–23 | Parking usage varies by hour |
| 2 | `minute` | 0–59 | Fine-grained within the hour |
| 3 | `day_of_week` | 0–6 | Mon=0, Sun=6, weekday vs weekend |
| 4 | `is_weekend` | 0 or 1 | Weekends have ~50% less usage |
| 5 | `hour_sin` | -1 to 1 | sin(2π × hour/24) — cyclic encoding |
| 6 | `hour_cos` | -1 to 1 | cos(2π × hour/24) — cyclic encoding |
| 7 | `is_morning_peak` | 0 or 1 | 1 if hour is 8–11 AM |
| 8 | `is_afternoon_peak` | 0 or 1 | 1 if hour is 2–4 PM |
| 9 | `is_night` | 0 or 1 | 1 if hour is 10 PM–5 AM |

### Why cyclic encoding (sin/cos)?

**Problem:** Hour 23 and Hour 0 are 1 hour apart in reality, but the model sees them as 23 units apart.

**Solution:** Map hours to a circle using sin and cos:
```
Hour  0: sin=0.00, cos=1.00
Hour  6: sin=1.00, cos=0.00
Hour 12: sin=0.00, cos=-1.00
Hour 18: sin=-1.00, cos=0.00
Hour 23: sin=-0.26, cos=0.97  ← Close to Hour 0!
```

Now the model "knows" that 11 PM and 1 AM are close together.

---

## 9. Backend & API

### Framework: FastAPI
- Modern Python web framework
- **Async** — handles multiple requests efficiently
- **Auto-documentation** — visit `/docs` for Swagger UI
- **Type validation** — catches bad requests automatically

### Server: Uvicorn
- ASGI server (Asynchronous Server Gateway Interface)
- Runs FastAPI in production mode
- Port 8000 by default

### API Endpoints (10 total):

#### Detection
| Endpoint | Method | What it does |
|----------|--------|-------------|
| `/api/detect` | POST | Upload image → full detection pipeline → annotated result |
| `/api/detect/sample` | GET | Generate random synthetic parking scene |

#### Parking Status
| Endpoint | Method | What it does |
|----------|--------|-------------|
| `/api/parking/status` | GET | Current slot statuses from last detection |
| `/api/parking/history` | GET | Occupancy over last 24 hours |
| `/api/parking/heatmap` | GET | 7×24 usage pattern matrix |
| `/api/parking/simulate` | POST | Generate 14 days of fake historical data |

#### Prediction
| Endpoint | Method | What it does |
|----------|--------|-------------|
| `/api/prediction/train` | POST | Train RandomForest on historical data |
| `/api/prediction/forecast` | GET | Predict occupancy for next 6 hours |
| `/api/prediction/slots` | GET | Per-slot availability predictions |
| `/api/prediction/peak_hours` | GET | Identify busiest hours of day |

### Example API response (POST /api/detect):
```json
{
  "status": "success",
  "mode": "yolo_detection",
  "total_vehicles": 8,
  "total_occupied": 8,
  "total_empty": 4,
  "total_slots": 12,
  "occupancy_pct": 66.7,
  "grid_rows": 3,
  "grid_cols": 4,
  "confidence_note": "Grid: 3×4 slots derived from 8 detected vehicles.",
  "detections": [...],
  "slots": [...],
  "diagnostics": {
    "detection": {"raw_full": 6, "raw_tiled": 9, "post_nms": 8, "post_filter": 8},
    "grid": {"rows": 3, "cols": 4, "slot_w": 95, "slot_h": 72},
    "mapping": {"matched": 8, "unmatched": 0}
  },
  "result_image": "base64_encoded_jpg..."
}
```

---

## 10. Frontend — Dashboard

### Technology:
- **HTML5** — structure
- **Tailwind CSS** (CDN) — utility-first styling, dark theme
- **Chart.js** (CDN) — line charts for history and forecast
- **Vanilla JavaScript** — no framework needed, direct DOM manipulation

### Dashboard Sections:
1. **Header** — ParkSense AI logo, live badge, clock
2. **Controls** — Generate Sample, Upload Image, Simulate, Train, Refresh
3. **Stats Row** — 5 cards: Vehicles, Occupied, Available, Total Slots, Occupancy%
4. **Confidence Note** — explains grid quality
5. **Parking Lot View** — annotated image with grid overlay
6. **Slot Grid** — cards showing each slot (red=occupied, green=empty)
7. **Slot Predictions** — per-slot availability forecast
8. **Charts** — Occupancy History (24h) + Forecast (6h)
9. **Heatmap** — 7×24 canvas showing weekly patterns
10. **Peak Hours** — busiest times identified

### How data flows to UI:
```
User clicks button
  → JavaScript fetch() to API endpoint
  → Backend processes and returns JSON
  → renderDetectionResult(data) updates:
     - Image container (base64 → <img>)
     - 5 stat cards (from JSON numbers)
     - Slot grid (dynamic columns matching grid_cols)
     - Confidence note
     - Mode badge (YOLOv8 or Simulated)
```

---

## 11. Database

### Engine: SQLite
- File-based database (no server needed)
- Stored at: `data/parking.db`
- Zero configuration

### Tables:

**`slot_snapshots`** — Every slot state at every detection:
```
id | timestamp           | slot_id | slot_name | status   | confidence
1  | 2026-04-01T10:30:00 | 1       | A1        | occupied | 0.87
2  | 2026-04-01T10:30:00 | 2       | A2        | empty    | 0.00
```

**`occupancy_summary`** — One row per detection/simulation event:
```
id | timestamp           | total_occupied | total_empty | total_slots | occupancy_rate
1  | 2026-04-01T10:30:00 | 8              | 4           | 12          | 0.6667
```

---

## 12. Every File Explained

```
MLT PROJECT/
├── run.py                          ← Entry point. Runs Uvicorn server on port 8000.
├── requirements.txt                ← pip install dependencies
├── test_system.py                  ← 41 automated checks (server, API, DB, ML)
├── test_pipeline.py                ← 23 pipeline invariant checks
│
├── ml_model/
│   ├── detector.py                 ← YOLOv8s + SAHI tiling + NMS + filtering
│   │                                  Main class: VehicleDetector
│   │                                  Key method: .detect(image) → (detections, diagnostics)
│   │
│   ├── slot_estimator.py           ← Virtual grid generation
│   │                                  Function: generate_slot_grid(detections, image_shape)
│   │                                  Returns: (slots_list, grid_info_dict)
│   │
│   ├── slot_mapper.py              ← Greedy center-distance vehicle→slot assignment
│   │                                  Function: map_vehicles_to_slots(slots, detections)
│   │                                  Returns: (mapped_slots, stats_dict)
│   │
│   ├── predictor.py                ← RandomForestRegressor wrapper
│   │                                  Class: ParkingPredictor
│   │                                  Methods: .train(), .predict_range(), .get_peak_hours()
│   │
│   ├── simulator.py                ← Generates fake but realistic historical data
│   │                                  Functions: generate_historical_data(), generate_parking_image()
│   │
│   └── parking_config.py           ← 12 hardcoded slots for SYNTHETIC images only
│                                      Used by: simulator.py, detect/sample endpoint
│
├── backend/
│   ├── app.py                      ← FastAPI app setup, CORS, static files, startup
│   ├── database.py                 ← SQLite operations (store/query snapshots)
│   └── routers/
│       ├── detection.py            ← POST /api/detect, GET /api/detect/sample
│       ├── parking.py              ← Status, history, heatmap, simulate
│       └── prediction.py           ← Train, forecast, slots, peak_hours
│
├── frontend/
│   ├── index.html                  ← Dashboard HTML (Tailwind CSS dark theme)
│   ├── js/app.js                   ← All dashboard JavaScript
│   └── css/styles.css              ← Custom CSS (cards, badges, animations)
│
└── data/
    ├── parking.db                  ← SQLite database (auto-created)
    └── images/
        └── latest_detection.jpg    ← Last annotated detection image
```

---

## 13. How to Run & Demo

### Setup (first time):
```bash
cd "MLT PROJECT"
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run.py
```

### Demo Script (in order):
1. Open `http://localhost:8000`
2. Click **"Generate Sample"** → shows synthetic parking lot with colored slots
3. Click **"Upload Image"** → select a real parking lot photo
   - System detects vehicles, shows grid overlay, updates all stats
4. Click **"Simulate 14 Days"** → generates 1,345 data points in database
5. Click **"Train Predictor"** → trains RandomForest, shows R² score
6. View **charts** (history, forecast), **heatmap**, **peak hours**
7. Click **"Refresh"** to reload all analytics

### Key URLs:
- Dashboard: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`

---

## 14. Viva Questions

### Detection Questions

**Q: What is YOLOv8?**
A: YOLO = You Only Look Once. It's a single-stage object detection neural network. Version 8 is the latest by Ultralytics. We use YOLOv8s (small variant, 11.2M parameters), pre-trained on COCO dataset which includes vehicle classes.

**Q: What is the difference between single-stage and two-stage detectors?**
A: Two-stage (like Faster R-CNN) first generates region proposals, then classifies each. Single-stage (like YOLO) predicts bounding boxes and classes in one pass → much faster but historically slightly less accurate. YOLOv8 has closed the accuracy gap.

**Q: What is SAHI and why do you use it?**
A: Slicing Aided Hyper Inference. In large aerial images, vehicles are tiny (maybe 30×20 pixels). YOLO resizes images to 1280px, making them even smaller. SAHI cuts the image into overlapping 640×640 tiles, runs YOLO on each tile where vehicles appear larger, then merges results.

**Q: What is NMS (Non-Maximum Suppression)?**
A: When multiple tiles overlap, the same vehicle may be detected twice. NMS compares all detection boxes: if two boxes overlap more than 45% (IoU > 0.45), it keeps the one with higher confidence and removes the other.

**Q: What is IoU (Intersection over Union)?**
A: IoU = (area of overlap between two boxes) / (total area covered by both boxes). IoU = 1.0 means perfect overlap. IoU = 0 means no overlap. We use IoU > 0.45 as the threshold for "same vehicle".

**Q: What is confidence threshold?**
A: The minimum score a detection must have to be kept. We use 0.15 (15%), which is low — this means we accept even uncertain detections to maximize recall (catching all vehicles). The area filter removes false positives later.

**Q: What classes does your model detect?**
A: Car (COCO class 2), Motorcycle (3), Bus (5), Truck (7). These are the vehicle-relevant classes from the 80 COCO classes.

**Q: Why not fine-tune YOLOv8 on parking-specific data?**
A: The pre-trained COCO model already detects vehicles well. Fine-tuning would require a labeled parking dataset (like PKLot with 695K images). For this project, pre-trained weights + SAHI is sufficient. Fine-tuning could be a future enhancement.

### Prediction Questions

**Q: Why RandomForest and not a neural network for prediction?**
A: RandomForest is better for our use case: (1) small dataset (~1,345 points), (2) trains in < 2 seconds, (3) interpretable — we can show feature importances, (4) no need for GPU. Neural networks need much more data and tuning.

**Q: What features does your model use?**
A: 9 time-based features: hour, minute, day_of_week, is_weekend, hour_sin, hour_cos, is_morning_peak, is_afternoon_peak, is_night. The cyclic sin/cos features handle the 23→0 hour wraparound.

**Q: What is R² score?**
A: Coefficient of determination. R²=1 is perfect, R²=0 means predicting the mean. Our R²=0.72 means the model explains 72% of the variance in parking occupancy using just time features.

**Q: What is overfitting and how do you prevent it?**
A: Overfitting = model memorizes training data but fails on new data. We prevent it with: (1) max_depth=10 limits tree complexity, (2) Random Forest averages 100 trees (reduces variance), (3) 70/30 train/test split to validate. Our train R²=0.90 vs test R²=0.72 shows slight overfitting but acceptable for a prototype.

**Q: What is an ensemble method?**
A: Combining multiple base models (here: 100 decision trees) to get a better prediction than any single model. RandomForest is a "bagging" ensemble — each tree trains on a random bootstrap sample with random feature subsets.

### Architecture Questions

**Q: Why FastAPI?**
A: Modern Python framework with async support, automatic request validation, built-in API documentation (/docs), and native JSON response handling. Much faster than Flask for I/O-bound tasks.

**Q: Why SQLite?**
A: Zero configuration, file-based, no separate database server needed. Perfect for a demo project. Production would use PostgreSQL or MySQL.

**Q: How does the virtual slot grid work?**
A: We compute median vehicle width/height from detections, add 20% gap, find the bounding rectangle of all vehicles with 30% padding, and divide into a row×column grid. Each cell gets a name like A1, B2.

**Q: How is vehicle-to-slot mapping done?**
A: Greedy center-distance assignment. Sort vehicles by confidence (best first). Each vehicle is assigned to the nearest unassigned slot by Euclidean distance between centers. This guarantees 1:1 mapping with no duplicates.

### General Questions

**Q: What are the limitations?**
A: (1) No fine-tuned model — relies on COCO pretrained weights. (2) Slot grid is approximate — doesn't know real lot layout. (3) Prediction trained on simulated data. (4) Single image — no video/real-time feed.

**Q: How would you deploy this in production?**
A: (1) Connect to CCTV camera for live feed. (2) Fine-tune YOLOv8 on parking datasets. (3) Deploy on cloud (AWS/GCP) with Docker. (4) Use PostgreSQL for multi-user access. (5) Add mobile app.

**Q: What is the COCO dataset?**
A: Common Objects in Context — 330K images with 80 object classes including person, car, truck, bus, motorcycle, etc. Used to pre-train object detection models. Created by Microsoft.

**Q: What ML techniques are used in this project?**
A: Two:
1. **Deep Learning** — YOLOv8 CNN for object detection (image → bounding boxes)
2. **Ensemble Learning** — RandomForest for regression (time features → occupancy rate)
Plus: NMS (post-processing), feature engineering (cyclic encoding), spatial algorithms (center-distance mapping).

---

## 15. Algorithms & Math

### NMS (Non-Maximum Suppression):
```
Input: list of boxes sorted by confidence (descending)
Output: filtered list with duplicates removed

while boxes remain:
    take the highest-confidence box → add to KEEP list
    compute IoU with every remaining box
    remove any box with IoU > threshold (0.45)
    repeat
```

### IoU Formula:
```
IoU = Intersection Area / Union Area

where:
  Intersection = overlap between two boxes
  Union = total area covered by both boxes (without double-counting)

IoU = Area(A ∩ B) / Area(A ∪ B)
    = Area(A ∩ B) / (Area(A) + Area(B) - Area(A ∩ B))
```

### Euclidean Distance (for mapping):
```
distance = √((x₁ - x₂)² + (y₁ - y₂)²)

where (x₁,y₁) = vehicle center, (x₂,y₂) = slot center
```

### Cyclic Encoding:
```
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)

This maps 24 hours to a unit circle, so:
  Hour 0  → (sin=0, cos=1)     (top of circle)
  Hour 6  → (sin=1, cos=0)     (right)
  Hour 12 → (sin=0, cos=-1)    (bottom)
  Hour 18 → (sin=-1, cos=0)    (left)
  Hour 23 → close to Hour 0    ← This is the key benefit!
```

### Occupancy Rate:
```
occupancy_rate = occupied_slots / total_slots
occupancy_pct  = occupancy_rate × 100
```

---

## 16. Limitations

| What | Limitation | How to fix |
|------|-----------|-----------|
| Detection accuracy | COCO pretrained, not parking-specific | Fine-tune on PKLot/CNRPark dataset |
| Slot grid | Approximate, doesn't match real lot lines | Use parking line detection or manual config |
| Prediction data | Trained on simulated data, not real | Collect real parking data over weeks |
| Real-time | Single image upload, no live feed | Add RTSP camera support |
| Scale | SQLite, single server | PostgreSQL + Docker + cloud |
| Occlusion | Can't detect cars hidden behind others | Use multiple camera angles |
| Night vision | Poor detection in low light | Use infrared cameras or image enhancement |

---

*ParkSense AI — MLT Major Project — Complete Study Guide*
