<div align="center">

<!-- Animated Header -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f172a,50:1e40af,100:3b82f6&height=220&section=header&text=🅿️%20ParkSense%20AI&fontSize=52&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Intelligent%20Parking%20Analytics%20System&descSize=18&descAlignY=55&descColor=93c5fd" width="100%"/>

<!-- Badges -->
<p>
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=white" />
<img src="https://img.shields.io/badge/FastAPI-0.104-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
<img src="https://img.shields.io/badge/scikit--learn-RF-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
<img src="https://img.shields.io/badge/OpenCV-4.8-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
</p>
<p>
<img src="https://img.shields.io/badge/Tests-41%2F41%20Passing-brightgreen?style=flat-square" />
<img src="https://img.shields.io/badge/Pipeline-23%2F23%20Verified-brightgreen?style=flat-square" />
<img src="https://img.shields.io/badge/API%20Endpoints-10-blue?style=flat-square" />
<img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" />
</p>

**Real-time vehicle detection, dynamic slot mapping, and predictive occupancy analytics — powered by Deep Learning & Machine Learning.**

[🚀 Quick Start](#-quick-start) · [📊 Features](#-features) · [🧠 ML Techniques](#-ml-techniques-used) · [🏗️ Architecture](#️-system-architecture) · [📡 API](#-api-endpoints) · [🔮 Future Scope](#-future-scope)

</div>

---

## 📌 About The Project

**ParkSense AI** is an AI-powered smart parking system built as a **Major Project** for **Machine Learning Techniques (MLT)**. It uses two core ML approaches:

| ML Technique | Model | Purpose |
|:---:|:---:|:---|
| 🧠 **Deep Learning** | YOLOv8s (CNN) | Detect vehicles in parking lot images |
| 🌲 **Ensemble Learning** | RandomForest | Predict future parking occupancy |

> **The core idea:** Upload any parking lot image → AI detects every vehicle → creates a virtual slot grid → maps vehicles to slots → shows real-time analytics → predicts future availability.

### ❓ Problem Statement

- 🚗 **30% of urban traffic** = drivers circling for parking
- ⛽ Wastes fuel, increases emissions, causes congestion
- 📉 No real-time visibility into parking lot availability
- 🔮 No way to predict when spots will free up

### ✅ Our Solution

An end-to-end AI system that:
1. **Sees** — Detects vehicles using YOLOv8 deep learning
2. **Organizes** — Creates a virtual parking grid dynamically
3. **Maps** — Assigns each vehicle to a slot (1:1, no duplicates)
4. **Analyzes** — Shows occupied/empty counts, occupancy %
5. **Predicts** — Forecasts future availability using RandomForest

---

## 🎯 Features

<table>
<tr>
<td width="50%">

### 🔍 Vehicle Detection
- YOLOv8s with **11.2M parameters**
- Detects **cars, trucks, buses, motorcycles**
- SAHI tiling for **any image resolution**
- Area + aspect ratio filtering

</td>
<td width="50%">

### 🗺️ Dynamic Slot Grid
- **No hardcoded** slot positions
- Grid generated from detected vehicle sizes
- Named slots: **A1, A2, B1, B2...**
- Every slot has **real (x,y) coordinates**

</td>
</tr>
<tr>
<td>

### 📈 Predictive Analytics
- RandomForest with **9 engineered features**
- **Cyclic time encoding** (sin/cos)
- 6-hour occupancy forecast
- Peak hour identification

</td>
<td>

### 🖥️ Live Dashboard
- **Dark-themed** modern UI
- Real-time stats & charts
- 7×24 usage heatmap
- Responsive slot grid cards

</td>
</tr>
</table>

---

## 🧠 ML Techniques Used

### 1️⃣ YOLOv8 — Object Detection (Deep Learning)

```
📸 Parking Image → 🧠 YOLOv8s CNN → 📦 Bounding Boxes + Classes + Confidence
```

| Aspect | Detail |
|--------|--------|
| **Model** | YOLOv8s (small) — single-stage detector |
| **Architecture** | CSPDarknet backbone → FPN neck → Detection head |
| **Parameters** | 11.2 million |
| **Pre-trained on** | COCO dataset (80 classes, 330K images) |
| **Vehicle classes** | Car (2), Motorcycle (3), Bus (5), Truck (7) |
| **Confidence threshold** | 0.15 (low → high recall) |
| **Inference resolution** | 1280px (full image) + 640px (tiles) |

**Why YOLOv8?**
- ⚡ Single-pass detection (real-time capable)
- 🎯 State-of-the-art accuracy on COCO
- 📦 Easy deployment via `ultralytics` package
- 🚗 Already knows vehicle classes — no fine-tuning needed

**Enhancement — SAHI Tiling:**
```
Large image (3000×2000) with tiny cars
         ↓
Split into 640×640 overlapping tiles
         ↓
YOLO runs on each tile (cars appear bigger)
         ↓
Coordinates shifted back to original space
         ↓
NMS merges duplicates (IoU > 0.45 removed)
```

### 2️⃣ RandomForest — Occupancy Prediction (Classical ML)

```
⏰ Time Features → 🌲 100 Decision Trees → 📊 Occupancy Rate (0.0–1.0)
```

| Aspect | Detail |
|--------|--------|
| **Model** | RandomForestRegressor (ensemble) |
| **Trees** | 100 decision trees |
| **Max depth** | 10 (prevents overfitting) |
| **Features** | 9 engineered time features |
| **Target** | Occupancy rate (0.0 to 1.0) |
| **Train R²** | ~0.90 |
| **Test R²** | ~0.72 |
| **Training time** | < 2 seconds |

**The 9 Features:**

| # | Feature | Why it matters |
|---|---------|---------------|
| 1 | `hour` | Usage varies by hour |
| 2 | `minute` | Fine-grained timing |
| 3 | `day_of_week` | Weekday vs weekend patterns |
| 4 | `is_weekend` | Weekends have ~50% less usage |
| 5 | `hour_sin` | Cyclic encoding — 11PM close to 1AM |
| 6 | `hour_cos` | Cyclic encoding (cosine component) |
| 7 | `is_morning_peak` | Flag for 8–11 AM rush |
| 8 | `is_afternoon_peak` | Flag for 2–4 PM rush |
| 9 | `is_night` | Flag for 10PM–5AM low period |

**Why RandomForest?**
- 📊 Works great on small datasets (~1,345 points)
- ⚡ Trains in < 2 seconds
- 🔍 Interpretable — shows feature importance
- 🛡️ Resistant to overfitting (ensemble of 100 trees)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│              Dark-themed Dashboard (Tailwind + Chart.js)        │
└──────────────────────────┬──────────────────────────────────────┘
                           │  HTTP / REST
┌──────────────────────────▼──────────────────────────────────────┐
│                     FastAPI BACKEND                              │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │ /api/detect   │  │ /api/parking │  │ /api/prediction    │    │
│  │ Upload+Detect │  │ Status+Hist  │  │ Train+Forecast     │    │
│  └──────┬───────┘  └──────┬───────┘  └────────┬───────────┘    │
└─────────┼─────────────────┼───────────────────┼─────────────────┘
          │                 │                   │
┌─────────▼─────────┐ ┌────▼────┐  ┌───────────▼───────────┐
│  DETECTION ENGINE │ │ SQLite  │  │  PREDICTION ENGINE    │
│  ┌──────────────┐ │ │   DB    │  │  ┌─────────────────┐  │
│  │ YOLOv8s      │ │ └────────┘  │  │ RandomForest    │  │
│  │ + SAHI Tiles │ │             │  │ 100 trees       │  │
│  │ + NMS Merge  │ │             │  │ 9 features      │  │
│  │ + Filtering  │ │             │  └─────────────────┘  │
│  └──────┬───────┘ │             └───────────────────────┘
│  ┌──────▼───────┐ │
│  │ Grid Gen     │ │
│  │ + Mapper     │ │
│  └──────────────┘ │
└───────────────────┘
```

### 🔄 Detection Pipeline (Step by Step)

```
📸 Image Upload
    │
    ▼
🔍 STEP 1: YOLOv8s — Full image @ 1280px
    │        → raw detections (bounding boxes)
    ▼
🧩 STEP 2: SAHI — 640px tiles with 25% overlap
    │        → more detections from tiles
    ▼
🔗 STEP 3: NMS Merge — Remove duplicates (IoU > 0.45)
    │
    ▼
🧹 STEP 4: Filter — Remove noise
    │        → too small (< 0.05% image)
    │        → too large (> 25% image)
    │        → too elongated (aspect > 5:1)
    ▼
📐 STEP 5: Grid — Create virtual slot grid
    │        → median vehicle size + 20% gap
    │        → bounding region + 30% padding
    │        → rows × cols named A1, A2, B1...
    ▼
📍 STEP 6: Map — Assign vehicles to slots
    │        → center-distance, greedy, 1:1
    ▼
🎨 STEP 7: Annotate — Draw on image
    │        → green grid (empty) + red grid (occupied)
    │        → vehicle bounding boxes + labels
    │        → stats bar at bottom
    ▼
💾 STEP 8: Store — SQLite + return JSON to frontend
```

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology | Role |
|:-----:|:----------:|:-----|
| 🧠 | **YOLOv8s** | Vehicle detection (CNN, 11.2M params, COCO pretrained) |
| 🧩 | **SAHI** | Sliding-window tiling for small object detection |
| 🌲 | **scikit-learn** | RandomForestRegressor for occupancy prediction |
| 🖼️ | **OpenCV** | Image processing, drawing, encoding |
| ⚡ | **FastAPI** | REST API backend (async, auto-docs) |
| 🗄️ | **SQLite** | Lightweight database for snapshots |
| 🎨 | **Tailwind CSS** | Dashboard styling (dark theme) |
| 📊 | **Chart.js** | Interactive charts (history, forecast) |
| 🔢 | **NumPy** | Numerical operations |
| 🐍 | **Python 3.10+** | Core language |

</div>

---

## 📁 Project Structure

```
ParkSense-AI/
│
├── 🧠 ml_model/                    # Machine Learning Core
│   ├── detector.py                  # YOLOv8s + SAHI + NMS + filtering
│   ├── slot_estimator.py            # Virtual parking grid generator
│   ├── slot_mapper.py               # Vehicle → slot spatial mapper
│   ├── predictor.py                 # RandomForest predictor
│   ├── simulator.py                 # Realistic data generator
│   └── parking_config.py            # Synthetic lot configuration
│
├── ⚡ backend/                      # FastAPI Server
│   ├── app.py                       # App setup, CORS, static files
│   ├── database.py                  # SQLite operations
│   └── routers/
│       ├── detection.py             # POST /detect, GET /detect/sample
│       ├── parking.py               # Status, history, heatmap, simulate
│       └── prediction.py            # Train, forecast, slots, peak_hours
│
├── 🎨 frontend/                     # Dashboard UI
│   ├── index.html                   # Main dashboard (dark theme)
│   ├── js/app.js                    # Dashboard logic & rendering
│   └── css/styles.css               # Custom styling & animations
│
├── 📄 run.py                        # Entry point (Uvicorn server)
├── 📋 requirements.txt              # Python dependencies
├── 🧪 test_system.py                # 41 system verification checks
├── 🧪 test_pipeline.py              # 23 pipeline invariant checks
│
├── 📖 README.md                     # ← You are here
├── 🎤 PRESENTATION.md               # Demo flow + Q&A for presentation
├── 📝 PROJECT_REPORT.md             # Full report for teacher evaluation
└── 📚 STUDY_GUIDE.md                # Complete viva preparation guide
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Aayush9808/smart-parking-system.git
cd smart-parking-system

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
python run.py
```

### 🌐 Access

| URL | What |
|-----|------|
| `http://localhost:8000` | Dashboard |
| `http://localhost:8000/docs` | API Documentation (Swagger) |

---

## 🎮 Demo Walkthrough

<table>
<tr><td>

### Step 1: Generate Sample
Click **"Generate Sample"** to see a synthetic parking lot with 12 slots (random occupancy)

### Step 2: Upload Real Image
Click **"Upload Image"** → Select any parking lot photo → YOLO detects vehicles → grid overlay + analytics

### Step 3: Simulate History
Click **"Simulate 14 Days"** → Generates 1,345 realistic data points in the database

</td><td>

### Step 4: Train Predictor
Click **"Train Predictor"** → RandomForest trains in ~1 sec → shows R² score (~0.72)

### Step 5: View Analytics
- 📈 Occupancy History (24 hours)
- 🔮 Occupancy Forecast (6 hours)
- 🌡️ 7×24 Usage Heatmap
- ⏰ Peak Hours identification

</td></tr>
</table>

---

## 📡 API Endpoints

<details>
<summary><strong>🔍 Detection (2 endpoints)</strong></summary>

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/detect` | Upload image → full detection pipeline → annotated result |
| `GET` | `/api/detect/sample` | Generate random synthetic parking scene |

**POST /api/detect** response includes:
- `detections` — list of vehicles with bounding boxes, confidence, class
- `slots` — mapped slot grid with status, coordinates, names
- `diagnostics` — detection counts at every pipeline stage
- `result_image` — base64-encoded annotated image

</details>

<details>
<summary><strong>🅿️ Parking Status (4 endpoints)</strong></summary>

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/parking/status` | Current slot statuses from last detection |
| `GET` | `/api/parking/history` | Occupancy over last 24 hours |
| `GET` | `/api/parking/heatmap` | 7×24 usage pattern matrix |
| `POST` | `/api/parking/simulate` | Generate 14 days of historical data |

</details>

<details>
<summary><strong>🔮 Prediction (4 endpoints)</strong></summary>

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/prediction/train` | Train RandomForest on historical data |
| `GET` | `/api/prediction/forecast` | Predict occupancy for next 6–24 hours |
| `GET` | `/api/prediction/slots` | Per-slot availability predictions |
| `GET` | `/api/prediction/peak_hours` | Identify busiest hours of day |

</details>

---

## 📊 Results & Testing

<div align="center">

| Metric | Result |
|:------:|:------:|
| 🧪 System Tests | **41/41 Passing** |
| 🔬 Pipeline Tests | **23/23 Passing** |
| 🎯 Prediction R² (test) | **0.72** |
| 🎯 Prediction R² (train) | **0.90** |
| ⚡ Training Time | **< 2 seconds** |
| 🔌 API Endpoints | **10** |

</div>

### ✅ Verified Invariants
- `occupied + empty == total_slots` — **always holds**
- `total_vehicles == len(detections)` — **always holds**
- 0 detections → 0 slots → 0% occupancy — **no fake data**
- Same image twice → **identical results** (deterministic)
- Every slot has **real (x,y) coordinates** — no phantom slots

---

## 🗃️ Database Schema

```sql
-- Every slot state at every detection event
CREATE TABLE slot_snapshots (
    id          INTEGER PRIMARY KEY,
    timestamp   TEXT NOT NULL,
    slot_id     INTEGER NOT NULL,
    slot_name   TEXT NOT NULL,          -- A1, B2, etc.
    status      TEXT CHECK(status IN ('occupied','empty')),
    confidence  REAL DEFAULT 1.0
);

-- One summary row per detection/simulation event
CREATE TABLE occupancy_summary (
    id              INTEGER PRIMARY KEY,
    timestamp       TEXT NOT NULL UNIQUE,
    total_occupied  INTEGER NOT NULL,
    total_empty     INTEGER NOT NULL,
    total_slots     INTEGER NOT NULL,
    occupancy_rate  REAL NOT NULL       -- 0.0 to 1.0
);
```

---

## 🔮 Future Scope

| Enhancement | Description |
|:---:|:---|
| 🎯 **Fine-tuned Model** | Train YOLOv8 on parking-specific datasets (PKLot, CNRPark) for higher accuracy |
| 📹 **Live Video Feed** | RTSP/webcam support for continuous real-time monitoring |
| ☁️ **Cloud Deployment** | Docker + AWS/GCP for production-scale deployment |
| 📱 **Mobile App** | Flutter/React Native app for drivers to check availability on-the-go |
| 🔑 **License Plate Recognition** | ANPR for automated entry/exit tracking and billing |
| 🤖 **Slot Line Detection** | Computer vision to detect actual painted slot lines instead of virtual grid |
| 📊 **Advanced Prediction** | LSTM/Transformer models for better time-series forecasting |
| 🌙 **Night Vision** | Infrared camera support + image enhancement for low-light detection |
| 💳 **Payment Integration** | Online booking and payment for reserved parking slots |
| 🗺️ **Multi-Lot Support** | Monitor and compare multiple parking lots from one dashboard |

---

## 🧑‍💻 Contributing

```bash
# Fork the repo, then:
git clone https://github.com/YOUR_USERNAME/smart-parking-system.git
cd smart-parking-system
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python run.py

# Run tests to verify
python test_system.py        # 41 checks
python test_pipeline.py      # 23 checks
```

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [📖 README.md](README.md) | Project overview (you're reading it) |
| [🎤 PRESENTATION.md](PRESENTATION.md) | Presentation guide with demo flow & Q&A |
| [📝 PROJECT_REPORT.md](PROJECT_REPORT.md) | Full project report for evaluation |
| [📚 STUDY_GUIDE.md](STUDY_GUIDE.md) | Complete viva preparation (20+ Q&A) |

---

## 📜 License

This project is for educational purposes (MLT Major Project).

---

<div align="center">

### ⭐ Star this repo if you found it useful!

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f172a,50:1e40af,100:3b82f6&height=120&section=footer&fontSize=14&fontColor=93c5fd&animation=fadeIn" width="100%"/>

<p>
<strong>ParkSense AI</strong> — Built with ❤️ for MLT Major Project<br>
<em>YOLOv8 · RandomForest · FastAPI · SAHI · OpenCV · Chart.js</em>
</p>

</div>
