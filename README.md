# ParkSense AI

**Intelligent Parking Analytics** — Real-time vehicle detection, occupancy estimation, and predictive analytics for parking lots.

Built with YOLOv8 + SAHI tiling, RandomForest prediction, and a modern dark-themed dashboard.

---

## Key Features

- **Vehicle Detection** — YOLOv8s with SAHI sliding-window tiling detects cars, buses, trucks, and motorcycles in aerial/CCTV parking images
- **Dynamic Slot Estimation** — Estimates total parking capacity from detected vehicle sizes and spatial distribution (no hardcoded layouts)
- **Occupancy Analytics** — Real-time occupied/available counts, occupancy percentage, and confidence scoring
- **Predictive Forecasting** — RandomForest model trained on historical data predicts future occupancy by hour
- **Usage Heatmap** — 7×24 day-hour heatmap showing parking usage patterns
- **Peak Hour Detection** — Identifies busiest hours and high-occupancy periods
- **Live Dashboard** — Modern dark-themed UI with real-time updates, charts, and slot grid visualization

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Detection | YOLOv8s (pretrained COCO) + SAHI tiling + OpenCV |
| Prediction | scikit-learn RandomForestRegressor |
| Backend | FastAPI + Uvicorn |
| Database | SQLite |
| Frontend | HTML + Tailwind CSS + Chart.js + Vanilla JS |
| Language | Python 3.10+ |

## Project Structure

```
├── backend/
│   ├── app.py              # FastAPI application
│   ├── database.py          # SQLite operations
│   └── routers/
│       ├── detection.py     # Upload & detect endpoints
│       ├── parking.py       # Status, history, heatmap
│       └── prediction.py    # Train, forecast, peak hours
├── ml_model/
│   ├── detector.py          # YOLOv8s + SAHI tiling engine
│   ├── slot_estimator.py    # Virtual slot grid generator
│   ├── slot_mapper.py       # Center-distance vehicle→slot mapper
│   ├── predictor.py         # RandomForest predictor
│   ├── simulator.py         # Historical data generator
│   └── parking_config.py    # Slot layout & constants
├── frontend/
│   ├── index.html           # Dashboard
│   ├── js/app.js            # Dashboard logic
│   └── css/styles.css       # Custom styles
├── run.py                   # Entry point
├── requirements.txt         # Dependencies
├── README.md
├── PRESENTATION.md          # Presentation guide (demo flow + Q&A)
├── PROJECT_REPORT.md        # Full report for evaluation
└── STUDY_GUIDE.md           # Complete study reference (viva prep)
```

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/parksense-ai.git
cd parksense-ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
python run.py
```

Open **http://localhost:8000** in your browser.

## Demo Flow

1. **Generate Sample** — Creates a synthetic parking scene with random occupancy
2. **Upload Image** — Upload a real parking lot image → YOLOv8 detects all vehicles → shows bounding boxes + analytics
3. **Simulate 14 Days** — Generates 1300+ historical data points for the prediction model
4. **Train Predictor** — Trains RandomForest on the simulated data (R² ≈ 0.7–0.8)
5. **View Analytics** — Occupancy history chart, 6-hour forecast, usage heatmap, and peak hours

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/detect` | Upload image → vehicle detection |
| GET | `/api/detect/sample` | Generate synthetic parking scene |
| GET | `/api/parking/status` | Current parking status |
| GET | `/api/parking/history` | Occupancy history (24h) |
| GET | `/api/parking/heatmap` | 7×24 usage heatmap |
| POST | `/api/parking/simulate` | Generate historical data |
| POST | `/api/prediction/train` | Train prediction model |
| GET | `/api/prediction/forecast` | Occupancy forecast |
| GET | `/api/prediction/slots` | Per-slot predictions |
| GET | `/api/prediction/peak_hours` | Peak hour analysis |

## Detection Pipeline

```
Input Image
    │
    ├─→ Full-image pass (YOLOv8s @ 1280px)
    │
    ├─→ SAHI tiled pass (640px tiles, 25% overlap)
    │
    └─→ NMS merge → final detections
              │
              └─→ Dynamic capacity estimation
                        │
                        └─→ Slot grid + analytics
```

---

**ParkSense AI** — MLT Major Project
