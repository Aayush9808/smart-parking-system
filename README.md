<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f1117,50:1e40af,100:3b82f6&height=200&section=header&text=🅿️%20ParkSense%20AI&fontSize=48&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=AI-Based%20Smart%20Parking%20System&descSize=18&descAlignY=55&descColor=93c5fd" width="100%"/>

<p>
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/YOLOv8s-Car_Detection-00FFFF?style=for-the-badge&logo=yolo&logoColor=white" />
<img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
<img src="https://img.shields.io/badge/scikit--learn-Prediction-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
<img src="https://img.shields.io/badge/OpenCV-Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
</p>

**Upload a parking lot image → AI detects every car → generates a slot grid → maps cars to slots → shows real-time analytics.**

</div>

---

## 🎯 What is This?

**ParkSense AI** is an AI-powered smart parking system that:

1. **Detects cars** in parking lot images using YOLOv8 deep learning
2. **Creates a parking grid** dynamically (no hardcoded slots)
3. **Maps each car** to the nearest slot (1:1, no duplicates)
4. **Shows analytics** — occupied, empty, occupancy %, charts
5. **Predicts future occupancy** using a RandomForest model

> Think of it like Google Maps for parking — but powered by computer vision.

---

## 🧠 How Does It Work? (Simple Explanation)

```
📸 You upload a parking lot photo
         ↓
🔍 YOLOv8 AI scans the image and finds every CAR
         ↓
📐 System creates a virtual parking grid based on car sizes
         ↓
📍 Each car gets assigned to the nearest parking slot
         ↓
📊 Dashboard shows: 6 cars detected, 15 slots, 9 empty, 40% full
         ↓
🔮 RandomForest predicts: "At 5 PM, parking will be 85% full"
```

**That's it.** No manual counting, no sensors, no hardware. Just a camera + AI.

---

## 🖥️ Tech Stack

| What | Technology | Why |
|------|-----------|-----|
| **Car Detection** | YOLOv8s (Deep Learning) | Best real-time object detector, pretrained on 80 classes |
| **Small Car Detection** | SAHI Tiling | Splits large images into tiles to find tiny/distant cars |
| **Prediction** | RandomForest (scikit-learn) | Predicts future occupancy from time patterns |
| **Backend** | FastAPI | Fast, modern Python API framework |
| **Database** | SQLite | Lightweight, zero-config database |
| **Frontend** | HTML + CSS + JavaScript | Clean dark-themed dashboard with Chart.js |
| **Image Processing** | OpenCV | Drawing bounding boxes, annotations |

---

## 📁 Project Structure

```
ParkSense-AI/
│
├── ml/                          # 🧠 Machine Learning Core
│   ├── detector.py              # YOLOv8 + SAHI car detection
│   ├── slot_generator.py        # Dynamic parking grid creator
│   ├── mapper.py                # Car → slot spatial mapping
│   └── config.py                # ML hyperparameters
│
├── backend/                     # ⚡ FastAPI Server
│   ├── app.py                   # Main application
│   ├── database.py              # SQLite operations
│   ├── schemas.py               # API response models
│   └── routes/
│       ├── detection.py         # POST /detect, GET /sample
│       ├── parking.py           # Status, history, heatmap
│       └── prediction.py        # Train, forecast, peak hours
│
├── services/                    # 🔧 Business Logic
│   ├── analytics.py             # Occupancy calculations
│   ├── predictor.py             # RandomForest model
│   └── simulator.py             # Historical data generator
│
├── frontend/                    # 🎨 Dashboard UI
│   ├── index.html               # Main page
│   ├── css/styles.css           # Dark theme styles
│   └── js/app.js                # Dashboard logic
│
├── config/
│   └── settings.py              # All configuration
│
├── run.py                       # Entry point
├── requirements.txt             # Dependencies
├── README.md                    # ← You are here
└── TECHNICAL.md                 # Deep technical docs
```

---

## 🚀 How to Run (Step by Step)

### Prerequisites
- **Python 3.10+** (check: `python --version`)
- **pip** (comes with Python)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Aayush9808/smart-parking-system.git
cd smart-parking-system

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
python run.py
```

### Open in Browser
| URL | What |
|-----|------|
| http://localhost:8000 | Dashboard |
| http://localhost:8000/docs | API Documentation |

> **First run downloads YOLOv8s model (~22 MB) automatically.**

---

## 🎮 Demo Flow (What to Click)

| Step | Action | What Happens |
|------|--------|-------------|
| 1 | Click **"Generate Sample"** | Creates a synthetic parking lot with random cars |
| 2 | Click **"Upload Image"** | Upload any parking lot photo → YOLOv8 detects cars |
| 3 | Click **"Simulate 14 Days"** | Generates 337 data points for training |
| 4 | Click **"Train Predictor"** | Trains RandomForest → shows R² score |
| 5 | View charts | Occupancy history, forecast, heatmap, peak hours |

---

## 📡 API Endpoints

| Method | Endpoint | What It Does |
|--------|----------|-------------|
| `POST` | `/api/detect` | Upload image → full detection pipeline |
| `GET` | `/api/detect/sample` | Generate synthetic parking scene |
| `GET` | `/api/parking/status` | Current parking status |
| `GET` | `/api/parking/history` | Occupancy over time |
| `GET` | `/api/parking/heatmap` | 7×24 usage heatmap |
| `POST` | `/api/parking/simulate` | Generate 14 days of data |
| `POST` | `/api/prediction/train` | Train the prediction model |
| `GET` | `/api/prediction/forecast` | Predict next 6-48 hours |
| `GET` | `/api/prediction/peak_hours` | Busiest hours of the day |
| `GET` | `/health` | System health check |

---

## 📊 What the Dashboard Shows

- **Stats Bar** — Cars detected, total slots, occupied, empty, occupancy %
- **Detection Image** — Annotated with car boxes + slot grid overlay
- **Slot Grid** — Visual grid of all slots (green = empty, red = occupied)
- **Diagnostics** — Pipeline step counts (full pass, SAHI tiles, NMS, filter)
- **History Chart** — Occupancy trend over time
- **Forecast Chart** — Predicted occupancy for upcoming hours
- **Heatmap** — 7-day × 24-hour usage patterns
- **Peak Hours** — Top 5 busiest hours

---

## 🧠 ML Techniques Used

### 1. YOLOv8 — Car Detection (Deep Learning)
- **What**: A CNN that looks at an image once and finds all objects
- **Model**: YOLOv8s (small variant, 11.2M parameters)
- **Trained on**: COCO dataset (330K images, 80 classes)
- **We use**: Only class 2 = "car"
- **Enhancement**: SAHI tiling splits large images into 640px overlapping tiles

### 2. RandomForest — Occupancy Prediction (Classical ML)
- **What**: 100 decision trees vote together on the predicted occupancy
- **Input**: 9 time-based features (hour, day, is_weekend, cyclic encoding...)
- **Output**: Occupancy rate (0.0 to 1.0)
- **R² Score**: ~0.97 on training data

---

## ❓ FAQ

**Q: Does this need a GPU?**
A: No. YOLOv8s runs fine on CPU. Detection takes 2-5 seconds.

**Q: Does it detect only cars?**
A: Yes. This version focuses on high-reliability **car-only** detection.

**Q: Where are the parking slots defined?**
A: Nowhere hardcoded. Slots are **generated dynamically** from detected car sizes.

**Q: What if no cars are detected?**
A: 0 cars → 0 slots → 0% occupancy. No fake data ever.

---

## 📜 License

Educational purposes (MLT Major Project).

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f1117,50:1e40af,100:3b82f6&height=100&section=footer" width="100%"/>

**ParkSense AI v2.0** — Rebuilt from scratch at industry level

*YOLOv8 · RandomForest · FastAPI · SAHI · OpenCV · Chart.js*

</div>
