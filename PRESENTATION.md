# 🅿️ ParkSense AI — Presentation Guide

> **Subject:** Machine Learning Techniques (MLT) — Major Project
> **Title:** AI-Based Smart Parking System with Slot Detection and Prediction

---

## 🎯 Problem Statement

- Urban parking wastes **30% of city traffic** — drivers circling looking for spots
- No real-time visibility into parking lot occupancy
- No way to predict when spaces will be free

**Our Solution:** An AI-powered system that **detects vehicles** in parking lot images, **maps them to slots**, shows **real-time occupancy**, and **predicts future availability**.

---

## 🏗️ System Architecture

```
                    ┌─────────────────────┐
                    │   Parking Lot Image  │
                    └──────────┬──────────┘
                               ▼
                ┌──────────────────────────────┐
                │   YOLOv8s + SAHI Tiling      │  ← Vehicle Detection
                │   (2-pass: full + tiles)     │
                └──────────────┬───────────────┘
                               ▼
                ┌──────────────────────────────┐
                │   Area & Aspect Filtering    │  ← Noise Removal
                └──────────────┬───────────────┘
                               ▼
                ┌──────────────────────────────┐
                │   Virtual Slot Grid          │  ← Grid Generation
                │   (median vehicle size →     │
                │    row-col grid A1,B2...)    │
                └──────────────┬───────────────┘
                               ▼
                ┌──────────────────────────────┐
                │   Center-Distance Mapper     │  ← Vehicle → Slot
                │   (greedy, 1:1 assignment)   │
                └──────────────┬───────────────┘
                               ▼
                ┌──────────────────────────────┐
                │   FastAPI Backend            │
                │   + SQLite + RandomForest    │
                └──────────────┬───────────────┘
                               ▼
                ┌──────────────────────────────┐
                │   Dashboard (Dark Theme)     │
                │   Stats + Charts + Grid      │
                └──────────────────────────────┘
```

---

## 🛠️ Technologies Used

| Component | Technology | Why? |
|-----------|-----------|------|
| Object Detection | **YOLOv8s** (COCO pretrained) | Real-time, accurate, detects cars/trucks/buses |
| Small Object Handling | **SAHI Tiling** (640px tiles) | Catches small/distant vehicles in aerial views |
| Prediction | **RandomForestRegressor** | Fast training, interpretable, good on time features |
| Backend API | **FastAPI** | Async, auto-docs, fast |
| Database | **SQLite** | Lightweight, zero-config |
| Frontend | **Tailwind CSS + Chart.js** | Modern UI, responsive charts |

---

## 📊 ML Models — Key Points for Q&A

### YOLOv8s (Detection)
- **What:** Pretrained object detector (11.2M parameters)
- **Classes detected:** Car, Motorcycle, Bus, Truck
- **Input:** Parking lot image (any resolution)
- **Output:** Bounding boxes with class + confidence score
- **Enhancement:** SAHI tiling splits large images into 640px overlapping tiles → detects small cars → NMS merges duplicates

### RandomForest (Prediction)
- **What:** Ensemble of 100 decision trees
- **Features (9):** hour, minute, day_of_week, is_weekend, hour_sin, hour_cos, is_morning_peak, is_afternoon_peak, is_night
- **Target:** Occupancy rate (0.0 to 1.0)
- **Performance:** R² ≈ 0.72 on test data
- **Use:** Predict parking occupancy for the next 6–24 hours

---

## 🖥️ Live Demo Flow

### Step 1: Generate Sample
Click **"Generate Sample"** → Shows synthetic parking lot with 12 slots

### Step 2: Upload Real Image
Click **"Upload Image"** → Upload any parking lot photo → YOLO detects vehicles → Grid overlay appears

### Step 3: Simulate Historical Data
Click **"Simulate 14 Days"** → Generates 1,345 realistic data points

### Step 4: Train Predictor
Click **"Train Predictor"** → RandomForest trains in ~1 second → Shows R² score

### Step 5: View Analytics
- **Occupancy History** chart (24 hours)
- **Forecast** chart (next 6 hours)
- **Heatmap** (7 days × 24 hours)
- **Peak Hours** identification

---

## 📈 Results

| Metric | Value |
|--------|-------|
| Vehicle Detection | YOLOv8s + SAHI (conf ≥ 0.15) |
| Prediction Accuracy | R² = 0.72 (test), R² = 0.90 (train) |
| API Endpoints | 10 REST endpoints |
| Test Coverage | 41/41 system tests + 23/23 pipeline tests |
| Data Pipeline | Detect → Filter → Grid → Map → Store → Predict |

---

## 🔑 Key Innovations

1. **No Hardcoded Slots** — Grid is generated dynamically from detected vehicle sizes
2. **SAHI Tiling** — Handles any image resolution by splitting into tiles
3. **Area Filtering** — Removes false positives (too small/large/elongated boxes)
4. **1:1 Slot Mapping** — Each vehicle maps to exactly one slot, no duplicates
5. **Full Diagnostics** — API returns detection counts at every pipeline stage

---

## 🙋 Expected Viva Questions & Answers

**Q: Why YOLOv8 and not Faster R-CNN?**
A: YOLOv8 is single-pass (faster inference), easier to deploy, and pretrained on COCO which includes vehicle classes.

**Q: Why not use a pre-defined parking layout?**
A: Dynamic grid generation works on ANY parking lot image without manual configuration.

**Q: How does SAHI improve detection?**
A: Large images have small vehicles that YOLO misses at low resolution. SAHI slices the image into overlapping tiles so each vehicle appears larger in its tile.

**Q: What features does the predictor use?**
A: 9 time-based features including cyclic hour encoding (sin/cos) to handle the 23→0 hour wrap-around.

**Q: What is NMS?**
A: Non-Maximum Suppression — merges overlapping detections from different tiles by keeping the highest-confidence box and removing boxes with IoU > 0.45.

---

*Built for MLT Major Project — ParkSense AI*
