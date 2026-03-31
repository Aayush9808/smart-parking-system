"""Full end-to-end verification of the Smart Parking System."""

import json
import sys

import cv2
import numpy as np
import requests

BASE = "http://localhost:8000"
PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}: PASS {detail}")
    else:
        FAIL += 1
        print(f"  ❌ {name}: FAIL {detail}")


print("=" * 60)
print("  SMART PARKING SYSTEM — FULL VERIFICATION")
print("=" * 60)

# ── 1. Server health ─────────────────────────────────────────
print("\n[1] Server Health")
try:
    r = requests.get(f"{BASE}/", timeout=5)
    check("Server responds", r.status_code == 200, f"(HTTP {r.status_code})")
    check("Returns HTML", "Smart Parking" in r.text)
except Exception as e:
    check("Server responds", False, str(e))
    print("SERVER NOT RUNNING — aborting.")
    sys.exit(1)

# ── 2. Static files ──────────────────────────────────────────
print("\n[2] Static Files")
r1 = requests.get(f"{BASE}/static/js/app.js")
r2 = requests.get(f"{BASE}/static/css/styles.css")
check("JS file served", r1.status_code == 200 and "generateSample" in r1.text)
check("CSS file served", r2.status_code == 200 and ".card" in r2.text)

# ── 3. Sample Detection ──────────────────────────────────────
print("\n[3] Sample Detection (GET /api/detect/sample)")
r = requests.get(f"{BASE}/api/detect/sample")
d = r.json()
check("Returns success", d["status"] == "success")
check("Has 12 slots", len(d["slots"]) == 12)
check("Has result image", len(d.get("result_image", "")) > 100)
check("Mode = simulation", d["mode"] == "simulation")
check("Slot data correct", all("name" in s and "status" in s for s in d["slots"]))
sample_occupied = d["total_occupied"]
print(f"     → Occupied: {d['total_occupied']}, Empty: {d['total_empty']}")

# ── 4. Upload Real Image Detection ───────────────────────────
print("\n[4] Upload Image Detection (POST /api/detect)")
# Create a test image with a car-like rectangle
img = np.full((500, 900, 3), 60, dtype=np.uint8)
cv2.rectangle(img, (60, 50), (150, 185), (180, 180, 180), -1)
_, buf = cv2.imencode(".jpg", img)
files = {"file": ("test.jpg", buf.tobytes(), "image/jpeg")}
r = requests.post(f"{BASE}/api/detect", files=files)
d = r.json()
check("Upload succeeds", d["status"] == "success")
check("Mode = yolo_detection", d["mode"] == "yolo_detection")
check("Has result image", len(d.get("result_image", "")) > 100)
check("Detections is list", isinstance(d.get("detections"), list))
print(f"     → YOLO detected {len(d['detections'])} vehicles, {d['total_occupied']} occupied")

# ── 5. Parking Status ────────────────────────────────────────
print("\n[5] Parking Status (GET /api/parking/status)")
r = requests.get(f"{BASE}/api/parking/status")
d = r.json()
check("Returns ok", d["status"] == "ok")
check("Has slots data", len(d.get("slots", [])) == 12)
check("Data persisted from detection", d["total_slots"] == 12)

# ── 6. Simulation ────────────────────────────────────────────
print("\n[6] Simulation (POST /api/parking/simulate)")
r = requests.post(f"{BASE}/api/parking/simulate")
d = r.json()
check("Simulation ok", d["status"] == "ok")
check("Generated 1000+ points", d["data_points"] > 1000)
print(f"     → {d['data_points']} snapshots generated")

# ── 7. History ────────────────────────────────────────────────
print("\n[7] Occupancy History (GET /api/parking/history)")
r = requests.get(f"{BASE}/api/parking/history")
d = r.json()
check("History ok", d["status"] == "ok")
check("Has data points", d["count"] > 0)
if d["count"] > 0:
    rec = d["data"][0]
    check("Record has timestamp", "timestamp" in rec)
    check("Record has occupancy", "total_occupied" in rec)
print(f"     → {d['count']} history records")

# ── 8. Heatmap ────────────────────────────────────────────────
print("\n[8] Heatmap (GET /api/parking/heatmap)")
r = requests.get(f"{BASE}/api/parking/heatmap")
d = r.json()
check("Heatmap ok", d["status"] == "ok")
check("Grid is 7x24", len(d["heatmap"]) == 7 and len(d["heatmap"][0]) == 24)
has_nonzero = any(v > 0 for row in d["heatmap"] for v in row)
check("Heatmap has real data", has_nonzero)

# ── 9. Train Prediction Model ────────────────────────────────
print("\n[9] Train Prediction (POST /api/prediction/train)")
r = requests.post(f"{BASE}/api/prediction/train")
d = r.json()
check("Training succeeded", d["status"] == "trained")
check("R² score > 0.5", d.get("test_score", 0) > 0.5, f"(R²={d.get('test_score')})")
check("Has feature importances", len(d.get("feature_importances", {})) > 0)
print(f"     → R² train={d.get('train_score')}, test={d.get('test_score')}")
print(f"     → Features: {list(d.get('feature_importances', {}).keys())}")

# ── 10. Forecast ──────────────────────────────────────────────
print("\n[10] Forecast (GET /api/prediction/forecast)")
r = requests.get(f"{BASE}/api/prediction/forecast")
d = r.json()
check("Forecast ok", d["status"] == "ok")
check("Has predictions", len(d.get("predictions", [])) > 0)
if d.get("predictions"):
    p = d["predictions"][0]
    check("Prediction has hour", "hour" in p)
    check("Prediction has occupied", "predicted_occupied" in p)
    print(f"     → {len(d['predictions'])} time points forecasted")
    print(f"     → Sample: {p['hour']} → {p['predicted_occupied']} occupied, {p['predicted_empty']} empty")

# ── 11. Slot Predictions ─────────────────────────────────────
print("\n[11] Slot Predictions (GET /api/prediction/slots)")
r = requests.get(f"{BASE}/api/prediction/slots")
d = r.json()
check("Slot preds ok", d["status"] == "ok")
check("Has 12 slot predictions", len(d.get("slot_predictions", [])) == 12)
occupied_slots = [s for s in d.get("slot_predictions", []) if s["current_status"] == "occupied"]
with_eta = [s for s in occupied_slots if s.get("prediction") and s["prediction"].get("estimated_free_in_minutes")]
print(f"     → {len(occupied_slots)} occupied, {len(with_eta)} have ETA predictions")

# ── 12. Peak Hours ────────────────────────────────────────────
print("\n[12] Peak Hours (GET /api/prediction/peak_hours)")
r = requests.get(f"{BASE}/api/prediction/peak_hours")
d = r.json()
check("Peak hours ok", d["status"] == "ok")
check("Has hourly data", len(d.get("hourly", [])) == 24)
check("Has busiest hour", d.get("busiest_hour") is not None)
if d.get("busiest_hour"):
    print(f"     → Busiest: {d['busiest_hour']['hour_label']} ({round(d['busiest_hour']['predicted_rate']*100)}% full)")
    print(f"     → Peak hours: {[p['hour_label'] for p in d.get('peak_hours', [])]}")

# ── 13. Database verification ─────────────────────────────────
print("\n[13] Database Verification")
import sqlite3
db_path = "data/parking.db"
conn = sqlite3.connect(db_path)
cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM slot_snapshots")
snap_count = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM occupancy_summary")
sum_count = cur.fetchone()[0]
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cur.fetchall()]
conn.close()
check("DB file exists and readable", snap_count > 0)
check("slot_snapshots has data", snap_count > 1000, f"({snap_count} rows)")
check("occupancy_summary has data", sum_count > 100, f"({sum_count} rows)")
check("Tables exist", "slot_snapshots" in tables and "occupancy_summary" in tables)

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  RESULTS: {PASS} PASSED, {FAIL} FAILED")
if FAIL == 0:
    print("  🎉 ALL CHECKS PASSED — SYSTEM IS FULLY WORKING")
else:
    print(f"  ⚠️  {FAIL} CHECK(S) FAILED — SEE ABOVE")
print("=" * 60)
