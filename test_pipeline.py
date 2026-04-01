"""
Pipeline verification — tests the rebuilt detection → grid → mapping chain.
Run this after starting the server to verify the core logic is sound.
"""

import json
import sys
import requests
import cv2
import numpy as np

BASE = "http://localhost:8000"
PASS = FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name} {detail}")
    else:
        FAIL += 1
        print(f"  ❌ {name} {detail}")


def test_with_image(label, img):
    """Upload an image and validate every part of the response."""
    print(f"\n── {label} ──")
    _, buf = cv2.imencode(".jpg", img)
    r = requests.post(f"{BASE}/api/detect", files={"file": ("test.jpg", buf.tobytes(), "image/jpeg")})
    d = r.json()

    check("Returns success", d["status"] == "success")
    check("Mode = yolo_detection", d["mode"] == "yolo_detection")

    n_det = len(d["detections"])
    n_slots = d["total_slots"]
    n_occ = d["total_occupied"]
    n_empty = d["total_empty"]
    occ_pct = d["occupancy_pct"]

    print(f"     Detections: {n_det}")
    print(f"     Slots: {n_slots} (occ={n_occ}, empty={n_empty}, {occ_pct}%)")

    # ── Core invariants ──
    check("Slots is list", isinstance(d["slots"], list))
    check("Detections is list", isinstance(d["detections"], list))
    check("occupied + empty == total_slots", n_occ + n_empty == n_slots,
          f"({n_occ} + {n_empty} = {n_occ + n_empty}, expected {n_slots})")
    check("total_vehicles == detections count",
          d["total_vehicles"] == n_det, f"({d['total_vehicles']} vs {n_det})")

    if n_det > 0:
        check("total_slots >= detections",
              n_slots >= n_det, f"({n_slots} >= {n_det})")
        check("occupied count matches detections",
              n_occ == n_det, f"({n_occ} == {n_det})")

        # Slot names should be row-col format (A1, B2, etc.)
        names = [s["name"] for s in d["slots"]]
        check("Slot names are row-col format",
              all(len(n) >= 2 and n[0].isalpha() and n[1:].isdigit() for n in names),
              f"(sample: {names[:5]})")

        # Every slot should have real coordinates (not 0,0)
        has_coords = all(s["x2"] > s["x1"] and s["y2"] > s["y1"] for s in d["slots"])
        check("All slots have real coordinates", has_coords)

        # Occupied slots should have class_name and confidence > 0
        occ_slots = [s for s in d["slots"] if s["status"] == "occupied"]
        check("Occupied slots have confidence > 0",
              all(s["confidence"] > 0 for s in occ_slots))
        check("Occupied slots have class_name",
              all(s["class_name"] in ("car", "truck", "bus", "motorcycle") for s in occ_slots))

        # No duplicates in slot names
        check("No duplicate slot names",
              len(names) == len(set(names)), f"({len(names)} slots, {len(set(names))} unique)")

        # Diagnostics present
        check("Has diagnostics", "diagnostics" in d)
        if "diagnostics" in d:
            diag = d["diagnostics"]
            print(f"     Diagnostics:")
            print(f"       Detection: full={diag['detection']['raw_full']}, "
                  f"tiled={diag['detection']['raw_tiled']}, "
                  f"nms={diag['detection']['post_nms']}, "
                  f"filtered={diag['detection']['post_filter']}")
            print(f"       Grid: {diag['grid']['rows']}x{diag['grid']['cols']} "
                  f"(slot {diag['grid']['slot_w']}x{diag['grid']['slot_h']}px)")
            print(f"       Mapping: matched={diag['mapping']['matched']}, "
                  f"unmatched={diag['mapping']['unmatched']}")
    else:
        check("No detections → 0 slots (correct)", n_slots == 0)

    check("Has result image", len(d.get("result_image", "")) > 100)
    check("Has confidence note", len(d.get("confidence_note", "")) > 0)

    return d


# ── Tests ─────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  PIPELINE VERIFICATION — detection → grid → mapping")
print("=" * 60)

# Test 1: Empty image (no vehicles)
print("\n[1] Blank image — should detect 0 vehicles")
blank = np.full((600, 800, 3), 128, dtype=np.uint8)
test_with_image("Blank gray image", blank)

# Test 2: Synthetic parking lot with car-shaped objects
print("\n[2] Synthetic scene with colored rectangles")
scene = np.full((800, 1200, 3), 80, dtype=np.uint8)
# Draw road markings
cv2.line(scene, (0, 400), (1200, 400), (200, 200, 200), 2)
# Draw car-like rectangles
colors = [(180, 180, 220), (60, 60, 200), (200, 200, 200), (50, 50, 50)]
for i, (x, y) in enumerate([(100, 150), (300, 150), (500, 500), (700, 500)]):
    cv2.rectangle(scene, (x, y), (x + 80, y + 120), colors[i % 4], -1)
test_with_image("Synthetic parking scene", scene)

# Test 3: Verify sample endpoint (synthetic simulation)
print("\n[3] Sample endpoint (GET /api/detect/sample)")
r = requests.get(f"{BASE}/api/detect/sample")
d = r.json()
s_occ = d["total_occupied"]
s_empty = d["total_empty"]
s_total = d["total_slots"]
check("Sample: occupied + empty == total", s_occ + s_empty == s_total,
      f"({s_occ} + {s_empty} = {s_total})")
check("Sample: mode = simulation", d["mode"] == "simulation")
check("Sample: all slots have names",
      all("name" in s for s in d["slots"]))

# Test 4: Verify consistency — detect twice gives same structure
print("\n[4] Consistency — same image twice")
test_img = np.random.randint(40, 120, (500, 700, 3), dtype=np.uint8)
_, buf = cv2.imencode(".jpg", test_img)
r1 = requests.post(f"{BASE}/api/detect", files={"file": ("t.jpg", buf.tobytes(), "image/jpeg")})
r2 = requests.post(f"{BASE}/api/detect", files={"file": ("t.jpg", buf.tobytes(), "image/jpeg")})
d1, d2 = r1.json(), r2.json()
check("Same image → same detection count",
      d1["total_vehicles"] == d2["total_vehicles"],
      f"({d1['total_vehicles']} vs {d2['total_vehicles']})")
check("Same image → same slot count",
      d1["total_slots"] == d2["total_slots"],
      f"({d1['total_slots']} vs {d2['total_slots']})")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  PIPELINE RESULTS: {PASS} PASSED, {FAIL} FAILED")
if FAIL == 0:
    print("  🎉 PIPELINE IS SOLID — ALL INVARIANTS HOLD")
else:
    print(f"  ⚠️  {FAIL} CHECK(S) FAILED")
print("=" * 60)
