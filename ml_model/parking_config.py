"""
Parking lot configuration — defines slot positions, colors, and layout constants.
"""

# Synthetic parking lot image dimensions
IMAGE_WIDTH = 900
IMAGE_HEIGHT = 500

# Parking slot definitions
# Each slot: id, name, row, x1, y1, x2, y2 (bounding box in pixels)
PARKING_SLOTS = [
    # Row 1 (Top) — Slots A1 to A6
    {"id": 1,  "name": "A1", "row": 1, "x1": 50,  "y1": 40, "x2": 160, "y2": 195},
    {"id": 2,  "name": "A2", "row": 1, "x1": 180, "y1": 40, "x2": 290, "y2": 195},
    {"id": 3,  "name": "A3", "row": 1, "x1": 310, "y1": 40, "x2": 420, "y2": 195},
    {"id": 4,  "name": "A4", "row": 1, "x1": 440, "y1": 40, "x2": 550, "y2": 195},
    {"id": 5,  "name": "A5", "row": 1, "x1": 570, "y1": 40, "x2": 680, "y2": 195},
    {"id": 6,  "name": "A6", "row": 1, "x1": 700, "y1": 40, "x2": 810, "y2": 195},
    # Row 2 (Bottom) — Slots B1 to B6
    {"id": 7,  "name": "B1", "row": 2, "x1": 50,  "y1": 285, "x2": 160, "y2": 440},
    {"id": 8,  "name": "B2", "row": 2, "x1": 180, "y1": 285, "x2": 290, "y2": 440},
    {"id": 9,  "name": "B3", "row": 2, "x1": 310, "y1": 285, "x2": 420, "y2": 440},
    {"id": 10, "name": "B4", "row": 2, "x1": 440, "y1": 285, "x2": 550, "y2": 440},
    {"id": 11, "name": "B5", "row": 2, "x1": 570, "y1": 285, "x2": 680, "y2": 440},
    {"id": 12, "name": "B6", "row": 2, "x1": 700, "y1": 285, "x2": 810, "y2": 440},
]

# ── Colours (BGR for OpenCV) ──────────────────────────────────────────────────
COLOR_EMPTY      = (46, 204, 113)       # Green
COLOR_OCCUPIED   = (231, 76, 60)        # Red
COLOR_SLOT_LINE  = (255, 255, 255)      # White
COLOR_ASPHALT    = (60, 60, 65)         # Dark gray
COLOR_LANE       = (75, 75, 80)         # Slightly lighter
COLOR_LANE_MARK  = (200, 200, 50)       # Yellow lane markings

# Car body colours used when generating synthetic images
CAR_COLORS = [
    (41, 128, 185),   # Blue
    (192, 57, 43),    # Red
    (44, 62, 80),     # Dark blue
    (127, 140, 141),  # Gray
    (243, 156, 18),   # Orange
    (255, 255, 255),  # White
    (30, 30, 30),     # Black
    (142, 68, 173),   # Purple
]
