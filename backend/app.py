"""
FastAPI application — wires together routers, static files, and CORS.
"""

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.database import init_db  # noqa: E402
from backend.routers import detection, parking, prediction  # noqa: E402

app = FastAPI(
    title="ParkSense AI",
    description="Intelligent Parking Analytics — YOLOv8 + SAHI + RandomForest",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(detection.router, prefix="/api", tags=["detection"])
app.include_router(parking.router, prefix="/api", tags=["parking"])
app.include_router(prediction.router, prefix="/api", tags=["prediction"])

# Frontend static files
frontend_dir = PROJECT_ROOT / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

# Generated images
images_dir = PROJECT_ROOT / "data" / "images"
images_dir.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(images_dir)), name="images")


@app.get("/")
async def serve_dashboard():
    return FileResponse(str(frontend_dir / "index.html"))


@app.on_event("startup")
async def startup():
    init_db()
    print()
    print("  \033[1;36m ParkSense AI \033[0m")
    print("  ─────────────────────────────────")
    print("  Dashboard : http://localhost:8000")
    print("  API Docs  : http://localhost:8000/docs")
    print("  ─────────────────────────────────")
    print()
