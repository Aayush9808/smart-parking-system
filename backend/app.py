"""
FastAPI Application — ParkSense AI Backend
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os

from config.settings import BASE_DIR
from backend.database import init_db
from backend.routes import detection, parking, prediction

# Initialize database
init_db()

# Create app
app = FastAPI(
    title="ParkSense AI",
    description="AI-Based Smart Parking System — Car Detection + Analytics",
    version="2.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(detection.router)
app.include_router(parking.router)
app.include_router(prediction.router)

# Serve frontend
frontend_dir = os.path.join(BASE_DIR, "frontend")
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/")
async def serve_dashboard():
    return FileResponse(os.path.join(frontend_dir, "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0", "system": "ParkSense AI"}
