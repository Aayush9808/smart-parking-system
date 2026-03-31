#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
#  Smart Parking System — one-command setup & launch
# ──────────────────────────────────────────────────────────────────────────────
set -e

cd "$(dirname "$0")"

echo "==========================================="
echo "  Smart Parking System — Setup"
echo "==========================================="

# 1. Create virtual environment (if missing)
if [ ! -d "venv" ]; then
    echo "[1/3] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[1/3] Virtual environment already exists."
fi

# 2. Activate & install dependencies
echo "[2/3] Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

# 3. Launch
echo "[3/3] Starting server..."
echo ""
echo "  Dashboard → http://localhost:8000"
echo "  API docs  → http://localhost:8000/docs"
echo ""
python run.py
