#!/bin/bash
set -e

echo "=== System Check ==="
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
else
    echo "Warning: nvidia-smi not found. Proceeding with CPU/Auto device."
fi

echo "=== Dependency Setup ==="
sudo apt update
sudo apt install -y python3.11 python3.11-venv build-essential

python3.11 -m venv .venv
source .venv/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-gpu.txt
python3 -m pip install -r requirements.txt

echo "=== Dataset Conversion ==="
python scripts/convert_json_to_yolo.py

echo "=== Model Training ==="
export CUDA_VISIBLE_DEVICES=0
python scripts/train_yolo11.py --epochs 50 --batch 4 --workers 2 --imgsz 640 --device 0
