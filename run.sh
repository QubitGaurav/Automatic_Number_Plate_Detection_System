#!/bin/bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv build-essential

python3.10 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements-gpu.txt

python scripts/convert_json_to_yolo.py
python scripts/train_yolo11.py --epochs 50 --batch 8 --imgsz 640 --device 0