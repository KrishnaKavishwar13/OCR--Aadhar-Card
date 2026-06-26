"""
train_yolo.py
-------------
This script trains a YOLOv8 model on your document dataset.
It handles both pretrained weights (.pt) and scratch training (.yaml).
"""

import torch
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

def main():
    # ✅ OPTION 1: Start from pretrained COCO weights (Recommended)
    model_path = "yolov8n.pt"

    # ✅ OPTION 2: Start from scratch (use YAML architecture)
    # model_path = "yolov8n.yaml"

    # Fix for PyTorch 2.6 pickle restriction (only needed if using .pt weights)
    torch.serialization.add_safe_globals([tasks.DetectionModel])

    # Load YOLO model
    model = YOLO(model_path)

    # Detect available device
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"🚀 Training on: {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

    # Start training
    model.train(
        data="data.yaml",         # path to dataset config file
        epochs=80,                # number of training epochs
        imgsz=640,                # image size (resize all images to 640x640)
        batch=16,                 # batch size (reduce if you hit OOM errors)
        project="training",       # output folder
        name="aadhar_yolov8n",    # experiment name
        pretrained=True,          # use pretrained COCO weights if .pt is used
        device=device             # 👈 force GPU if available
    )

    print("✅ Training completed! Check the 'training/' folder for results.")

if __name__ == "__main__":
    main()
