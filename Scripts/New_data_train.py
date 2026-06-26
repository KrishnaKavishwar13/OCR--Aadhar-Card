"""
new_data_train.py
-----------------
This script fine-tunes a YOLOv8 model on a new dataset (datasetB).
It loads the best weights from datasetA and continues training.
"""

import torch
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks


def main():
    # ✅ Path to the trained model from datasetA
    prev_model_path = "training/aadhar_yolov8n/weights/best.pt"

    # ✅ Load model with pretrained weights
    torch.serialization.add_safe_globals([tasks.DetectionModel])
    model = YOLO(prev_model_path)

    # ✅ Detect device (GPU/CPU)
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"🚀 Training on: {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

    # ✅ Train on new dataset (datasetB)
    model.train(
        data="dataB.yaml",          # 👈 update with your datasetB config
        epochs=50,                     # fine-tuning ke liye kam epochs rakhte hain
        imgsz=640,
        batch=16,
        project="training_B",           # output folder for datasetB training
        name="yolov8n_finetune_B",     # experiment name
        device=device
    )

    print("✅ Training on datasetB completed! Check the 'training_B/' folder for results.")


if __name__ == "__main__":
    main()
