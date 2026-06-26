# Document OCR + YOLOv8 Pipeline

## 📌 Overview
This project extracts structured details (Name, DOB, Gender, Aadhar Number, Address) 
from Indian Aadhar card images using:
- **YOLOv8** (for field detection)
- **EasyOCR** (for text recognition)
- **Regex & cleaning** (for number/date formats)

## 📂 Project Structure
- `dataset/` → training/validation dataset in YOLO format
- `data.yaml` → dataset config
- `scripts/train_yolo.py` → train YOLOv8
- `scripts/run_inference.py` → run inference + OCR + CSV export
- `scripts/utils.py` → helper functions
- `training/` → auto-generated results
- `inference/` → test images, extracted CSV

## 🚀 Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
