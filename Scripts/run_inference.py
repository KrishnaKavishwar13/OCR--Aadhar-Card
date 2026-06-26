"""
run_inference.py
----------------
1. Load YOLOv8 model.
2. Run detection on all Aadhaar test images.
3. Use Tesseract OCR (EasyOCR fallback).
4. Clean Aadhaar fields with regex.
5. Save results into CSV.
"""
import os
import cv2
import pytesseract
import easyocr
import pandas as pd
from ultralytics import YOLO
from supervision import Detections
from utils import clean_aadhar_number, clean_dob, clean_gender

import torch

# Monkeypatch torch.load to disable weights_only=True default
_original_load = torch.load
def safe_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = safe_load

# Windows users: update if needed
tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

MODEL_PATH = "training/aadhar_yolov8n/weights/best.pt"
TEST_FOLDER = "inference/test/images"
OUTPUT_CSV = "inference/extracted_csv/aadhar_details.csv"

def extract_text(crop, reader=None):
    try:
        text = pytesseract.image_to_string(crop, config="--psm 6", lang="eng")
        if text.strip():
            return text.strip()
    except Exception:
        pass
    if reader:
        return " ".join(reader.readtext(crop, detail=0)).strip()
    return ""

def main():
    model = YOLO(MODEL_PATH)
    reader = easyocr.Reader(['en'], gpu=True)

    all_results = []

    for filename in os.listdir(TEST_FOLDER):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(TEST_FOLDER, filename)
            print(f"🔍 Processing {filename}")

            image = cv2.imread(image_path)
            results = model.predict(image_path, conf=0.4)
            detections = Detections.from_ultralytics(results[0])

            output = {
                'File': filename,
                'Document_Type': 'Aadhar',
                'Name': '',
                'Document_Number': '',
                'Date_of_Birth': '',
                'Address': '',
                'Gender': ''
            }

            for bbox, label in zip(detections.xyxy, detections.data['class_name']):
                x1, y1, x2, y2 = map(int, bbox)
                crop = image[y1:y2, x1:x2]

                text = extract_text(crop, reader)

                if label == 'AADHAR_NUMBER':
                    output['Document_Number'] = clean_aadhar_number(text)
                elif label == 'DATE_OF_BIRTH':
                    output['Date_of_Birth'] = clean_dob(text)
                elif label == 'NAME' and not output['Name']:
                    output['Name'] = text
                elif label == 'ADDRESS' and not output['Address']:
                    output['Address'] = text
                elif label == 'GENDER':
                    output['Gender'] = clean_gender(text)

            all_results.append(output)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    pd.DataFrame(all_results).to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()