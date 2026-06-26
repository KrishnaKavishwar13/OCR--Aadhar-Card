"""
test_single_image.py
--------------------
1. Load YOLOv8 model.
2. Run detection on Aadhaar image.
3. Use Tesseract OCR (EasyOCR fallback).
4. Regex clean Aadhaar number, DOB, Gender.
5. Print structured details.
"""

import cv2
import re
import pytesseract
from ultralytics import YOLO
from supervision import Detections
import easyocr
from utils import clean_aadhar_number, clean_dob, clean_gender

# Windows users: update path to your installed Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ==== CONFIG ====
MODEL_PATH = r"E:\OCR-aadhar\training_B\yolov8n_finetune_B\weights\best.pt"
IMAGE_PATH = r"E:\OCR-aadhar\dataset\Single_sample_test\E:\OCR-aadhar\dataset\Single_sample_test\AAdhar card bittu.jpg"

def extract_text(crop, reader=None):
    """Try Tesseract, fallback to EasyOCR."""
    try:
        text = pytesseract.image_to_string(crop, config="--psm 6", lang="eng")
        if text.strip():
            return text.strip()
    except Exception:
        pass
    if reader:
        return " ".join(reader.readtext(crop, detail=0)).strip()
    return ""

def test_single_image(image_path):
    model = YOLO(MODEL_PATH)
    reader = easyocr.Reader(['en'], gpu=True)

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return

    results = model.predict(image_path, conf=0.4)
    detections = Detections.from_ultralytics(results[0])

    output = {
        'Document_Type': 'Aadhar',
        'Document_Number': '',
        'Date_of_Birth': '',
        'Gender': '',
        'Name': '',
        'Address': ''
    }

    for bbox, label in zip(detections.xyxy, detections.data['class_name']):
        x1, y1, x2, y2 = map(int, bbox)
        crop = image[y1:y2, x1:x2]

        text = extract_text(crop, reader)

        if re.search(r"\d{4}\s\d{4}\s\d{4}", text):
            output['Document_Number'] = clean_aadhar_number(text)
        elif re.search(r"\d{1,2}[-/]\d{1,2}[-/]\d{4}", text):
            output['Date_of_Birth'] = clean_dob(text)
        elif text.lower() in ["male", "female", "others"]:
            output['Gender'] = clean_gender(text)
        elif label == 'NAME' and not output['Name']:
            output['Address'] = text
        elif label == 'ADDRESS' and not output['Address']:
            output['Name'] = text

    print("\n✅ Extracted Aadhar Details:")
    for k, v in output.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    test_single_image(IMAGE_PATH)
