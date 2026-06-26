import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import re
import pytesseract
from ultralytics import YOLO
from supervision import Detections
import easyocr
from utils import clean_aadhar_number, clean_dob, clean_gender

# Path to Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# YOLO model
MODEL_PATH = r"E:\OCR-aadhar\training_B\yolov8n_finetune_B\weights\best.pt"

model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['en'], gpu=True)

# ---------- OCR Logic ----------
def extract_text(crop):
    try:
        text = pytesseract.image_to_string(crop, config="--psm 6", lang="eng")
        if text.strip():
            return text.strip()
    except:
        pass
    return " ".join(reader.readtext(crop, detail=0)).strip()

def clean_gender(text):
    t = text.lower()
    t = re.sub(r'[^a-z]', '', t)   # remove numbers, spaces, symbols

    if 'male' in t:
        return 'Male'
    if 'female' in t:
        return 'Female'
    if 'other' in t:
        return 'Other'
    return ''

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

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

        text = extract_text(crop)

        if re.search(r"\d{4}\s\d{4}\s\d{4}", text):
            output['Document_Number'] = clean_aadhar_number(text)
        elif re.search(r"\d{1,2}[-/]\d{1,2}[-/]\d{4}", text):
            output['Date_of_Birth'] = clean_dob(text)
        elif text.lower() in ["male", "female", "others"]:
            output['Gender'] = clean_gender(text)
        elif label == 'NAME' and not output['Name']:
            output['Name'] = text
        elif label == 'ADDRESS' and not output['Address']:
            output['Address'] = text

    return output


# ---------- GUI ----------
def upload_and_process():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )
    if not file_path:
        return

    result = process_image(file_path)

    if not result:
        messagebox.showerror("Error", "Failed to process image")
        return

    output_box.delete(1.0, tk.END)
    for k, v in result.items():
        output_box.insert(tk.END, f"{k}: {v}\n")


root = tk.Tk()
root.title("Aadhaar OCR System")
root.geometry("600x500")
root.configure(bg="#f2f2f2")

title = tk.Label(root, text="Aadhaar Data Extraction", font=("Arial", 20, "bold"), bg="#f2f2f2")
title.pack(pady=20)

btn = tk.Button(
    root,
    text="Upload Aadhaar Image",
    command=upload_and_process,
    font=("Arial", 14),
    bg="#4CAF50",
    fg="white",
    padx=20,
    pady=10
)
btn.pack(pady=10)

output_box = tk.Text(root, height=15, width=60, font=("Consolas", 12))
output_box.pack(pady=20)

root.mainloop()
