import os
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.models import User
from .models import Document
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from ultralytics import YOLO
import cv2
import pytesseract
import re

# --- CONFIG ---
import torch
import ultralytics.nn.tasks as tasks

# Monkeypatch torch.load to disable weights_only=True default
_original_load = torch.load
def safe_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = safe_load

# Construct absolute path to the model relative to the project root or use fixed path
# Assuming the project root is 3 levels up from here? No, views.py is in Aadhar_interface/aadhar_ocr_app/ocr/
# Let's use a robust way to find the model or default to a known location
# e:/OCR-aadhar/training/aadhar_yolov8n/weights/best.pt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODEL_PATH = os.path.join(BASE_DIR, "training", "aadhar_yolov8n", "weights", "best.pt")

# Fallback if training not done, but we expect it to be there.
if not os.path.exists(MODEL_PATH):
    print(f"WARNING: Model not found at {MODEL_PATH}")

tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

model = YOLO(MODEL_PATH)

def extract_text_from_box(image, box):
    x1, y1, x2, y2 = map(int, box)
    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(gray, lang='eng', config='--psm 6')
    return text.strip()

def index(request):
    # This view is deprecated, redirecting to login
    return login_view(request)

    return render(request, 'ocr/index.html', context)

def login_view(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        
        user = authenticate(request, username=email, password=password)
        if user is not None:
            login(request, user)
            return redirect('dashboard')
        else:
            messages.error(request, 'Invalid email or password')
            
    return render(request, 'login.html')

def register_view(request):
    if request.method == 'POST':
        full_name = request.POST.get('full-name')
        email = request.POST.get('email')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm-password')

        if password != confirm_password:
            messages.error(request, 'Passwords do not match')
            return render(request, 'register.html')
        
        if User.objects.filter(username=email).exists():
            messages.error(request, 'Email already registered')
            return render(request, 'register.html')
        
        try:
            # Create user with email as username
            user = User.objects.create_user(username=email, email=email, password=password)
            user.first_name = full_name
            user.save()
            messages.success(request, 'Account created successfully! Please login.')
            return redirect('login')
        except Exception as e:
            messages.error(request, f'Error creating account: {str(e)}')
            
    return render(request, 'register.html')

def logout_view(request):
    logout(request)
    return redirect('login')

def dashboard_view(request):
    documents = Document.objects.all().order_by('-uploaded_at')
    context = {'documents': documents}
    return render(request, 'dashboard1.html', context)

def admin_login_view(request):
    return render(request, 'admin_login_page.html')

def admin_dashboard_view(request):
    return render(request, 'admin_dashboard.html')

def document_upload_view(request):
    if request.method == 'POST' and request.FILES.get('aadhar_image'):
        uploaded_file = request.FILES['aadhar_image']
        fs = FileSystemStorage()
        file_path = fs.save(uploaded_file.name, uploaded_file)
        full_path = fs.path(file_path)

        # Run YOLO
        results = model.predict(source=full_path, conf=0.4, verbose=False)
        image = cv2.imread(full_path)

        details = {
            'Document_Type': 'Aadhar',
            'AADHAR_NUMBER': '',
            'DATE_OF_BIRTH': '',
            'GENDER': '',
            'NAME': '',
        }

        detected_texts = []

        # Collect all box texts
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            for box in boxes:
                text = extract_text_from_box(image, box)
                if text.strip():
                    detected_texts.append(text.strip())

        # --- JUGAAD LOGIC 🔥 ---
        combined_text = " ".join(detected_texts)
        combined_text = re.sub(r'\n+', ' ', combined_text)
        print("DEBUG TEXT:", combined_text)

        # 🔹 1. Extract Aadhar Number (12-digit pattern)
        aadhar_matches = re.findall(r'\b\d{4}\s?\d{4}\s?\d{4}\b', combined_text)
        if aadhar_matches:
            details['AADHAR_NUMBER'] = aadhar_matches[0].replace(" ", "")

        # 🔹 2. Extract DOB (with / or -)
        dob_matches = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b', combined_text)
        if dob_matches:
            details['DATE_OF_BIRTH'] = dob_matches[0]

        # 🔹 3. Extract Gender
        if re.search(r'\bmale\b', combined_text, re.I):
            details['GENDER'] = "Male"
        elif re.search(r'\bfemale\b', combined_text, re.I):
            details['GENDER'] = "Female"
        elif re.search(r'\bothers\b', combined_text, re.I):
            details['GENDER'] = "Others"

        # 🔹 4. Extract Name (anything before DOB or Gender)
        # First, remove known non-name keywords so they don't get combined with the name
        clean_text = re.sub(r'\b(Government|Republic|India|Male|Female|Others|DOB|Year|Birth|of|the|To)\b', ' ', combined_text, flags=re.I)
        # Collapse multiple spaces
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        name_candidates = re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', clean_text)
        if name_candidates:
            details['NAME'] = name_candidates[0].strip()
        
        try:
            user = request.user if request.user.is_authenticated else None
            document = Document.objects.create(
                user=user,
                image=file_path,  # FileSystemStorage save returns relative path
                name=details['NAME'],
                aadhar_number=details['AADHAR_NUMBER'],
                dob=details['DATE_OF_BIRTH'],
                gender=details['GENDER'],
                extracted_data=details,
                status='PENDING'
            )
            document.save()
        except Exception as e:
            print(f"Error saving document: {e}")
            messages.error(request, 'Error saving document to database.')

        context = {
            'details': details,
            'image_url': fs.url(file_path),
            'document_id': document.id if 'document' in locals() else None
        }
        return render(request, 'extraction_results_page.html', context)

    return render(request, 'document_upload_page.html')

def extraction_results_view(request):
    return render(request, 'extraction_results_page.html')

def verification_result_view(request):
    return render(request, 'verification_result_page.html')

def success_page_view(request):
    return render(request, 'success_page.html')

def error_page_view(request):
    return render(request, 'error_page.html')

def admin_view_document_details(request):
    return render(request, 'admin_view_document_details_page.html')