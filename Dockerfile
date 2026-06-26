FROM python:3.10-slim

# Install system dependencies including Tesseract OCR and libgl for OpenCV
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory to the project root
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire OCR-aadhar directory into the container
COPY . /app/

# Set the working directory to where manage.py is located
WORKDIR /app/Aadhar_interface/aadhar_ocr_app

# Run database migrations
RUN python manage.py migrate

# Expose the port Render expects
EXPOSE 8000

# Start Gunicorn server (timeout increased because PyTorch can take time to load)
CMD ["gunicorn", "aadhar_ocr_app.wsgi:application", "--bind", "0.0.0.0:8000", "--timeout", "120"]
