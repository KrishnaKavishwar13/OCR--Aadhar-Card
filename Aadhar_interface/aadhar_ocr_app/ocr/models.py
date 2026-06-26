from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class Document(models.Model):
    STATUS_CHOICES = [
        ('PENDING', 'Pending Verification'),
        ('VERIFIED', 'Verified'),
        ('REJECTED', 'Rejected'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    image = models.ImageField(upload_to='documents/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    
    # Extracted fields for easy access/filtering
    name = models.CharField(max_length=255, blank=True, null=True)
    aadhar_number = models.CharField(max_length=20, blank=True, null=True)
    dob = models.CharField(max_length=20, blank=True, null=True)
    gender = models.CharField(max_length=10, blank=True, null=True)
    
    # Store full JSON result for flexibility
    extracted_data = models.JSONField(blank=True, null=True)

    def __str__(self):
        username = self.user.username if self.user else "Anonymous"
        return f"{username} - {self.uploaded_at.strftime('%Y-%m-%d %H:%M')}"
