from django.contrib import admin
from .models import Document

# Register your models here.
@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ('user', 'uploaded_at', 'status', 'name', 'aadhar_number')
    list_filter = ('status', 'uploaded_at')
    search_fields = ('user__username', 'name', 'aadhar_number')
    readonly_fields = ('uploaded_at', 'extracted_data')
