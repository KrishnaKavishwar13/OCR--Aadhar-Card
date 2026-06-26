
from django.urls import path
from . import views

urlpatterns = [
    path('', views.document_upload_view, name='home'),

    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('upload/', views.document_upload_view, name='document_upload'),
    path('results/', views.extraction_results_view, name='results'),
    path('verification/', views.verification_result_view, name='verification_result'),
    path('success/', views.success_page_view, name='success'),
    path('error/', views.error_page_view, name='error'),
    
    # Admin routes
    path('admin-login/', views.admin_login_view, name='admin_login'),
    path('admin-dashboard/', views.admin_dashboard_view, name='admin_dashboard'),
    path('admin-document/', views.admin_view_document_details, name='admin_document_details'),
]
