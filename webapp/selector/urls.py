from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_dataset, name='upload'),
    path('preview/', views.preview_dataset, name='preview_dataset'),
    path('analyze/', views.analyze_dataset, name='analyze'),
]
