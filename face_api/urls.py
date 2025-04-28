# face_api/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('reconocer/', views.reconocer, name='reconocer_rostro'),
]
