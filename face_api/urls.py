# face_api/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('home/', views.homepage, name='homepage'),
    path('reconocer/', views.reconocer, name='reconocer_rostro'),
]
