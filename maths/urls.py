# maths/urls.py
from django.urls import path
from . import views

app_name = "maths"

urlpatterns = [
    path("", views.classpad_main, name="main"),  # /maths/
    
]
