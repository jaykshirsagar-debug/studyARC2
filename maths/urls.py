# maths/urls.py
from django.urls import path
from . import views

app_name = "maths"

urlpatterns = [
    # Main ClassPad screen
    path("", views.maths, name="maths"),

]
