# maths/urls.py
from django.urls import path
from . import views

app_name = "maths"

urlpatterns = [
    path("", views.maths, name="maths"),
    path("graph-data/", views.graph_data, name="graph-data"),
]

