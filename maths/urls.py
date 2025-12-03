from django.urls import path
from . import views

urlpatterns = [
    path('dashboard/maths', views.maths, name='maths'),
]
