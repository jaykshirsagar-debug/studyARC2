# studyarc_project/urls.py
from django.contrib import admin
from django.urls import path, include
from core import views as core_views
from accounts import views as accounts_views

urlpatterns = [
    path('admin/', admin.site.urls),

    path('', core_views.home, name='home'),
    path('dashboard/', core_views.dashboard, name='dashboard'),

    # maths app (namespaced)
    path('maths/', include(('maths.urls', 'maths'), namespace='maths')),
    path('accounts/', include(('accounts.urls', 'accounts'), namespace='accounts'))
    
]
