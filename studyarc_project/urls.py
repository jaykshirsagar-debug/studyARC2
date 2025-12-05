# studyarc_project/urls.py
from django.contrib import admin
from django.urls import path, include
from core import views as core_views
from accounts import views as accounts_views

urlpatterns = [
    path('admin/', admin.site.urls),

    path('', core_views.home, name='home'),
    path('dashboard/', core_views.dashboard, name='dashboard'),

    # auth
    path('login/', accounts_views.login_view, name='login'),
    path('logout/', accounts_views.logout_view, name='logout'),
    path('register/', accounts_views.register_view, name='register'),

    # maths app (namespaced)
    path('maths/', include(('maths.urls', 'maths'), namespace='maths')),
]
