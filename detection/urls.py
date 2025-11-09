from django.urls import path
from . import views
from . import views_hybrid

urlpatterns = [
    path('', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('profile/', views.profile_view, name='profile'),
    path('detect/', views.detect_view, name='detect'),
    path('report/<int:report_id>/pdf/', views.download_report_pdf, name='download_report'),
]

