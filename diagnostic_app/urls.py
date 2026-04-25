from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('api/models/', views.get_models, name='get_models'),
    path('api/preprocess/', views.preprocess, name='preprocess'),
    path('api/analyze/', views.analyze, name='analyze'),
    path('api/export-report/', views.export_report, name='export_report'),
]
