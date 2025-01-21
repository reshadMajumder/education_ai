from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('process-frame/', views.process_frame, name='process_frame'),
    path('clear-canvas/', views.clear_canvas, name='clear_canvas'),
]