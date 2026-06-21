from django.urls import path
from . import views

urlpatterns = [
    path('', views.valuation_root, name='valuation_root'),
    path('<str:symbol>/', views.valuation_view, name='valuation_detail'),
    path('api/<str:symbol>/', views.valuation_api, name='valuation_api'),
    path('api/<str:symbol>/status/', views.valuation_refresh_status_api, name='valuation_refresh_status_api'),
]
