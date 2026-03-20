"""institutional_chips/urls.py"""
from django.urls import path
from .views import chips_view, refresh_tw_api, refresh_us_api, refresh_status_api

urlpatterns = [
    path('', chips_view, name='chips'),
    path('api/refresh/tw/', refresh_tw_api, name='chips_refresh_tw'),
    path('api/refresh/us/', refresh_us_api, name='chips_refresh_us'),
    path('api/refresh-status/<str:market>/', refresh_status_api, name='chips_refresh_status'),
]
