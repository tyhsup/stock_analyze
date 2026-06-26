from django.urls import path
from .views import chips_view, refresh_tw_api, refresh_us_api, refresh_status_api, api_industry_flow, api_us_stocks_list
from .us_chips_views import api_us_money_flow

urlpatterns = [
    path('', chips_view, name='chips'),
    path('api/refresh/tw/', refresh_tw_api, name='chips_refresh_tw'),
    path('api/refresh/us/', refresh_us_api, name='chips_refresh_us'),
    path('api/refresh-status/<str:market>/', refresh_status_api, name='chips_refresh_status'),
    path('api/industry-flow/', api_industry_flow, name='api_industry_flow'),
    path('api/us-stocks/', api_us_stocks_list, name='api_us_stocks_list'),
    path('api/us-money-flow/', api_us_money_flow, name='api_us_money_flow'),
]
