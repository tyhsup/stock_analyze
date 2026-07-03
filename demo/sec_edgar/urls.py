from django.urls import path
from .views import (
    dashboard_view, 
    api_13f_holdings, 
    api_insider_trades, 
    api_sync_13f, 
    api_sync_insider,
    api_financial_data,
    api_institution_holdings,
    api_sync_institution_13f
)

urlpatterns = [
    path('', dashboard_view, name='sec_edgar_dashboard'),
    path('api/13f/<str:ticker>/', api_13f_holdings, name='api_13f_holdings'),
    path('api/insider/<str:ticker>/', api_insider_trades, name='api_insider_trades'),
    path('api/financial/<str:ticker>/', api_financial_data, name='api_financial_data'),
    path('api/sync/13f/', api_sync_13f, name='api_sync_13f'),
    path('api/sync/insider/', api_sync_insider, name='api_sync_insider'),
    path('api/institution/holdings/', api_institution_holdings, name='api_institution_holdings'),
    path('api/sync/institution/', api_sync_institution_13f, name='api_sync_institution_13f'),
]
