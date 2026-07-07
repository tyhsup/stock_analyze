from django.urls import path
from . import views

app_name = 'agents'

urlpatterns = [
    path('api/chat/', views.chat_api, name='chat_api'),
    path('chat-panel/', views.chat_panel_template, name='chat_panel'),
    path('audit/', views.model_audit_view, name='model_audit'),
    path('api/audit/', views.model_audit_api, name='model_audit_api'),
]
