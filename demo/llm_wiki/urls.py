from django.urls import path
from . import views

urlpatterns = [
    path('', views.wiki_home, name='wiki_home'),
    path('chat/', views.wiki_chat_api, name='wiki_chat'),
    path('index/', views.wiki_index_api, name='wiki_index'),
    path('write/', views.wiki_write_api, name='wiki_write'),
]
