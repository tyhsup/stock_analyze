"""
URL configuration for demo project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
#from member.views import show_member_info
#from member.views import receive_Data
#from ClassServices.views import studentMethod
from stock_Django.views import home, News_display, refresh_status_api, news_refresh_api, smart_advisor_analysis

urlpatterns = [
    path('admin/', admin.site.urls),
    path('home/', home, name='home'),
    path('', home, name='index'),
    path('news/', News_display, name='news'),
    path('chips/', include('institutional_chips.urls')),
    path('valuation/', include('valuation.urls')),
    path('api/refresh-status/<str:ticker>/', refresh_status_api, name='refresh_status'),
    path('api/news-refresh/<str:ticker>/', news_refresh_api, name='news_refresh'),
    path('api/smart-advisor/<str:ticker>/', smart_advisor_analysis, name='smart_advisor'),
]
