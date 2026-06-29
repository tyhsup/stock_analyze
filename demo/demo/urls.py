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
from stock_Django.views import (
    home, News_display, refresh_status_api, news_refresh_api, 
    smart_advisor_analysis, gemini_advisor_analysis,
    macrotrends_financials_api, macrotrends_ratios_api
)
from stock_Django.scheduler_views import (
    scheduler_home, jobs_list_or_create, trigger_job_immediately,
    delete_job, update_job_heartbeat, llm_parse_prompt, get_llm_usage
)

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
    path('api/gemini-advisor/<str:ticker>/', gemini_advisor_analysis, name='gemini_advisor'),
    path('api/macrotrends/financials', macrotrends_financials_api, name='macrotrends_financials'),
    path('api/macrotrends/ratios', macrotrends_ratios_api, name='macrotrends_ratios'),
    
    # 整合排程器 Dashboard 路由
    path('scheduler/', scheduler_home, name='scheduler_home'),
    path('scheduler/api/jobs', jobs_list_or_create, name='scheduler_list_jobs'),
    path('scheduler/api/jobs/<int:job_id>', delete_job, name='scheduler_delete_job'),
    path('scheduler/api/jobs/execute/<int:job_id>', trigger_job_immediately, name='scheduler_execute_job'),
    path('scheduler/api/jobs/heartbeat/<int:job_id>', update_job_heartbeat, name='scheduler_heartbeat_job'),
    path('scheduler/api/llm/parse', llm_parse_prompt, name='scheduler_llm_parse'),
    path('scheduler/api/llm/usage', get_llm_usage, name='scheduler_llm_usage'),
    path('sec-edgar/', include('sec_edgar.urls')),
    path('wiki/', include('llm_wiki.urls')),
]
