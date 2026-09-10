# -*- coding: utf-8 -*-
"""
Django Celery 配置入口
包含 Redis Broker/Backend、自定義高精度 Decimal JSON 序列化器與連線池防護。
"""
import os
import json
from decimal import Decimal
import datetime
from celery import Celery
from kombu.serialization import register

# 設置 Django settings 模組環境變數
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'demo.settings')

# 自定義金融高精度 JSON 編碼器 (防禦要求：防止 Decimal 序列化崩潰)
class FinancialJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        elif isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        elif hasattr(obj, 'item'):
            return obj.item()
        return super().default(obj)

def financial_json_dumps(obj):
    return json.dumps(obj, cls=FinancialJSONEncoder)

def financial_json_loads(s):
    return json.loads(s)

# 註冊 kombu 自定義序列化格式
register(
    'financial_json',
    financial_json_dumps,
    financial_json_loads,
    content_type='application/x-financial-json',
    content_encoding='utf-8'
)

app = Celery('demo')

# 從 Django settings 讀取 CELERY_ 開頭的配置
app.config_from_object('django.conf:settings', namespace='CELERY')

# 防禦要求：防禦可見性超時與 Redis 連線池溢出
app.conf.update(
    task_serializer='financial_json',
    result_serializer='financial_json',
    accept_content=['financial_json', 'json'],
    task_acks_late=True,
    task_reject_on_worker_cancel=True,
    broker_transport_options={
        'max_connections': 100,
        'visibility_timeout': 43200,  # 12 小時防禦大型批次被 Redis 重複派發
    },
    worker_prefetch_multiplier=1,
)

# 自動發現所有已註冊 Django App 中的 tasks.py
app.autodiscover_tasks()

@app.task(bind=True, ignore_result=True)
def debug_task(self):
    print(f'Celery Debug Request: {self.request!r}')
