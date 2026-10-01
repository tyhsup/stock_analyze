# -*- coding: utf-8 -*-
"""
系統健康檢查視圖 (Health Check Endpoint)
支援 Liveness (進程存活) 與 Readiness (資料庫與快取就緒) 探測。
具備敏感資訊防洩漏保護與例外安全機制。
"""
import logging
from django.http import JsonResponse
from django.db import connection
from django.conf import settings

logger = logging.getLogger(__name__)

def health_check(request):
    """
    健康檢查端點：
    - ?type=liveness: 僅檢測 Web 容器進程是否正常響應
    - 預設 / ?type=readiness: 檢測核心依賴 (MySQL 資料庫與 Redis Broker)
    """
    probe_type = request.GET.get('type', 'readiness').lower()

    # 1. 存活探測 (Liveness Probe)
    if probe_type == 'liveness':
        return JsonResponse({"status": "alive", "service": "django_web"}, status=200)

    # 2. 就緒探測 (Readiness Probe)
    checks = {
        "database": "unknown",
        "redis": "unknown"
    }
    all_healthy = True

    # 檢測 MySQL 連線
    try:
        connection.ensure_connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()
        checks["database"] = "ok"
    except Exception as e:
        all_healthy = False
        checks["database"] = "unreachable"
        # 僅記錄在伺服器端內部日誌，不對外公開敏感錯誤或連線憑證
        logger.error(f"[HealthCheck] Database connection check failed: {type(e).__name__}")

    # 檢測 Redis 連線
    try:
        broker_url = getattr(settings, 'CELERY_BROKER_URL', 'redis://127.0.0.1:6379/0')
        import redis
        client = redis.Redis.from_url(broker_url, socket_timeout=2, socket_connect_timeout=2)
        if client.ping():
            checks["redis"] = "ok"
        else:
            all_healthy = False
            checks["redis"] = "ping_failed"
    except Exception as e:
        all_healthy = False
        checks["redis"] = "unreachable"
        logger.error(f"[HealthCheck] Redis connection check failed: {type(e).__name__}")

    response_data = {
        "status": "healthy" if all_healthy else "unhealthy",
        "service": "django_web",
        "checks": checks
    }

    status_code = 200 if all_healthy else 503
    return JsonResponse(response_data, status=status_code)
