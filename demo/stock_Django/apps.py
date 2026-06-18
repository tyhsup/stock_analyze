from django.apps import AppConfig
import os
import sys


class StockDjangoConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'stock_Django'

    def ready(self):
        # 僅在主進程中啟動背景排程器（防止 runserver reload 模式重複啟動）
        if os.environ.get('RUN_MAIN') == 'true':
            from django.conf import settings
            GEMINI_TASK_DIR = str(settings.BASE_DIR.parent / "Gemini_task")
            if GEMINI_TASK_DIR not in sys.path:
                sys.path.insert(0, GEMINI_TASK_DIR)
                
            from app.scheduler import start_scheduler
            import logging
            logger = logging.getLogger("scheduler.main")
            logger.info("[Django Ready] 正在啟動排程器背景線程...")
            start_scheduler(num_workers=2)
