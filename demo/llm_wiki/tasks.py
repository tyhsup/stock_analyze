# -*- coding: utf-8 -*-
"""
LLM Wiki Celery Tasks
提供排程自動執行知識庫索引差異同步。
"""

import logging
from celery import shared_task
from .services.gemini_rag import rag_service

logger = logging.getLogger(__name__)


@shared_task(bind=True, name="llm_wiki.tasks.sync_wiki_index_task", max_retries=2, default_retry_delay=300)
def sync_wiki_index_task(self, dry_run=False):
    """
    Celery Beat 定時觸發的 Wiki 索引差異同步任務。
    """
    logger.info(f"[Celery] 開始執行 Wiki 索引差異同步 (dry_run={dry_run})...")
    try:
        summary = rag_service.sync_index(dry_run=dry_run)
        logger.info(f"[Celery] Wiki 索引同步成功: status={summary['status']}, added={summary['added_count']}, modified={summary['modified_count']}, deleted={summary['deleted_count']}")
        return {
            "status": summary["status"],
            "added": summary["added_count"],
            "modified": summary["modified_count"],
            "deleted": summary["deleted_count"],
            "total_chunks": summary.get("indexed_chunks", 0)
        }
    except Exception as exc:
        logger.error(f"[Celery] Wiki 索引同步任務失敗: {exc}")
        raise self.retry(exc=exc)
