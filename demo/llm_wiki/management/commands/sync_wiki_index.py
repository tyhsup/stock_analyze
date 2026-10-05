# -*- coding: utf-8 -*-
"""
Django Management Command: sync_wiki_index
執行 LLM Wiki 知識庫 SHA-256 差異比對與增量向量同步。
"""

import sys
import logging
from django.core.management.base import BaseCommand
from llm_wiki.services.gemini_rag import rag_service

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "執行 LLM Wiki 知識庫 SHA-256 差異比對與向量同步 (支援 --dry-run)"

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="僅執行差異掃描並輸出報告，不呼叫 Embedding API 且不寫入索引",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=20,
            help="分批提交之文件數量 (預設: 20)",
        )

    def handle(self, *args, **options):
        dry_run = options.get("dry_run", False)
        batch_size = options.get("batch_size", 20)

        self.stdout.write(self.style.NOTICE(f"[sync_wiki_index] 啟動 Wiki 索引同步 (模式: {'DRY-RUN' if dry_run else '真實同步'})..."))

        def progress_cb(current, total, file_name):
            if current % 5 == 0 or current == total:
                self.stdout.write(f"  -> 進度: [{current}/{total}] 正在處理: {file_name}")

        try:
            summary = rag_service.sync_index(
                dry_run=dry_run,
                batch_size=batch_size,
                on_progress=progress_cb
            )

            self.stdout.write("=" * 60)
            self.stdout.write(self.style.SUCCESS(f"同步狀態: {summary['status']}"))
            self.stdout.write(f"Vault 實體檔案數: {summary['total_vault']}")
            self.stdout.write(f"索引收錄檔案數: {summary['total_indexed']}")
            self.stdout.write(f"未變更文件數: {summary['unchanged_count']}")
            self.stdout.write(self.style.WARNING(f"待新增文件數 (Added): {summary['added_count']}"))
            self.stdout.write(self.style.WARNING(f"待更新文件數 (Modified): {summary['modified_count']}"))
            self.stdout.write(self.style.NOTICE(f"待清理文件數 (Deleted): {summary['deleted_count']}"))

            if not dry_run:
                self.stdout.write(self.style.SUCCESS(f"已清理過期 Chunks: {summary.get('chunks_removed', 0)}"))
                self.stdout.write(self.style.SUCCESS(f"已新增向量 Chunks: {summary.get('chunks_added', 0)}"))
                self.stdout.write(self.style.SUCCESS(f"當前索引總 Chunks: {summary.get('indexed_chunks', 0)}"))
                if summary.get("backup_path"):
                    self.stdout.write(f"備份目錄: {summary['backup_path']}")
            else:
                self.stdout.write(self.style.NOTICE("[DRY-RUN 結束] 未消耗任何 Embedding 配額，未修改索引檔。"))

            self.stdout.write("=" * 60)

        except Exception as e:
            self.stderr.write(self.style.ERROR(f"[sync_wiki_index] 執行失敗: {e}"))
            sys.exit(1)
