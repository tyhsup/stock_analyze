# -*- coding: utf-8 -*-
"""
LLM Wiki Views
提供知識庫瀏覽、RAG 問答、手動差異同步與即時筆記寫入 API。
"""

import json
import logging
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .services.gemini_rag import rag_service
from .services.obsidian_sync import write_markdown_file, list_markdown_files

logger = logging.getLogger(__name__)


def wiki_home(request):
    """Wiki 首頁，提供目錄樹與同步狀態展示。"""
    files = list_markdown_files()
    rag_service.check_and_reload()

    # 取得輕量差異統計
    pending_diff = {"added": [], "modified": [], "deleted": []}
    try:
        diff_res = rag_service.diff_vault()
        pending_diff = {
            "added": len(diff_res.get("added", [])),
            "modified": len(diff_res.get("modified", [])),
            "deleted": len(diff_res.get("deleted", [])),
        }
    except Exception as e:
        logger.warning(f"[WikiViews] 計算差異統計失敗 (非阻斷): {e}")

    context = {
        'files': files,
        'index_status': "已載入" if rag_service.index else "未建立",
        'document_count': len(rag_service.documents),
        'pending_added': pending_diff.get("added", 0),
        'pending_modified': pending_diff.get("modified", 0),
        'pending_deleted': pending_diff.get("deleted", 0),
        'has_pending': (pending_diff.get("added", 0) + pending_diff.get("modified", 0) + pending_diff.get("deleted", 0)) > 0
    }
    return render(request, 'llm_wiki/index.html', context)


@csrf_exempt
def wiki_chat_api(request):
    """RAG 問答 API"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            query = data.get('query', '').strip()
            if not query:
                return JsonResponse({'error': 'Query is empty'}, status=400)

            result = rag_service.chat(query)
            return JsonResponse(result)
        except Exception as e:
            logger.error(f"[WikiViews] 問答 API 異常: {e}")
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request'}, status=400)


@csrf_exempt
def wiki_sync_api(request):
    """
    知識庫差異同步 API (支援 dry_run 預檢與真實同步)
    POST /wiki/sync/
    { "dry_run": false }
    """
    if request.method in ['POST', 'GET']:
        dry_run = False
        if request.method == 'POST':
            try:
                if request.body:
                    data = json.loads(request.body)
                    dry_run = bool(data.get('dry_run', False))
            except Exception:
                pass
        else:
            dry_run = request.GET.get('dry_run', 'false').lower() in ('true', '1')

        try:
            summary = rag_service.sync_index(dry_run=dry_run)
            safe_response = {
                "status": str(summary.get("status", "success")),
                "added_count": int(summary.get("added_count", 0)),
                "modified_count": int(summary.get("modified_count", 0)),
                "deleted_count": int(summary.get("deleted_count", 0)),
                "unchanged_count": int(summary.get("unchanged_count", 0)),
                "total_vault": int(summary.get("total_vault", 0)),
                "total_indexed": int(summary.get("total_indexed", 0)),
                "chunks_added": int(summary.get("chunks_added", 0)),
                "chunks_removed": int(summary.get("chunks_removed", 0)),
                "indexed_chunks": int(summary.get("indexed_chunks", 0)),
                "added_files": [str(f) for f in summary.get("added_files", [])[:10]],
                "modified_files": [str(f) for f in summary.get("modified_files", [])[:10]],
                "deleted_files": [str(f) for f in summary.get("deleted_files", [])[:10]],
            }
            return JsonResponse(safe_response)
        except Exception as e:
            logger.error(f"[WikiViews] 差異同步失敗: {e}")
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)


@csrf_exempt
def wiki_index_api(request):
    """全量重建索引 API (建議改用 wiki_sync_api)"""
    if request.method == 'POST':
        try:
            count = rag_service.build_index()
            return JsonResponse({'status': 'success', 'chunks_indexed': count})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request'}, status=400)


@csrf_exempt
def wiki_write_api(request):
    """寫入筆記並自動進行增量更新"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            sub_path = data.get('path')
            content = data.get('content')
            metadata = data.get('metadata', {})

            if not sub_path or not content:
                return JsonResponse({'error': 'path and content are required'}, status=400)

            filepath = write_markdown_file(sub_path, content, metadata)

            # 即時增量更新該檔案之向量索引
            try:
                rag_service.incremental_update([filepath])
            except Exception as e:
                logger.warning(f"[WikiViews] 寫入後增量索引失敗 (非阻斷): {e}")

            return JsonResponse({'status': 'success', 'filepath': filepath})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request'}, status=400)
