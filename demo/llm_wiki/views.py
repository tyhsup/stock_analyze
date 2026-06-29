from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

from .services.gemini_rag import rag_service
from .services.obsidian_sync import write_markdown_file, list_markdown_files

def wiki_home(request):
    files = list_markdown_files()
    context = {
        'files': files,
        'index_status': "已載入" if rag_service.index else "未建立",
        'document_count': len(rag_service.documents)
    }
    return render(request, 'llm_wiki/index.html', context)

@csrf_exempt
def wiki_chat_api(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            query = data.get('query', '')
            if not query:
                return JsonResponse({'error': 'Query is empty'}, status=400)
                
            result = rag_service.chat(query)
            return JsonResponse(result)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request'}, status=400)

@csrf_exempt
def wiki_index_api(request):
    if request.method == 'POST':
        try:
            count = rag_service.build_index()
            return JsonResponse({'status': 'success', 'chunks_indexed': count})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request'}, status=400)

@csrf_exempt
def wiki_write_api(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            sub_path = data.get('path')
            content = data.get('content')
            metadata = data.get('metadata', {})
            
            if not sub_path or not content:
                return JsonResponse({'error': 'path and content are required'}, status=400)
                
            filepath = write_markdown_file(sub_path, content, metadata)
            
            # Optionally trigger index rebuild in background
            # rag_service.build_index()
            
            return JsonResponse({'status': 'success', 'filepath': filepath})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request'}, status=400)
