import os
import faiss
import numpy as np
from dotenv import load_dotenv
from .obsidian_sync import list_markdown_files, read_markdown_file

# Try importing the Google GenAI SDK (newer version)
try:
    from google import genai
    from google.genai import types
    USE_OLD_SDK = False
except ImportError:
    # Fallback to the older SDK if that's what's installed
    import google.generativeai as genai
    USE_OLD_SDK = True

# Load API key using standard Django project paths
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(base_dir, 'stock_Django', '.env')
if not os.path.exists(env_path):
    env_path = os.path.join(base_dir, '.env')
load_dotenv(env_path)

# Fallback to user home config if still not found
if not os.getenv("GEMINI_API_KEY"):
    load_dotenv(os.path.join(os.path.expanduser("~"), ".gemini", "antigravity", ".env"))

# Constants
EMBEDDING_MODEL = "models/gemini-embedding-2" # Gemini Embedding 2
CHAT_MODEL = "gemini-3.5-flash" # Gemini 3.5 Flash
INDEX_DIR = os.path.join(base_dir, 'llm_wiki', 'data')

class GeminiRAG:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            print("WARNING: GEMINI_API_KEY not found in .env")
            
        if not USE_OLD_SDK:
            self.client = genai.Client(api_key=self.api_key)
        else:
            genai.configure(api_key=self.api_key)
            
        self.index_path = os.path.join(INDEX_DIR, "wiki.index")
        self.metadata_path = os.path.join(INDEX_DIR, "metadata.npy")
        
        self.index = None
        self.documents = [] # Store metadata mapping
        
        if not os.path.exists(INDEX_DIR):
            os.makedirs(INDEX_DIR, exist_ok=True)
            
        self.load_index()

    def get_embedding(self, text):
        import time
        max_retries = 8
        base_delay = 3
        for attempt in range(max_retries):
            try:
                if not USE_OLD_SDK:
                    response = self.client.models.embed_content(
                        model=EMBEDDING_MODEL,
                        contents=text
                    )
                    emb = response.embeddings[0].values
                else:
                    result = genai.embed_content(
                        model=EMBEDDING_MODEL,
                        content=text,
                        task_type="retrieval_document"
                    )
                    emb = result['embedding']
                
                # 每次成功呼叫後強制冷卻 0.5 秒，平滑發送流量，防止突發超限
                time.sleep(0.5)
                return emb
            except Exception as e:
                # 判斷是否為 429/配額超限錯誤
                err_msg = str(e).lower()
                if "429" in err_msg or "quota" in err_msg or "limit" in err_msg or "exhausted" in err_msg:
                    delay = base_delay * (2 ** attempt)
                    print(f"遇到 API 配額超限 (429)，等待 {delay} 秒後進行重試 ({attempt + 1}/{max_retries})...")
                    time.sleep(delay)
                else:
                    raise e
        raise Exception("取得嵌入向量失敗：已達最大重試次數，配額依然超限。")

    def generate_answer(self, prompt):
        if not USE_OLD_SDK:
            response = self.client.models.generate_content(
                model=CHAT_MODEL,
                contents=prompt
            )
            return response.text
        else:
            model = genai.GenerativeModel(CHAT_MODEL)
            response = model.generate_content(prompt)
            return response.text

    def build_index(self):
        print("Building vector index from Vault...")
        files = list_markdown_files()
        
        embeddings = []
        self.documents = []
        
        next_id = 1
        for file in files:
            try:
                post = read_markdown_file(file)
                content = post.content
                if not content.strip():
                    continue
                    
                # Simple chunking
                chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
                
                for idx, chunk in enumerate(chunks):
                    emb = self.get_embedding(chunk)
                    embeddings.append(emb)
                    self.documents.append({
                        "chunk_id": next_id,
                        "file": os.path.basename(file),
                        "path": file,
                        "chunk_index": idx,
                        "text": chunk
                    })
                    next_id += 1
            except Exception as e:
                print(f"Error processing {file}: {e}")
                
        if embeddings:
            emb_matrix = np.array(embeddings).astype('float32')
            dimension = emb_matrix.shape[1]
            
            # 使用 IndexIDMap 包裹 IndexFlatL2 以便未來能進行 remove_ids
            sub_index = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIDMap(sub_index)
            
            ids = np.array([doc["chunk_id"] for doc in self.documents]).astype('int64')
            self.index.add_with_ids(emb_matrix, ids)
            
            # Save index and metadata
            faiss.write_index(self.index, self.index_path)
            np.save(self.metadata_path, self.documents)
            print(f"Built index with {len(self.documents)} chunks.")
            return len(self.documents)
        return 0

    def load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                loaded_index = faiss.read_index(self.index_path)
                self.documents = np.load(self.metadata_path, allow_pickle=True).tolist()
                
                # 自動相容升級：如果讀取出來的不是 IndexIDMap，就地升級它
                if not isinstance(loaded_index, faiss.IndexIDMap):
                    print("偵測到舊版 Flat 索引格式，開始就地升級為 IndexIDMap...")
                    dimension = loaded_index.d
                    sub_index = faiss.IndexFlatL2(dimension)
                    self.index = faiss.IndexIDMap(sub_index)
                    
                    # 重新為 documents 分配 chunk_id
                    for idx, doc in enumerate(self.documents):
                        doc["chunk_id"] = idx + 1
                        
                    # 從舊索引中抽取所有向量重建
                    if loaded_index.ntotal > 0:
                        vectors = loaded_index.reconstruct_n(0, loaded_index.ntotal)
                        ids = np.array([doc["chunk_id"] for doc in self.documents]).astype('int64')
                        self.index.add_with_ids(vectors, ids)
                        
                    # 寫回磁碟存檔，完成升級
                    faiss.write_index(self.index, self.index_path)
                    np.save(self.metadata_path, self.documents)
                    print("舊版 Flat 索引升級完成。")
                else:
                    self.index = loaded_index
                    # 確保 documents 中每個 doc 都至少有 chunk_id，防微小缺失
                    modified = False
                    for idx, doc in enumerate(self.documents):
                        if "chunk_id" not in doc:
                            doc["chunk_id"] = idx + 1
                            modified = True
                    if modified:
                        np.save(self.metadata_path, self.documents)
                
                print("FAISS index loaded successfully.")
            except Exception as e:
                print(f"載入 FAISS 索引失敗，將在下次執行時重建: {e}")
                self.index = None
                self.documents = []
        else:
            print("No FAISS index found. Please build index.")

    def incremental_update(self, changed_files):
        """僅對變更的 Markdown 檔案更新向量索引。"""
        if not changed_files:
            print("無變更檔案，跳過 RAG 增量更新。")
            return
            
        print(f"RAG 開始增量更新，變按檔案數: {len(changed_files)}")
        self.load_index()
        
        if self.index is None or not self.documents:
            print("本地無索引或 metadata 損毀，執行全量重建...")
            self.build_index()
            return
            
        # 1. 篩選出需要移除的舊 chunks IDs
        changed_basenames = [os.path.basename(f) for f in changed_files]
        old_chunks_to_remove = [doc for doc in self.documents if doc["file"] in changed_basenames]
        
        if old_chunks_to_remove:
            old_ids = [doc["chunk_id"] for doc in old_chunks_to_remove]
            print(f"正在從索引中移除舊 chunks，共計 {len(old_ids)} 個向量...")
            # 從 FAISS 索引中刪除
            self.index.remove_ids(np.array(old_ids).astype('int64'))
            # 從 metadata 列表中移除
            self.documents = [doc for doc in self.documents if doc["file"] not in changed_basenames]
            
        # 2. 讀取並處理變更的檔案，產生新 chunks 的 Embedding
        embeddings = []
        new_docs = []
        
        # 決定自增 ID 起始值
        max_id = max([doc.get("chunk_id", 0) for doc in self.documents]) if self.documents else 0
        next_id = max_id + 1
        
        for file in changed_files:
            # 如果檔案在本地被刪除了（例如刪除 Source），我們就不新增它，此時已完成 remove_ids 即可
            if not os.path.exists(file):
                print(f"檔案已在本地刪除，僅清理其向量索引: {os.path.basename(file)}")
                continue
                
            try:
                post = read_markdown_file(file)
                content = post.content
                if not content.strip():
                    continue
                
                # 分塊 (Chunking)
                chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
                for idx, chunk in enumerate(chunks):
                    emb = self.get_embedding(chunk)
                    embeddings.append(emb)
                    new_docs.append({
                        "chunk_id": next_id,
                        "file": os.path.basename(file),
                        "path": file,
                        "chunk_index": idx,
                        "text": chunk
                    })
                    next_id += 1
            except Exception as e:
                print(f"增量處理檔案失敗 {file}: {e}")
                
        # 3. 追加新向量與 metadata 並存檔
        if embeddings:
            emb_matrix = np.array(embeddings).astype('float32')
            new_ids = np.array([doc["chunk_id"] for doc in new_docs]).astype('int64')
            self.index.add_with_ids(emb_matrix, new_ids)
            self.documents.extend(new_docs)
            
        # 4. 寫回磁碟存檔
        faiss.write_index(self.index, self.index_path)
        np.save(self.metadata_path, self.documents)
        print(f"RAG 增量更新成功！新增了 {len(new_docs)} 個 chunks，目前總計 {len(self.documents)} 個 chunks。")

    def chat(self, query):
        if self.index is None or len(self.documents) == 0:
            return "知識庫尚未建立索引，請先點擊建立索引。"
            
        # 1. 偵測查詢中是否提及知識庫中的特定檔名 (不分大小寫)
        query_lower = query.lower()
        mentioned_files = set()
        matched_docs = []
        
        # 收集所有不重複的檔名
        unique_files = {}
        for doc in self.documents:
            unique_files[doc['file'].lower()] = doc['file']
            
        # 檢查是否有任何檔名被提及
        for fname_lower, fname_orig in unique_files.items():
            if fname_lower in query_lower:
                mentioned_files.add(fname_orig)
                
        # 若提及特定檔名，將該檔案的 chunks 優先加載 (限制單一檔案最多 8 個 chunks 以防 Context 爆炸)
        if mentioned_files:
            print(f"[RAG] 偵測到查詢提及特定檔案: {mentioned_files}")
            for doc in self.documents:
                if doc['file'] in mentioned_files:
                    if len([d for d in matched_docs if d['file'] == doc['file']]) < 8:
                        matched_docs.append(doc)
                        
        # 2. 進行標準向量檢索 (K 提高為 5 以豐富關聯內容)
        query_emb = np.array([self.get_embedding(query)]).astype('float32')
        k = 5
        distances, indices = self.index.search(query_emb, k)
        
        contexts = []
        citations = []
        seen_chunks = set()
        
        # 優先排入檔名匹配的內容
        for doc in matched_docs:
            chunk_key = f"{doc['file']}_{doc['chunk_index']}"
            if chunk_key not in seen_chunks:
                contexts.append(f"來源: {doc['file']}\n內容:\n{doc['text']}")
                citations.append(doc['file'])
                seen_chunks.add(chunk_key)
                
        # 補上向量相似的內容
        for idx in indices[0]:
            if idx < len(self.documents) and idx >= 0:
                doc = self.documents[idx]
                chunk_key = f"{doc['file']}_{doc['chunk_index']}"
                if chunk_key not in seen_chunks:
                    contexts.append(f"來源: {doc['file']}\n內容:\n{doc['text']}")
                    citations.append(doc['file'])
                    seen_chunks.add(chunk_key)
                    
        context_str = "\n\n---\n\n".join(contexts)
        
        prompt = f"""
        你是一個知識庫問答機器人。請根據以下檢索到的參考內容，來回答使用者的問題。
        如果參考內容中無法找到答案，請明確告知「知識庫中無法確認」。
        在回答時，請引用來源檔案名稱。
        
        參考內容：
        {context_str}
        
        問題：
        {query}
        """
        
        answer = self.generate_answer(prompt)
        
        # Deduplicate citations
        unique_citations = list(set(citations))
        
        return {
            "answer": answer,
            "citations": unique_citations
        }

rag_service = GeminiRAG()
