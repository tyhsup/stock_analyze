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
        if not USE_OLD_SDK:
            response = self.client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text
            )
            return response.embeddings[0].values
        else:
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']

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
        
        for file in files:
            try:
                post = read_markdown_file(file)
                content = post.content
                if not content.strip():
                    continue
                    
                # Simple chunking (for production, use LangChain or LlamaIndex chunkers)
                chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
                
                for idx, chunk in enumerate(chunks):
                    emb = self.get_embedding(chunk)
                    embeddings.append(emb)
                    self.documents.append({
                        "file": os.path.basename(file),
                        "path": file,
                        "chunk_index": idx,
                        "text": chunk
                    })
            except Exception as e:
                print(f"Error processing {file}: {e}")
                
        if embeddings:
            emb_matrix = np.array(embeddings).astype('float32')
            # FAISS inner product for cosine similarity (assuming normalized, but let's use L2 for safety if unnormalized)
            self.index = faiss.IndexFlatL2(emb_matrix.shape[1])
            self.index.add(emb_matrix)
            
            # Save index and metadata
            faiss.write_index(self.index, self.index_path)
            np.save(self.metadata_path, self.documents)
            print(f"Built index with {len(self.documents)} chunks.")
            return len(self.documents)
        return 0

    def load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            self.documents = np.load(self.metadata_path, allow_pickle=True).tolist()
            print("FAISS index loaded.")
        else:
            print("No FAISS index found. Please build index.")

    def chat(self, query):
        if self.index is None or len(self.documents) == 0:
            return "知識庫尚未建立索引，請先點擊建立索引。"
            
        # Get query embedding
        query_emb = np.array([self.get_embedding(query)]).astype('float32')
        
        # Search Top K
        k = 3
        distances, indices = self.index.search(query_emb, k)
        
        contexts = []
        citations = []
        
        for idx in indices[0]:
            if idx < len(self.documents) and idx >= 0:
                doc = self.documents[idx]
                contexts.append(f"來源: {doc['file']}\n內容:\n{doc['text']}")
                citations.append(doc['file'])
                
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
