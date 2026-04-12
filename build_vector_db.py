import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# 初始化
client = chromadb.PersistentClient(path="./chroma_db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_or_create_collection(name="codebase_memory", embedding_function=ef)

# 針對 Python 代碼優化的切分器
splitter = RecursiveCharacterTextSplitter.from_language(
    language="python", 
    chunk_size=1000,   # 每個代碼區塊約 1000 字
    chunk_overlap=100  # 區塊間保留 100 字重疊，避免邏輯斷裂
)

def process_codebase(directory):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".py"):
                path = os.path.join(root, filename)
                with open(path, 'r', encoding='utf-8') as f:
                    code = f.read()
                    # 進行智慧切分
                    chunks = splitter.split_text(code)
                    
                    for i, chunk in enumerate(chunks):
                        collection.add(
                            documents=[chunk],
                            metadatas=[{"file": filename, "type": "code"}],
                            ids=[f"{filename}_chunk_{i}"]
                        )
    print("代碼庫向量化完成！")

if __name__ == "__main__":
    # 指向你的程式碼資料夾
    process_codebase("./myDjango")