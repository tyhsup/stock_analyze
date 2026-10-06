# -*- coding: utf-8 -*-
"""
LLM Wiki Gemini RAG Service
支援 FAISS 向量檢索、SHA-256 差異比對、跨平台路徑解析、原子替換與進程間熱重載。
"""

import os
import sys
import glob
import time
import shutil
import hashlib
import logging
from datetime import datetime
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Tuple

import faiss
import numpy as np
from dotenv import load_dotenv

from .obsidian_sync import list_markdown_files, read_markdown_file, get_vault_dir

logger = logging.getLogger(__name__)

# Try importing the Google GenAI SDK (newer version)
try:
    from google import genai
    from google.genai import types
    USE_OLD_SDK = False
except ImportError:
    import google.generativeai as genai
    USE_OLD_SDK = True

# Load API key using standard Django project paths
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(base_dir, 'stock_Django', '.env')
if not os.path.exists(env_path):
    env_path = os.path.join(base_dir, '.env')
load_dotenv(env_path)

if not os.getenv("GEMINI_API_KEY"):
    load_dotenv(os.path.join(os.path.expanduser("~"), ".gemini", "antigravity", ".env"))

# Constants
EMBEDDING_MODEL = "models/gemini-embedding-2"
CHAT_MODEL = "gemini-3.5-flash"
INDEX_DIR = os.path.join(base_dir, 'llm_wiki', 'data')
BACKUP_DIR = os.path.join(INDEX_DIR, 'backup')
LOCK_FILE = os.path.join(INDEX_DIR, '.sync.lock')

# 檔案鎖相容性處理 (Linux 容器原生支援 fcntl，Windows 本機防禦回退)
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    fcntl = None
    HAS_FCNTL = False


def compute_content_hash(text: str) -> str:
    """計算文本內容之 SHA-256 雜湊值 (UTF-8 編碼)。"""
    if not text:
        return hashlib.sha256(b"").hexdigest()
    if isinstance(text, str):
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    return hashlib.sha256(str(text).encode("utf-8", errors="replace")).hexdigest()


class GeminiRAG:
    """Gemini RAG 向量知識庫服務類別"""

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("[GeminiRAG] WARNING: GEMINI_API_KEY not found in .env")

        if not USE_OLD_SDK:
            self.client = genai.Client(api_key=self.api_key) if self.api_key else None
        else:
            if self.api_key:
                genai.configure(api_key=self.api_key)
            self.client = None

        self.index_path = os.path.join(INDEX_DIR, "wiki.index")
        self.metadata_path = os.path.join(INDEX_DIR, "metadata.npy")

        self.index = None
        self.documents: List[Dict[str, Any]] = []
        self.last_metadata_mtime: float = 0.0

        if not os.path.exists(INDEX_DIR):
            os.makedirs(INDEX_DIR, exist_ok=True)
        if not os.path.exists(BACKUP_DIR):
            os.makedirs(BACKUP_DIR, exist_ok=True)

        self.load_index()

    @contextmanager
    def _file_lock(self, timeout: float = 5.0):
        """
        獲取同步排他鎖，防止 web 與 celery_worker 併發寫入損壞索引。
        """
        os.makedirs(INDEX_DIR, exist_ok=True)
        lock_fd = None
        start_time = time.time()
        acquired = False

        if HAS_FCNTL:
            try:
                lock_fd = open(LOCK_FILE, "w")
                while time.time() - start_time < timeout:
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        acquired = True
                        break
                    except (IOError, OSError):
                        time.sleep(0.2)

                if not acquired:
                    raise TimeoutError(f"無法在 {timeout} 秒內取得同步檔案鎖 {LOCK_FILE}，其他程序正在同步中。")
                yield
            finally:
                if lock_fd:
                    try:
                        if acquired:
                            fcntl.flock(lock_fd, fcntl.LOCK_UN)
                        lock_fd.close()
                    except Exception:
                        pass
        else:
            # Windows 或無 fcntl 環境：簡易檔案鎖
            while time.time() - start_time < timeout:
                try:
                    lock_fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                    acquired = True
                    break
                except (FileExistsError, OSError):
                    time.sleep(0.2)

            if not acquired:
                raise TimeoutError(f"其他同步程序正在進行中 (鎖檔案存在: {LOCK_FILE})。")
            try:
                yield
            finally:
                if acquired and lock_fd is not None:
                    try:
                        os.close(lock_fd)
                        if os.path.exists(LOCK_FILE):
                            os.remove(LOCK_FILE)
                    except Exception:
                        pass

    def check_and_reload(self) -> bool:
        """
        跨進程熱重載偵測：若磁碟上 metadata.npy 被其他行程更新，則就地重載索引。
        """
        if not os.path.exists(self.metadata_path) or not os.path.exists(self.index_path):
            return False
        try:
            disk_mtime = os.path.getmtime(self.metadata_path)
            if disk_mtime > self.last_metadata_mtime:
                logger.info(f"[GeminiRAG] 偵測到磁碟索引更新 (mtime: {disk_mtime} > {self.last_metadata_mtime})，執行熱重載...")
                self.load_index()
                return True
        except Exception as e:
            logger.warning(f"[GeminiRAG] 檢查磁碟索引狀態異常: {e}")
        return False

    def get_embedding(self, text: str) -> List[float]:
        """呼叫 Gemini Embedding API 生成 768 維度向量，具備 8 次指數退避與速率保護。"""
        max_retries = 8
        base_delay = 3
        clean_text = (text or "").strip()
        if not clean_text:
            clean_text = "Empty content"

        for attempt in range(max_retries):
            try:
                if not USE_OLD_SDK:
                    result = self.client.models.embed_content(
                        model=EMBEDDING_MODEL,
                        contents=clean_text
                    )
                    return result.embeddings[0].values
                else:
                    result = genai.embed_content(
                        model=EMBEDDING_MODEL,
                        content=clean_text
                    )
                    return result['embedding']
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "quota" in err_str.lower() or "resource" in err_str.lower():
                    wait_time = base_delay * (2 ** attempt)
                    logger.warning(f"[GeminiRAG] Embedding 速率限制 (429/配額)，正在進行第 {attempt+1}/{max_retries} 次重試，等待 {wait_time} 秒...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"[GeminiRAG] Embedding 生成失敗 (非429): {e}")
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(2)
        raise RuntimeError("[GeminiRAG] 超過最大重試次數，無法取得 Embedding。")

    def generate_answer(self, prompt: str) -> str:
        """生成回答內容"""
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

    def backup_index(self) -> Optional[str]:
        """
        在寫入或修改索引前，對現有 wiki.index 與 metadata.npy 進行快照備份。
        保留最近 5 份備份，清理更早的備份。
        """
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            return None

        os.makedirs(BACKUP_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_dir = os.path.join(BACKUP_DIR, f"backup_{ts}")
        os.makedirs(target_dir, exist_ok=True)

        try:
            shutil.copy2(self.index_path, os.path.join(target_dir, "wiki.index"))
            shutil.copy2(self.metadata_path, os.path.join(target_dir, "metadata.npy"))
            logger.info(f"[GeminiRAG] 已成功備份當前索引至: {target_dir}")

            # 清理過期備份 (保留最新 5 個)
            all_backups = sorted(
                [os.path.join(BACKUP_DIR, d) for d in os.listdir(BACKUP_DIR) if os.path.isdir(os.path.join(BACKUP_DIR, d))],
                key=os.path.getmtime
            )
            if len(all_backups) > 5:
                for old in all_backups[:-5]:
                    shutil.rmtree(old, ignore_errors=True)
            return target_dir
        except Exception as e:
            logger.warning(f"[GeminiRAG] 備份索引失敗 (非阻斷): {e}")
            return None

    def _atomic_save(self, index: Any, documents: List[Dict[str, Any]]):
        """
        原子寫入保證 (Atomic Replacement)：
        透過寫入臨時檔案並使用 os.replace 原子替換，防止寫入中斷損毀索引檔案。
        """
        os.makedirs(INDEX_DIR, exist_ok=True)
        tmp_index = os.path.join(INDEX_DIR, "wiki.index.tmp")
        tmp_meta = os.path.join(INDEX_DIR, "metadata.tmp.npy")

        # 1. 寫入暫存檔
        faiss.write_index(index, tmp_index)
        np.save(tmp_meta, documents)

        # 2. 原子替換
        os.replace(tmp_index, self.index_path)
        os.replace(tmp_meta, self.metadata_path)

        self.index = index
        self.documents = documents
        if os.path.exists(self.metadata_path):
            self.last_metadata_mtime = os.path.getmtime(self.metadata_path)

    def load_index(self):
        """從磁碟載入 FAISS 索引與 metadata，支援舊格式自動升級與跨平台路徑正規化。"""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                loaded_index = faiss.read_index(self.index_path)
                self.documents = np.load(self.metadata_path, allow_pickle=True).tolist()

                # 自動相容升級：如果讀取出來的不是 IndexIDMap，就地升級它
                if not isinstance(loaded_index, faiss.IndexIDMap):
                    logger.info("[GeminiRAG] 偵測到舊版 Flat 索引格式，就地升級為 IndexIDMap...")
                    dimension = loaded_index.d
                    sub_index = faiss.IndexFlatL2(dimension)
                    self.index = faiss.IndexIDMap(sub_index)

                    for idx, doc in enumerate(self.documents):
                        doc["chunk_id"] = idx + 1

                    if loaded_index.ntotal > 0:
                        vectors = loaded_index.reconstruct_n(0, loaded_index.ntotal)
                        ids = np.array([doc["chunk_id"] for doc in self.documents]).astype('int64')
                        self.index.add_with_ids(vectors, ids)

                    self._atomic_save(self.index, self.documents)
                    logger.info("[GeminiRAG] 舊版 Flat 索引升級完成。")
                else:
                    self.index = loaded_index
                    modified = False
                    for idx, doc in enumerate(self.documents):
                        if "chunk_id" not in doc:
                            doc["chunk_id"] = idx + 1
                            modified = True
                    if modified:
                        np.save(self.metadata_path, self.documents)

                self._normalize_document_paths()
                if os.path.exists(self.metadata_path):
                    self.last_metadata_mtime = os.path.getmtime(self.metadata_path)
                logger.info(f"[GeminiRAG] FAISS index loaded successfully with {len(self.documents)} chunks.")
            except Exception as e:
                logger.error(f"[GeminiRAG] 載入 FAISS 索引失敗: {e}")
                self.index = None
                self.documents = []
        else:
            logger.info("[GeminiRAG] No FAISS index found. Please build or sync index.")

    def _normalize_document_paths(self):
        """
        跨平台路徑動態轉譯 (Windows vs Docker Linux)：
        將 metadata 內部可能殘留的 Windows 實體絕對路徑轉換為當前環境可用的路徑。
        """
        current_vault = get_vault_dir()
        needs_save = False

        for doc in self.documents:
            raw_path = str(doc.get("path", ""))
            rel_path = doc.get("rel_path")

            if not rel_path:
                norm_raw = raw_path.replace("\\", "/")
                k_idx = norm_raw.find("/knowledge/")
                if k_idx != -1:
                    rel_path = norm_raw[k_idx + len("/knowledge/"):].lstrip("/")
                elif ":" in norm_raw:
                    rel_path = norm_raw.split(":", 1)[1].lstrip("/")
                else:
                    rel_path = doc.get("file", os.path.basename(raw_path))

                doc["rel_path"] = rel_path
                needs_save = True

            doc["path"] = os.path.normpath(os.path.join(current_vault, rel_path))

        if needs_save:
            try:
                np.save(self.metadata_path, self.documents)
                if os.path.exists(self.metadata_path):
                    self.last_metadata_mtime = os.path.getmtime(self.metadata_path)
                logger.info(f"[GeminiRAG] 已成功正規化 {len(self.documents)} 筆 Chunks 之相對路徑。")
            except Exception as e:
                logger.warning(f"[GeminiRAG] 儲存正規化 metadata 異常 (非阻斷): {e}")

    def diff_vault(self) -> Dict[str, Any]:
        """
        掃描 Vault 實體與現有 FAISS Metadata 進行 SHA-256 內容雜湊比對。
        回傳結構:
        {
            "added": [rel_path, ...],
            "modified": [rel_path, ...],
            "deleted": [rel_path, ...],
            "unchanged_count": int,
            "total_vault": int,
            "total_indexed": int,
            "vault_hashes": {rel_path: hash},
            "meta_hashes": {rel_path: hash}
        }
        特別注意：舊 chunks 若無 content_hash，會依 chunk_index 拼接舊 text 算出基準 hash 並回填，
        避免 33,987 筆 chunks 被誤判為 modified！
        """
        self.check_and_reload()
        current_vault = get_vault_dir()
        all_vault_files = list_markdown_files()

        # 1. 整理當前 Vault 實體檔案之相對路徑與 SHA-256 雜湊
        vault_files_map: Dict[str, str] = {}  # rel_path -> full_path
        vault_hashes: Dict[str, str] = {}     # rel_path -> sha256(post.content)

        for full_p in all_vault_files:
            try:
                rel = os.path.relpath(full_p, current_vault).replace("\\", "/")
                vault_files_map[rel] = full_p
                post = read_markdown_file(full_p)
                content = post.content if post and hasattr(post, "content") else ""
                vault_hashes[rel] = compute_content_hash(content)
            except Exception as e:
                logger.warning(f"[GeminiRAG] 讀取 Vault 檔案雜湊失敗 {full_p}: {e}")

        # 2. 整理現有 Metadata 中的檔案與其 content_hash
        # 將 documents 依 rel_path 分組
        docs_by_rel: Dict[str, List[Dict[str, Any]]] = {}
        for d in self.documents:
            rp = d.get("rel_path")
            if not rp:
                raw_path = str(d.get("path", "")).replace("\\", "/")
                k_idx = raw_path.find("/knowledge/")
                rp = raw_path[k_idx + len("/knowledge/"):].lstrip("/") if k_idx != -1 else d.get("file", "")
            rp_clean = str(rp).replace("\\", "/")
            docs_by_rel.setdefault(rp_clean, []).append(d)

        meta_hashes: Dict[str, str] = {}
        needs_meta_save = False

        for rel, doc_list in docs_by_rel.items():
            # 優先讀取第一筆的 content_hash
            existing_hash = doc_list[0].get("content_hash")
            if not existing_hash:
                # 依 chunk_index 排序並拼接所有舊 text，算出基準 hash
                sorted_docs = sorted(doc_list, key=lambda x: x.get("chunk_index", 0))
                reconstructed_text = "".join(x.get("text", "") for x in sorted_docs)
                existing_hash = compute_content_hash(reconstructed_text)
                for x in doc_list:
                    x["content_hash"] = existing_hash
                needs_meta_save = True

            meta_hashes[rel] = existing_hash

        if needs_meta_save:
            try:
                np.save(self.metadata_path, self.documents)
                if os.path.exists(self.metadata_path):
                    self.last_metadata_mtime = os.path.getmtime(self.metadata_path)
                logger.info(f"[GeminiRAG] 已成功回填 {len(meta_hashes)} 份文件的 content_hash。")
            except Exception as e:
                logger.warning(f"[GeminiRAG] 回填 content_hash 儲存異常: {e}")

        # 3. 三方集合比對
        vault_rels = set(vault_hashes.keys())
        indexed_rels = set(meta_hashes.keys())

        added = sorted(list(vault_rels - indexed_rels))
        deleted = sorted(list(indexed_rels - vault_rels))
        
        common = vault_rels & indexed_rels
        modified = []
        unchanged_count = 0

        for rel in common:
            if vault_hashes[rel] != meta_hashes[rel]:
                modified.append(rel)
            else:
                unchanged_count += 1

        modified.sort()

        return {
            "added": added,
            "modified": modified,
            "deleted": deleted,
            "unchanged_count": unchanged_count,
            "total_vault": len(vault_rels),
            "total_indexed": len(indexed_rels),
            "vault_hashes": vault_hashes,
            "meta_hashes": meta_hashes
        }

    def sync_index(self, dry_run: bool = False, batch_size: int = 20, on_progress: Optional[Any] = None) -> Dict[str, Any]:
        """
        執行基於 SHA-256 content_hash 的精確差異同步。
        - dry_run=True: 僅回傳比對報告，不進行任何 Embedding 與磁碟寫入。
        - dry_run=False:
          1. 獲取排他檔案鎖
          2. 自動備份既有索引
          3. remove_ids 移除 modified 與 deleted 檔案的舊向量
          4. 僅對 added 與 modified 檔案進行分塊與 Embedding (支援 429 退避與間隔)
          5. 每 batch_size 個檔案執行一次部分原子提交
          6. 完成後進行最終原子替換並更新 mtime
        """
        diff = self.diff_vault()
        added = diff["added"]
        modified = diff["modified"]
        deleted = diff["deleted"]

        summary = {
            "status": "dry_run" if dry_run else "success",
            "added_count": len(added),
            "modified_count": len(modified),
            "deleted_count": len(deleted),
            "unchanged_count": diff["unchanged_count"],
            "total_vault": diff["total_vault"],
            "total_indexed": diff["total_indexed"],
            "added_files": added,
            "modified_files": modified,
            "deleted_files": deleted,
            "indexed_chunks": len(self.documents),
            "chunks_added": 0,
            "chunks_removed": 0
        }

        if dry_run:
            logger.info(f"[GeminiRAG] Dry-Run 差異掃描完成: 新增 {len(added)}, 修改 {len(modified)}, 刪除 {len(deleted)}, 不變 {diff['unchanged_count']}")
            return summary

        # 若無任何變更，直接返回
        if not added and not modified and not deleted:
            logger.info("[GeminiRAG] 知識庫與索引完全一致，無須同步。")
            summary["status"] = "up_to_date"
            return summary

        # 進入真實同步階段：獲取檔案鎖
        with self._file_lock(timeout=10.0):
            # 重新加載確保基底最新
            self.load_index()
            if self.index is None:
                raise RuntimeError("[GeminiRAG] 索引加載失敗，無法進行差異同步。")

            # 1. 備份現有索引
            backup_path = self.backup_index()
            summary["backup_path"] = backup_path

            current_vault = get_vault_dir()
            vault_hashes = diff["vault_hashes"]

            # 2. 收集需要刪除舊向量的 rel_paths (modified + deleted)
            targets_to_purge = set(modified) | set(deleted)
            if targets_to_purge:
                ids_to_remove = []
                remaining_docs = []
                for doc in self.documents:
                    rp = str(doc.get("rel_path", "")).replace("\\", "/")
                    if rp in targets_to_purge:
                        ids_to_remove.append(doc["chunk_id"])
                    else:
                        remaining_docs.append(doc)

                if ids_to_remove:
                    logger.info(f"[GeminiRAG] 正在自 FAISS 移除 {len(ids_to_remove)} 筆過期向量 (來自 {len(targets_to_purge)} 份文件)...")
                    self.index.remove_ids(np.array(ids_to_remove).astype('int64'))
                    summary["chunks_removed"] = len(ids_to_remove)

                self.documents = remaining_docs

            # 3. 處理需要新增/更新的檔案 (added + modified)
            files_to_embed = added + modified
            total_files = len(files_to_embed)
            logger.info(f"[GeminiRAG] 開始處理新增與修改文件，共計 {total_files} 檔...")

            max_id = max([doc.get("chunk_id", 0) for doc in self.documents]) if self.documents else 0
            next_id = max_id + 1

            batch_docs: List[Dict[str, Any]] = []
            batch_embeddings: List[List[float]] = []
            total_chunks_added = 0

            for f_idx, rel_path in enumerate(files_to_embed, 1):
                full_path = os.path.normpath(os.path.join(current_vault, rel_path))
                if not os.path.exists(full_path):
                    logger.warning(f"[GeminiRAG] 檔案不存在，跳過: {full_path}")
                    continue

                try:
                    post = read_markdown_file(full_path)
                    content = post.content if post and hasattr(post, "content") else ""
                    if not content.strip():
                        continue

                    file_hash = vault_hashes.get(rel_path) or compute_content_hash(content)
                    chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]

                    for c_idx, chunk in enumerate(chunks):
                        emb = self.get_embedding(chunk)
                        batch_embeddings.append(emb)
                        batch_docs.append({
                            "chunk_id": next_id,
                            "file": os.path.basename(full_path),
                            "path": full_path,
                            "rel_path": rel_path,
                            "content_hash": file_hash,
                            "chunk_index": c_idx,
                            "text": chunk
                        })
                        next_id += 1
                        time.sleep(0.15)  # 溫和間隔防 429

                    total_chunks_added += len(chunks)
                    if on_progress:
                        try:
                            on_progress(f_idx, total_files, rel_path)
                        except Exception:
                            pass

                except Exception as e:
                    logger.error(f"[GeminiRAG] 處理檔案 {rel_path} 產生向量失敗: {e}")

                # 分批提交 (每 batch_size 檔執行一次原子存檔)
                if f_idx % batch_size == 0 and batch_embeddings:
                    emb_mat = np.array(batch_embeddings).astype('float32')
                    ids_mat = np.array([d["chunk_id"] for d in batch_docs]).astype('int64')
                    self.index.add_with_ids(emb_mat, ids_mat)
                    self.documents.extend(batch_docs)
                    self._atomic_save(self.index, self.documents)
                    logger.info(f"[GeminiRAG] [分批提交] 進度: {f_idx}/{total_files} 檔已儲存 (累計 {len(self.documents)} chunks)")
                    batch_embeddings = []
                    batch_docs = []

            # 4. 提交最後剩餘的 chunks
            if batch_embeddings:
                emb_mat = np.array(batch_embeddings).astype('float32')
                ids_mat = np.array([d["chunk_id"] for d in batch_docs]).astype('int64')
                self.index.add_with_ids(emb_mat, ids_mat)
                self.documents.extend(batch_docs)

            # 最終原子儲存
            self._atomic_save(self.index, self.documents)
            summary["chunks_added"] = total_chunks_added
            summary["indexed_chunks"] = len(self.documents)
            logger.info(f"[GeminiRAG] 差異同步成功完成！移除 {summary['chunks_removed']} chunks，新增 {total_chunks_added} chunks，目前總計 {len(self.documents)} chunks。")
            return summary

    def build_index(self):
        """全量重建索引 (耗費大量 Token，建議優先使用 sync_index)。"""
        logger.info("[GeminiRAG] Building vector index from Vault...")
        files = list_markdown_files()
        embeddings = []
        self.documents = []
        next_id = 1
        current_vault = get_vault_dir()

        for file in files:
            try:
                post = read_markdown_file(file)
                content = post.content if post and hasattr(post, "content") else ""
                if not content.strip():
                    continue

                chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
                try:
                    rel_path = os.path.relpath(file, current_vault).replace("\\", "/")
                except Exception:
                    rel_path = os.path.basename(file)

                file_hash = compute_content_hash(content)

                for idx, chunk in enumerate(chunks):
                    emb = self.get_embedding(chunk)
                    embeddings.append(emb)
                    self.documents.append({
                        "chunk_id": next_id,
                        "file": os.path.basename(file),
                        "path": file,
                        "rel_path": rel_path,
                        "content_hash": file_hash,
                        "chunk_index": idx,
                        "text": chunk
                    })
                    next_id += 1
                    time.sleep(0.15)
            except Exception as e:
                logger.error(f"[GeminiRAG] Error processing {file}: {e}")

        if embeddings:
            emb_matrix = np.array(embeddings).astype('float32')
            dimension = emb_matrix.shape[1]
            sub_index = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIDMap(sub_index)
            ids = np.array([doc["chunk_id"] for doc in self.documents]).astype('int64')
            self.index.add_with_ids(emb_matrix, ids)
            self._atomic_save(self.index, self.documents)
            logger.info(f"[GeminiRAG] Built index with {len(self.documents)} chunks.")
            return len(self.documents)
        return 0

    def incremental_update(self, changed_files: List[str]):
        """
        針對指定檔案清單進行增量更新。
        已升級為以 rel_path 作為唯一鍵，消除同名檔誤刪問題。
        """
        if not changed_files:
            logger.info("[GeminiRAG] 無變更檔案，跳過增量更新。")
            return

        self.check_and_reload()
        current_vault = get_vault_dir()

        # 標準化變更檔案為 rel_path 清單
        target_rels = set()
        for f in changed_files:
            if os.path.isabs(f):
                try:
                    target_rels.add(os.path.relpath(f, current_vault).replace("\\", "/"))
                except Exception:
                    target_rels.add(os.path.basename(f))
            else:
                target_rels.add(f.replace("\\", "/"))

        with self._file_lock(timeout=10.0):
            self.backup_index()

            # 1. 移除舊 chunks
            ids_to_remove = [doc["chunk_id"] for doc in self.documents if doc.get("rel_path") in target_rels]
            if ids_to_remove and self.index:
                self.index.remove_ids(np.array(ids_to_remove).astype('int64'))
                self.documents = [doc for doc in self.documents if doc.get("rel_path") not in target_rels]

            # 2. 重新 Embedding 存在的檔案
            new_docs = []
            new_embs = []
            max_id = max([doc.get("chunk_id", 0) for doc in self.documents]) if self.documents else 0
            next_id = max_id + 1

            for rel in target_rels:
                full_path = os.path.normpath(os.path.join(current_vault, rel))
                if not os.path.exists(full_path):
                    continue

                try:
                    post = read_markdown_file(full_path)
                    content = post.content if post and hasattr(post, "content") else ""
                    if not content.strip():
                        continue

                    file_hash = compute_content_hash(content)
                    chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
                    for idx, chunk in enumerate(chunks):
                        emb = self.get_embedding(chunk)
                        new_embs.append(emb)
                        new_docs.append({
                            "chunk_id": next_id,
                            "file": os.path.basename(full_path),
                            "path": full_path,
                            "rel_path": rel,
                            "content_hash": file_hash,
                            "chunk_index": idx,
                            "text": chunk
                        })
                        next_id += 1
                        time.sleep(0.15)
                except Exception as e:
                    logger.error(f"[GeminiRAG] 增量處理 {rel} 失敗: {e}")

            if new_embs and self.index:
                emb_mat = np.array(new_embs).astype('float32')
                id_mat = np.array([d["chunk_id"] for d in new_docs]).astype('int64')
                self.index.add_with_ids(emb_mat, id_mat)
                self.documents.extend(new_docs)

            self._atomic_save(self.index, self.documents)
            logger.info(f"[GeminiRAG] 增量更新完成，變更 {len(target_rels)} 檔，新增 {len(new_docs)} chunks。")

    def extract_query_intent(self, query: str) -> Dict[str, Any]:
        """
        智慧解析使用者提問中的時間與文檔類型意圖 (Hybrid Search Intent Parser)。
        回傳:
        {
            "time_patterns": ['2026-10', ...],
            "doc_types": ['plans', ...],
            "is_recent": bool
        }
        """
        import re
        q = (query or "").strip().lower()
        time_patterns = []
        doc_types = []
        is_recent = False

        # 1. 時間解析
        # 匹配具體年份+月份: 2026-10, 2026年10月, 2026/10, 2026.10
        full_match = re.search(r'(20\d\d)[年\-_/\.](\d{1,2})月?', q)
        if full_match:
            year = full_match.group(1)
            month = int(full_match.group(2))
            if 1 <= month <= 12:
                time_patterns.append(f"{year}-{month:02d}")
        else:
            # 匹配單純月份: 10月份, 10月, 9月份 (預設當前年份 2026)
            month_match = re.search(r'(?<!\d)(\d{1,2})月份?', q)
            if month_match:
                month = int(month_match.group(1))
                if 1 <= month <= 12:
                    current_year = datetime.now().year if datetime.now().year >= 2026 else 2026
                    time_patterns.append(f"{current_year}-{month:02d}")

        # 匹配相對時間詞: 最近, 最新, 近期
        if any(w in q for w in ["最近", "最新", "近期"]):
            is_recent = True
            current_year = datetime.now().year if datetime.now().year >= 2026 else 2026
            time_patterns.extend([f"{current_year}-10", f"{current_year}-09"])

        # 2. 文檔類型解析
        type_keywords = {
            "plans": ["計畫", "計劃", "規劃", "方案", "plan"],
            "walkthroughs": ["成果", "紀錄", "記錄", "工作紀錄", "walkthrough", "完成", "總結"],
            "errors": ["錯誤", "問題", "修復", "修正", "bug", "error", "異常", "失效"]
        }
        for dtype, kws in type_keywords.items():
            if any(k in q for k in kws):
                doc_types.append(dtype)

        return {
            "time_patterns": list(set(time_patterns)),
            "doc_types": list(set(doc_types)),
            "is_recent": is_recent
        }

    def chat(self, query: str) -> Dict[str, Any]:
        """
        雙路混合檢索 (Dual-path Hybrid Metadata Search) 問答。
        - 路徑 1: 結構化元數據過濾候選池 (時間/類型/特定檔名過濾優先注入)
        - 路徑 2: 全域 FAISS Top-K 向量語意檢索
        - 雙路融合: 路徑 1 置頂 + 路徑 2 補充去重 + 單檔上限 3 chunks + 總上限 10 chunks 防 context 膨脹。
        """
        self.check_and_reload()
        if self.index is None or len(self.documents) == 0:
            return {"answer": "知識庫尚未建立索引，請先執行差異同步建立索引。", "citations": []}

        query_lower = query.lower()
        intent = self.extract_query_intent(query)
        time_patterns = intent.get("time_patterns", [])
        doc_types = intent.get("doc_types", [])

        # ==========================================
        # 路徑 1: 結構化元數據與特定檔名過濾候選池
        # ==========================================
        path1_docs: List[Dict[str, Any]] = []

        # 1-1. 檢查特定檔名提及
        unique_files = {}
        for doc in self.documents:
            unique_files[doc['file'].lower()] = doc['file']

        mentioned_files = set()
        for fname_lower, fname_orig in unique_files.items():
            if fname_lower in query_lower:
                mentioned_files.add(fname_orig)

        if mentioned_files:
            logger.info(f"[HybridSearch] 偵測到提及特定檔案: {mentioned_files}")
            for doc in self.documents:
                if doc['file'] in mentioned_files:
                    path1_docs.append(doc)

        # 1-2. 依時間與類型元數據過濾候選檔案
        if time_patterns or doc_types:
            logger.info(f"[HybridSearch] 偵測到時間/類型意圖: 時間={time_patterns}, 類型={doc_types}")
            matched_by_intent: Dict[str, List[Dict[str, Any]]] = {}

            for doc in self.documents:
                rel_path = str(doc.get("rel_path") or doc.get("file", "")).replace("\\", "/")

                # 時間比對
                time_matched = True
                if time_patterns:
                    time_matched = any(tp in rel_path for tp in time_patterns)

                # 類型比對 (plans, walkthroughs, errors)
                type_matched = True
                if doc_types:
                    type_matched = any(f"{dt}/" in rel_path or dt in rel_path for dt in doc_types)

                if time_matched and type_matched:
                    fname = doc.get("file")
                    matched_by_intent.setdefault(fname, []).append(doc)

            # 針對過濾出的各個檔案，依 chunk_index 排序，每個檔案挑選前 2 塊 (標題與精華摘要)
            for fname, dlist in matched_by_intent.items():
                sorted_dlist = sorted(dlist, key=lambda x: x.get("chunk_index", 0))
                # 取首 2 塊，若總候選池未滿 10 塊則追加
                for d in sorted_dlist[:2]:
                    if len(path1_docs) < 10 and d not in path1_docs:
                        path1_docs.append(d)

        # ==========================================
        # 路徑 2: 全域 FAISS Top-K 向量語意檢索
        # ==========================================
        path2_docs: List[Dict[str, Any]] = []
        try:
            query_emb = np.array([self.get_embedding(query)]).astype('float32')
            k = 5
            distances, indices = self.index.search(query_emb, k)
            for idx in indices[0]:
                if 0 <= idx < len(self.documents):
                    path2_docs.append(self.documents[idx])
        except Exception as e:
            logger.warning(f"[HybridSearch] 向量檢索異常: {e}")

        # ==========================================
        # 雙路融合 (Deduplicated Fusion & Budget Guard)
        # ==========================================
        contexts = []
        citations = []
        seen_chunks = set()
        file_chunk_counts: Dict[str, int] = {}
        MAX_TOTAL_CHUNKS = 10
        MAX_PER_FILE_CHUNKS = 3

        # 優先排入路徑 1 (元數據與時間過濾結果)
        for doc in path1_docs:
            if len(contexts) >= MAX_TOTAL_CHUNKS:
                break
            fname = doc.get("file", "unknown")
            if file_chunk_counts.get(fname, 0) >= MAX_PER_FILE_CHUNKS:
                continue

            chunk_key = f"{fname}_{doc.get('chunk_index', 0)}"
            if chunk_key not in seen_chunks:
                contexts.append(f"【來源檔案: {fname} (區塊 {doc.get('chunk_index', 0)})】\n{doc.get('text', '')}")
                citations.append(fname)
                seen_chunks.add(chunk_key)
                file_chunk_counts[fname] = file_chunk_counts.get(fname, 0) + 1

        # 補充排入路徑 2 (向量語意相似結果)
        for doc in path2_docs:
            if len(contexts) >= MAX_TOTAL_CHUNKS:
                break
            fname = doc.get("file", "unknown")
            if file_chunk_counts.get(fname, 0) >= MAX_PER_FILE_CHUNKS:
                continue

            chunk_key = f"{fname}_{doc.get('chunk_index', 0)}"
            if chunk_key not in seen_chunks:
                contexts.append(f"【來源檔案: {fname} (區塊 {doc.get('chunk_index', 0)})】\n{doc.get('text', '')}")
                citations.append(fname)
                seen_chunks.add(chunk_key)
                file_chunk_counts[fname] = file_chunk_counts.get(fname, 0) + 1

        context_str = "\n\n" + ("=" * 40) + "\n\n".join(contexts)

        # 構造帶有清晰指引的 Prompt
        prompt = f"""
你是一個專業的專案知識庫 AI 助手。請根據以下檢索到的參考內容，精準、結構化地回答使用者的問題。
若使用者詢問特定月份或類型的計畫/紀錄，請根據參考來源中的各檔案名稱與內文進行條列彙整說明。
如果參考內容中確實無相關資料，請明確告知「知識庫中無法確認」。
回答時請務必在各條目後標註或引用對應的來源檔案名稱。

參考內容：
{context_str}

使用者的問題：
{query}
"""
        answer = self.generate_answer(prompt)
        return {
            "answer": answer,
            "citations": list(dict.fromkeys(citations))  # 保持順序去重
        }


# 全域單例
rag_service = GeminiRAG()
