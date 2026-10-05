# -*- coding: utf-8 -*-
"""
Obsidian 知識庫同步模組 (Obsidian Sync Service)
支援跨平台 (Docker Linux 容器環境與 Windows 本機環境) 動態路徑解析與 Markdown 同步。
"""

import os
import glob
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import frontmatter

logger = logging.getLogger(__name__)

DEFAULT_WIN_VAULT = r"E:\Obsidian_knowledge\antigravity_knowledge\knowledge"
DEFAULT_CONTAINER_VAULT = "/app/knowledge_vault"


def get_vault_dir() -> str:
    """
    動態解析知識庫根目錄路徑。
    優先順序：
    1. 環境變數 OBSIDIAN_VAULT_DIR (來自 .env)
    2. Docker 容器標準掛載點 /app/knowledge_vault (若該目錄存在)
    3. Windows 宿主機原生路徑 E:\\Obsidian_knowledge\\antigravity_knowledge\\knowledge (若該目錄存在)
    4. 回退預設值
    """
    env_vault = os.getenv("OBSIDIAN_VAULT_DIR", "").strip()
    if env_vault and os.path.exists(env_vault):
        return os.path.normpath(env_vault)

    # 容器化掛載優先檢查
    if os.path.exists(DEFAULT_CONTAINER_VAULT):
        return DEFAULT_CONTAINER_VAULT

    # Windows 宿主機路徑檢查
    if os.path.exists(DEFAULT_WIN_VAULT):
        return os.path.normpath(DEFAULT_WIN_VAULT)

    # 若皆不存在，回傳環境變數設定值或容器預設
    return env_vault or DEFAULT_CONTAINER_VAULT


# 模組預設值相容
VAULT_DIR = get_vault_dir()


def ensure_vault_exists() -> str:
    """確保知識庫及其標準子目錄結構存在"""
    vault_dir = get_vault_dir()
    try:
        if not os.path.exists(vault_dir):
            os.makedirs(vault_dir, exist_ok=True)
            logger.info(f"[ObsidianSync] 已建立知識庫目錄: {vault_dir}")

        # 標準結構資料夾
        for folder in ["specs", "plans", "walkthroughs", "errors", "references"]:
            sub_folder = os.path.join(vault_dir, folder)
            os.makedirs(sub_folder, exist_ok=True)
        os.makedirs(os.path.join(vault_dir, "references", "notebooklm"), exist_ok=True)
    except Exception as e:
        logger.warning(f"[ObsidianSync] 建立知識庫子目錄異常 (可能是唯讀掛載): {e}")

    return vault_dir


def list_markdown_files() -> List[str]:
    """
    遞迴列出知識庫中所有 Markdown 檔案。
    自動過濾 .git, .obsidian, .trash 等隱藏設定資料夾。
    """
    vault_dir = ensure_vault_exists()
    all_files = glob.glob(os.path.join(vault_dir, "**", "*.md"), recursive=True)
    
    clean_files = []
    for f in all_files:
        norm = os.path.normpath(f)
        # 排除隱藏與暫存目錄
        parts = norm.split(os.sep)
        if any(p.startswith(".") for p in parts):
            continue
        clean_files.append(norm)

    return sorted(clean_files)


def resolve_file_path(filepath: str) -> str:
    """若為相對路徑，自動拼接當前 VAULT_DIR"""
    vault_dir = get_vault_dir()
    if os.path.isabs(filepath):
        return os.path.normpath(filepath)
    return os.path.normpath(os.path.join(vault_dir, filepath))


def read_markdown_file(filepath: str):
    """讀取 Markdown 筆記並解析 Frontmatter 元數據 (具備 YAML 語法解析容錯 fallback)"""
    full_path = resolve_file_path(filepath)
    with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
        raw_text = f.read()
    try:
        post = frontmatter.loads(raw_text)
    except Exception as e:
        # 當 frontmatter 含有非法轉義字元時，降級為純文字 Post 物件
        logger.warning(f"[ObsidianSync] Frontmatter 解析失敗，降級為純文字讀取 ({os.path.basename(full_path)}): {e}")
        post = frontmatter.Post(raw_text)
    return post


def write_markdown_file(sub_path: str, content: str, metadata: Optional[dict] = None) -> str:
    """安全寫入 Markdown 檔案至知識庫"""
    vault_dir = ensure_vault_exists()
    filepath = resolve_file_path(sub_path)
    
    # 確保父資料夾存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if metadata is None:
        metadata = {}
    
    if 'date' not in metadata:
        metadata['date'] = datetime.now().strftime("%Y-%m-%d")
        
    post = frontmatter.Post(content, **metadata)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(frontmatter.dumps(post))
        
    logger.info(f"[ObsidianSync] 成功寫入 Markdown 檔案: {filepath}")
    return filepath
