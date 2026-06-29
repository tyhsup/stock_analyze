import os
import glob
import re
from datetime import datetime

# 全域資料庫路徑
VAULT_DIR = r"C:\Users\許廷宇\.gemini\config\knowledge"

# 專案根目錄
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

def ensure_folders():
    """確保全域 Wiki 資料夾結構存在"""
    if not os.path.exists(VAULT_DIR):
        print(f"建立全域 WIKI 目錄: {VAULT_DIR}")
        os.makedirs(VAULT_DIR, exist_ok=True)
    
    for folder in ["specs", "plans", "walkthroughs", "errors", "references"]:
        folder_path = os.path.join(VAULT_DIR, folder)
        if not os.path.exists(folder_path):
            print(f"建立分類目錄: {folder_path}")
            os.makedirs(folder_path, exist_ok=True)

def truncate_content(filepath, max_lines=200):
    """
    如果檔案過大，進行截斷處理，保留前後部分，並加上說明。
    避免過大檔案破壞 RAG 索引的效率。
    """
    if not os.path.exists(filepath):
        return None
        
    file_size = os.path.getsize(filepath)
    # 如果小於 100 KB，直接完整讀取
    if file_size < 100 * 1024:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
            
    # 大檔案進行行數截斷
    print(f"檔案 {os.path.basename(filepath)} 較大 ({file_size / 1024:.2f} KB)，進行截斷處理...")
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    if len(lines) <= max_lines * 2:
        return "".join(lines)
        
    header = lines[:max_lines]
    footer = lines[-max_lines:]
    
    truncated_text = (
        "".join(header) +
        f"\n\n... [中間共 {len(lines) - max_lines * 2} 行日誌/內容已省略。完整檔案請參見專案路徑: {filepath}] ...\n\n" +
        "".join(footer)
    )
    return truncated_text

def format_markdown(content, title, doc_type):
    """加上 Frontmatter"""
    date_str = datetime.now().strftime("%Y-%m-%d")
    # 移除原有的 frontmatter (如果有的話)，避免重複
    clean_content = content
    if content.strip().startswith("---"):
        parts = re.split(r'^---$', content.strip(), maxsplit=2, flags=re.MULTILINE)
        if len(parts) >= 3:
            clean_content = parts[2].strip()
            
    frontmatter = f"""---
title: {title}
type: {doc_type}
date: {date_str}
---

"""
    return frontmatter + clean_content

def process_file(src_name, dest_subfolder, dest_name, title, doc_type, wrap_code=None):
    """處理單一檔案的複製、格式轉換與分類"""
    src_path = os.path.join(PROJECT_DIR, src_name)
    if not os.path.exists(src_path):
        print(f"來源檔案不存在，跳過: {src_name}")
        return
        
    print(f"處理檔案: {src_name} -> {dest_subfolder}/{dest_name}")
    content = truncate_content(src_path)
    if content is None:
        return
        
    # 如果需要用程式碼區塊包裹 (例如 log 或 sql)
    if wrap_code:
        content = f"```{wrap_code}\n{content}\n```"
        
    formatted_content = format_markdown(content, title, doc_type)
    
    dest_path = os.path.join(VAULT_DIR, dest_subfolder, dest_name)
    # 確保子目錄存在
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    with open(dest_path, 'w', encoding='utf-8') as f:
        f.write(formatted_content)
    print(f"成功寫入: {dest_path}")

def main():
    ensure_folders()
    
    # 檔案對照清單
    # (來源檔名, 目標子資料夾, 目標檔名, 標題, 文件類型, 程式碼包裹類型)
    files_to_copy = [
        # 1. 計劃與規格類 (plans)
        ("gemma_plan_result.txt", "plans", "gemma_plan_result.md", "量化交易模型升級開發計畫書", "plans", None),
        ("gemma_v3_gnn_plan.md", "plans", "gemma_v3_gnn_plan.md", "v3.0 結構化關聯建模 (Relational Modeling) 技術架構建議書", "plans", None),
        ("gemma_us_sector_fix.md", "plans", "gemma_us_sector_fix.md", "v3.0 GNN 美股產業資訊缺失解決方案技術報告", "plans", None),
        ("gemma_finbert_result.md", "plans", "gemma_finbert_result.md", "v2.0 引入 FinBERT 深度語義特徵強化實作指引", "plans", None),
        
        # 2. 參考資料類 (references)
        ("gemma_reasoner_out.txt", "references", "gemma_reasoner_out.md", "產業投資者摘要與資料庫操作優化建議", "references", None),
        ("db_schema.txt", "references", "db_schema.md", "資料庫結構表格與欄位清單 (Schema 1)", "references", "markdown"),
        ("db_schema2.txt", "references", "db_schema2.md", "資料庫結構表格與欄位清單 (Schema 2)", "references", "markdown"),
        ("reasoning_prompt_ai.txt", "references", "reasoning_prompt_ai.md", "AI 模型架構 Deep Research 提示詞", "references", "markdown"),
        ("reasoning_prompt_tw.txt", "references", "reasoning_prompt_tw.md", "台股交易與籌碼分析優化提示詞", "references", "markdown"),
        ("stock_db_backup.sql", "references", "stock_db_backup_sql.md", "台股與美股資料庫備份 SQL 綱要", "references", "sql"),
        ("stock_batch_update.log", "references", "stock_batch_update_log.md", "股票資料批次更新執行日誌", "references", "text"),
        ("stock_investor_final.log", "references", "stock_investor_final_log.md", "三大法人籌碼分析最終執行日誌", "references", "text"),
        ("stock_investor_tpex.log", "references", "stock_investor_tpex_log.md", "上櫃三大法人籌碼分析執行日誌", "references", "text"),
    ]
    
    for src_name, dest_subfolder, dest_name, title, doc_type, wrap_code in files_to_copy:
        process_file(src_name, dest_subfolder, dest_name, title, doc_type, wrap_code)

if __name__ == "__main__":
    main()
