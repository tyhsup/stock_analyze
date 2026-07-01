import os
import glob
import re

VAULT_DIR = r"C:\Users\許廷宇\.gemini\config\knowledge"

RELATION_RULES = {
    "gemma_plan_result.md": {
        "keywords": ["Roadmap", "開發計畫書", "升級開發計畫", "v1.0", "v2.0", "v3.0", "v4.0"],
        "desc": "量化交易模型升級開發計畫書"
    },
    "gemma_finbert_result.md": {
        "keywords": ["FinBERT", "SentimentProbabilityModel", "情緒分析", "語義特徵", "情緒機率"],
        "desc": "v2.0 引入 FinBERT 深度語義特徵強化實作指引"
    },
    "gemma_v3_gnn_plan.md": {
        "keywords": ["GNN", "Node Transformer", "鄰接矩陣", "Adjacency Matrix", "關聯建模"],
        "desc": "v3.0 結構化關聯建模技術架構建議書"
    },
    "gemma_us_sector_fix.md": {
        "keywords": ["美股產業", "Sector", "stocks_us", "美股節點", "邊界缺失"],
        "desc": "v3.0 GNN 美股產業資訊缺失解決方案技術報告"
    },
    "gemma_reasoner_out.md": {
        "keywords": ["get_industry_investor_summary", "OP_Fun", "資料庫操作", "優化建議"],
        "desc": "產業投資者摘要與資料庫操作優化建議"
    },
    "db_schema.md": {
        "keywords": ["Table: ", "auth_group", "stocks_tw", "valuation_valuationresult", "DailyPriceTW"],
        "desc": "資料庫結構表格與欄位清單 (Schema 1)"
    },
    "db_schema2.md": {
        "keywords": ["Table: ", "financial_raw_tw", "financial_raw_us", "stock_cost_us"],
        "desc": "資料庫結構表格與欄位清單 (Schema 2)"
    },
    "reasoning_prompt_ai.md": {
        "keywords": ["IntegratedStockPredModel", "stock_cost_AI.py", "Deep Research", "模型架構"],
        "desc": "AI 模型架構 Deep Research 提示詞"
    },
    "reasoning_prompt_tw.md": {
        "keywords": ["00981A", "update_tw_stocks", "update_tw_prices", "特別股", "ETF"],
        "desc": "台股交易與籌碼分析優化提示詞"
    },
    "stock_db_backup_sql.md": {
        "keywords": ["CREATE TABLE", "INSERT INTO", "stock_db_backup", "SQL 綱要"],
        "desc": "台股與美股資料庫備份 SQL 綱要"
    },
    "stock_batch_update_log.md": {
        "keywords": ["批次更新", "yf.download", "update_tw_stocks.py", "DailyPriceTW"],
        "desc": "股票資料批次更新執行日誌"
    },
    "stock_investor_final_log.md": {
        "keywords": ["三大法人籌碼", "stock_investor", "買賣超股數"],
        "desc": "三大法人籌碼分析最終執行日誌"
    },
    "stock_investor_tpex_log.md": {
        "keywords": ["上櫃三大法人", "tpex", "櫃買中心"],
        "desc": "上櫃三大法人籌碼分析執行日誌"
    }
}

def parse_frontmatter(content):
    """從 Markdown 內容中抽取 Frontmatter 的 title 與 source"""
    if not content.strip().startswith("---"):
        return {}
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}
    fm_text = parts[1]
    fields = {}
    for line in fm_text.split("\n"):
        if ":" in line:
            k, v = line.split(":", 1)
            # 移除外層雙引號、單引號與前後空格
            v_clean = v.strip().strip('"').strip("'")
            fields[k.strip()] = v_clean
    return fields

def annotate_files(changed_files=None):
    """
    對知識庫中的 Markdown 檔案進行增量或全量關聯性標註。
    若指定 changed_files，則僅掃描與處理變更或新增的檔案。
    """
    print("開始進行知識庫檔案關聯性標註...")
    
    # 1. 第一步：全量掃描所有檔案，建立 NotebookLM 筆記本分群
    all_files = glob.glob(os.path.join(VAULT_DIR, "**", "*.md"), recursive=True)
    notebook_groups = {} # 結構: { "LLM wiki建構要點": [ { filename, title, filepath } ] }
    
    for filepath in all_files:
        # 跳過大小為 0 的空檔案
        if os.path.exists(filepath) and os.path.getsize(filepath) == 0:
            continue
            
        # 跳過我們稍後要自動產生的筆記本首頁檔本身，避免自己循環群組
        if "_notebook_index.md" in filepath.lower():
            continue
            
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            fields = parse_frontmatter(content)
            source_val = fields.get("source", "")
            
            # 匹配 "NotebookLM (筆記本名稱)"
            match = re.match(r'^NotebookLM\s*\((.*)\)$', source_val)
            if match:
                nb_name = match.group(1).strip()
                title_val = fields.get("title", os.path.basename(filepath))
                notebook_groups.setdefault(nb_name, []).append({
                    "filename": os.path.basename(filepath),
                    "title": title_val,
                    "filepath": filepath
                })
        except Exception as e:
            pass

    # 2. 第二步：為每個筆記本自動建立或更新一個 Wiki 索引首頁檔案
    # 這樣在 Obsidian 中點選筆記本 Wikilink (例如 [[LLM_wiki建構要點_notebook_index]]) 就不會是空連結
    generated_indices = []
    for nb_name, sources in notebook_groups.items():
        clean_nb_name = re.sub(r'[\\/*?:"<>|# ]', "_", nb_name)
        index_filename = f"{clean_nb_name}_notebook_index.md".lower()
        index_filepath = os.path.join(VAULT_DIR, "references", "notebooklm", index_filename)
        
        # 組裝筆記本索引頁內容
        from datetime import datetime
        links_list = []
        for src in sorted(sources, key=lambda x: x["title"]):
            link_name = os.path.splitext(src["filename"])[0]
            links_list.append(f"* [[{link_name}]] - {src['title']}")
            
        index_content = f"""---
title: "NotebookLM 筆記本：{nb_name}"
type: references
date: "{datetime.now().strftime('%Y-%m-%d')}"
source: "NotebookLM ({nb_name})"
---

# NotebookLM 筆記本：{nb_name}

本筆記本在雲端同步的相關來源資料已自動歸類如下：

### 筆記本來源文件目錄
{chr(10).join(links_list)}
"""
        try:
            # 寫入或更新首頁
            with open(index_filepath, 'w', encoding='utf-8') as f:
                f.write(index_content)
            generated_indices.append(index_filepath)
            # 使用 ascii/safe 印出以防 cp950 編碼崩潰
            safe_index_filename = index_filename.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
            safe_print_index = re.sub(r'[^\x00-\x7F\u4e00-\u9fa5]', '?', safe_index_filename)
            print(f" [首頁產生/更新] {safe_print_index} (共 {len(sources)} 個 Sources)")
        except Exception as e:
            print(f"產生筆記本首頁失敗: {e}")

    # 3. 第三步：過濾出本次需要處理的目標檔案
    if changed_files is not None:
        # 本次變更檔案 + 自動產生的首頁檔案都需要做關聯
        target_files = [f for f in (list(changed_files) + generated_indices) if os.path.exists(f)]
    else:
        target_files = glob.glob(os.path.join(VAULT_DIR, "**", "*.md"), recursive=True)
        
    if not target_files:
        print("未偵測到任何需要關聯處理的檔案。")
        return
        
    print(f"待處理檔案數: {len(target_files)}")
    
    for filepath in target_files:
        # 跳過大小為 0 的空檔案
        if os.path.exists(filepath) and os.path.getsize(filepath) == 0:
            continue
            
        filename = os.path.basename(filepath)
        # 跳過索引首頁本身的關聯標註，因為它自己就是目錄
        if "_notebook_index.md" in filename:
            continue
            
        try:
            # 使用 ascii/safe 印出以防 cp950 編碼崩潰
            safe_filename = filename.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
            # 過濾掉可能引起 cp950 崩潰的特定字元
            safe_print_name = re.sub(r'[^\x00-\x7F\u4e00-\u9fa5]', '?', safe_filename)
            print(f"\n處理檔案: {safe_print_name}")
        except Exception:
            print("\n處理檔案: [Unicode Name File]")
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # 清除先前可能已存在的「關聯知識點」區塊，避免重複累加
            clean_content = re.sub(r'\n+---\n+### 關聯知識點 \(Related Knowledge\).*$', '', content, flags=re.DOTALL).strip()
            
            relations = []
            
            # (A) 靜態規則關聯比對
            for ref_file, rules in RELATION_RULES.items():
                if ref_file == filename:
                    continue
                    
                is_matched = False
                for kw in rules["keywords"]:
                    if kw.lower() in clean_content.lower() or kw.lower() in filename.lower():
                        is_matched = True
                        break
                        
                if filename == "gemma_plan_result.md" and ref_file in ["gemma_finbert_result.md", "gemma_v3_gnn_plan.md", "gemma_us_sector_fix.md"]:
                    is_matched = True
                if ref_file == "gemma_plan_result.md" and filename in ["gemma_finbert_result.md", "gemma_v3_gnn_plan.md", "gemma_us_sector_fix.md"]:
                    is_matched = True
                    
                db_files = ["db_schema.md", "db_schema2.md", "stock_db_backup_sql.md"]
                if filename in db_files and ref_file in db_files:
                    is_matched = True
                    
                log_files = ["stock_batch_update_log.md", "stock_investor_final_log.md", "stock_investor_tpex_log.md"]
                if filename in log_files and ref_file in log_files:
                    is_matched = True
                    
                if is_matched:
                    link_name = os.path.splitext(ref_file)[0]
                    relations.append(f"* [[{link_name}]] - {rules['desc']}")
            
            # (B) NotebookLM 同筆記本分群關聯
            fields = parse_frontmatter(content)
            source_val = fields.get("source", "")
            match = re.match(r'^NotebookLM\s*\((.*)\)$', source_val)
            
            if match:
                nb_name = match.group(1).strip()
                clean_nb_name = re.sub(r'[\\/*?:"<>|# ]', "_", nb_name)
                index_link = f"{clean_nb_name}_notebook_index".lower()
                
                # 指向該筆記本首頁
                relations.append(f"* [[{index_link}]] - NotebookLM 筆記本：{nb_name} 首頁索引")
                
                # 列出同屬於這本筆記本的其他 Sources 檔案
                same_nb_sources = notebook_groups.get(nb_name, [])
                other_sources = [s for s in same_nb_sources if s["filename"] != filename]
                
                if other_sources:
                    relations.append(f"* 同屬此筆記本的相關文件：")
                    # 按字母排序
                    for s in sorted(other_sources, key=lambda x: x["title"]):
                        link_name = os.path.splitext(s["filename"])[0]
                        relations.append(f"  * [[{link_name}]] - {s['title']}")
                    
            if relations:
                print(f" -> 建立關聯: {len(relations)} 個連結")
                relation_block = "\n\n---\n### 關聯知識點 (Related Knowledge)\n" + "\n".join(relations)
                new_content = clean_content + relation_block
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(new_content)
            else:
                print(" -> 未發現相關聯檔案。")
        except Exception as e:
            print(f"處理檔案 {filename} 關聯標記出錯: {e}")
            
    print("\n關聯性標註完成！")

if __name__ == "__main__":
    annotate_files()
