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

def annotate_files(changed_files=None):
    """
    對知識庫中的 Markdown 檔案進行增量或全量關聯性標註。
    若指定 changed_files，則僅掃描與處理變更或新增的檔案。
    """
    print("開始進行知識庫檔案關聯性標註...")
    
    if changed_files is not None:
        target_files = [f for f in changed_files if os.path.exists(f)]
    else:
        target_files = glob.glob(os.path.join(VAULT_DIR, "**", "*.md"), recursive=True)
        
    if not target_files:
        print("未偵測到任何需要關聯處理的檔案。")
        return
        
    print(f"待處理檔案數: {len(target_files)}")
    
    for filepath in target_files:
        filename = os.path.basename(filepath)
        print(f"\n處理檔案: {filename}")
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # 清除先前可能已存在的「關聯知識點」區塊，避免重複累加
            clean_content = re.sub(r'\n+---\n+### 關聯知識點 \(Related Knowledge\).*$', '', content, flags=re.DOTALL).strip()
            
            relations = []
            
            # 根據規則進行關聯比對
            for ref_file, rules in RELATION_RULES.items():
                if ref_file == filename:
                    continue # 不要自己關聯自己
                    
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
