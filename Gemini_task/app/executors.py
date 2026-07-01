import os
import sys
import logging

# 動態添加 mydjango/demo 到 Python path 中，使 stock_Django module 可以正確載入
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "demo"))
if DEMO_DIR not in sys.path:
    sys.path.insert(0, DEMO_DIR)

# 設定 logger
logger = logging.getLogger("scheduler.executors")

try:
    from stock_Django.stock_cost import StockCostManager
    from stock_Django.stock_investor import StockInvestorManager
    from stock_Django.stock_investor_tpex import TPExInvestorManager
    from stock_Django.stock_investor_us import USStockInvestorManager
except ImportError as e:
    logger.error(f"無法載入 stock_Django 中的管理器: {e}")
    raise e

def execute_tw_stock_cost(remarks: str = None):
    """
    台股股價與籌碼面更新任務執行器
    若 remarks 中包含特定的股票代碼（如 2330），則執行單一更新，否則全量更新。
    """
    manager = StockCostManager()
    ticker = (remarks or "").strip()
    
    if ticker:
        # 如果使用者輸入了多個代號，如 "2330, 2454"，逐一更新
        tickers = [t.strip() for t in ticker.replace("，", ",").split(",") if t.strip()]
        if len(tickers) > 1:
            logger.info(f"開始更新多個台股代號: {tickers}")
            success = True
            for t in tickers:
                if not manager.update_single_ticker(t):
                    success = False
            if not success:
                raise Exception(f"部分台股股價更新失敗: {tickers}")
        else:
            logger.info(f"開始更新單一台股: {tickers[0]}")
            if not manager.update_single_ticker(tickers[0]):
                raise Exception(f"更新單一台股股價失敗: {tickers[0]}")
    else:
        logger.info("開始更新全台股股價 (上市及上櫃)")
        manager.update_all_cost()

def execute_us_stock_cost(remarks: str = None):
    """
    美股股價與籌碼面更新任務執行器
    若 remarks 中包含特定的股票代碼（如 AAPL），則執行單一更新，否則全量更新。
    """
    manager = StockCostManager()
    ticker = (remarks or "").strip()
    
    if ticker:
        tickers = [t.strip().upper() for t in ticker.replace("，", ",").split(",") if t.strip()]
        if len(tickers) > 1:
            logger.info(f"開始更新多個美股代號: {tickers}")
            success = True
            for t in tickers:
                if not manager.update_single_ticker(t):
                    success = False
            if not success:
                raise Exception(f"部分美股股價更新失敗: {tickers}")
        else:
            logger.info(f"開始更新單一美股: {tickers[0]}")
            if not manager.update_single_ticker(tickers[0]):
                raise Exception(f"更新單一美股股價失敗: {tickers[0]}")
    else:
        logger.info("開始更新全美股股價")
        manager.update_us_cost()

def execute_twse_investor(remarks: str = None):
    """
    台股三大法人買賣超 (上市) 更新任務執行器
    """
    logger.info("開始更新台股三大法人買賣超 (上市)")
    manager = StockInvestorManager()
    manager.update_investor_data()

def execute_tpex_investor(remarks: str = None):
    """
    台股三大法人買賣超 (上櫃) 更新任務執行器
    """
    logger.info("開始更新台股三大法人買賣超 (上櫃)")
    manager = TPExInvestorManager()
    manager.update_all_tpex_investors()

def execute_us_investor(remarks: str = None):
    """
    美股三大法人持股更新任務執行器
    若 remarks 中指定了代號 (如 AAPL,NVDA)，則更新該清單。
    否則預設更新 ['AAPL', 'NVDA', 'TSLA']。
    """
    manager = USStockInvestorManager()
    ticker_str = (remarks or "").strip()
    
    if ticker_str:
        tickers = [t.strip().upper() for t in ticker_str.replace("，", ",").split(",") if t.strip()]
    else:
        tickers = ['AAPL', 'NVDA', 'TSLA']
        
    logger.info(f"開始更新美股法人持股: {tickers}")
    manager.update_investor_db(tickers)

def execute_tw_stock_price_only(remarks: str = None):
    """
    僅更新台灣股價更新任務執行器
    若 remarks 中包含特定的股票代碼（如 2330），則執行單一更新，否則全量更新。
    """
    execute_tw_stock_cost(remarks)

def execute_us_stock_price_only(remarks: str = None):
    """
    僅更新美國股價更新任務執行器
    若 remarks 中包含特定的股票代碼（如 AAPL），則執行單一更新，否則全量更新。
    """
    execute_us_stock_cost(remarks)

def execute_tw_listed_list_update(remarks: str = None):
    """
    台灣上市公司清單更新任務執行器
    """
    logger.info("開始更新台灣上市公司清單...")
    from stock_Django.mySQL_OP import OP_Fun
    import requests
    import pandas as pd
    from io import StringIO
    
    op = OP_Fun()
    url = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
    res = requests.get(url, timeout=15)
    res.encoding = 'big5'
    
    dfs = pd.read_html(StringIO(res.text))
    if not dfs:
        raise Exception("無法讀取 HTML 表格")
    df = dfs[0]
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    
    all_stocks = []
    current_type = ""
    for _, row in df.iterrows():
        val = str(row['有價證券代號及名稱'])
        if "　" not in val and len(val) > 0:
            current_type = val
        elif current_type == "股票" and "　" in val:
            symbol, name = val.split("　", 1)
            all_stocks.append({"symbol": symbol, "name": name, "market": "sii"})
            
    if all_stocks:
        op.upsert_stocks_info(pd.DataFrame(all_stocks), market='tw')
        logger.info(f"更新台灣上市公司成功，共 {len(all_stocks)} 筆")
    else:
        raise Exception("未找到任何台灣上市公司資料")

def execute_tw_otc_list_update(remarks: str = None):
    """
    台灣上櫃公司清單更新任務執行器
    """
    logger.info("開始更新台灣上櫃公司清單...")
    from stock_Django.mySQL_OP import OP_Fun
    import requests
    import pandas as pd
    from io import StringIO
    
    op = OP_Fun()
    url = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"
    res = requests.get(url, timeout=15)
    res.encoding = 'big5'
    
    dfs = pd.read_html(StringIO(res.text))
    if not dfs:
        raise Exception("無法讀取 HTML 表格")
    df = dfs[0]
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    
    all_stocks = []
    current_type = ""
    for _, row in df.iterrows():
        val = str(row['有價證券代號及名稱'])
        if "　" not in val and len(val) > 0:
            current_type = val
        elif current_type == "股票" and "　" in val:
            symbol, name = val.split("　", 1)
            all_stocks.append({"symbol": symbol, "name": name, "market": "otc"})
            
    if all_stocks:
        op.upsert_stocks_info(pd.DataFrame(all_stocks), market='tw')
        logger.info(f"更新台灣上櫃公司成功，共 {len(all_stocks)} 筆")
    else:
        raise Exception("未找到任何台灣上櫃公司資料")

def execute_us_stock_list_update(remarks: str = None):
    """
    美國上市公司清單更新任務執行器
    """
    logger.info("開始更新美國上市公司清單...")
    from stock_Django.mySQL_OP import OP_Fun
    from stock_Django.scraper_us import get_us_stock_list
    
    op = OP_Fun()
    us_stocks = get_us_stock_list()
    if not us_stocks.empty:
        op.upsert_stocks_info(us_stocks, market='us')
        logger.info(f"更新美國上市公司成功，共 {len(us_stocks)} 筆")
    else:
        raise Exception("未找到任何美國上市公司資料")

def execute_sync_notebooklm(remarks: str = None):
    """
    自動化確認 NotebookLM 有無新筆記或新內容，並下載同步至 Wiki 知識庫中，接著自動標記關聯與重建索引。
    """
    logger.info("開始檢查 NotebookLM 筆記本更新狀態...")
    import sys
    import json
    import re
    from datetime import datetime
    
    # 加入 notebooklm-mcp-cli 模組路徑
    sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "notebooklm-mcp-cli", "src")))
    try:
        from notebooklm_tools.core.auth import load_cached_tokens
        from notebooklm_tools.core.client import NotebookLMClient
        from notebooklm_tools.services.sources import get_source_content
    except ImportError as e:
        logger.error(f"無法載入 NotebookLM CLI 元件: {e}")
        raise e
        
    tokens = load_cached_tokens()
    if not tokens:
        raise Exception("未找到 NotebookLM 認證資訊，請先執行 'nlm login' 進行登入。")
        
    client = NotebookLMClient(
        cookies=tokens.cookies,
        csrf_token=tokens.csrf_token,
        session_id=tokens.session_id,
        build_label=getattr(tokens, "build_label", "")
    )
    
    # 讀取雲端筆記本清單
    try:
        list_res = client.list_notebooks()
        if isinstance(list_res, list):
            notebooks = list_res
        else:
            notebooks = list_res.get("notebooks", [])
    except Exception as e:
        raise Exception(f"獲取 Notebook 列表失敗: {e}")
        
    logger.info(f"成功取得雲端筆記本清單，共 {len(notebooks)} 個筆記本")
    
    # 本地狀態檔案，紀錄上次同步的修改時間與已同步的 sources 對照
    state_file = r"C:\Users\許廷宇\.gemini\config\knowledge\references\notebooklm\sync_state.json"
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    
    state = {}
    if os.path.exists(state_file):
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                raw_state = json.load(f)
                # 相容舊格式轉換
                if "notebooks" not in raw_state:
                    state = {"notebooks": {}}
                    for nb_id, modified_at in raw_state.items():
                        state["notebooks"][nb_id] = {
                            "modified_at": modified_at,
                            "sources": {}  # 舊 sources 留空，待本次或後續同步補全
                        }
                else:
                    state = raw_state
        except Exception:
            state = {"notebooks": {}}
    else:
        state = {"notebooks": {}}
            
    def get_val(obj, key, default=""):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    # 決定要同步的筆記本
    target_notebooks = []
    filter_keyword = (remarks or "").strip().lower()
    
    notebook_states = state.setdefault("notebooks", {})
    
    for nb in notebooks:
        nb_id = get_val(nb, "id")
        nb_title = get_val(nb, "title")
        nb_modified = get_val(nb, "modified_at")
        
        # 如果 remarks 指定了關鍵字，進行過濾
        if filter_keyword and filter_keyword not in nb_title.lower() and filter_keyword not in nb_id:
            continue
            
        nb_record = notebook_states.get(nb_id, {})
        last_sync_modified = nb_record.get("modified_at", "")
        # 如果最後修改時間不同，則判定有更新
        if nb_modified != last_sync_modified:
            target_notebooks.append(nb)
            
    if not target_notebooks:
        logger.info("所有筆記本皆在最新狀態，無須同步。")
        return "所有筆記本皆在最新狀態，無須同步。"
        
    logger.info(f"偵測到有 {len(target_notebooks)} 個筆記本需要同步更新。")
    
    synced_files = []
    for nb in target_notebooks:
        nb_id = get_val(nb, "id")
        nb_title = get_val(nb, "title")
        logger.info(f"開始同步筆記本: {nb_title} ({nb_id})...")
        
        nb_record = notebook_states.setdefault(nb_id, {
            "modified_at": "",
            "sources": {}
        })
        synced_sources = nb_record.setdefault("sources", {})
        
        try:
            sources = client.get_notebook_sources_with_types(nb_id)
            logger.info(f"筆記本包含 {len(sources)} 個 sources")
            
            for src in sources:
                src_id = get_val(src, "id")
                src_title = get_val(src, "title")
                
                if not src_title or src_title.strip() == "":
                    continue
                    
                # 檔名清理，移除特殊字元
                clean_title = re.sub(r'[\\/*?:"<>|#]', "", src_title).strip()
                clean_title = clean_title.replace(" ", "_")
                dest_filename = f"{clean_title}.md"
                dest_path = os.path.join(r"C:\Users\許廷宇\.gemini\config\knowledge\references\notebooklm", dest_filename)
                
                # 方案 B：如果該 source_id 已經存在於同步狀態中，且本地實體檔案確實存在，則直接跳過下載
                if src_id in synced_sources and os.path.exists(dest_path):
                    logger.info(f" - [跳過已存在] source: {src_title}")
                    continue
                    
                logger.info(f" - [下載中] 讀取 source: {src_title}...")
                try:
                    src_content_res = get_source_content(client, src_id)
                    raw_text = src_content_res.get("content", "")
                    
                    if not raw_text.strip():
                        continue
                    
                    # 轉義內文中的 [[ 與 ]]，防止 Obsidian 誤判為 wikilink 產生空檔案
                    raw_text = raw_text.replace("[[", "\\[\\[").replace("]]", "\\]\\]")
                        
                    # 加上 Frontmatter，用雙引號包裹 title 與 source 以免冒號等特殊字元導致 YAML 解析失敗
                    safe_title = src_title.replace('"', '\\"')
                    safe_source = f"NotebookLM ({nb_title})".replace('"', '\\"')
                    frontmatter = f"---\ntitle: \"{safe_title}\"\ntype: references\ndate: {datetime.now().strftime('%Y-%m-%d')}\nsource: \"{safe_source}\"\n---\n\n"
                    
                    with open(dest_path, "w", encoding="utf-8") as f:
                        f.write(frontmatter + raw_text)
                    synced_files.append(dest_path)
                    
                    # 更新 source 同步狀態
                    synced_sources[src_id] = {
                        "title": src_title,
                        "last_synced": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                except Exception as e:
                    logger.error(f"   下載 source '{src_title}' 失敗: {e}")
                    
            # 同步成功，更新本地修改時間狀態
            nb_record["modified_at"] = get_val(nb, "modified_at")
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"同步筆記本 '{nb_title}' 失敗: {e}")
            
    if synced_files:
        logger.info(f"成功同步了 {len(synced_files)} 個文件。開始自動後處理優化...")
        
        # 1. 執行 README 重命名防同名衝突
        changed_paths = synced_files
        try:
            from app.rename_readme_files import rename_readmes
            renamed_paths = rename_readmes(changed_files=synced_files)
            if renamed_paths:
                # 重新組合變更路徑
                changed_paths = []
                for p in synced_files:
                    for r in renamed_paths:
                        if os.path.dirname(p) == os.path.dirname(r) and os.path.basename(p).lower() == "readme.md":
                            changed_paths.append(r)
                            break
                    else:
                        if os.path.exists(p):
                            changed_paths.append(p)
        except Exception as e:
            logger.error(f"更名後處理失敗: {e}")
            
        # 2. 執行關聯性標籤標記 (僅增量變更檔案)
        try:
            from app.annotate_wiki_relations import annotate_files
            annotate_files(changed_files=changed_paths)
        except Exception as e:
            logger.error(f"關聯性標記失敗: {e}")
            
        # 3. 增量更新 RAG 向量索引庫
        try:
            from llm_wiki.services.gemini_rag import rag_service
            rag_service.incremental_update(changed_files=changed_paths)
            logger.info("RAG 向量索引庫增量更新成功！")
        except Exception as e:
            logger.error(f"RAG 索引增量更新失敗: {e}")
            
        return f"成功同步了 {len(synced_files)} 個文件並增量重建索引。"
        
    return "無須同步。"

# 任務執行器對照表
EXECUTORS = {
    "tw_stock_cost": execute_tw_stock_cost,
    "us_stock_cost": execute_us_stock_cost,
    "tw_stock_price_only": execute_tw_stock_price_only,
    "us_stock_price_only": execute_us_stock_price_only,
    "twse_investor": execute_twse_investor,
    "tpex_investor": execute_tpex_investor,
    "us_investor": execute_us_investor,
    "tw_listed_list_update": execute_tw_listed_list_update,
    "tw_otc_list_update": execute_tw_otc_list_update,
    "us_stock_list_update": execute_us_stock_list_update,
    "sync_notebooklm": execute_sync_notebooklm
}
