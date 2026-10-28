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

from .self_healing_scheduler import SelfHealingScheduler
_healing_scheduler = SelfHealingScheduler()


def execute_tw_stock_cost(remarks: str = None):
    """
    台股股價與籌碼面更新任務執行器
    若 remarks 中包含特定的股票代碼（如 2330），則執行單一更新，否則全量更新。
    已整合自癒保護機制。
    """
    def _task():
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
            
    _healing_scheduler.execute_with_healing(_task)

def execute_us_stock_cost(remarks: str = None):
    """
    美股股價與籌碼面更新任務執行器
    若 remarks 中包含特定的股票代碼（如 AAPL），則執行單一更新，否則全量更新。
    已整合自癒保護機制。
    """
    def _task():
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
            
    _healing_scheduler.execute_with_healing(_task)

def execute_twse_investor(remarks: str = None):
    """
    台股三大法人買賣超 (上市) 更新任務執行器
    已整合自癒保護機制。
    """
    def _task():
        logger.info("開始更新台股三大法人買賣超 (上市)")
        manager = StockInvestorManager()
        manager.update_investor_data()
        
    _healing_scheduler.execute_with_healing(_task)

def execute_tpex_investor(remarks: str = None):
    """
    台股三大法人買賣超 (上櫃) 更新任務執行器
    已整合自癒保護機制。
    """
    def _task():
        logger.info("開始更新台股三大法人買賣超 (上櫃)")
        manager = TPExInvestorManager()
        manager.update_all_tpex_investors()
        
    _healing_scheduler.execute_with_healing(_task)

def execute_us_investor(remarks: str = None):
    """
    美股三大法人持股更新任務執行器
    若 remarks 中指定了代號 (如 AAPL,NVDA)，則更新該清單。
    否則預設更新 ['AAPL', 'NVDA', 'TSLA']。
    已整合自癒保護機制。
    """
    def _task():
        import re
        manager = USStockInvestorManager()
        ticker_str = (remarks or "").strip()
        
        # 僅允許英文字母、逗號及半形/全形空格，防止將錯誤日誌或鎖庫訊息誤判為代碼
        if ticker_str and re.match(r'^[a-zA-Z,\s，]+$', ticker_str):
            tickers = [t.strip().upper() for t in ticker_str.replace("，", ",").split(",") if t.strip()]
        else:
            tickers = ['AAPL', 'NVDA', 'TSLA']
            
        logger.info(f"開始更新美股法人持股: {tickers}")
        manager.update_investor_db(tickers)
        
    _healing_scheduler.execute_with_healing(_task)


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
    state_file = r"E:\Obsidian_knowledge\antigravity_knowledge\knowledge\references\notebooklm\sync_state.json"
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
                dest_path = os.path.join(r"E:\Obsidian_knowledge\antigravity_knowledge\knowledge\references\notebooklm", dest_filename)
                
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

def execute_update_macro_data(remarks: str = None):
    """
    宏觀總體經濟數據 (FRED & 台灣央行) 更新任務執行器
    remarks 可以為 "us", "tw" 或空 (代表兩者都更新)。
    """
    def _task():
        country = (remarks or "").strip().lower()
        from django.core.management import call_command
        logger.info(f"開始更新總體經濟數據，國家篩選: {country or '全部'}")
        if country in ["us", "tw"]:
            call_command('update_macro_data', country=country)
        else:
            call_command('update_macro_data')
    _healing_scheduler.execute_with_healing(_task)


def _get_taipei_today_str() -> str:
    """
    取得台灣標準時間 (Asia/Taipei, UTC+8) 的當天日期字串 (YYYY-MM-DD)。
    防禦容器或 Django UTC 時區設定導致跨日判斷偏差。
    """
    from datetime import datetime, timezone, timedelta
    tz_taipei = timezone(timedelta(hours=8))
    return datetime.now(tz_taipei).strftime("%Y-%m-%d")


def _classify_ticker_market(ticker: str) -> str:
    """
    根據股票代號格式判斷所屬市場 ('tw' 或 'us')。
    """
    t_upper = str(ticker).strip().upper()
    clean = t_upper.replace(".TWO", "").replace(".TW", "")
    if clean.isdigit() or t_upper.endswith(".TW") or t_upper.endswith(".TWO"):
        return "tw"
    return "us"


def _normalize_date_str(raw_val) -> str:
    """
    將各種可能的日期表示 (YYYY-MM-DD, YYYY/MM/DD, ISO 8601) 統一正規化為 YYYY-MM-DD。
    若無法解析則回傳空字串。
    """
    if not raw_val:
        return ""
    s = str(raw_val).strip()
    if not s:
        return ""
    from datetime import datetime
    s_head = s[:10].replace("/", "-")
    try:
        return datetime.strptime(s_head, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M", "%Y/%m/%d"):
            try:
                return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
    return ""


def _resolve_news_target_tickers(remarks: str = None, market_filter: str = "all") -> list:
    """
    解析新聞定時更新之目標個股清單，回傳 [(ticker, market), ...]。
    1. 若 remarks 指定代號 (如 '2330, AAPL')，先執行 Unicode NFKC 正規化與白名單驗證後解析指定代號。
    2. 若 remarks 為空，自動掃描 NewsExcelManager.data_dir 下所有既有個股 Excel 檔案。
    3. 若目錄為空，回退至預設核心觀察股清單。
    """
    import re
    import unicodedata
    from stock_Django.news_excel import NewsExcelManager

    clean_remarks = unicodedata.normalize("NFKC", (remarks or "")).strip()
    # 清理重試時附加的 Error 前綴
    if "Error:" in clean_remarks and "|" in clean_remarks:
        clean_remarks = clean_remarks.split("|", 1)[-1].strip()
    elif clean_remarks.startswith("Error:"):
        clean_remarks = ""

    raw_tickers = []
    if clean_remarks and re.match(r"^[A-Za-z0-9\.\-_,\s]+$", clean_remarks):
        for part in clean_remarks.split(","):
            t = part.strip().upper()
            if t and not t.startswith("_"):
                raw_tickers.append(t)

    if not raw_tickers:
        try:
            news_mgr = NewsExcelManager()
            if news_mgr.data_dir and news_mgr.data_dir.exists():
                for xlsx_path in sorted(news_mgr.data_dir.glob("*.xlsx")):
                    stem = xlsx_path.stem.strip()
                    if not stem or stem.startswith("_") or stem.lower() == "news_data":
                        continue
                    if not re.match(r"^[A-Za-z0-9_]+$", stem):
                        continue
                    # 還原檔名中的底線後綴 (例如 2330_TW -> 2330.TW)
                    if stem.endswith("_TWO"):
                        ticker_name = stem[:-4] + ".TWO"
                    elif stem.endswith("_TW"):
                        ticker_name = stem[:-3] + ".TW"
                    else:
                        ticker_name = stem.upper()
                    raw_tickers.append(ticker_name)
        except Exception as e:
            logger.warning(f"掃描 news_data 目錄失敗，將使用預設觀察清單: {e}")

    if not raw_tickers:
        raw_tickers = ["2330", "2454", "0050", "AAPL", "NVDA", "MSFT", "GOOG"]

    resolved = []
    seen = set()
    mf = (market_filter or "all").strip().lower()
    for t in raw_tickers:
        mkt = _classify_ticker_market(t)
        if mf in ("tw", "us") and mkt != mf:
            continue
        if t not in seen:
            seen.add(t)
            resolved.append((t, mkt))

    return resolved


def _filter_and_analyze_today_news(
    ticker: str,
    raw_items: list,
    today_str: str,
    news_mgr,
    agent
) -> tuple:
    """
    嚴格篩選發布日期為 today_str (YYYY-MM-DD) 之當日新聞，
    執行分析前冪等預檢 (Idempotency)，僅對尚未分析入庫的當日新聞呼叫 AgentNewsAnalyzer，
    最後寫入對應 Excel。

    Returns:
        (today_matched_count, newly_analyzed_count, written_count)
    """
    import concurrent.futures

    if not raw_items:
        return (0, 0, 0)

    # 1. 嚴格當日日期過濾 (僅保留正規化後發布日期 == today_str)
    today_items = []
    seen_batch = set()
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        item_date = _normalize_date_str(item.get("日期", ""))
        title = str(item.get("標題", "")).strip()
        if item_date == today_str and title:
            item["日期"] = item_date
            dedup_key = (title.lower(), item_date)
            if dedup_key not in seen_batch:
                seen_batch.add(dedup_key)
                today_items.append(item)

    if not today_items:
        return (0, 0, 0)

    # 2. 分析前冪等預檢：讀取該檔股票既有當日新聞，避免重複呼叫 LLM 消耗配額
    existing_keys = set()
    existing_links = set()
    try:
        existing_rows = news_mgr.read_news(ticker, limit=300)
        for row in existing_rows:
            r_date = _normalize_date_str(row.get("日期", ""))
            r_title = str(row.get("標題", "")).strip()
            r_link = str(row.get("連結", "")).strip()
            r_summary = str(row.get("分析摘要", "")).strip()
            # 已存在且具備有效分析摘要者視為已完成
            if r_date == today_str and r_summary and not r_summary.startswith(("分析異常", "AI 分析")):
                if r_title:
                    existing_keys.add((r_title, r_date))
                if r_link:
                    existing_links.add(r_link)
    except Exception as e:
        logger.warning(f"[{ticker}] 讀取既有新聞進行預檢時發生非阻斷異常: {e}")

    items_to_analyze = []
    for item in today_items:
        t_str = str(item.get("標題", "")).strip()
        d_str = str(item.get("日期", "")).strip()[:10]
        l_str = str(item.get("連結", "")).strip()
        if (t_str, d_str) in existing_keys or (l_str and l_str in existing_links):
            continue
        items_to_analyze.append(item)

    if not items_to_analyze:
        logger.info(f"[{ticker}] 當日 ({today_str}) 共 {len(today_items)} 則新聞皆已分析入庫，跳過重複 AI 分析。")
        return (len(today_items), 0, 0)

    # 3. 執行 AI 新聞分析 (Gemini 3.8 Flash -> Ollama -> 規則兜底)
    enhanced_results = []
    for item in items_to_analyze:
        lang_label = item.get("語言", "zh-TW")
        if agent:
            ai_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                future = ai_executor.submit(
                    agent.analyze_news,
                    {
                        "title": item.get("標題", ""),
                        "date": item.get("日期", ""),
                        "content": item.get("內容", ""),
                        "link": item.get("連結", ""),
                        "source": item.get("來源", ""),
                        "language": lang_label,
                    },
                    is_recent=True,
                )
                analysis = future.result(timeout=120)
                item["正負分析"] = analysis.get("positive_negative_analysis", "中立")
                item["市場"] = analysis.get("market", item.get("市場", "TW"))
                conf_val = float(analysis.get("confidence", 0.85))
                item["信心度"] = f"{conf_val:.0%}"
                item["影響範疇"] = analysis.get("impact_scope", "短期")
                item["分析摘要"] = analysis.get("reasoning_summary", "分析完成。")
            except concurrent.futures.TimeoutError:
                logger.warning(f"[{ticker}] AI 新聞分析逾時 (120s): {item.get('標題', '')[:30]}")
                item["信心度"] = "N/A"
                item["影響範疇"] = "短期"
                item["分析摘要"] = "AI 分析逾時 (120s)"
            except Exception as e:
                logger.warning(f"[{ticker}] AI 新聞分析異常: {e}")
                item["信心度"] = "N/A"
                item["影響範疇"] = "短期"
                item["分析摘要"] = f"分析異常: {str(e)[:80]}"
            finally:
                ai_executor.shutdown(wait=False)
        else:
            item["信心度"] = "N/A"
            item["影響範疇"] = "短期"
            item["分析摘要"] = "AI 分析未啟用"

        enhanced_results.append(item)

    # 4. 寫入 Excel
    written = news_mgr.write_news(ticker, enhanced_results)
    return (len(today_items), len(enhanced_results), written)


def execute_daily_news_update(remarks: str = None, market_filter: str = "all"):
    """
    定時更新當天日期 (YYYY-MM-DD) 的所有新聞並進行 AI 新聞分析執行器。
    支援透過 remarks 指定單一或多個股票代號，若留空則自動掃描 news_data 追蹤清單與總體市場頭條。
    已整合 SelfHealingScheduler 自癒保護與單一標的異常隔離。
    """
    def _task():
        from stock_Django.news_scraper_cnyes import CnyesScraper
        from stock_Django.news_scraper_en import CnbcCliScraper
        from stock_Django.news_excel import NewsExcelManager
        from stock_Django.agent_news_analyzer import AgentNewsAnalyzer
        from stock_Django.data_freshness import _set_status, _news_status_key

        today_str = _get_taipei_today_str()
        targets = _resolve_news_target_tickers(remarks=remarks, market_filter=market_filter)
        logger.info(
            f"[DailyNews] 開始執行當日 ({today_str}) 新聞定時更新與 AI 分析 | "
            f"市場篩選: {market_filter} | 目標個股數: {len(targets)}"
        )

        cnyes_scraper = CnyesScraper()
        cnbc_scraper = CnbcCliScraper()
        news_mgr = NewsExcelManager()
        try:
            agent = AgentNewsAnalyzer()
        except Exception as e:
            logger.error(f"[DailyNews] 初始化 AgentNewsAnalyzer 失敗，將降級處理: {e}")
            agent = None

        total_matched = 0
        total_analyzed = 0
        total_written = 0
        failed_tickers = []

        for idx, (ticker, mkt) in enumerate(targets, start=1):
            news_key = _news_status_key(ticker)
            _set_status(news_key, "running", 20, f"[排程] 正在抓取 {ticker} 當日 ({today_str}) 新聞...")
            try:
                raw_results = []
                if mkt == "tw":
                    zh_items = cnyes_scraper.fetch_news(ticker, market="tw", limit=30)
                    for it in zh_items:
                        it["語言"] = "zh-TW"
                    raw_results.extend(zh_items)
                else:
                    zh_items = cnyes_scraper.fetch_news(ticker, market="us", limit=20)
                    for it in zh_items:
                        it["語言"] = "zh-TW"
                    raw_results.extend(zh_items)

                    en_items = cnbc_scraper.fetch_news(ticker, limit=25)
                    for it in en_items:
                        it["語言"] = "en"
                    raw_results.extend(en_items)

                matched, analyzed, written = _filter_and_analyze_today_news(
                    ticker=ticker,
                    raw_items=raw_results,
                    today_str=today_str,
                    news_mgr=news_mgr,
                    agent=agent,
                )
                total_matched += matched
                total_analyzed += analyzed
                total_written += written

                _set_status(
                    news_key,
                    "done",
                    100,
                    f"完成 {ticker} 當日 ({today_str}) 新聞更新：符合當日 {matched} 則，新分析 {analyzed} 則，寫入 {written} 則",
                )
                logger.info(
                    f"[DailyNews] ({idx}/{len(targets)}) {ticker} [{mkt.upper()}] 完成 | "
                    f"當日新聞: {matched} 則 | 新 AI 分析: {analyzed} 則 | 新增寫入: {written} 則"
                )
            except Exception as ticker_err:
                failed_tickers.append(ticker)
                logger.error(f"[DailyNews] 個股 {ticker} 新聞更新發生異常 (已隔離): {ticker_err}")
                _set_status(news_key, "error", 0, f"排程更新異常: {str(ticker_err)[:80]}")

        # 若未指定特定個股，同步抓取當日市場總體頭條寫入 _general.xlsx
        clean_rem = (remarks or "").strip()
        if not clean_rem or clean_rem.startswith("Error:"):
            try:
                general_raw = []
                mf = (market_filter or "all").strip().lower()
                if mf in ("all", "tw"):
                    tw_gen = cnyes_scraper.search_news("台股", limit=25)
                    for it in tw_gen:
                        it["語言"] = "zh-TW"
                    general_raw.extend(tw_gen)
                if mf in ("all", "us"):
                    us_gen = cnbc_scraper.fetch_news("US Market", limit=20)
                    for it in us_gen:
                        it["語言"] = "en"
                    general_raw.extend(us_gen)

                if general_raw:
                    g_matched, g_analyzed, g_written = _filter_and_analyze_today_news(
                        ticker="_general",
                        raw_items=general_raw,
                        today_str=today_str,
                        news_mgr=news_mgr,
                        agent=agent,
                    )
                    total_matched += g_matched
                    total_analyzed += g_analyzed
                    total_written += g_written
                    logger.info(
                        f"[DailyNews] 總體市場頭條 (_general) 完成 | "
                        f"當日新聞: {g_matched} 則 | 新 AI 分析: {g_analyzed} 則 | 新增寫入: {g_written} 則"
                    )
            except Exception as gen_err:
                logger.warning(f"[DailyNews] 總體市場頭條 (_general) 更新異常 (非阻斷): {gen_err}")

        if targets and len(failed_tickers) == len(targets):
            raise RuntimeError(f"所有目標個股新聞更新均失敗: {failed_tickers}")

        summary_msg = (
            f"當日 ({today_str}) 新聞定時更新與 AI 分析完成：共掃描 {len(targets)} 檔標的，"
            f"符合當日新聞 {total_matched} 則，執行 AI 分析 {total_analyzed} 則，新增寫入 {total_written} 則。"
        )
        logger.info(f"[DailyNews] {summary_msg}")
        return summary_msg

    return _healing_scheduler.execute_with_healing(_task)


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
    "sync_notebooklm": execute_sync_notebooklm,
    "update_macro_data": execute_update_macro_data,
    "update_macro_data_us": lambda remarks=None: execute_update_macro_data(remarks="us"),
    "update_macro_data_tw": lambda remarks=None: execute_update_macro_data(remarks="tw"),
    "daily_news_update": lambda remarks=None: execute_daily_news_update(remarks=remarks, market_filter="all"),
    "tw_news_daily": lambda remarks=None: execute_daily_news_update(remarks=remarks, market_filter="tw"),
    "us_news_daily": lambda remarks=None: execute_daily_news_update(remarks=remarks, market_filter="us"),
}
