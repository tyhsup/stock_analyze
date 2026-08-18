import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from django.db import connection, transaction

logger = logging.getLogger(__name__)

# 取得專案根目錄 (e:\Infinity\mydjango)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# 定義 CLI 工具路徑
TWSE_CLI_PATH = BASE_DIR / "twse-cli-v2" / "bin" / "twse-cli.exe"
TPEX_CLI_PATH = BASE_DIR / "tpex-cli" / "bin" / "tpex-pp-cli.exe"


def parse_roc_date(roc_date_str: Any) -> Optional[str]:
    """
    解析民國年字串 (如 1150818 或 115/08/18) 轉為西元 YYYY-MM-DD。
    防禦 6 碼與 7 碼民國年。
    """
    if not roc_date_str:
        return None
    s = str(roc_date_str).strip().replace('/', '').replace('-', '')
    if not s.isdigit():
        return None
    
    try:
        if len(s) == 7:
            year = int(s[:3]) + 1911
            month = int(s[3:5])
            day = int(s[5:])
        elif len(s) == 6:
            year = int(s[:2]) + 1911
            month = int(s[2:4])
            day = int(s[4:])
        else:
            return None
        return f"{year:04d}-{month:02d}-{day:02d}"
    except (ValueError, IndexError):
        return None


def clean_numeric(value: Any, default: float = 0.0) -> float:
    """
    清理可能帶有逗號、空字串、橫線或無效字元的數值，安全轉為 float。
    """
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    
    s = str(value).strip().replace(',', '').replace('%', '').replace('+', '')
    if not s or s in ('--', 'N/A', 'NaN', 'null', 'None'):
        return default
    try:
        return float(s)
    except ValueError:
        return default


def get_latest_market_date() -> str:
    """
    從 stock_investor 取得最新交易日期，作為 TWSE 缺少內建日期時的基準日期；
    若查無紀錄則 fallback 至今日日期 (YYYY-MM-DD)。
    """
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT MAX(date) FROM stock_investor")
            row = cursor.fetchone()
            if row and row[0]:
                d = row[0]
                return d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)
    except Exception as e:
        logger.warning(f"無法從 stock_investor 取得最新日期: {e}")
    return datetime.now().strftime('%Y-%m-%d')


def fetch_twse_margin(target_date: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    透過 twse-cli-v2 抓取集中市場 (上市) 全體股票當日融資融券餘額。
    """
    if not TWSE_CLI_PATH.exists():
        logger.error(f"TWSE CLI 執行檔不存在: {TWSE_CLI_PATH}")
        return []

    date_str = target_date or get_latest_market_date()
    cmd = [str(TWSE_CLI_PATH), "exchange-report", "list-exchangereport-12", "--agent"]
    
    logger.info(f"執行 TWSE 融資融券 CLI: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=60,
            cwd=str(BASE_DIR)
        )
        if result.returncode != 0:
            logger.error(f"TWSE CLI 執行失敗 (code {result.returncode}): {result.stderr}")
            return []
        
        stdout_clean = result.stdout.strip()
        if not stdout_clean:
            logger.warning("TWSE CLI 回傳空內容。")
            return []
        
        raw_data = json.loads(stdout_clean)
        # 支援 list、results、data、records
        if isinstance(raw_data, list):
            records = raw_data
        elif isinstance(raw_data, dict):
            records = raw_data.get('results', raw_data.get('data', raw_data.get('records', [])))
        else:
            records = []
        
        parsed_list = []
        for item in records:
            if not isinstance(item, dict):
                continue
            
            # TWSE JSON 支援 Key matching 與 Index-based fallback
            vals = list(item.values())
            keys = list(item.keys())
            
            # 取得股票代號 (通常為第 0 個欄位)
            symbol = str(vals[0] if len(vals) > 0 else '').strip()
            if not symbol or not symbol[0].isalnum():
                continue
            
            # 根據 TWSE list-exchangereport-12 欄位固定順序映射：
            # 0: 代號, 1: 名稱, 2: 融券今日餘額, 3: 融券前日餘額, 4: 融券現券償還, 5: 融券買進, 6: 融券賣出, 7: 融券限額
            # 8: 融資今日餘額, 9: 融資前日餘額, 10: 融資現金償還, 11: 融資買進, 12: 融資賣出, 13: 融資限額
            short_balance = clean_numeric(vals[2] if len(vals) > 2 else 0)
            short_covering = clean_numeric(vals[5] if len(vals) > 5 else 0)
            short_sale = clean_numeric(vals[6] if len(vals) > 6 else 0)
            margin_balance = clean_numeric(vals[8] if len(vals) > 8 else 0)
            margin_purchase = clean_numeric(vals[11] if len(vals) > 11 else 0)
            margin_sales = clean_numeric(vals[12] if len(vals) > 12 else 0)
            
            parsed_list.append({
                'date': date_str,
                'number': symbol,
                'margin_purchase': margin_purchase,
                'margin_sales': margin_sales,
                'margin_balance': margin_balance,
                'short_sale': short_sale,
                'short_covering': short_covering,
                'short_balance': short_balance,
                'margin_utilization_rate': 0.0,
                'short_utilization_rate': 0.0
            })
        
        logger.info(f"TWSE 成功解析 {len(parsed_list)} 筆上市融資融券資料 (日期: {date_str})。")
        return parsed_list
    except subprocess.TimeoutExpired:
        logger.error("TWSE CLI 執行逾時 (60s)。")
        return []
    except Exception as e:
        logger.error(f"fetch_twse_margin 發生例外: {e}")
        return []


def fetch_tpex_margin() -> List[Dict[str, Any]]:
    """
    透過 tpex-cli 抓取櫃買市場 (上櫃) 全體股票當日融資融券餘額。
    """
    if not TPEX_CLI_PATH.exists():
        logger.error(f"TPEx CLI 執行檔不存在: {TPEX_CLI_PATH}")
        return []

    cmd = [str(TPEX_CLI_PATH), "tpex-mainboard-margin-balance", "--agent"]
    logger.info(f"執行 TPEx 融資融券 CLI: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=60,
            cwd=str(BASE_DIR)
        )
        if result.returncode != 0:
            logger.error(f"TPEx CLI 執行失敗 (code {result.returncode}): {result.stderr}")
            return []
        
        stdout_clean = result.stdout.strip()
        if not stdout_clean:
            logger.warning("TPEx CLI 回傳空內容。")
            return []
        
        raw_data = json.loads(stdout_clean)
        if isinstance(raw_data, list):
            records = raw_data
        elif isinstance(raw_data, dict):
            records = raw_data.get('results', raw_data.get('data', raw_data.get('records', [])))
        else:
            records = []
        
        fallback_date = get_latest_market_date()
        parsed_list = []
        for item in records:
            if not isinstance(item, dict):
                continue
            
            symbol = str(item.get("SecuritiesCompanyCode", "")).strip()
            if not symbol or not symbol[0].isalnum():
                continue
            
            raw_roc_date = item.get("Date")
            date_str = parse_roc_date(raw_roc_date) or fallback_date
            
            margin_purchase = clean_numeric(item.get("MarginPurchase"))
            margin_sales = clean_numeric(item.get("MarginSales"))
            margin_balance = clean_numeric(item.get("MarginPurchaseBalance"))
            short_covering = clean_numeric(item.get("ShortConvering"))
            short_sale = clean_numeric(item.get("ShortSale"))
            short_balance = clean_numeric(item.get("ShortSaleBalance"))
            margin_util_rate = clean_numeric(item.get("MarginPurchaseUtilizationRate"))
            short_util_rate = clean_numeric(item.get("ShortSaleUtilizationRate"))
            
            parsed_list.append({
                'date': date_str,
                'number': symbol,
                'margin_purchase': margin_purchase,
                'margin_sales': margin_sales,
                'margin_balance': margin_balance,
                'short_sale': short_sale,
                'short_covering': short_covering,
                'short_balance': short_balance,
                'margin_utilization_rate': margin_util_rate,
                'short_utilization_rate': short_util_rate
            })
            
        logger.info(f"TPEx 成功解析 {len(parsed_list)} 筆上櫃融資融券資料。")
        return parsed_list
    except subprocess.TimeoutExpired:
        logger.error("TPEx CLI 執行逾時 (60s)。")
        return []
    except Exception as e:
        logger.error(f"fetch_tpex_margin 發生例外: {e}")
        return []


def save_margin_data_to_db(data_list: List[Dict[str, Any]], batch_size: int = 500) -> int:
    """
    將清洗後的融資融券資料批次寫入 MySQL stock_margin_balance 表，
    採用 INSERT ... ON DUPLICATE KEY UPDATE 保證冪等性 (Idempotency)。
    """
    if not data_list:
        logger.warning("無資料需要寫入資料庫。")
        return 0

    sql = """
        INSERT INTO stock_margin_balance (
            date, number, margin_purchase, margin_sales, margin_balance,
            short_sale, short_covering, short_balance,
            margin_utilization_rate, short_utilization_rate
        ) VALUES (
            %(date)s, %(number)s, %(margin_purchase)s, %(margin_sales)s, %(margin_balance)s,
            %(short_sale)s, %(short_covering)s, %(short_balance)s,
            %(margin_utilization_rate)s, %(short_utilization_rate)s
        )
        ON DUPLICATE KEY UPDATE
            margin_purchase = VALUES(margin_purchase),
            margin_sales = VALUES(margin_sales),
            margin_balance = VALUES(margin_balance),
            short_sale = VALUES(short_sale),
            short_covering = VALUES(short_covering),
            short_balance = VALUES(short_balance),
            margin_utilization_rate = VALUES(margin_utilization_rate),
            short_utilization_rate = VALUES(short_utilization_rate);
    """
    
    total_saved = 0
    with transaction.atomic():
        with connection.cursor() as cursor:
            for i in range(0, len(data_list), batch_size):
                chunk = data_list[i:i + batch_size]
                cursor.executemany(sql, chunk)
                total_saved += len(chunk)
            
    logger.info(f"成功批次儲存/更新 {total_saved} 筆融資融券資料至 stock_margin_balance。")
    return total_saved


def sync_all_margin_data(target_date: Optional[str] = None) -> Dict[str, int]:
    """
    同步全市場（上市 + 上櫃）融資融券資料。
    """
    twse_data = fetch_twse_margin(target_date=target_date)
    tpex_data = fetch_tpex_margin()
    
    twse_saved = save_margin_data_to_db(twse_data) if twse_data else 0
    tpex_saved = save_margin_data_to_db(tpex_data) if tpex_data else 0
    
    return {
        'twse_count': twse_saved,
        'tpex_count': tpex_saved,
        'total_count': twse_saved + tpex_saved
    }
