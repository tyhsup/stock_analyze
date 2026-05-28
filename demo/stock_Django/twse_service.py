"""
twse_service.py

透過呼叫本地編譯的 twse-cli.exe，即時取得 TWSE/TPEX 市場資訊。
提供以下功能：
  - get_sbl_volume(ticker): 取得指定台股代碼當日可借券賣出股數
  - _run_cli(args): 通用 CLI 呼叫包裝，含超時與錯誤處理

快取策略：使用 dict-based in-memory cache，TTL = 5 分鐘，
避免對同一 ticker 在短時間內重複呼叫 CLI，防止流量過多。
"""

import subprocess
import json
import logging
import os
import time
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# twse-cli.exe 絕對路徑（固定指向 twse-cli-v2/bin 目錄）
_CLI_BASE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "twse-cli-v2", "bin"
)
_CLI_EXE = os.path.join(_CLI_BASE_DIR, "twse-cli.exe")

# In-memory 快取：{ key: { "data": ..., "ts": float } }
_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_TTL_SECONDS = 300  # 5 分鐘


def _is_cache_valid(key: str) -> bool:
    """判斷快取是否仍在有效期內。"""
    entry = _CACHE.get(key)
    if not entry:
        return False
    return (time.time() - entry["ts"]) < _CACHE_TTL_SECONDS


def _run_cli(args: list, timeout: int = 30) -> Optional[dict]:
    """
    通用 CLI 呼叫包裝。
    回傳 dict（JSON 解析結果），或在失敗時回傳 None。

    參數：
        args: CLI 子命令與旗標，例如 ["sbl", "--agent"]
        timeout: 等待 CLI 回應的最長時間（秒）
    """
    if not os.path.exists(_CLI_EXE):
        logger.error(f"twse-cli.exe 不存在於路徑：{_CLI_EXE}")
        return None

    cmd = [_CLI_EXE] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )
        raw = result.stdout.strip()
        if not raw:
            logger.warning(f"twse-cli 無輸出，指令：{cmd}，stderr：{result.stderr[:200]}")
            return None
        return json.loads(raw)
    except subprocess.TimeoutExpired:
        logger.error(f"twse-cli 呼叫超時（>{timeout}s），指令：{cmd}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"twse-cli JSON 解析失敗：{e}，原始輸出前 200 字：{result.stdout[:200]}")
        return None
    except Exception as e:
        logger.error(f"twse-cli 呼叫異常：{e}")
        return None


def get_sbl_volume(ticker: str) -> Optional[str]:
    """
    取得指定台股代碼當日可借券賣出股數（Securities Borrowing and Lending）。

    回傳值：
        格式化後的股數字串（如 "12,700,800"），或 None（資料不存在 / 非台股）。

    注意：
        - 上市股（.TW 後綴或純數字）讀取 TWSEAvailableVolume
        - 上櫃股（.TWO 後綴）讀取 GRETAIAvailableVolume
        - CLI 一次回傳全市場所有股票，故以快取 key="sbl_all" 整批儲存

    參數：
        ticker: 股票代碼，例如 "2330"、"2330.TW"、"6488.TWO"
    """
    # 標準化：去除後綴取得純代碼
    clean = str(ticker).upper().replace(".TW", "").replace(".TWO", "").strip()
    is_otc = ".TWO" in str(ticker).upper()

    # 不處理非台股（含英文字母的美股）
    if not clean.isdigit():
        return None

    cache_key = "sbl_all"

    # 嘗試讀取快取
    if not _is_cache_valid(cache_key):
        logger.info("SBL 快取已過期或不存在，正在呼叫 twse-cli sbl...")
        data = _run_cli(["sbl", "--agent"])
        if data and "results" in data:
            # 將陣列轉為以代碼為 key 的 dict，加速後續查詢
            index: Dict[str, dict] = {}
            for item in data["results"]:
                twse_code = str(item.get("TWSECode", "")).strip()
                gretai_code = str(item.get("GRETAICode", "")).strip()
                if twse_code and twse_code != "_":
                    index[twse_code] = item
                if gretai_code and gretai_code != "_":
                    index[gretai_code] = item
            _CACHE[cache_key] = {"data": index, "ts": time.time()}
        else:
            logger.warning("SBL 資料取得失敗，回傳 None")
            return None

    index = _CACHE[cache_key]["data"]
    item = index.get(clean)

    if not item:
        logger.debug(f"SBL 資料中找不到代碼：{clean}")
        return None

    # 依上市/上櫃選擇對應欄位
    if is_otc:
        vol = item.get("GRETAIAvailableVolume", "")
    else:
        vol = item.get("TWSEAvailableVolume", "")

    # 若上市欄位為空，嘗試上櫃欄位（少數股票同時存在兩市場）
    if not vol or vol.strip() == "":
        vol = item.get("GRETAIAvailableVolume", "") or item.get("TWSEAvailableVolume", "")

    return vol.strip() if vol and vol.strip() else None
