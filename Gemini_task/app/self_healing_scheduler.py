import time
import sys
import os
import logging
import traceback
from typing import Callable, Any

# 將當前路徑的父目錄與 demo 目錄加入 Python path，以便正確 import
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "demo"))
if DEMO_DIR not in sys.path:
    sys.path.insert(0, DEMO_DIR)

from stock_Django.adaptive_retry import CircuitBreaker
from stock_Django.scraper_utils import report_429, delay_controller
from stock_Django.mySQL_OP import OP_Fun
from sqlalchemy import text

logger = logging.getLogger("scheduler.self_healing")

class SelfHealingScheduler:
    """
    OAV 自癒任務排程器。
    包裹執行任務，具備故障偵測、自適應 PID 控流回饋、資料庫重置與最多 3 次自癒重試。
    """
    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 30.0):
        self.breaker = CircuitBreaker(failure_threshold=failure_threshold, recovery_timeout=recovery_timeout)
        self.max_retries = 3

    def verify_system_state(self) -> bool:
        """
        [Verify] 驗證系統狀態 (輕量化測試)
        測試 MySQL 資料庫連線與基本網路通道。
        """
        logger.info("[Verify] 正在執行輕量化自檢驗證...")
        
        # 1. 驗證資料庫連線
        try:
            op = OP_Fun()
            with op.engine.connect() as conn:
                conn.execute(text("SELECT 1")).fetchone()
            logger.info("[Verify] 資料庫自檢通過")
        except Exception as e:
            logger.error(f"[Verify] 資料庫自檢失敗，資料庫不可用: {e}")
            return False

        # 2. 驗證網路通道 (測試 TWSE 首頁)
        import requests
        try:
            resp = requests.head("https://www.twse.com.tw/zh/page/trading/exchange/MI_INDEX.html", timeout=5)
            if resp.status_code < 500:
                logger.info("[Verify] 網路通道自檢通過")
            else:
                logger.warning(f"[Verify] 網路通道自檢異常，HTTP 狀態碼: {resp.status_code}")
                return False
        except Exception as e:
            logger.error(f"[Verify] 網路通道自檢失敗，無法連線網路: {e}")
            return False

        return True

    def execute_with_healing(self, task_func: Callable[..., Any], *args, **kwargs) -> Any:
        """
        [Observe-Act-Verify] 具備自癒能力的任務執行包裝器
        """
        if not self.breaker.can_execute():
            logger.critical(f"斷路器處於開啟 (OPEN) 狀態。任務 {task_func.__name__} 拒絕執行")
            raise RuntimeError("Service temporarily unavailable due to open circuit breaker.")

        retry_count = 0
        while retry_count <= self.max_retries:
            try:
                # --- Observe: 觀察並執行任務 ---
                if retry_count > 0:
                    logger.info(f"正在嘗試第 {retry_count}/{self.max_retries} 次自癒重試...")
                
                result = task_func(*args, **kwargs)
                
                # 執行成功，觀測並重置斷路器
                self.breaker.observe_success()
                return result

            except Exception as e:
                # --- Observe: 偵測故障根源 ---
                tb = traceback.format_exc()
                logger.error(f"偵測到任務 {task_func.__name__} 執行異常。\nException: {e}\nTraceback:\n{tb}")
                
                # 每次執行失敗，皆應向斷路器回報
                self.breaker.observe_failure()
                
                retry_count += 1
                if retry_count > self.max_retries or self.breaker.state == "OPEN":
                    logger.critical(f"任務 {task_func.__name__} 已達最大自癒重試次數 ({self.max_retries}) 或斷路器已開啟 (OPEN)，自癒宣告失敗")
                    raise e


                # --- Act: 實作自適應自癒決策 ---
                error_msg = str(e).lower()
                
                if "rate limit" in error_msg or "429" in error_msg or "too many requests" in error_msg:
                    # 自適應控流：回報 429，拉大控制器延遲
                    logger.warning("[Act] 判定為流量限制，觸發自適應 PID 控流 Hard Backoff")
                    report_429()
                    backoff_sec = delay_controller.current_delay
                elif "connection" in error_msg or "timeout" in error_msg or "lost connection" in error_msg:
                    # 資料庫或網路中斷：重置資料庫連接池，並計算指數退避
                    logger.warning("[Act] 判定為連線或逾時異常，執行資料庫與資源重置")
                    try:
                        # 強制重建 OP_Fun engine 單例
                        OP_Fun._engine = None
                        logger.info("[Act] 資料庫連線池已清除重設")
                    except Exception as ex:
                        logger.error(f"[Act] 重置資料庫連線池失敗: {ex}")
                    
                    backoff_sec = 2.0 ** retry_count
                else:
                    # 其他未知錯誤，使用基本退避
                    logger.warning("[Act] 判定為未知業務邏輯錯誤，執行標準指數退避")
                    backoff_sec = 2.0 ** retry_count

                # --- Verify: 驗證自癒條件是否具備 ---
                logger.info(f"[Act] 執行自癒冷卻中，休眠 {backoff_sec:.1f} 秒...")
                time.sleep(backoff_sec)

                verified = self.verify_system_state()
                if not verified:
                    logger.warning("[Verify] 自檢驗證失敗，當前環境暫不具備重試條件，延長冷卻時間...")
                    time.sleep(5.0)  # 額外等待 5 秒
                else:
                    logger.info("[Verify] 自檢驗證成功，具備重試條件，準備進行下一輪重試")
