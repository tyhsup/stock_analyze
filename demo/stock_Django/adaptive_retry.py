import time
import random
import logging

logger = logging.getLogger("stock_Django.adaptive_retry")

class PIDAdaptiveDelay:
    """
    使用離散 PID 控制算法的自適應爬蟲延遲控制器。
    結合 2-5 秒隨機延遲規範，動態調整請求間的間隔時間。
    """
    def __init__(self, target_latency: float = 1.5, Kp: float = 0.5, Ki: float = 0.1, Kd: float = 0.2):
        self.target_latency = target_latency
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        
        self.integral = 0.0
        self.last_error = 0.0
        
        # 預設基礎延遲區間 2 到 5 秒
        self.min_delay = 2.0
        self.max_delay = 30.0
        self.current_delay = random.uniform(2.0, 5.0)

    def calculate_delay(self, current_latency: float, is_success: bool) -> float:
        """
        根據最近一次請求的延遲與是否成功，動態計算下一次請求的延遲時間。
        """
        if not is_success:
            # 遇到失敗或 429 限制，大幅度增加誤差以拉長延遲 (Hard Backoff)
            error = 10.0
        else:
            # 計算與目標延遲的偏離度
            error = current_latency - self.target_latency
            
        # 限制積分項防溢出 (Anti-windup)
        self.integral = max(-10.0, min(10.0, self.integral + error))
        
        derivative = error - self.last_error
        self.last_error = error
        
        # PID 輸出計算
        pid_output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        
        # 結合隨機抖動 (Jitter) 的基礎延遲
        base_jitter = random.uniform(2.0, 5.0)
        
        # 新延遲 = 基礎抖動延遲 + PID 修正值
        self.current_delay = max(self.min_delay, min(self.max_delay, base_jitter + pid_output))
        
        logger.debug(f"PID 調整：當前延遲 {self.current_delay:.2f} 秒 (Error: {error:.2f}, Integral: {self.integral:.2f})")
        return self.current_delay

    def handle_429(self):
        """
        強制拉大延遲至最大限制，實作退避。
        """
        self.current_delay = self.max_delay
        self.integral = 10.0  # 設為積分上限
        logger.warning(f"偵測到 429 限制，已將爬蟲延遲重置為最大值：{self.max_delay} 秒")


class CircuitBreaker:
    """
    斷路器，用於自癒排程器中的服務故障隔離。
    支援 CLOSED, OPEN, HALF-OPEN 三種狀態，重試上限為 3 次。
    """
    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = "CLOSED"
        self.failure_count = 0
        self.last_failure_time = 0.0

    def observe_success(self):
        """
        觀測到執行成功，重置狀態。
        """
        self.failure_count = 0
        self.state = "CLOSED"
        logger.info("服務執行成功，斷路器恢復為 CLOSED 狀態")

    def observe_failure(self):
        """
        觀測到執行失敗，累加計數並評估是否開啟斷路器。
        """
        self.failure_count += 1
        logger.warning(f"服務執行失敗，當前失敗次數：{self.failure_count}/{self.failure_threshold}")
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.last_failure_time = time.time()
            logger.critical(f"失敗次數達到閾值，斷路器開啟 (OPEN)，進入冷卻期 ({self.recovery_timeout} 秒)")

    def can_execute(self) -> bool:
        """
        評估目前是否允許執行。
        """
        if self.state == "CLOSED":
            return True
            
        if self.state == "OPEN":
            # 檢查冷卻時間是否已過
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF-OPEN"
                logger.info("冷卻時間已過，防禦斷路器切換至 HALF-OPEN 狀態，允許測試性請求")
                return True
            return False
            
        if self.state == "HALF-OPEN":
            return True
            
        return False
