import time
import random
import requests
from stock_Django.adaptive_retry import PIDAdaptiveDelay

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1"
]

# 全局自適應爬蟲延遲控制器單例
delay_controller = PIDAdaptiveDelay()

def get_random_ua():
    return random.choice(USER_AGENTS)

def smart_delay(latency: float = None, is_success: bool = True):
    """
    自適應延遲，模擬人類行為以避開反爬蟲。
    若傳入實際請求延遲（latency）與是否成功（is_success），將使用 PID 控制算法動態調整下一個延遲；
    否則，依據當前控制器算出的延遲時間進行休眠。
    """
    if latency is not None:
        delay_controller.calculate_delay(latency, is_success)
        
    delay_sec = delay_controller.current_delay
    time.sleep(delay_sec)

def report_429():
    """
    回報遭遇到 429 (Too Many Requests) 限制，強制拉大下一個延遲。
    """
    delay_controller.handle_429()

def get_session():
    session = requests.Session()
    session.headers.update({"User-Agent": get_random_ua()})
    return session

