import subprocess
import os
import json
from typing import Optional, Dict, Any

class EdgarCLIBridge:
    """
    封裝 pp-edgar (Go CLI) 的呼叫。
    """
    def __init__(self):
        self._available: Optional[bool] = None
        # Windows 的預設安裝路徑
        self.default_paths = [
            "edgar-pp-cli",  # 系統 PATH 優先
            r"C:\Users\許廷宇\AppData\Local\Programs\PrintingPress\bin\edgar-pp-cli.exe",
        ]
        self.resolved_path: Optional[str] = None

    def get_executable_path(self) -> str:
        """
        尋找可執行的 pp-edgar 檔案路徑。
        """
        if self.resolved_path:
            return self.resolved_path

        for path in self.default_paths:
            try:
                # 測試是否可用
                result = subprocess.run(
                    [path, "doctor"],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='ignore',
                    timeout=5,
                    env=self._get_env()
                )
                if result.returncode in [0, 1]:  # doctor 沒認證會回傳 1，但代表可執行
                    self.resolved_path = path
                    return path
            except (FileNotFoundError, subprocess.SubprocessError):
                continue
        
        # 若都找不到，預設使用第一個，讓後續呼叫報錯
        return "edgar-pp-cli"

    def _get_env(self) -> Dict[str, str]:
        """
        獲取合併了環境變數與 .env 中設定的 env dict。
        """
        env = os.environ.copy()
        email = os.getenv("COMPANY_PP_CONTACT_EMAIL", "sum998888@gmail.com")
        env["COMPANY_PP_CONTACT_EMAIL"] = email
        return env

    def is_available(self) -> bool:
        """
        檢查 pp-edgar CLI 是否可用。
        """
        if self._available is not None:
            return self._available

        exe = self.get_executable_path()
        try:
            result = subprocess.run(
                [exe, "doctor"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
                timeout=5,
                env=self._get_env()
            )
            # 只要不是找不到檔案的錯誤，且能基本通訊，就視為可用
            self._available = True
        except (FileNotFoundError, subprocess.SubprocessError):
            self._available = False

        return self._available

    def run_command(self, args: list, timeout: int = 60) -> Dict[str, Any]:
        """
        執行 pp-edgar 指令並解析 JSON 輸出。
        """
        exe = self.get_executable_path()
        cmd = [exe] + args
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
                timeout=timeout,
                env=self._get_env()
            )
            if result.returncode != 0:
                raise RuntimeError(f"pp-edgar error: {result.stderr or result.stdout}")
            
            if not result.stdout.strip():
                return {}
                
            return json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse JSON from pp-edgar output: {result.stdout}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to execute pp-edgar command: {str(e)}") from e

    def insider_summary(self, ticker: str, since: str = "12mo", senior_only: bool = False) -> Dict[str, Any]:
        """
        呼叫 insider-summary 並回傳解析後的 dict。
        """
        args = ["insider-summary", ticker, "--since", since, "--json"]
        if senior_only:
            args.append("--senior-only")
        return self.run_command(args)
