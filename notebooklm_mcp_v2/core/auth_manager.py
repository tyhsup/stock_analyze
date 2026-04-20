"""
認證管理器 v2（Windows AES-GCM + DPAPI 完整版）
支援 Chrome v80+ 的 AES-256-GCM Cookie 解密與 Fernet 加密儲存。
"""

import base64
import json
import logging
import os
import re
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

DEFAULT_VAULT_DIR = Path.home() / ".notebooklm_mcp_v2"
VAULT_FILE = "credentials.vault"
KEY_FILE = "master.key"


class AuthManager:
    """
    認證管理器：從 Chrome Profile 提取 cookies 並以 Fernet 加密儲存。
    支援 Chrome v80+ 的 AES-256-GCM Cookie 加密格式（DPAPI 解密主金鑰）。
    """

    REQUIRED_COOKIES = [
        "__Secure-1PSID",
        "__Secure-1PSIDTS",
        "__Secure-1PSIDCC",
        "__Secure-3PSID",
        "__Secure-3PSIDTS",
        "__Secure-3PSIDCC",
        "SID",
        "HSID",
        "SSID",
        "APISID",
        "SAPISID",
        "NID",
    ]

    def __init__(self, vault_dir: Optional[str] = None):
        self.vault_dir = Path(vault_dir) if vault_dir else DEFAULT_VAULT_DIR
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        self._fernet: Optional[Fernet] = None
        self._cookies: Optional[Dict[str, str]] = None
        self._csrf_token: Optional[str] = None
        self._init_encryption()

    # ── Fernet 加密初始化 ──────────────────────────────────────

    def _init_encryption(self) -> None:
        """初始化 Fernet 加密金鑰，若不存在則自動生成"""
        key_path = self.vault_dir / KEY_FILE
        if key_path.exists():
            with open(key_path, "rb") as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(key_path, "wb") as f:
                f.write(key)
            logger.info(f"已生成新的加密金鑰：{key_path}")
        self._fernet = Fernet(key)

    def _encrypt(self, data: str) -> bytes:
        return self._fernet.encrypt(data.encode("utf-8"))

    def _decrypt(self, token: bytes) -> str:
        return self._fernet.decrypt(token).decode("utf-8")

    # ── 憑證持久化 ─────────────────────────────────────────────

    def save_credentials(
        self, cookies: Dict[str, str], csrf_token: str
    ) -> None:
        """將認證資訊加密後儲存至本地 vault"""
        payload = json.dumps({"cookies": cookies, "csrf_token": csrf_token})
        encrypted = self._encrypt(payload)
        vault_path = self.vault_dir / VAULT_FILE
        with open(vault_path, "wb") as f:
            f.write(encrypted)
        logger.info(f"認證資訊已加密儲存：{vault_path}")

    def load_credentials(
        self,
    ) -> Optional[Tuple[Dict[str, str], str]]:
        """從加密 vault 載入認證資訊"""
        vault_path = self.vault_dir / VAULT_FILE
        if not vault_path.exists():
            logger.warning("未找到已儲存的認證資訊")
            return None
        try:
            with open(vault_path, "rb") as f:
                encrypted = f.read()
            payload = json.loads(self._decrypt(encrypted))
            self._cookies = payload["cookies"]
            self._csrf_token = payload["csrf_token"]
            logger.info("已成功載入加密認證資訊")
            return (self._cookies, self._csrf_token)
        except Exception as e:
            logger.error(f"載入認證資訊失敗：{e}")
            return None

    # ── Chrome Cookie 解密（Windows DPAPI + AES-GCM）─────────

    @staticmethod
    def _get_chrome_master_key(profile_dir: str) -> Optional[bytes]:
        """
        從 Chrome Local State 提取並用 DPAPI 解密 AES 主金鑰。
        適用於 Chrome v80+（Windows 環境）。
        """
        profile_path = Path(profile_dir)
        # Local State 位於 User Data 根目錄（profile_dir 的上一層）
        local_state_candidates = [
            profile_path / "Local State",
            profile_path.parent / "Local State",
        ]

        local_state_path = None
        for p in local_state_candidates:
            if p.exists():
                local_state_path = p
                break

        if not local_state_path:
            logger.warning("找不到 Chrome Local State 檔案，無法解密 v10+ Cookies")
            return None

        try:
            with open(local_state_path, "r", encoding="utf-8") as f:
                local_state = json.load(f)

            encrypted_key_b64 = local_state.get("os_crypt", {}).get(
                "encrypted_key", ""
            )
            if not encrypted_key_b64:
                return None

            encrypted_key = base64.b64decode(encrypted_key_b64)
            # 移除前 5 個位元組的 "DPAPI" 前綴
            encrypted_key = encrypted_key[5:]

            # 使用 Windows DPAPI 解密
            import win32crypt  # type: ignore
            master_key = win32crypt.CryptUnprotectData(
                encrypted_key, None, None, None, 0
            )[1]
            return master_key

        except ImportError:
            logger.error("缺少 pypiwin32，無法使用 DPAPI 解密")
            return None
        except Exception as e:
            logger.error(f"主金鑰提取失敗：{e}")
            return None

    @staticmethod
    def _decrypt_cookie_value(
        encrypted_value: bytes,
        master_key: Optional[bytes],
    ) -> str:
        """
        解密單一 Cookie 值。
        - v10/v11 前綴：AES-256-GCM（Chrome v80+）
        - 無前綴：舊版 DPAPI 直接加密
        """
        if not encrypted_value:
            return ""

        try:
            if encrypted_value[:3] in (b"v10", b"v11"):
                if not master_key:
                    return ""
                from Crypto.Cipher import AES  # type: ignore

                # 結構：[version 3b][nonce 12b][ciphertext][auth_tag 16b]
                nonce = encrypted_value[3:15]
                payload = encrypted_value[15:]
                cipher = AES.new(master_key, AES.MODE_GCM, nonce)
                decrypted = cipher.decrypt(payload)
                return decrypted[:-16].decode("utf-8", errors="replace")
            else:
                # 舊版 DPAPI
                import win32crypt  # type: ignore

                result = win32crypt.CryptUnprotectData(
                    encrypted_value, None, None, None, 0
                )
                return result[1].decode("utf-8", errors="replace")
        except Exception as e:
            logger.debug(f"Cookie 解密失敗（略過）：{e}")
            return ""

    def extract_cookies_from_chrome(
        self, profile_dir: str
    ) -> Dict[str, str]:
        """
        從 Chrome Profile 完整提取並解密 NotebookLM 所需 cookies。

        Args:
            profile_dir: Chrome User Data 目錄（如 C:\\...\\Chrome\\User Data）
                         或直接指向 Default Profile 目錄

        Note:
            Chrome 必須完全關閉，否則 SQLite 資料庫處於鎖定狀態。
        """
        # 搜尋 Cookies 資料庫位置（新版在 Network 子目錄）
        profile_path = Path(profile_dir)
        cookie_db_candidates = [
            profile_path / "Default" / "Network" / "Cookies",
            profile_path / "Default" / "Cookies",
            profile_path / "Network" / "Cookies",
            profile_path / "Cookies",
        ]

        cookies_db = None
        for candidate in cookie_db_candidates:
            if candidate.exists():
                cookies_db = candidate
                break

        if not cookies_db:
            raise FileNotFoundError(
                f"找不到 Chrome Cookies 資料庫。請確認路徑：{profile_dir}"
            )

        logger.info(f"找到 Cookies 資料庫：{cookies_db}")

        # 提取主金鑰（用於 AES-GCM 解密）
        master_key = self._get_chrome_master_key(str(profile_dir))

        # 複製到臨時目錄以解除 SQLite 鎖定
        tmp_dir = tempfile.mkdtemp()
        tmp_db = Path(tmp_dir) / "Cookies"
        shutil.copy2(str(cookies_db), str(tmp_db))

        cookies: Dict[str, str] = {}
        try:
            conn = sqlite3.connect(str(tmp_db))
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name, value, encrypted_value FROM cookies "
                "WHERE host_key LIKE '%google.com%'"
            )
            for name, value, encrypted_value in cursor.fetchall():
                if name not in self.REQUIRED_COOKIES:
                    continue
                # 優先使用 encrypted_value 解密
                if encrypted_value:
                    decrypted = self._decrypt_cookie_value(
                        encrypted_value, master_key
                    )
                    if decrypted and decrypted.isascii():
                        cookies[name] = decrypted
                        continue
                # fallback：直接使用 value（舊格式未加密）
                if value and isinstance(value, str) and value.isascii():
                    cookies[name] = value

            conn.close()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        logger.info(f"成功提取 {len(cookies)} 個 cookies（共需 {len(self.REQUIRED_COOKIES)} 個）")
        return cookies

    # ── CSRF Token 提取 ────────────────────────────────────────

    def extract_csrf_from_page(self, page_html: str) -> str:
        """從 NotebookLM 頁面 HTML 提取 CSRF Token（AT key）"""
        patterns = [
            r'"AT","([^"]+)"',
            r"AT['\"]:\s*['\"]([^'\"]{20,})['\"]",
            r"SNlM0e[\"']:\s*[\"']([^\"']+)",
            r"initData.*?\"([A-Za-z0-9_\-]{30,}==)\"",
        ]
        for pattern in patterns:
            match = re.search(pattern, page_html)
            if match:
                token = match.group(1)
                logger.info(f"已提取 CSRF Token（長度 {len(token)}）")
                return token
        raise ValueError("無法從頁面 HTML 中提取 CSRF Token，請確認 cookies 有效")

    async def fetch_csrf_token(self, cookies: Dict[str, str]) -> str:
        """GET 請求 NotebookLM 首頁以提取 CSRF Token"""
        import httpx

        async with httpx.AsyncClient(
            cookies=cookies,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            },
            follow_redirects=True,
            timeout=30.0,
        ) as client:
            resp = await client.get("https://notebooklm.google.com/")
            if resp.status_code != 200:
                raise ValueError(
                    f"NotebookLM 首頁請求失敗（{resp.status_code}），"
                    "請確認 cookies 有效且已登入 Google 帳號"
                )
            return self.extract_csrf_from_page(resp.text)

    # ── 屬性 ───────────────────────────────────────────────────

    @property
    def cookies(self) -> Optional[Dict[str, str]]:
        return self._cookies

    @property
    def csrf_token(self) -> Optional[str]:
        return self._csrf_token

    def is_authenticated(self) -> bool:
        return bool(self._cookies and self._csrf_token)
