"""
互動式認證設定工具
協助從 Chrome Profile 提取認證資訊並加密儲存。
"""

import asyncio
import sys
from pathlib import Path

CHROME_PROFILES = {
    "Windows": {
        "Chrome": Path.home() / "AppData" / "Local" / "Google" / "Chrome" / "User Data",
        "Edge": Path.home() / "AppData" / "Local" / "Microsoft" / "Edge" / "User Data",
        "Brave": Path.home() / "AppData" / "Local" / "BraveSoftware" / "Brave-Browser" / "User Data",
    }
}


async def run_setup():
    """互動式認證設定"""
    print("=" * 60)
    print("NotebookLM MCP Server v2.0 - 認證設定工具")
    print("=" * 60)
    print()

    # 直接引入 core 模組，不透過 package __init__
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from notebooklm_mcp_v2.core.auth_manager import AuthManager
    
    auth = AuthManager()
    
    profile_dir = ""
    if len(sys.argv) > 1:
        profile_dir = sys.argv[1]
    else:
        # 顯示常見 Chrome Profile 路徑
        import os
        user = os.environ.get("USERNAME", "User")
        print("常見 Chrome Profile 路徑（請選擇或自行輸入）：")
        print()
        
        options = {}
        idx = 1
        for browser, path_obj in CHROME_PROFILES["Windows"].items():
            if path_obj.exists():
                print(f"  {idx}. [{browser}] {path_obj}")
                options[str(idx)] = str(path_obj)
                idx += 1
        
        print()
        print("請輸入選項編號或直接貼上 Chrome Profile 路徑（先關閉 Chrome）：")
        user_input = input().strip()
        
        if user_input in options:
            profile_dir = options[user_input]
        else:
            profile_dir = user_input
    
    if not Path(profile_dir).exists():
        print(f"錯誤：路徑不存在 - {profile_dir}")
        return
    
    print(f"\n正在從 Chrome Profile 提取 cookies：{profile_dir}")
    cookies = {}
    try:
        cookies = auth.extract_cookies_from_chrome(profile_dir)
    except Exception as e:
        print(f"提取 cookies 失敗：{e}")
    
    if not cookies or "__Secure-1PSID" not in cookies:
        print("\n[警告] 無法從瀏覽器提取有效的 cookies 或是缺少核心認證 Cookie。")
        print("（可能原因：Chrome 版本過新使用了 v20 安全機制，或瀏覽器未關閉）")
        print("請開啟瀏覽器進入 NotebookLM (https://notebooklm.google.com/)")
        print("按 F12 進入「應用程式(Application) -> Cookies」")
        print("搜尋並手動複製以下兩個 Cookie 的值：")
        
        psid = input("\n請貼上 __Secure-1PSID 的值: ").strip()
        psidts = input("請貼上 __Secure-1PSIDTS 的值: ").strip()
        
        if psid and psidts:
            cookies = {
                "__Secure-1PSID": psid,
                "__Secure-1PSIDTS": psidts,
                # 一些額外的通用 fallback
                "__Secure-1PSIDCC": "",
                "__Secure-3PSID": psid,
                "__Secure-3PSIDTS": psidts,
            }
        else:
            print("必須輸入 Cookie 才能完成認證。")
            return
    else:
        print(f"成功提取 {len(cookies)} 個 cookies")
    
    print("正在提取 CSRF Token...")
    try:
        csrf_token = await auth.fetch_csrf_token(cookies)
        print(f"成功取得 CSRF Token（長度 {len(csrf_token)}）")
    except Exception as e:
        print(f"提取 CSRF Token 失敗：{e}")
        return
    
    auth.save_credentials(cookies, csrf_token)
    vault_path = auth.vault_dir / "credentials.vault"
    print(f"\n認證資訊已加密儲存至：{vault_path}")
    print("\n設定完成！可以啟動 MCP Server：")
    print("  python -m notebooklm_mcp_v2.server_v2")
    print()


if __name__ == "__main__":
    asyncio.run(run_setup())
