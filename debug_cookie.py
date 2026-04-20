import sys
sys.path.insert(0, 'E:\\Infinity\\mydjango')
from notebooklm_mcp_v2.core.auth_manager import AuthManager
import os
import sqlite3
import shutil
import tempfile
from pathlib import Path
import traceback

profile = Path(os.environ['LOCALAPPDATA']) / 'Google/Chrome/User Data'
master_key = AuthManager._get_chrome_master_key(str(profile))
db_path = profile / 'Default/Network/Cookies'

tmp_db = Path(tempfile.gettempdir()) / f'tmp_cookies_{os.urandom(4).hex()}.db'
shutil.copy2(str(db_path), str(tmp_db))

conn = sqlite3.connect(str(tmp_db))
cursor = conn.cursor()
cursor.execute('SELECT host_key, name, value, encrypted_value FROM cookies WHERE host_key LIKE "%google.com%" AND name="SID"')
rows = cursor.fetchall()
if rows:
    host, name, value, encrypted_value = rows[0]
    print(f'Attempting to decrypt SID. encrypted_value starts with: {encrypted_value[:3]}')
    try:
        from Crypto.Cipher import AES
        nonce = encrypted_value[3:15]
        payload = encrypted_value[15:]
        cipher = AES.new(master_key, AES.MODE_GCM, nonce)
        decrypted = cipher.decrypt(payload)
        print(f'Decrypted payload length: {len(decrypted)}')
        print(f'Decrypted payload: {decrypted[:-16].decode("utf-8", errors="replace")}')
    except Exception as e:
        traceback.print_exc()
else:
    print('No SID found')
