#!/bin/sh
set -e

echo "[Entrypoint] 啟動前置檢測程序..."

# 防禦競態條件 (Race Condition)：若指定 DB_HOST 且非空，持續輪詢直至資料庫埠號開啟就緒
if [ -n "$DB_HOST" ]; then
    echo "[Entrypoint] 正在檢測 MySQL 連線狀態 ($DB_HOST:${DB_PORT:-3306})..."
    python - << 'EOF'
import os
import sys
import time
import socket

host = os.getenv('DB_HOST', 'localhost')
port = int(os.getenv('DB_PORT', '3306'))
timeout = 60
start_time = time.time()

while time.time() - start_time < timeout:
    try:
        with socket.create_connection((host, port), timeout=2):
            print(f"[Entrypoint] MySQL 資料庫服務就緒 ({host}:{port})！")
            sys.exit(0)
    except (socket.error, ConnectionRefusedError):
        print(f"[Entrypoint] 等待 MySQL 資料庫初始化中 ({host}:{port})...")
        time.sleep(2)

print(f"[Entrypoint] 錯誤：等待 MySQL 資料庫連線超時 ({host}:{port})！", file=sys.stderr)
sys.exit(1)
EOF
fi

# 自動執行資料庫遷移
echo "[Entrypoint] 執行 Django 資料庫遷移 (migrate)..."
python manage.py migrate --noinput

# 生產環境自動收集靜態檔案至 staticfiles 目錄
if [ "$COLLECT_STATIC" = "1" ] || [ "$DEBUG" = "False" ] || [ "$DEBUG" = "false" ]; then
    echo "[Entrypoint] 收集靜態檔案 (collectstatic)..."
    python manage.py collectstatic --noinput --clear || echo "[Entrypoint] 靜態檔案收集跳過或已完成。"
fi

echo "[Entrypoint] 啟動應用服務進程: $@"
exec "$@"
