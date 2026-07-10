import sqlite3

db_path = "e:/Infinity/mydjango/Gemini_task/scheduler.db"
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# 將處於 running 狀態的任務重設為 failed
cur.execute(
    "UPDATE jobs SET status = 'failed', remarks = '系統異常中斷：排程行程卡死已手動終止' WHERE status = 'running'"
)
conn.commit()
print(f"成功更新 {cur.rowcount} 筆卡死的任務狀態。")
conn.close()
