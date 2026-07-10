import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_path = os.path.join(BASE_DIR, 'Gemini_task', 'scheduler.db')

print(f"Connecting to database: {db_path}")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    cursor.execute("ALTER TABLE jobs ADD COLUMN last_heartbeat DATETIME")
    conn.commit()
    print("Column last_heartbeat successfully added to jobs table.")
except sqlite3.OperationalError as e:
    # If the column already exists, SQLite will throw an OperationalError
    if "duplicate column name" in str(e).lower() or "already exists" in str(e).lower():
         print("Column last_heartbeat already exists. Skipping migration.")
    else:
         print(f"Error during migration: {e}")
finally:
    conn.close()
