import os
import sys

def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
        
    log_path = r"C:\Users\許廷宇\.gemini\antigravity-ide\brain\9627c572-9cec-4de3-ad71-eebff3f7fd04\.system_generated\tasks\task-208.log"
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return
        
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            print(f.read())
    except Exception as e:
        print(f"Read error: {e}")

if __name__ == '__main__':
    main()
