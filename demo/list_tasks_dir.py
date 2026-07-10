import os
import sys

def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
        
    tasks_dir = r"C:\Users\許廷宇\.gemini\antigravity-ide\brain\9627c572-9cec-4de3-ad71-eebff3f7fd04\.system_generated\tasks"
    if not os.path.exists(tasks_dir):
        print(f"Tasks directory not found: {tasks_dir}")
        return
        
    try:
        files = os.listdir(tasks_dir)
        print(f"Files in tasks dir: {files}")
    except Exception as e:
        print(f"List error: {e}")

if __name__ == '__main__':
    main()
