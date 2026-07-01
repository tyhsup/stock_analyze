import os
import sys
import re
import argparse
import glob
from datetime import datetime

# 設定常數路徑
BRAIN_BASE_DIR = r"C:\Users\許廷宇\.gemini\antigravity-ide\brain"
VAULT_DIR = r"C:\Users\許廷宇\.gemini\config\knowledge"

def get_latest_brain_dir():
    """
    尋找最新被修改且含有實體計畫或工作紀錄的 brain 對話資料夾。
    """
    if not os.path.exists(BRAIN_BASE_DIR):
        print(f"錯誤：找不到 brain 根目錄 {BRAIN_BASE_DIR}")
        return None
        
    subdirs = glob.glob(os.path.join(BRAIN_BASE_DIR, "*"))
    subdirs = [d for d in subdirs if os.path.isdir(d)]
    
    # 依修改時間排序
    subdirs.sort(key=lambda d: os.path.getmtime(d), reverse=True)
    
    # 遍歷尋找最近含有 md 檔案的資料夾
    for d in subdirs:
        if os.path.exists(os.path.join(d, "implementation_plan.md")) or os.path.exists(os.path.join(d, "walkthrough.md")):
            return d
            
    # 退而求其次，回傳最新修改的資料夾
    if subdirs:
        return subdirs[0]
    return None

def sanitize_slug(text):
    """
    將中英文字串轉為適用於檔名的 slug。
    """
    # 移除特殊字元，空白轉底線
    clean = re.sub(r'[\\/*?:"<>|#\s]', "_", text)
    # 壓縮重複的底線
    clean = re.sub(r'_+', "_", clean)
    return clean.strip("_")

def strip_frontmatter(content):
    """
    移除 Markdown 內容中既有的 Frontmatter 與 H1 大標題，防止寫入新檔時重複。
    """
    text = content.strip()
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            text = parts[2].strip()
            
    # 移除開頭的第一行 H1 標題 (# Title)
    text = re.sub(r'^#\s+.*?\n', '', text).strip()
    return text

def sync_session(request_name, root_cause=None):
    """
    主同步流程：將本次會話的計畫、工作紀錄與錯誤修補報告同步至全域知識庫。
    """
    brain_dir = get_latest_brain_dir()
    if not brain_dir:
        print("錯誤：找不到有效的對話資料夾 (brain directory)")
        return
        
    conv_id = os.path.basename(brain_dir)
    print(f"偵測到最新對話 ID: {conv_id}")
    print(f"對話來源路徑: {brain_dir}")
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    slug = sanitize_slug(request_name)
    
    # 建立目標目錄
    os.makedirs(os.path.join(VAULT_DIR, "plans"), exist_ok=True)
    os.makedirs(os.path.join(VAULT_DIR, "walkthroughs"), exist_ok=True)
    os.makedirs(os.path.join(VAULT_DIR, "errors"), exist_ok=True)
    
    # 1. 同步計畫 (Plans)
    plan_src = os.path.join(brain_dir, "implementation_plan.md")
    if os.path.exists(plan_src):
        try:
            with open(plan_src, "r", encoding="utf-8") as f:
                content = f.read()
            clean_body = strip_frontmatter(content)
            
            title = f"開發計畫書：{request_name}"
            subtitle = f"需求名稱：{request_name}"
            
            fm_content = f"""---
title: "{title}"
subtitle: "{subtitle}"
type: plans
date: "{date_str}"
source: "Antigravity Session ({conv_id})"
---

# {title}

{clean_body}
"""
            dest_path = os.path.join(VAULT_DIR, "plans", f"{date_str}_{slug}_plan.md")
            with open(dest_path, "w", encoding="utf-8") as f:
                f.write(fm_content)
            print(f" [同步成功] 計畫書 -> plans/{os.path.basename(dest_path)}")
        except Exception as e:
            print(f" [同步失敗] 計畫書: {e}")
            
    # 2. 同步工作紀錄 (Walkthroughs)
    wt_src = os.path.join(brain_dir, "walkthrough.md")
    if os.path.exists(wt_src):
        try:
            with open(wt_src, "r", encoding="utf-8") as f:
                content = f.read()
            clean_body = strip_frontmatter(content)
            
            title = f"工作紀錄與成果：{request_name}"
            subtitle = f"需求名稱：{request_name}"
            
            fm_content = f"""---
title: "{title}"
subtitle: "{subtitle}"
type: walkthroughs
date: "{date_str}"
source: "Antigravity Session ({conv_id})"
---

# {title}

{clean_body}
"""
            dest_path = os.path.join(VAULT_DIR, "walkthroughs", f"{date_str}_{slug}_walkthrough.md")
            with open(dest_path, "w", encoding="utf-8") as f:
                f.write(fm_content)
            print(f" [同步成功] 工作紀錄 -> walkthroughs/{os.path.basename(dest_path)}")
        except Exception as e:
            print(f" [同步失敗] 工作紀錄: {e}")

    # 3. 處理錯誤日誌 (Errors)
    err_src = os.path.join(brain_dir, "error_report.md")
    # 如果本地有 error_report.md，或者使用者有傳入 root_cause，就進行錯誤同步
    if os.path.exists(err_src) or root_cause:
        try:
            clean_body = ""
            if os.path.exists(err_src):
                with open(err_src, "r", encoding="utf-8") as f:
                    content = f.read()
                clean_body = strip_frontmatter(content)
            else:
                # 自動產生一個標準格式的錯誤回報範本
                clean_body = f"""## 錯誤症狀與問題背景
在執行「{request_name}」需求時遇到了系統異常。

## 根本原因 (Root Cause)
{root_cause}

## 解決與修補方案 (Suggested Fix)
已針對程式碼進行防錯性處理，並完成測試驗證。
"""
            
            title = f"問題修復報告：{request_name}"
            subtitle = f"錯誤根因：{root_cause if root_cause else '系統檢測錯誤'}"
            
            fm_content = f"""---
title: "{title}"
subtitle: "{subtitle}"
type: errors
date: "{date_str}"
source: "Antigravity Session ({conv_id})"
---

# {title}

{clean_body}
"""
            dest_path = os.path.join(VAULT_DIR, "errors", f"{date_str}_{slug}_error.md")
            with open(dest_path, "w", encoding="utf-8") as f:
                f.write(fm_content)
            print(f" [同步成功] 錯誤日誌 -> errors/{os.path.basename(dest_path)}")
        except Exception as e:
            print(f" [同步失敗] 錯誤日誌: {e}")

    # 4. 自動觸發關聯性標註與首頁重建
    print("\n執行知識庫檔案關聯性標註...")
    try:
        from app.annotate_wiki_relations import annotate_files
        annotate_files()
    except ImportError:
        # 如果路徑問題，改用 os.system 執行
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "annotate_wiki_relations.py")
        os.system(f"python {script_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="同步本次對話歷程至全域知識庫")
    parser.add_argument("--request-name", required=True, help="本需求或工作的主題名稱")
    parser.add_argument("--root-cause", help="若為錯誤修復，請提供問題的根本原因（做為副標註）")
    
    args = parser.parse_args()
    sync_session(args.request_name, args.root_cause)
