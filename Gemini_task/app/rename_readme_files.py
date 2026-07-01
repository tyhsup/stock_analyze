import os
import glob
import re

VAULT_DIR = r"C:\Users\許廷宇\.gemini\config\knowledge"

def rename_readmes(changed_files=None):
    """
    重新命名知識庫中的 README.md 檔案以防止同名衝突。
    若指定了 changed_files，則僅處理這些檔案中的 README.md。
    """
    print("開始執行 README 重命名防同名衝突...")
    
    if changed_files is not None:
        target_files = [f for f in changed_files if os.path.exists(f) and os.path.basename(f).lower() == "readme.md"]
    else:
        target_files = []
        md_files = glob.glob(os.path.join(VAULT_DIR, "**", "*.md"), recursive=True)
        for filepath in md_files:
            if os.path.basename(filepath).lower() == "readme.md":
                target_files.append(filepath)
                
    if not target_files:
        print("未偵測到任何需要處理的 README.md 檔案。")
        return []

    renamed_files = []
    for filepath in target_files:
        filename = os.path.basename(filepath)
        old_dir = os.path.dirname(filepath)
        parent_dir_name = os.path.basename(old_dir)
        
        # 決定新名稱
        if parent_dir_name.lower() in ["scaffold", "docs", "src", "tests", "lib"]:
            grandparent_dir = os.path.dirname(old_dir)
            grandparent_name = os.path.basename(grandparent_dir)
            new_filename = f"{grandparent_name}_{parent_dir_name}_readme.md".lower()
        else:
            new_filename = f"{parent_dir_name}_readme.md".lower()
            
        new_filepath = os.path.join(old_dir, new_filename)
        
        print(f"發現同名檔案: {filepath} -> 預計重新命名為: {new_filename}")
        
        # 1. 讀取並更新 Frontmatter 中的 title
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            new_title = f"{parent_dir_name.replace('_', ' ').title()} 說明文件"
            if content.strip().startswith("---"):
                title_pattern = r'(title:\s*)([^\n]+)'
                if re.search(title_pattern, content):
                    content = re.sub(title_pattern, r'\1' + f'"{new_title}"', content)
                else:
                    content = content.replace("---\n", f"---\ntitle: \"{new_title}\"\n", 1)
            else:
                from datetime import datetime
                frontmatter = f"---\ntitle: \"{new_title}\"\ntype: references\ndate: {datetime.now().strftime('%Y-%m-%d')}\n---\n\n"
                content = frontmatter + content
                
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            print(f"更新檔案內容標題失敗: {e}")
            continue
            
        # 2. 實體重新命名
        try:
            # 如果新檔名跟舊檔名一樣 (例如已經被重命名過了)，就跳過
            if filepath.lower() == new_filepath.lower():
                continue
            if os.path.exists(new_filepath):
                os.remove(new_filepath)
            os.rename(filepath, new_filepath)
            print(f" [成功] 已重新命名為: {new_filepath}")
            renamed_files.append(new_filepath)
        except Exception as e:
            print(f" [失敗] 無法重新命名: {e}")
            
    return renamed_files
