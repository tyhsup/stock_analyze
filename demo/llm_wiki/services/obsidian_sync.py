import os
import glob
import frontmatter
from datetime import datetime

VAULT_DIR = r"C:\Users\許廷宇\.gemini\config\knowledge"

def ensure_vault_exists():
    if not os.path.exists(VAULT_DIR):
        os.makedirs(VAULT_DIR, exist_ok=True)
    # Create standard folders
    for folder in ["specs", "plans", "walkthroughs", "errors", "references"]:
        os.makedirs(os.path.join(VAULT_DIR, folder), exist_ok=True)
    os.makedirs(os.path.join(VAULT_DIR, "references", "notebooklm"), exist_ok=True)

def list_markdown_files():
    ensure_vault_exists()
    return glob.glob(os.path.join(VAULT_DIR, "**", "*.md"), recursive=True)

def read_markdown_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        post = frontmatter.load(f)
    return post

def write_markdown_file(sub_path, content, metadata=None):
    ensure_vault_exists()
    filepath = os.path.join(VAULT_DIR, sub_path)
    
    # Ensure subdirectories exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if metadata is None:
        metadata = {}
    
    if 'date' not in metadata:
        metadata['date'] = datetime.now().strftime("%Y-%m-%d")
        
    post = frontmatter.Post(content, **metadata)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(frontmatter.dumps(post))
        
    return filepath
