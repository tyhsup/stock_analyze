import os
import glob
import pandas as pd
import sys
from agent_news_analyzer import AgentNewsAnalyzer

# 強制控制台輸出使用 UTF-8，避免 cp950 編碼錯誤
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

def clean_and_migrate_columns(df):
    """
    使用模糊匹配將中文欄位轉換為英文。
    """
    # 定義模糊匹配規則
    rules = {
        'title': ['標題', 'title', 'Title'],
        'date': ['日期', '時間', 'date', 'Date', '發布時間', 'Publish Date'],
        'content': ['內容', '內文', 'content', 'Content', '描述'],
        'link': ['連結', 'URL', 'link', 'Link', 'url'],
        'source': ['來源', 'source', 'Source', '媒體'],
        'positive_negative_analysis': ['正負分析', '情緒', 'sentiment', 'Sentiment'],
        'sentiment_score': ['情緒分析結果', '分數', 'score', 'Score', 'sentiment_score']
    }
    
    new_df = pd.DataFrame()
    original_cols = df.columns.tolist()
    used_cols = set()

    for eng_key, keywords in rules.items():
        found = False
        for col in original_cols:
            if any(k in str(col) for k in keywords):
                if eng_key not in new_df.columns:
                    new_df[eng_key] = df[col]
                else:
                    # 如果有多個匹配欄位，嘗試合併
                    new_df[eng_key] = new_df[eng_key].fillna(df[col])
                used_cols.add(col)
                found = True
        
        if not found:
            new_df[eng_key] = None

    # 保留原本沒被匹配到的其他欄位 (如果是英文的)
    for col in original_cols:
        if col not in used_cols and not any('\u4e00' <= char <= '\u9fff' for char in str(col)):
            new_df[col] = df[col]
            
    # 確保所有必要的 Agent 產出欄位都存在
    agent_fields = ['market', 'confidence', 'impact_scope', 'reasoning_summary']
    for field in agent_fields:
        if field not in new_df.columns:
            new_df[field] = None
            
    return new_df

def migrate_historical_news():
    data_dir = "E:/Infinity/mydjango/demo/newsapp/news_data"
    
    if not os.path.exists(data_dir):
        print(f"目錄 {data_dir} 不存在。")
        return
        
    excel_files = glob.glob(os.path.join(data_dir, "*.xlsx"))
    
    if not excel_files:
        print(f"在 {data_dir} 中找不到 Excel 檔案。")
        return
        
    analyzer = AgentNewsAnalyzer()
    
    for file_path in excel_files:
        print(f"正在處理: {os.path.basename(file_path)}")
        try:
            df = pd.read_excel(file_path)
            
            # 1. 欄位清洗
            df = clean_and_migrate_columns(df)
            
            # 2. 檢查是否需要 AI 分析 (僅在配額允許時執行)
            # 注意：這裡我們先執行清洗，即便 AI 失敗也會存檔
            needs_save = True
            
            # 嘗試處理前 3 筆缺失資料作為測試
            missing_indices = df[df['sentiment_score'].isna() | (df['sentiment_score'] == 0)].index.tolist()
            
            if missing_indices:
                print(f"  [提示] 發現 {len(missing_indices)} 條缺失分析的資料。")
                for index in missing_indices[:5]: # 每次嘗試處理少量資料，避免卡死
                    row = df.loc[index]
                    news_data = {
                        'title': row.get('title', ''),
                        'date': str(row.get('date', '')),
                        'content': row.get('content', ''),
                        'link': row.get('link', ''),
                        'source': row.get('source', '')
                    }
                    
                    if not news_data['title']: continue
                    
                    try:
                        print(f"  [AI 分析] {news_data['title'][:20]}...")
                        result = analyzer.analyze_news(news_data)
                        if result:
                            for key in ['market', 'positive_negative_analysis', 'sentiment_score', 
                                        'confidence', 'impact_scope', 'reasoning_summary']:
                                if key in result:
                                    df.at[index, key] = result[key]
                    except Exception as e:
                        if "429" in str(e):
                            print("  [停止] 已達 Groq 配額限制，停止本次 AI 分析，僅執行欄位清洗。")
                            break
                        print(f"  [跳過] 錯誤: {e}")
            
            # 3. 儲存更新後的檔案
            df.to_excel(file_path, index=False)
            print(f"  [完成] 檔案已標準化為英文欄位。")
            
        except Exception as e:
            print(f"處理檔案 {file_path} 時發生錯誤: {e}")

if __name__ == "__main__":
    migrate_historical_news()
