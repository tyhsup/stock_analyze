import os
import glob
import pandas as pd
from agent_news_analyzer import AgentNewsAnalyzer

def migrate_historical_news():
    data_dir = "E:/Infinity/mydjango/demo/newsapp/news_data"
    
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist.")
        return
        
    excel_files = glob.glob(os.path.join(data_dir, "*.xlsx"))
    
    if not excel_files:
        print(f"No Excel files found in {data_dir}.")
        return
        
    analyzer = AgentNewsAnalyzer()
    
    for file_path in excel_files:
        print(f"Processing {file_path}...")
        try:
            df = pd.read_excel(file_path)
            
            # Check if sentiment_score exists
            if 'sentiment_score' not in df.columns:
                df['market'] = ''
                df['positive_negative_analysis'] = ''
                df['sentiment_score'] = 0.0
                df['confidence'] = 0.0
                df['impact_scope'] = ''
                df['reasoning_summary'] = ''
                df['source'] = df.get('來源', '')
                df['link'] = df.get('連結', '')
                df['title'] = df.get('標題', '')
                df['date'] = df.get('發布時間', '')
                df['content'] = df.get('內文', '')
                
            for index, row in df.iterrows():
                # Only analyze if sentiment_score is missing or 0.0 for newly added columns
                if pd.isna(row.get('sentiment_score')) or row.get('sentiment_score') == 0.0:
                    news_data = {
                        'title': row.get('標題', row.get('title', '')),
                        'date': row.get('發布時間', row.get('date', '')),
                        'content': row.get('內文', row.get('content', '')),
                        'link': row.get('連結', row.get('link', '')),
                        'source': row.get('來源', row.get('source', ''))
                    }
                    
                    print(f"Analyzing row {index}: {news_data['title']}")
                    result = analyzer.analyze_news(news_data)
                    
                    df.at[index, 'market'] = result.get('market', '')
                    df.at[index, 'positive_negative_analysis'] = result.get('positive_negative_analysis', '')
                    df.at[index, 'sentiment_score'] = result.get('sentiment_score', 0.0)
                    df.at[index, 'confidence'] = result.get('confidence', 0.0)
                    df.at[index, 'impact_scope'] = result.get('impact_scope', '')
                    df.at[index, 'reasoning_summary'] = result.get('reasoning_summary', '')
                    df.at[index, 'source'] = result.get('source', '')
                    df.at[index, 'link'] = result.get('link', '')
                    df.at[index, 'title'] = result.get('title', '')
                    df.at[index, 'date'] = result.get('date', '')
                    df.at[index, 'content'] = result.get('content', '')
            
            # Save the updated dataframe back to the excel file
            df.to_excel(file_path, index=False)
            print(f"Successfully updated {file_path}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    print("Starting historical news migration...")
    migrate_historical_news()
    print("Migration completed.")
