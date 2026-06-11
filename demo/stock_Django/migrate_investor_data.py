import sys
import os
import pandas as pd
import numpy as np
import logging
from sqlalchemy import text

# 確保能 import 到 django app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from stock_Django import mySQL_OP
except ImportError:
    import mySQL_OP

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def migrate_data():
    sql_op = mySQL_OP.OP_Fun()
    
    # 1. 檢查舊表是否存在並動態取得欄位名稱
    with sql_op.engine.connect() as conn:
        table_exists = conn.execute(text("SHOW TABLES LIKE 'stock_investor'")).fetchone()
        if not table_exists:
            logger.error("舊資料表 'stock_investor' 不存在，無法進行遷移。")
            return
            
        # 獲取舊表的真實欄位名稱
        columns_info = conn.execute(text("SHOW COLUMNS FROM stock_investor")).fetchall()
        column_names = [col[0] for col in columns_info]
        logger.info(f"舊表真實欄位名稱: {column_names}")
        
        if not column_names:
            logger.error("無法取得舊表欄位。")
            return
            
        date_col = column_names[0]  # 第一個欄位代表日期
        num_col = column_names[1]   # 第二個欄位代表股票代號
        
        # 獲取所有不重複的交易日期列表
        logger.info(f"正在讀取所有交易日期列表 (使用欄位: {date_col})...")
        dates_result = conn.execute(text(f"SELECT DISTINCT `{date_col}` FROM stock_investor ORDER BY `{date_col}` ASC")).fetchall()
        dates_list = [r[0] for r in dates_result if r[0] is not None]
        
    total_dates = len(dates_list)
    logger.info(f"總共找到 {total_dates} 個交易日需要遷移。")
    
    # 2. 逐日遷移，徹底解決大表 OOM 與游標逾時問題
    processed_count = 0
    
    for idx, target_date in enumerate(dates_list):
        # 格式化日期以供查詢
        if hasattr(target_date, 'strftime'):
            date_query_val = target_date.strftime('%Y-%m-%d')
            date_str = target_date.strftime('%Y-%m-%d')
        else:
            date_query_val = str(target_date)
            date_str = str(target_date)
            
        try:
            # 讀取該日期的所有資料
            query = f"SELECT * FROM stock_investor WHERE `{date_col}` = :dt"
            df_day = pd.read_sql(text(query), sql_op.engine, params={"dt": date_query_val})
            
            if df_day.empty:
                continue
                
            migrated_df = pd.DataFrame()
            num_cols = len(df_day.columns)
            
            # 安全取得指定索引列資料的閉包
            def get_col_by_idx(col_idx):
                if col_idx < num_cols:
                    return pd.to_numeric(df_day.iloc[:, col_idx].astype(str).str.replace(',', ''), errors='coerce').fillna(0.0)
                return pd.Series(0.0, index=df_day.index)

            # 轉換至標準欄位
            migrated_df['date'] = df_day.iloc[:, 0]
            migrated_df['number'] = df_day.iloc[:, 1].astype(str).str.strip()
            migrated_df['name'] = df_day.iloc[:, 2].astype(str).str.strip() if num_cols > 2 else None
            
            # 法人買賣股數對齊
            migrated_df['foreign_buy'] = get_col_by_idx(3)
            migrated_df['foreign_sell'] = get_col_by_idx(4)
            migrated_df['foreign_net'] = get_col_by_idx(5)
            
            migrated_df['trust_buy'] = get_col_by_idx(9)
            migrated_df['trust_sell'] = get_col_by_idx(10)
            migrated_df['trust_net'] = get_col_by_idx(11)
            
            # 自營商 (自營 13 + 避險 16)
            migrated_df['dealer_buy'] = get_col_by_idx(13) + get_col_by_idx(16)
            migrated_df['dealer_sell'] = get_col_by_idx(14) + get_col_by_idx(17)
            migrated_df['dealer_net'] = get_col_by_idx(12)
            
            migrated_df['total_net'] = get_col_by_idx(19)
            
            # 特殊日期轉換容錯
            try:
                migrated_df['date'] = pd.to_datetime(migrated_df['date']).dt.date
            except:
                pass
                
            # 清除無效列
            migrated_df = migrated_df.dropna(subset=['date', 'number'])
            migrated_df = migrated_df[migrated_df['number'] != '']
            
            # 寫入新表
            if not migrated_df.empty:
                with sql_op.engine.begin() as conn:
                    cols = ", ".join([f"`{c}`" for c in migrated_df.columns])
                    placeholders = ", ".join([f":{c}" for c in migrated_df.columns])
                    update_cols = ", ".join([f"`{c}`=VALUES(`{c}`)" for c in migrated_df.columns if c not in ['date', 'number']])
                    
                    sql_query = f"""
                        INSERT INTO `stock_investor_tw` ({cols}) 
                        VALUES ({placeholders})
                        ON DUPLICATE KEY UPDATE {update_cols}
                    """
                    dict_list = migrated_df.to_dict(orient='records')
                    # 將 date 轉換為字串以確保 JSON/SQL 相容
                    for d in dict_list:
                        d['date'] = str(d['date'])
                    conn.execute(text(sql_query), dict_list)
                    
            processed_count += len(migrated_df)
            
            # 每處理 10% 或 20 個交易日印出一次進度
            if idx % 20 == 0 or idx == total_dates - 1:
                logger.info(f"進度: {idx + 1}/{total_dates} 天 ({date_str}) | 已成功遷移 {processed_count} 筆資料...")
                
        except Exception as e:
            logger.error(f"❌ 遷移交易日 {date_str} 時發生錯誤: {e}")
            
    logger.info("🎉 所有歷史資料遷移成功！")

if __name__ == "__main__":
    migrate_data()
