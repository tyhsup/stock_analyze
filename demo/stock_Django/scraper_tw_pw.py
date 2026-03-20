import asyncio
import pandas as pd
from playwright.sync_api import sync_playwright
from .mySQL_OP import OP_Fun
from .scraper_utils import smart_delay
import logging
from io import StringIO

logger = logging.getLogger(__name__)

def scrape_tw_financials_playwright(symbol, year, quarter):
    """使用 Playwright 模擬瀏覽器抓取 MOPS 財報 (mopsov 網域)"""
    roc_year = year - 1911
    # 使用較穩定的 mopsov 網域
    urls = {
        'IS': "https://mopsov.twse.com.tw/mops/web/t164sb04", # 損益
        'BS': "https://mopsov.twse.com.tw/mops/web/t164sb03", # 資產
        'CF': "https://mopsov.twse.com.tw/mops/web/t164sb05"  # 現流
    }
    
    all_items = []
    
    with sync_playwright() as p:
        # 開啟瀏覽器，若要觀察可設 headless=False
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        
        for stmt_type, url in urls.items():
            print(f"Scraping {stmt_type} for {symbol} {year}Q{quarter}...")
            try:
                page.goto(url, wait_until="networkidle", timeout=30000)
                
                # 1. 處理「公司代號」
                page.fill('input#co_id', symbol)
                
                # 2. 處理「年度/季別」顯示問題：切換 'isnew' 下拉選單為「歷史資料」
                # 根據 Subagent回報，必須切換才能看到年度輸入框
                if page.locator('select#isnew').count() > 0:
                    page.select_option('select#isnew', label="歷史資料")
                    smart_delay(1, 2)
                
                # 3. 填寫年度與季別
                page.fill('input#year', str(roc_year))
                page.select_option('select#season', label=str(quarter))
                
                # 4. 點擊「查詢」
                # 使用 Subagent 觀察到的按鈕點擊方式或鍵盤 Enter
                page.click('input[type="button"][value=" 查詢 "]')
                
                # 5. 等待結果表格出現
                try:
                    # MOPS 結果通常在 table.hasBorder
                    page.wait_for_selector('table.hasBorder', timeout=20000)
                except:
                    print(f"DEBUG: No table found for {stmt_type}. Possible no data or block.")
                    continue
                
                # 6. 解析 HTML - 使用 evaluate 確保字元正確性 (JavaScript Unicode)
                data_list = page.evaluate("""
                    () => {
                        const rows = Array.from(document.querySelectorAll('table.hasBorder tr'));
                        return rows.map(row => {
                            const cells = Array.from(row.querySelectorAll('td'));
                            return cells.map(c => c.innerText.trim());
                        }).filter(r => r.length >= 2);
                    }
                """)
                
                if data_list:
                    item_count = 0
                    for cells in data_list:
                        item = cells[0].replace('\xa0', ' ').strip()
                        amount_val = cells[1].replace('\xa0', ' ').strip()
                        
                        if not item or item in ["項目", "nan"] or "資產" in item or "損益" in item:
                            continue
                            
                        try:
                            # 清洗數字
                            s = str(amount_val).replace(',', '').replace('(', '-').replace(')', '').strip()
                            if s == '-' or s == '': continue
                            amount = float(s)
                            all_items.append({
                                'symbol': symbol, 'year': year, 'quarter': quarter,
                                'statement_type': stmt_type, 'item_name': item, 'amount': amount
                            })
                            item_count += 1
                        except: continue
                    print(f"DEBUG: Successfully captured {item_count} items from {stmt_type}")
                
            except Exception as e:
                print(f"DEBUG: Error scraping {stmt_type}: {e}")
            
            # 避免觸發反爬蟲
            smart_delay(3, 5)
            
        browser.close()
        
    return pd.DataFrame(all_items)

def scrape_tw_financials_multi(symbol, periods: list):
    """
    Efficiently scrape multiple periods for TW stock in one browser session.
    periods: list of (year, quarter) tuples.
    """
    urls = {
        'IS': "https://mopsov.twse.com.tw/mops/web/t164sb04",
        'BS': "https://mopsov.twse.com.tw/mops/web/t164sb03",
        'CF': "https://mopsov.twse.com.tw/mops/web/t164sb05"
    }
    all_items = []
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        
        for year, quarter in periods:
            roc_year = year - 1911
            for stmt_type, url in urls.items():
                logger.info(f"Scraping {symbol} {year}Q{quarter} {stmt_type}...")
                try:
                    page.goto(url, wait_until="networkidle", timeout=30000)
                    page.fill('input#co_id', symbol)
                    if page.locator('select#isnew').count() > 0:
                        page.select_option('select#isnew', label="歷史資料")
                        smart_delay(0.5, 1)
                    
                    page.fill('input#year', str(roc_year))
                    page.select_option('select#season', label=str(quarter))
                    page.click('input[type="button"][value=" 查詢 "]')
                    
                    try:
                        page.wait_for_selector('table.hasBorder', timeout=10000)
                    except:
                        continue
                    
                    data_list = page.evaluate("""
                        () => {
                            const rows = Array.from(document.querySelectorAll('table.hasBorder tr'));
                            return rows.map(row => {
                                const cells = Array.from(row.querySelectorAll('td'));
                                return cells.map(c => c.innerText.trim());
                            }).filter(r => r.length >= 2);
                        }
                    """)
                    
                    if data_list:
                        for cells in data_list:
                            item = cells[0].replace('\xa0', ' ').strip()
                            amount_val = cells[1].replace('\xa0', ' ').strip()
                            if not item or item in ["項目", "nan"] or "資產" in item or "損益" in item:
                                continue
                            try:
                                s = str(amount_val).replace(',', '').replace('(', '-').replace(')', '').strip()
                                if s == '-' or s == '': continue
                                amount = float(s)
                                all_items.append({
                                    'symbol': symbol, 'year': year, 'quarter': quarter,
                                    'statement_type': stmt_type, 'item_name': item, 'amount': amount
                                })
                            except: continue
                except Exception as e:
                    logger.error(f"Error {symbol} {year}Q{quarter} {stmt_type}: {e}")
                smart_delay(2, 4) # Delay between types/periods
                
        browser.close()
    return pd.DataFrame(all_items)

if __name__ == "__main__":
    op = OP_Fun()
    # Manual test logic here if needed
    pass
