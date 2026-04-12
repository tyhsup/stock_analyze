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
    # 新版 SPA 介面網址
    urls = {
        'IS': "https://mops.twse.com.tw/mops/#/web/t164sb04", # 損益
        'BS': "https://mops.twse.com.tw/mops/#/web/t164sb03", # 資產
        'CF': "https://mops.twse.com.tw/mops/#/web/t164sb05"  # 現流
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
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                
                # 1. 偵測介面版本並處理「公司代號」
                # 使用聯集選擇器等待新版 (#companyId) 或舊版 (#co_id) 出現
                try:
                    page.wait_for_selector('input#companyId, input#co_id', timeout=20000)
                except:
                    # 如果都等不到，可能是網頁載入極慢或路徑錯誤
                    print(f"DEBUG: Timeout waiting for company input on {stmt_type}")
                    continue

                # 判定實際出現的 ID
                is_new_mops = page.locator('input#companyId').count() > 0
                target_selector = 'input#companyId' if is_new_mops else 'input#co_id'
                page.fill(target_selector, symbol)
                
                # 2. 處理「年度/季別」：新版 SPA 需要點擊「自訂」標籤
                is_new_mops = 'companyId' in target_selector
                if is_new_mops:
                    # 點擊「自訂資料」標籤以顯示年份輸入框
                    # 使用更寬鬆的文本選擇器，因為可能是 button, label 或 div
                    custom_btn = page.locator(':text-is("自訂")')
                    if custom_btn.count() > 0:
                        custom_btn.first.click()
                        # 重要：新版點擊後會出現「載入中...」遮罩，必須等待它消失
                        try:
                            page.locator('div:has-text("載入中...")').wait_for(state="hidden", timeout=15000)
                        except:
                            # 如果沒出現或消失極快，就繼續
                            pass
                else:
                    # 舊版處理 'isnew' 下拉選單
                    if page.locator('select#isnew').count() > 0:
                        page.select_option('select#isnew', label="歷史資料")
                
                smart_delay(0.5, 1)
                
                # 3. 填寫年度與季別
                page.wait_for_selector('input#year', timeout=5000)
                page.fill('input#year', str(roc_year))
                page.select_option('select#season', label=str(quarter))
                
                # 4. 點擊「查詢」
                if is_new_mops:
                    page.click('button#searchBtn')
                else:
                    page.click('input[type="button"][value=" 查詢 "]')
                
                # 5. 等待結果表格出現 (新版可能需要多一點時間渲染)
                try:
                    # MOPS 結果通常在 table.hasBorder 或特定結果區塊
                    # 增加對「無資料」文字的偵測，避免超時等待
                    page.wait_for_selector('table.hasBorder, :text-is("查無資料")', timeout=15000)
                    if page.locator(':text-is("查無資料")').count() > 0:
                        print(f"DEBUG: No data found on MOPS for {symbol} {year}Q{quarter}")
                        continue
                except:
                    # 有時候沒資料會彈窗或顯示「無資料」
                    continue
                
                # 6. 解析 HTML - 優化 JS 解析以適應新版結構
                data_list = page.evaluate("""
                    () => {
                        const tables = Array.from(document.querySelectorAll('table.hasBorder'));
                        if (tables.length === 0) return [];
                        // 優先找尋包含「項目」或「會計項目」的表格
                        const mainTable = tables.find(t => t.innerText.includes('項目')) || 
                                         tables.sort((a, b) => b.rows.length - a.rows.length)[0];
                        if (!mainTable) return [];
                        
                        return Array.from(mainTable.rows).map(row => {
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
                    page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    
                    # 偵測版本並填寫公司代號
                    try:
                        page.wait_for_selector('input#companyId, input#co_id', timeout=20000)
                    except:
                        continue

                    is_new_mops = page.locator('input#companyId').count() > 0
                    target_selector = 'input#companyId' if is_new_mops else 'input#co_id'
                    page.fill(target_selector, symbol)
                    
                    is_new_mops = 'companyId' in target_selector
                    if is_new_mops:
                        custom_btn = page.locator(':text-is("自訂")')
                        if custom_btn.count() > 0:
                            custom_btn.first.click()
                            try:
                                page.locator('div:has-text("載入中...")').wait_for(state="hidden", timeout=15000)
                            except:
                                pass
                    else:
                        if page.locator('select#isnew').count() > 0:
                            page.select_option('select#isnew', label="歷史資料")
                    
                    smart_delay(0.5, 1)
                    page.wait_for_selector('input#year', timeout=10000)
                    page.fill('input#year', str(roc_year))
                    page.select_option('select#season', label=str(quarter))
                    
                    if is_new_mops:
                        page.click('button#searchBtn')
                    else:
                        page.click('input[type="button"][value=" 查詢 "]')
                    
                    try:
                        page.wait_for_selector('table.hasBorder', timeout=15000)
                    except:
                        continue
                    
                    data_list = page.evaluate("""
                        () => {
                            const tables = Array.from(document.querySelectorAll('table.hasBorder'));
                            const mainTable = tables.sort((a, b) => b.rows.length - a.rows.length)[0];
                            if (!mainTable) return [];
                            return Array.from(mainTable.rows).map(row => {
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
