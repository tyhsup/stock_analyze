import requests

prompt = """
在 v3.0 結構化關聯建模 (GNN) 計畫中，我們遭遇了一個問題：
目前本地資料庫中的美股表格 (`stocks_us`) 僅包含 symbol, name, market 等基本資訊，缺少台股那樣清晰的「產業別 (Sector)」欄位，這導致我們無法直接建立美股的產業鄰接矩陣。

請針對此狀況提供解決方案：
1. **補全建議**：推薦 1-2 個免費或低成本的 Python API 或爬蟲來源（如 yfinance, SEC EDGAR 等），說明如何從這些來源自動化提取美股的 Sector/Industry 標籤，並存入本地資料庫。
2. **替代方案**：如果暫時不補全資料庫，是否有「非產業別」的純數據驅動方法（例如：收益率相關度、ETF 成分股重疊度等）來構建美股節點的關聯？請說明實作邏輯。
3. **資料庫 Schema 建議**：應如何修改 `stocks_us` 或是增加一個 Metadata 表來整合這些資訊？

請以技術報告格式回覆。
"""

OLLAMA_URL = "http://localhost:11434/api/generate"
payload = {
    "model": "gemma4:26b",
    "prompt": prompt,
    "stream": False,
    "options": {"temperature": 0.3}
}

try:
    response = requests.post(OLLAMA_URL, json=payload, timeout=600)
    with open('gemma_us_sector_fix.md', 'w', encoding='utf-8') as f:
        f.write(response.json().get("response", "No response from Gemma"))
    print("Done")
except Exception as e:
    print(f"Error: {e}")
