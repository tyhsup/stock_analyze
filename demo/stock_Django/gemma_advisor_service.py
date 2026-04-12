import os
import logging
import json
import subprocess
import pandas as pd
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class GemmaAdvisorService:
    def __init__(self):
        # 腳本路徑相對於專案根目錄
        self.reasoner_path = os.path.abspath(os.path.join(os.getcwd(), ".agents", "helpers", "gemma_reasoner.py"))

    def generate_stock_report(self, ticker: str, technical_summary: str, sentiment_data: Dict[str, Any], valuation_data: Dict[str, Any]) -> str:
        """
        將所有數據合成一個 Prompt 並調用本地 Gemma 4 進行推理分析。
        """
        prompt = f"""
        請針對股票 {ticker} 進行深度投資分析。
        
        [技術面數據]
        {technical_summary}
        
        [輿情面 (Sentiment)]
        - 正面比例: {sentiment_data.get('positive', 0)}
        - 負面比例: {sentiment_data.get('negative', 0)}
        - 情緒強度: {sentiment_data.get('label', '中性')}
        
        [基本面/估值]
        - 公允價值: {valuation_data.get('fair_value', 'N/A')}
        - 上漲空間: {valuation_data.get('upside', 'N/A')*100:.1f}%
        - DCF估算值: {valuation_data.get('dcf', {}).get('implied_price', 'N/A')}
        
        任務：
        1. 結合以上數據，分析目前股價與估值的偏離原因。
        2. 判斷目前的 AI 預測趨勢 (看漲機率: {sentiment_data.get('score', 50)}%) 是否具備基本面支撐。
        3. 提供具體的操作風險提示（字數 200 字以內，繁體中文）。
        """
        
        try:
            # 調用本地推理腳本
            # 由於 gemma_reasoner.py 接受第一個參數作為 Query
            result = subprocess.run(
                ["python", self.reasoner_path, prompt],
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=300
            )
            
            if result.returncode == 0:
                report = result.stdout.strip()
                return report
            else:
                logger.error(f"Gemma reasoning failed: {result.stderr}")
                return "智慧分析暫時無法產生報告，請檢查本地模型狀態。"
                
        except Exception as e:
            logger.error(f"Failed to invoke Gemma Advisor: {e}")
            return f"智慧顧問異常: {str(e)}"

    def get_structured_advice(self, ticker: str, data: Dict[str, Any]) -> str:
        """
        簡化版接口，供 views.py 調用。
        """
        # 1. 整理技術面簡要
        kline = data.get('historical_data', pd.DataFrame())
        if not kline.empty:
            last = kline.iloc[-1]
            tech_info = f"收盤價: {last.get('Close', 0):.2f}, 5MA: {last.get('SMA_5', 0):.2f}, 趨勢: {data.get('trend_label', '未知')}"
        else:
            tech_info = "暫無技術面數據"
            
        # 2. 整理情緒面
        senti = data.get('sentiment_summary', {})
        
        # 3. 整理估值面
        val = data.get('valuation', {})
        
        return self.generate_stock_report(ticker, tech_info, senti, val)
