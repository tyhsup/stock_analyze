import os
import torch
import logging
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from transformers import BertTokenizer, BertForSequenceClassification
from ckiptagger import WS, POS, NER
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class NLPService:
    """
    NLP 服務類別，提供新聞情感分析與斷詞功能。
    使用單例模式確保模型僅載入一次。
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NLPService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = os.getenv('MODEL_DIR', 'E:/Infinity/webbug/')
        self.ckip_data_dir = os.getenv('CKIP_DATA_DIR', 'E:/Infinity/webbug/data')
        
        # BERT 模型路徑
        self.bert_tokenizer_path = os.path.join(self.model_dir, 'final_tokenizer_stock_news_BERT_1k')
        self.bert_model_path = os.path.join(self.model_dir, 'final_model_stock_news_BERT_1k')
        
        try:
            logger.info(f"正在從 {self.bert_model_path} 載入 BERT 模型 (設備: {self.device})...")
            self.tokenizer = BertTokenizer.from_pretrained(self.bert_tokenizer_path)
            self.model = BertForSequenceClassification.from_pretrained(self.bert_model_path).to(self.device)
            self.model.eval()
            
            # CKIP Tagger 模型 (設為可選)
            try:
                logger.info("正在載入 CKIP Tagger 模型...")
                self.ws = WS(self.ckip_data_dir)
                self.pos = POS(self.ckip_data_dir)
                self.ner = NER(self.ckip_data_dir)
                self.has_ckip = True
            except Exception as e_ckip:
                logger.warning(f"CKIP 載入失敗 (將跳過斷詞): {e_ckip}")
                self.has_ckip = False
            
            # 載入停用詞
            self.stop_words = self._load_stop_words()
            
            self._initialized = True
            logger.info("NLP 服務初始化完成。")
        except Exception as e:
            logger.error(f"NLP 服務初始化失敗: {e}")
            self._initialized = False

    def _load_stop_words(self) -> List[str]:
        stop_words_path = os.path.join(self.model_dir, 'cn_stop_words.txt')
        if os.path.exists(stop_words_path):
            with open(stop_words_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f]
        return []

    def clean_text(self, text: str) -> str:
        """過濾停用詞並進行簡單清洗"""
        if not self._initialized or not self.has_ckip:
            return text
            
        word_sentence_list = self.ws([text])
        filtered_words = [w for w in word_sentence_list[0] if w not in self.stop_words]
        return " ".join(filtered_words)

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        分析文本情緒。支援長文本分段處理。
        回傳: {'score': float (0-1), 'label': str ('positive'/'negative'/'neutral')}
        """
        if not self._initialized:
            return {'score': 0.5, 'label': 'neutral', 'error': 'Service not initialized'}

        if not text or not isinstance(text, str):
            return {'score': 0.5, 'label': 'neutral'}

        try:
            # BERT 限制通常為 512 token
            max_len = 512
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            
            # 如果文本較短，直接處理
            if len(tokens) <= max_len:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=max_len
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    score = torch.sigmoid(outputs.logits)[0].item()
            else:
                # 長文本分段處理
                scores = []
                # 每次取 510 個 token (保留 [CLS], [SEP])
                chunk_size = 510
                for i in range(0, len(tokens), chunk_size):
                    chunk_tokens = tokens[i : i + chunk_size]
                    if not chunk_tokens: continue
                    
                    # 補回特殊的 [CLS] 和 [SEP] (簡化處理)
                    input_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                    inputs = self.tokenizer(input_text, return_tensors="pt", padding="max_length", 
                                          truncation=True, max_length=max_len).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        scores.append(torch.sigmoid(outputs.logits)[0].item())
                
                score = sum(scores) / len(scores) if scores else 0.5
            
            # P0: Implementation Plan says score >= 0.65 positive, <= 0.35 negative
            label = 'neutral'
            if score >= 0.65: label = 'positive'
            elif score <= 0.35: label = 'negative'
                
            return {
                'score': score,
                'label': label
            }
        except Exception as e:
            logger.error(f"情緒分析執行錯誤: {e}")
            return {'score': 0.5, 'label': 'neutral'}

    def batch_analyze_sentiment(self, texts: List[str]) -> List[Dict[str, Any]]:
        """批量情緒分析 (優化效能)"""
        if not self._initialized or not texts:
            return [{'score': 0.5, 'label': 'neutral'}] * len(texts)

        try:
            # 真正批量處理：tokenize all with padding
            max_len = 512
            encoded_inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded_inputs)
                scores = torch.sigmoid(outputs.logits).cpu().numpy()

            results = []
            for s_arr in scores:
                score = s_arr[0]
                label = 'neutral'
                if score >= 0.65: label = 'positive'
                elif score <= 0.35: label = 'negative'
                results.append({'score': float(score), 'label': label})
            return results
        except Exception as e:
            logger.error(f"批量分析錯誤: {e}")
            return [{'score': 0.5, 'label': 'error'}] * len(texts)
