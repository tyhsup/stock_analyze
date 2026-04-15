$prompt = "請根據之前討論的四項優化建議：
1. 整合 FinBERT 語義分析取代單純的正負向機率
2. 引入 GNN / Node Transformer 構建股票關聯矩陣
3. Cross-Modal 多模態特徵融合 (取代原有的 Concatenate)
4. 程式架構解耦重構 (切割 Data Loaders 與 Model Config)

請排定一個分階段的版次修改計畫（例如：v1.0 基礎解耦, v2.0... 等），說明每個階段要實作的項目順序、為何這樣安排（相依性考量），以及各階段的測試/驗證重點。請以具體的計畫文件格式輸出。"

$prompt | Out-File -Encoding utf8 -FilePath e:/Infinity/mydjango/plan_query.txt

python .agents/helpers/gemma_reasoner.py "$(Get-Content e:/Infinity/mydjango/plan_query.txt -Raw)"
