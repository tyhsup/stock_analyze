# Gemini 雲端智慧投資建議：綜合性分析與技術指標協同判讀工作流 SOP (v2.0)

本工作流規範旨在提供 Gemini 雲端智慧投資建議系統在進行台美股個股與大盤分析時的標準作業程序（SOP）。

本文件採用**雙層架構**：
1. **上層：三維度過濾模型 (Three-Dimensional Filter Model)**：作為 Gemini 雲端智慧投資建議的綜合性分析模型，統合基本面、籌碼面與技術面。
2. **下層：四層級技術指標判讀 SOP (Technical Indicator Priority Hierarchy)**：作為技術面 Agent 專責分析技術指標時的內部 SOP，供三維度模型的 Level 3 執行階段調用。

Agent 在產出任何投資建議或分析結論前，必須嚴格遵守以下分析架構、過濾機制與防衛規範。

---

## 壹、 三維度過濾模型：綜合性分析框架 (Three-Dimensional Filter Model)

### 一、 核心邏輯與維度拆解

股票分析主要區分為「基本面」、「技術面」與「籌碼面」三大面向。單一面向皆有其分析盲點：

* 單看基本面容易買在股價發動前的漫長等待期。
* 單看技術面容易在假突破或短線拉拍時誤判方向。
* 單看籌碼面則可能誤跟主力的短線炒作與拉高出貨。

**三面整合的核心在於回答三個互補的關鍵問題：**

| 面向 | 核心問題 | 分析工具 | 適用時間維度 |
| :--- | :--- | :--- | :--- |
| **基本面** | 公司體質好不好？成長動能如何？ | ROE、PEG、月營收 YoY/MoM、毛利率、EPS | 中長期（季至年） |
| **籌碼面** | 錢往哪裡跑？誰在買賣？ | 三大法人買賣超、集保大戶、關鍵分點、融資融券 | 短期（日至週） |
| **技術面** | 價格是否走出趨勢？最佳進出場時機？ | K 線型態、均線、MACD、RSI、布林通道、量價關係 | 短中期（日至月） |

### 二、 市場分支邏輯 (Market Branching)

由於台股與美股的市場結構差異，三維度模型必須依據市場類型進行分支處理：

| 市場 | 適用模式 | 籌碼面處理方式 |
| :--- | :--- | :--- |
| **台股 (TW)** | 完整三維度（基本面 × 籌碼面 × 技術面） | 三大法人買賣超、集保大戶持股、關鍵分點/地緣分點、融資融券 |
| **美股 (US)** | 二維度（基本面 × 技術面） | 籌碼面標記為 `N/A`；可選擇性引用 13F 機構持倉變化、Options Flow、Dark Pool 數據作為輔助參考，但不納入加權計算 |

### 三、 實戰 3 步驟選股與交易 SOP

#### Level 1 (定位)：基本面 × 籌碼面 —— 確立趨勢大方向與基本面底氣

**核心邏輯**：尋找「有故事」且「資金認可」的標的。

##### 1-A. 基本面篩選（台股與美股通用）

* **ROE 體質檢驗**：優先選擇 ROE > 20% 的標的，代表公司運用股東資金創造獲利的能力強。ROE 決定估值天花板。
* **PEG 成長性驗證**：PEG 落於 0.75 - 1.2 區間為合理；PEG < 0.75 為潛在低估（需排除衰退陷阱）；PEG > 1.5 且 ROE < 10% 屬高風險，應絕對避開。
* **營收與獲利趨勢**：月營收 YoY 連續 3 個月以上成長，且毛利率呈現結構性上揚（產品升級而非裁員縮編），為高品質成長訊號。

| 情境 | ROE | PE | PEG | 實戰決策 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 高 (>20%) | 低 | <1 | 難得機會：建立部位區（搭配技術面確認） |
| 2 | 高 (>20%) | 中 | ≈1 | 合理價：分批建立部位區 |
| 3 | 中 (10-20%) | 高 | >1.2 | 觀察區：等待估值修正或基本面提升 |
| 4 | 低 (<10%) | 高 | >1.5 | 高風險：絕對避開，嚴防價值陷阱 |

##### 1-B. 籌碼面篩選（僅台股適用；美股此區塊標記 `N/A`）

* **三大法人追蹤**：外資/投信連續 5 日累計買超，且總張數明顯增加。
* **集保大戶持股**：400 張以上大戶持股比例連續增加（籌碼集中），10 張以下散戶持股減少。
* **融資融券交叉判讀**：融資減少 + 融券增加 + 法人買超 = 散戶離場、法人進場，正面訊號。融資增加 + 法人賣超 = 散戶追高、法人出貨，需警戒。
* **關鍵分點與地緣分點**：追蹤低檔大買、高檔大賣的高勝率券商分點，以及公司所在地之地緣分點或庫藏股分點。

##### 1-C. Level 1 防禦機制

* 基本面優良但股價已透支（漲幅過大、PEG 過高）時，不得強行建倉。
* 籌碼面極佳但基本面惡化時，法人可能僅為短線操作，不值得跟進。
* **Level 1 通過條件**：基本面至少 2 項達標（ROE + PEG 或 ROE + 營收趨勢），且籌碼面無明顯負面訊號（台股）。

---

#### Level 2 (確認)：籌碼面 × 技術面 —— 精準捕捉發動時機與過濾假突破

**核心邏輯**：將 Level 1 的假設，用價格行為與資金動能來驗證。

##### 2-A. 技術突破確認

* 當籌碼面結構良好（大戶鎖碼、法人買進）或基本面通過篩選（美股）時，技術面尋找以下突破時機：
  * 帶量突破盤整區間。
  * 站上關鍵均線（月線 20MA 或季線 60MA）。
  * 突破經典型態頸線（W 底、頭肩底）。

##### 2-B. 多重指標共振過濾假突破

1. **動能指標確認**：突破時需伴隨 MACD 在零軸上方形成黃金交叉（或柱狀體由負翻正），且 RSI 從超賣區反彈或突破 50 中線。
2. **量價背離驗證**：健康的突破必須「價漲量增」。若股價創高但成交量萎縮（價漲量縮），或 RSI/MACD 出現高檔頂背離，暗示追價力道不足，極可能為假突破陷阱。
3. **突破三重驗證機制**：
   * K 線實體收盤價完全站穩頸線/壓力位之外。
   * 等待拉回測試原壓力轉支撐（Pullback）且量縮守穩後再進場。
   * 若突破後 3 根 K 線內價格又縮回區間內，立即離場。

##### 2-C. Level 2 防禦機制

* 條件不齊全不強行建倉：當籌碼面極佳但技術面破底，或技術面突破但無成交量與法人買盤配合時，假突破風險極高，應耐心等待共振訊號。
* 當指標出現背離但大戶正在吸籌（Level 1 確認），可視為「積極的假跌破」，提高觀察權重而非立即放棄。

---

#### Level 3 (執行)：技術面 × 風險控制 —— 精準進出場與資金管理

**核心邏輯**：將所有分析結果轉化為具體的交易參數。此階段調用「技術面 Agent 四層級 SOP」（本文件貳部分）進行精細技術判讀。

##### 3-A. 條件式建倉（2B 與 123 法則協同）

* **左側試單（2B 法則）**：若價格創高/創低後未能站穩並迅速折回前高/前低，且 MACD 柱狀體背離，可建立輕倉底倉，將停損設於極端高/低點外側，提供極佳盈虧比。
* **右側加碼（123 法則）**：當價格順利突破頸線位完成型態，確認原趨勢被破壞並反轉時再行加碼。

##### 3-B. 嚴格風控與停利機制

* **技術停損**：依據型態頸線、關鍵均線支撐或前期波段高低點設定停損。台股可輔以「大戶平均成本區間」設定結構性止損點。
* **分批停利與移動停利**：達第一目標價（如型態等距測量幅）時平倉 50% 部位，剩餘部位利用均線（如 20MA）或新轉折點進行移動停利，讓利潤奔跑。
* **部位管理紀律**：採用分批建倉（如 3:2:1 或 1:2:3），順勢賺錢才加碼、賠錢堅決減碼停損，杜絕逢跌攤平。

---

### 四、 指標權重與衝突裁決機制 (Conflict Resolution)

當三面訊號出現矛盾時，Agent 必須依照以下決策矩陣進行裁決：

| 情境 | 基本面 | 籌碼面 | 技術面 | 裁決結論 |
| :--- | :--- | :--- | :--- | :--- |
| A | 正面 | 正面（台股）/ N/A（美股） | 正面（放量突破） | **強烈買進訊號**：三維共振，信心最高 |
| B | 正面 | 正面 | 負面（技術面破底） | **暫緩進場**：等待技術面止穩確認，籌碼面可能在吸籌階段 |
| C | 正面 | 負面（法人倒貨） | 正面（突破） | **高風險假突破**：法人出貨伴隨技術突破，極可能為拉高出貨 |
| D | 負面 | 正面 | 正面 | **短線投機**：基本面不支持長期持有，僅作短線跟單 |
| E | 負面 | 負面 | 正面 | **拒絕建議**：僅有技術面訊號，缺乏根本支撐，Agent 降低信賴等級 |
| F | 正面 | N/A（美股） | 正面 | **有效買進訊號**：二維共振（美股模式），信心中高 |

---

## 貳、 技術面 Agent 內部 SOP：四層級指標判讀與防衛機制

> 以下為技術面 Agent 專責分析技術指標時的內部標準作業程序，供三維度模型 Level 3 (執行) 階段調用。

### 一、 核心分析原則 (Core Analytical Principles)

1. **價格行為優先，指標確認在後 (Price Action First, Indicators Second)**
   * 所有技術指標（MACD、RSI、KD 等）皆基於歷史價格導出，具備先天的滯後性。
   * 指標的核心功能在於「動能驗證」與「強度過濾」，絕不可單憑指標交叉（如黃金交叉/死亡交叉）獨立作為進出場決策依據。
   * 所有判讀必須先建立在價格結構、經典型態與關鍵支撐壓力位（頸線）之上。

2. **嚴格區分波動性回調與結構性反轉 (Pullback vs. Structural Reversal)**
   * **波動性回調 (Pullback)**：既有趨勢中的短暫修正。特徵為成交量顯著萎縮（價跌量縮）、中短期均線扣抵值仍高於現價、MACD 快慢線維持在零軸之上、RSI 於 40-50 中軌區間獲得支撐。
   * **結構性反轉 (Structural Reversal)**：市場情緒與籌碼結構的根本性轉向。特徵為帶量突破或跌破關鍵頸線、均線走平並實質蓋頭下彎、指標出現嚴重頂/底背離與零軸下穿。

3. **多重時間框架與指標共振 (Multi-Timeframe & Indicator Confluence)**
   * **時間框架原則**：遵循「順大勢、逆小勢」。使用日線 (D1) 或週線 (W1) 確立大方向趨勢，使用 4 小時 (H4) 或 1 小時 (H1) 尋找精準進出場點。
   * **指標共振 (Confluence)**：高勝率訊號必須具備至少三個維度的共振（例如：價格放量突破頸線 + MACD 零軸之上金叉 + RSI 脫離超賣區回升）。

---

### 二、 四層級指標判讀優先順序 (Priority Hierarchy)

在處理多個技術指標出現分歧或矛盾時，Agent 必須依照以下優先層級進行裁決：

```
層級 1：價格行為與圖表型態 (Price Action & Patterns) — 權重 40%
    ↓
層級 2：成交量與量價關係 (Volume & Supply/Demand) — 權重 30%
    ↓
層級 3：趨勢指標與動能 (Trend & Momentum: MACD, MA) — 權重 20%
    ↓
層級 4：震盪動能與超買超賣 (Oscillators: RSI, KD, BBands) — 權重 10%
```

#### 1. 層級 1：價格行為與經典型態判讀
* **頭肩頂 / 頭肩底**：
  * 頭肩頂（頭部高於左右肩）：收盤價跌破頸線確認趨勢由漲轉跌。
  * 頭肩底（頭部低於左右肩）：收盤價放量突破頸線確認底部成立。
* **雙重頂 (M頂) / 雙重底 (W底)**：
  * 兩度測試相近阻力/支撐位。突破中間轉折高/低點（頸線）時型態確立。
* **關鍵防衛線**：以頸線 (Neckline) 為多空分界。

#### 2. 層級 2：成交量與量價九宮格驗證
* **價漲量增**：健康攻擊格局，順勢做多或續抱。
* **價漲量縮**：虛漲背離，追價意願不足，高檔假突破風險高。
* **價跌量增**：高檔出現為強烈空頭起跌；長線低檔出現則可能為恐慌盤拋售與落底換手。
* **價跌量縮**：籌碼沉澱與賣壓竭盡，多頭趨勢中為良性回檔。

#### 3. 層級 3：趨勢指標 (MACD & MA) 協同過濾
* **MACD 零軸位置分界**：
  | MACD 交叉情境 | 零軸位置 | 趨勢判定 | 訊號可靠性與 Agent 處置 |
  | :--- | :--- | :--- | :--- |
  | 黃金交叉 | 零軸之上 | 強烈多頭攻擊 | **高**：順勢建立多單或加碼 |
  | 黃金交叉 | 零軸之下 | 弱勢空頭反彈 | **低**：僅視為短線反彈，不宜重倉追高 |
  | 死亡交叉 | 零軸之下 | 強烈空頭下殺 | **高**：順勢平倉多單或建立空單 |
  | 死亡交叉 | 零軸之上 | 多頭回檔修正 | **中**：多單適度減碼或提昇停利點 |

#### 4. 層級 4：震盪指標 (RSI & 布林通道) 轉折確認
* **RSI 超買超賣**：RSI < 30 進入超賣區，等待回升至 30-50 區間確認反彈動能；RSI > 70 進入超買區警惕回檔。
* **布林通道 (Bollinger Bands)**：價格觸及布林上軌且 RSI > 70 提供雙重超買賣出確認；觸及下軌且 RSI < 30 提供雙重超賣買進確認。

---

### 三、 背離分析標準作業程序 (Divergence SOP)

背離是捕捉趨勢動態衰竭的核心先行指標。Agent 必須區分以下背離型態：

#### 1. 一般背離 (Regular Divergence — 預示趨勢反轉)
* **頂背離（高檔看跌背離）**：
  * 現象：價格持續創波段新高（Higher High），但 MACD 柱狀體/快線高點或 RSI 高點呈現下滑（Lower High）。
  * 意義：買盤推升力道衰竭，為強烈賣出或轉空警訊。
* **底背離（低檔看漲背離）**：
  * 現象：價格持續創波段新低（Lower Low），但 MACD 谷值或 RSI 低點逐漸墊高（Higher Low）。
  * 意義：賣壓邊際遞減，恐慌盤拋售結束，反彈彈升機率高。

#### 2. RSI 隱藏背離 (Hidden Divergence — 預示趨勢延續)
* **隱藏看漲背離（順勢買點）**：
  * 現象：上升趨勢中，股價創下較高的低點（Higher Low），但 RSI 卻創下較低的低點（Lower Low）。
  * 意義：下殺動能已耗盡，多頭趨勢短暫休息後即將發動下一波攻擊。
* **隱藏看跌背離（順勢賣點）**：
  * 現象：下降趨勢中，股價創下較低的高點（Lower High），但 RSI 卻創下較高的高點（Higher High）。
  * 意義：反彈動能耗盡，空頭趨勢延續。

---

### 四、 風險控制與假突破防衛機制 (Risk Control & Anti-Fakeout)

為過濾市場假突破陷阱並落實「小虧大賺」，Agent 生成的建議必須包含以下防護機制：

#### 1. 三重突破驗證機制 (Triple Verification)
1. **收盤價確認**：嚴禁因盤中價格短暫穿透頸線或布林邊界即判定突破。必須等待 K 線（H4 或 D1）收盤價實體完全收在頸線之外。
2. **量能與回測確認 (Pullback)**：突破時必須伴隨成交量放大。放量突破後，耐心等待價格以「量縮」方式回測原頸線（此時壓力轉支撐），守穩不破為最佳高勝率進場點。
3. **2B 法則過濾（失敗的突破）**：當價格創波段新高/新低後未能站穩，並迅速折回前期高點之下/低點之上時，構成 2B 假突破。此時可進行左側試單，將停損極窄化地設於該假突破之最高/最低點外側。

#### 2. 停損與停利計算標準
* **型態學結構停損**：
  * 多單停損：設定於頭肩底或 W 底之「右肩/右腳低點」下方。
  * 空單停損：設定於頭肩頂或 M 頂之「右肩/右頭高點」上方。
* **目標價計算（等距測量法）**：
  $$\text{最小目標價} = \text{頸線突破價} \pm H$$
  其中 $H$ 為頭部/頂點至頸線之垂直距離。
* **分批移動停利**：第一目標價達標後平倉 50% 部位，剩餘部位跟隨 20 MA 或近期轉折低點進行移動停利（Trailing Stop）。

---

## 參、 Gemini AI Agent 提示詞樣板與 JSON 輸出規範

Gemini 綜合性分析 Agent 在執行分析時，系統必須注入以下 System Prompt 並要求輸出固定 Schema。JSON Schema 僅規範分析邏輯的輸出結構，不預留特定 API 資料格式。

### 1. JSON 輸出 Schema 規範（v2.0 三維度整合版 — 台股範例）
```json
{
  "symbol": "2330.TW",
  "market": "TW",
  "analysis_timestamp": "2026-07-30T03:00:00Z",
  "analysis_mode": "Three_Dimensional",

  "fundamental_analysis": {
    "roe": 28.5,
    "peg": 0.95,
    "revenue_yoy_trend": "Consecutive_Growth_3M",
    "gross_margin_trend": "Structural_Expansion",
    "valuation_scenario": "Scenario_2_Fair_Value",
    "fundamental_verdict": "Positive"
  },

  "chip_analysis": {
    "available": true,
    "foreign_investors_5d": "Net_Buy",
    "investment_trust_5d": "Net_Buy",
    "large_holder_trend": "Concentration_Increasing",
    "margin_trading_signal": "Retail_Exiting_Institutional_Entering",
    "key_branch_activity": "Low_Accumulation_Detected",
    "chip_verdict": "Positive"
  },

  "trend_summary": {
    "primary_trend": "Bullish/Bearish/Neutral",
    "reversal_signal_detected": true,
    "reversal_type": "Head_and_Shoulders_Bottom / Double_Top / 2B_Fakeout / None",
    "confidence_score": 0.88
  },

  "priority_layers_evaluation": {
    "layer1_price_action": {
      "pattern": "W Bottom",
      "neckline_price": 980.0,
      "breakout_status": "Confirmed_By_Close"
    },
    "layer2_volume": {
      "volume_status": "Volume_Expanding",
      "price_volume_matrix": "Price_Up_Volume_Up"
    },
    "layer3_trend_momentum": {
      "macd_status": "Golden_Cross_Above_Zero",
      "ma_alignment": "Bullish_Alignment"
    },
    "layer4_oscillators": {
      "rsi_value": 58.5,
      "rsi_divergence": "Hidden_Bullish_Divergence"
    }
  },

  "conflict_resolution": {
    "scenario": "A",
    "description": "三維共振：基本面正面、籌碼面正面、技術面放量突破",
    "final_confidence_adjustment": "None"
  },

  "risk_management": {
    "entry_strategy": "Pullback_Entry",
    "entry_price_zone": "980.0 - 985.0",
    "stop_loss_price": 955.0,
    "stop_loss_basis": "W_Bottom_Right_Foot_Below",
    "take_profit_target_1": 1030.0,
    "take_profit_target_2": 1055.0,
    "risk_reward_ratio": 2.25,
    "position_sizing": "3:2:1_Batch_Entry"
  },

  "integrated_verdict_reasoning": "【基本面】ROE 28.5% 且 PEG 0.95 落於合理區間，月營收 YoY 連續 3 個月成長，毛利率結構性上揚。【籌碼面】外資與投信連續 5 日買超，集保大戶持股比例連續增加，融資餘額未異常增加。【技術面】價格已於日線級別以實體 K 線放量突破 980 頸線，且 MACD 於零軸之上維持金叉，RSI 呈現隱藏看漲背離。【綜合判定】三維度共振（情境 A），判定為結構性底部完成，建議待回測 980 頸線守穩後分批建倉。"
}
```

### 2. JSON 輸出 Schema 規範（v2.0 二維度版 — 美股範例）
```json
{
  "symbol": "AAPL",
  "market": "US",
  "analysis_timestamp": "2026-07-30T03:00:00Z",
  "analysis_mode": "Two_Dimensional",

  "fundamental_analysis": {
    "roe": 160.0,
    "peg": 1.1,
    "revenue_yoy_trend": "Consecutive_Growth_3M",
    "gross_margin_trend": "Stable",
    "valuation_scenario": "Scenario_2_Fair_Value",
    "fundamental_verdict": "Positive"
  },

  "chip_analysis": {
    "available": false,
    "supplementary_reference": {
      "13f_institutional_trend": "Major_Funds_Increasing_Position",
      "options_flow_signal": "Bullish_Call_Sweep_Detected"
    },
    "chip_verdict": "N/A"
  },

  "trend_summary": {
    "primary_trend": "Bullish",
    "reversal_signal_detected": false,
    "reversal_type": "None",
    "confidence_score": 0.82
  },

  "priority_layers_evaluation": {
    "layer1_price_action": {
      "pattern": "Breakout_Above_Consolidation",
      "neckline_price": 195.0,
      "breakout_status": "Confirmed_By_Close"
    },
    "layer2_volume": {
      "volume_status": "Volume_Expanding",
      "price_volume_matrix": "Price_Up_Volume_Up"
    },
    "layer3_trend_momentum": {
      "macd_status": "Golden_Cross_Above_Zero",
      "ma_alignment": "Bullish_Alignment"
    },
    "layer4_oscillators": {
      "rsi_value": 62.3,
      "rsi_divergence": "None"
    }
  },

  "conflict_resolution": {
    "scenario": "F",
    "description": "二維共振（美股模式）：基本面正面、技術面正面，籌碼面不適用",
    "final_confidence_adjustment": "None"
  },

  "risk_management": {
    "entry_strategy": "Pullback_Entry",
    "entry_price_zone": "195.0 - 197.0",
    "stop_loss_price": 188.0,
    "stop_loss_basis": "Consolidation_Low_Below",
    "take_profit_target_1": 210.0,
    "take_profit_target_2": 220.0,
    "risk_reward_ratio": 2.0,
    "position_sizing": "1:2:3_Batch_Entry"
  },

  "integrated_verdict_reasoning": "【基本面】ROE 極高且 PEG 1.1 落於合理區間，營收穩定成長。【籌碼面】美股不適用台股籌碼分析；輔助參考顯示 13F 主要基金增持、Options Flow 偵測到看漲大單掃貨。【技術面】價格放量突破 195 盤整區間，MACD 零軸之上金叉，RSI 62.3 處於健康多頭區間。【綜合判定】二維度共振（情境 F），信心中高，建議待回測 195 守穩後分批建倉。"
}
```

---

### 五、 核心指標與實戰工具矩陣

| 面向 | 分析工具/指標 | 實戰關鍵判讀規則 | 核心功能與注意事項 |
| :--- | :--- | :--- | :--- |
| **基本面** | ROE / PEG / 營收 | 高 ROE (>20%) + PEG 0.75-1.2 + 月營收 YoY 連續成長 | ROE 決定估值天花板，PEG 驗證成長溢價，營收與毛利率驗證獲利品質 |
| **籌碼面** | 三大法人 / 集保大戶 | 外資/投信連續多日買超 + 千張大戶持股比例連續增加 + 融資減 | 追蹤強勢資金（聰明錢），判斷籌碼趨向集中或發散（僅台股） |
| **籌碼面** | 關鍵/地緣分點 | 尋找低檔大買、高檔大賣的勝率券商 | 防範隔日沖分點強拉後隔天倒貨（僅台股） |
| **技術面** | 均線 (MA) & 扣抵值 | 短期 MA5/10，波段 MA20/60，長線 MA240 | 均線反映平均成本與趨勢；扣抵值預判均線翻轉 |
| **技術面** | MACD 指標 | 零軸以上黃金交叉（最強多頭）；高檔頂背離預警動能衰竭 | 衡量中長線波段動能與方向，盤整市易失真 |
| **技術面** | RSI 指標 | 50 多空分界；>70 超買、<30 超賣；隱藏看漲背離預示趨勢延續 | 買賣力道溫度計；單邊趨勢中易高/低檔鈍化 |
| **技術面** | 布林通道 (Bollinger) | 通道擠壓預示突破；強趨勢貼帶沿 ±2σ 運行 | 衡量波動率；貼帶時切忌逆勢摸頂抄底 |
| **技術面** | 量價關係九宮格 | 價漲量增（健康多頭）；價漲量縮（動能衰退/虛漲） | 成交量為燃料，驗證突破真偽 |

---

### 六、 避坑指南與風控紀律

1. **避免「單一指標信仰」**：指標皆具滯後性，是驗證動能而非預測未來的預言球，必須堅持「價格行為與型態優先，指標與籌碼確認在後」。
2. **條件不齊全不強行建倉**：當籌碼面極佳但技術面破底，或技術面突破但無成交量與法人買盤配合時，假突破風險極高。
3. **嚴格執行交易紀律與部位管理**：進場前先想好停損與風險承受度，採用分批建倉，順勢賺錢才加碼、賠錢堅決減碼停損，杜絕逢跌攤平。

---
*本 SOP 為 Gemini 雲端智慧投資建議系統運作之最高規範，修改或更新需經由四 Agent Council 協商決議。*
