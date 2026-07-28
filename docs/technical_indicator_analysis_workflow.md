# Gemini 雲端智慧投資建議：技術指標協同判讀工作流 SOP (v1.0)

本工作流規範旨在提供 Gemini 雲端智慧投資建議技術面 Agent 在進行台美股個股與大盤技術面分析時的標準作業程序（SOP）。Agent 在產出任何投資建議或技術分析結論前，必須嚴格遵守以下四層級分析架構、背離過濾與防衛機制。

---

## 一、 核心分析原則 (Core Analytical Principles)

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

## 二、 四層級指標判讀優先順序 (Priority Hierarchy)

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

### 1. 層級 1：價格行為與經典型態判讀
* **頭肩頂 / 頭肩底**：
  * 頭肩頂（頭部高於左右肩）：收盤價跌破頸線確認趨勢由漲轉跌。
  * 頭肩底（頭部低於左右肩）：收盤價放量突破頸線確認底部成立。
* **雙重頂 (M頂) / 雙重底 (W底)**：
  * 兩度測試相近阻力/支撐位。突破中間轉折高/低點（頸線）時型態確立。
* **關鍵防衛線**：以頸線 (Neckline) 為多空分界。

### 2. 層級 2：成交量與量價九宮格驗證
* **價漲量增**：健康攻擊格局，順勢做多或續抱。
* **價漲量縮**：虛漲背離，追價意願不足，高檔假突破風險高。
* **價跌量增**：高檔出現為強烈空頭起跌；長線低檔出現則可能為恐慌盤拋售與落底換手。
* **價跌量縮**：籌碼沉澱與賣壓竭盡，多頭趨勢中為良性回檔。

### 3. 層級 3：趨勢指標 (MACD & MA) 協同過濾
* **MACD 零軸位置分界**：
  | MACD 交叉情境 | 零軸位置 | 趨勢判定 | 訊號可靠性與 Agent 處置 |
  | :--- | :--- | :--- | :--- |
  | 黃金交叉 | 零軸之上 | 強烈多頭攻擊 | **高**：順勢建立多單或加碼 |
  | 黃金交叉 | 零軸之下 | 弱勢空頭反彈 | **低**：僅視為短線反彈，不宜重倉追高 |
  | 死亡交叉 | 零軸之下 | 強烈空頭下殺 | **高**：順勢平倉多單或建立空單 |
  | 死亡交叉 | 零軸之上 | 多頭回檔修正 | **中**：多單適度減碼或提昇停利點 |

### 4. 層級 4：震盪指標 (RSI & 布林通道) 轉折確認
* **RSI 超買超賣**：RSI < 30 進入超賣區，等待回升至 30-50 區間確認反彈動能；RSI > 70 進入超買區警惕回檔。
* **布林通道 (Bollinger Bands)**：價格觸及布林上軌且 RSI > 70 提供雙重超買賣出確認；觸及下軌且 RSI < 30 提供雙重超賣買進確認。

---

## 三、 背離分析標準作業程序 (Divergence SOP)

背離是捕捉趨勢動態衰竭的核心先行指標。Agent 必須區分以下背離型態：

### 1. 一般背離 (Regular Divergence — 預示趨勢反轉)
* **頂背離（高檔看跌背離）**：
  * 現象：價格持續創波段新高（Higher High），但 MACD 柱狀體/快線高點或 RSI 高點呈現下滑（Lower High）。
  * 意義：買盤推升力道衰竭，為強烈賣出或轉空警訊。
* **底背離（低檔看漲背離）**：
  * 現象：價格持續創波段新低（Lower Low），但 MACD 谷值或 RSI 低點逐漸墊高（Higher Low）。
  * 意義：賣壓邊際遞減，恐慌盤拋售結束，反彈彈升機率高。

### 2. RSI 隱藏背離 (Hidden Divergence — 預示趨勢延續)
* **隱藏看漲背離（順勢買點）**：
  * 現象：上升趨勢中，股價創下較高的低點（Higher Low），但 RSI 卻創下較低的低點（Lower Low）。
  * 意義：下殺動能已耗盡，多頭趨勢短暫休息後即將發動下一波攻擊。
* **隱藏看跌背離（順勢賣點）**：
  * 現象：下降趨勢中，股價創下較低的高點（Lower High），但 RSI 卻創下較高的高點（Higher High）。
  * 意義：反彈動能耗盡，空頭趨勢延續。

---

## 四、 風險控制與假突破防衛機制 (Risk Control & Anti-Fakeout)

為過濾市場假突破陷阱並落實「小虧大賺」，Agent 生成的建議必須包含以下防護機制：

### 1. 三重突破驗證機制 (Triple Verification)
1. **收盤價確認**：嚴禁因盤中價格短暫穿透頸線或布林邊界即判定突破。必須等待 K 線（H4 或 D1）收盤價實體完全收在頸線之外。
2. **量能與回測確認 (Pullback)**：突破時必須伴隨成交量放大。放量突破後，耐心等待價格以「量縮」方式回測原頸線（此時壓力轉支撐），守穩不破為最佳高勝率進場點。
3. **2B 法則過濾（失敗的突破）**：當價格創波段新高/新低後未能站穩，並迅速折回前期高點之下/低點之上時，構成 2B 假突破。此時可進行左側試單，將停損極窄化地設於該假突破之最高/最低點外側。

### 2. 停損與停利計算標準
* **型態學結構停損**：
  * 多單停損：設定於頭肩底或 W 底之「右肩/右腳低點」下方。
  * 空單停損：設定於頭肩頂或 M 頂之「右肩/右頭高點」上方。
* **目標價計算（等距測量法）**：
  $$\text{最小目標價} = \text{頸線突破價} \pm H$$
  其中 $H$ 為頭部/頂點至頸線之垂直距離。
* **分批移動停利**：第一目標價達標後平倉 50% 部位，剩餘部位跟隨 20 MA 或近期轉折低點進行移動停利（Trailing Stop）。

---

## 五、 Gemini AI Agent 提示詞樣板與 JSON 輸出規範

Gemini 技術面 Agent 在執行分析時，系統必須注入以下 System Prompt 並要求輸出固定 Schema：

### 1. JSON 輸出 Schema 規範
```json
{
  "symbol": "2330.TW",
  "analysis_timestamp": "2026-07-25T21:30:00Z",
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
  "risk_management": {
    "entry_strategy": "Pullback_Entry",
    "entry_price_zone": "980.0 - 985.0",
    "stop_loss_price": 955.0,
    "take_profit_target_1": 1030.0,
    "risk_reward_ratio": 2.25
  },
  "technical_verdict_reasoning": "價格已於日線級別以實體 K 線放量突破 980 頸線，且 MACD 於零軸之上維持金叉，RSI 呈現隱藏看漲背離。判定為結構性底部完成，建議待回測 980 頸線守穩後分批建倉。"
}
```

---
*本 SOP 為 Gemini 雲端智慧投資建議技術面 Agent 運作之最高規範，修改或更新需經由四 Agent Council 協商決議。*
