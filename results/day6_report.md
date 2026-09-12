# Day 6 預檢報告 — Go/No-Go Decision

## 結論:NO-GO — Pipeline 需要修正

T4/T5 DSI = 0.000,完全不落在 Maisak 2013 的 0.6-0.8 生物學區間。
原因是 T4/T5 細胞完全沒有 spike,不是因為連接體沒有先驗,是因為訊號沒有從光感受器傳到 T4/T5。

---

## 完成的事

1. **確認 Stonkfly visual.py 用 hex grid** ✅
   - `prepare.py` 第 61 行:`xy.append((h[0] - 0.5 * h[1], np.sqrt(3) / 2 * h[1]))`
   - 這是標準 axial hex → cartesian 轉換
   - `graph.npz` 同時保存 `hexes`(原始 hex 座標)和 `uv`(正規化矩形)

2. **編譯 C++ LIF 核心** ✅
   - `libmemory.so` 編譯成功,SHA-256 驗證通過
   - 0.1ms 步長,事件驅動,lazy subthreshold evolution

3. **實作 hex grid drifting grating 生成器** ✅
   - 直接在 hex cartesian 座標上生成正弦光柵
   - 8 方向 × 2 秒,3,335 個光感受器
   - 方向性驗證:0° vs 180° 差異 mean=0.620, max=1.0(反方向接近反相)

4. **跑完整 8 方向 × 2 秒實驗** ✅
   - 總 wall time ~50 秒(非常快)
   - 結果:T4/T5 在所有 8 個方向都是 0 spike

---

## 診斷結果

### 訊號傳遞鏈

| 層級 | 細胞類型 | 膜電位(mV) | spike/100ms | 狀態 |
|------|---------|-----------|-------------|------|
| 光感受器 | R1-R6 | -49 | ~41,000 | ✅ 正常 |
| Lamina | L1 | -54 | ~5,500 | ✅ 正常 |
| Lamina | L2 | -54 | ~1,300 | ✅ 正常 |
| Lamina | L3 | -54 | ~2,400 | ✅ 正常 |
| Medulla | Mi1 | **-55** | **0** | ❌ 被抑制 |
| Medulla | Tm3 | **-55** | **0** | ❌ 被抑制 |
| Medulla | Mi4 | -55 | 0 | ❌ 被抑制 |
| Lobula | T4 | **-52(rest)** | **0** | ❌ 無輸入 |
| Lobula | T5 | -52(rest) | 0 | ❌ 無輸入 |

### 根本原因

**Mi1 和 Tm3 的膜電位被推到 -55 mV,比 rest(-52)還低。**

- Mi1/Tm3 確實被 awaken(在 active list 裡,共 12,769 個 active cells)
- L1 → Mi1 和 L1 → Tm3 的連接確實存在(Tm3[0] 有 6 個 L1 上游)
- 但 Mi1/Tm3 的 incoming edges 中,抑制性(負權重)多於興奮性
  - Tm3[0]: 78 個 incoming edges,42 個負權重,36 個正權重
- Stonkfly 的 LIF 動力學只給 lamina tonic drive(`lamina_bias=12.0`),不給 medulla
- 沒有 tonic drive,Mi1/Tm3 只靠突觸輸入,而突觸輸入的興奮/抑制不平衡導致它們被壓制

### 為什麼這不是「連接體沒有先驗」

這個結果**不回答核心問題**。它只回答了「Stonkfly 的 LIF 動力學參數不適合驅動 T4/T5」。

核心問題是「連接體拓撲本身是否對運動偵測有先驗能力」。要回答這個問題,訊號必須能到達 T4/T5。目前訊號卡在 Mi1/Tm3,所以 T4/T5 的 DSI=0 不代表連接體沒有先驗,只代表 pipeline 有問題。

---

## 修正方案

### 方案 A:給 medulla 細胞 tonic drive(最小改動)

在 `_neural_step` 中,除了 `lamina_bias` 給 lamina,也給 Mi1/Tm3/Mi4/Tm9 一個小的 tonic drive,把它們的膜電位從 -55 推到接近 -47(接近閾值)。

**優點**:改動最小,只需調整 `brain.py` 的 `_neural_step`
**風險**:tonic drive 是非生物學的,可能影響 DSI 測量的正當性

### 方案 B:調整突觸權重平衡

檢查 `transmitter_signs` 的邏輯,確認 L1(興奮性,cholinergic)→ Mi1 的權重沒有被錯誤地標為抑制性。

**優點**:如果 bug 在 sign assignment,修正後訊號應該自然傳遞
**風險**:需要深入 `transmitters.py` 檢查

### 方案 C:直接刺激 T4/T5 上游(跳過 lamina)

不從 R1-R6 開始,直接給 Mi1/Tm3 一個時變的 drive 模擬它們收到運動訊號。這跳過了 lamina 的訊號傳遞問題,直接測試 Mi1/Tm3 → T4/T5 的迴路是否有方向選擇性。

**優點**:繞過傳遞問題,直接測核心問題
**風險**:偏離了「從光感受器開始」的生物學路徑

### 建議:先試方案 B(檢查 sign assignment)

如果 sign assignment 有 bug,修正後 Mi1/Tm3 應該能被 L1 的興奮性輸入驅動。這是最可能是根本問題的地方。

---

## 下一步

1. **檢查 `transmitters.py`** — 確認 L1(cholinergic)的 sign 是否正確標為 +1
2. **檢查 L1 → Mi1/Tm3 的實際權重** — 確認正權重確實存在且強度足夠
3. **如果 sign 正確,試方案 A** — 給 medulla 細胞小 tonic drive
4. **重新跑 Day 6 預檢** — 看修正後 T4/T5 是否有 spike

---

## 已建立的基礎設施

雖然預檢 NO-GO,但所有基礎設施都已就緒:

- `malecns-topology-prior/` repo 結構完整
- MaleCNS 資料下載 + SHA-256 驗證通過
- 連接體 import + graph.npz 編譯成功
- C++ LIF 核心編譯成功
- T4/T5 細胞識別完成(6,861 T4 + 6,720 T5 = 13,581)
- Hex grid drifting grating 生成器完成
- DSI 計算 + 分析管線完成
- 8 方向 × 2 秒實驗跑完(50 秒 wall time)

修正動力學參數後,重新跑只需 1 分鐘。預檢的目的就是發現這個問題——現在發現了,成本很低。
