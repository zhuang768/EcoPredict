<h1 align="center">🌊 EcoPredict</h1>
<p align="center"><b>社區級洪災／土石流即時風險預測與告警系統</b></p>
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi" />
  <img src="https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikitlearn" />
  <img src="https://img.shields.io/badge/Leaflet.js-1.9-199900?logo=leaflet" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>
<p align="center">
  <a href="https://ecopredict.pages.dev">🌐 Live Demo</a> ·
  <a href="https://ecopredict-api.onrender.com/healthz">🔌 API Status</a> ·
  <a href="docs/analysis_report.md">📄 Analysis Report</a>
</p>

---

## 問題陳述

**台灣每年颱風與豪雨季節，獨居長者、行動不便者是最脆弱的受災群體。**

傳統預警系統以縣市為單位發佈，精度不足以讓第一線社工判斷哪個社區、哪些人需要「立刻」聯絡。社工師在颱風夜面對數百筆關懷名單，缺乏量化依據決定優先順序，往往在事後才知道哪個山區部落已經道路中斷。

**EcoPredict 做三件事：**
1. 從中央氣象署即時抓取 1,000 個雨量觀測站資料與 196 鄉鎮降雨預報
2. 用隨機森林模型預測未來 0–48 小時的社區級淹水 / 土石流風險機率
3. 自動比對「弱勢關懷名單」，產生以「高風險 → 獨居/行動不便者優先」排序的告警清單

---

## 資料集

| 資料集 | 來源 | 更新頻率 | 說明 |
|--------|------|----------|------|
| 雨量觀測站 | [CWA O-A0002-001](https://opendata.cwa.gov.tw/dataset/all/O-A0002-001) | 每 10 分鐘 | 全台 1,000 站，含 1h/3h/6h/12h/24h/2天/3天累積雨量 |
| 鄉鎮天氣預報 | [CWA F-D0047-XXX](https://opendata.cwa.gov.tw/dataset/forecast) | 每 6 小時 | 21 個縣市 endpoint，196 鄉鎮 × 未來 4 天 × 每 3hr 降雨機率 |
| 颱風警報 | [CWA W-C0034-001](https://opendata.cwa.gov.tw/api/v1/rest/datastore/W-C0034-001) | 即時 | CAP 格式颱風警報，判斷是否為颱風警戒期間 |

> **API 申請**：前往 [opendata.cwa.gov.tw](https://opendata.cwa.gov.tw) 註冊帳號，即可取得免費 API 授權碼。申請後填入 `.env` 的 `CWA_API_TOKEN`。

---

## 系統架構

```
┌─────────────────────────────────────────────────────────────────┐
│                        資料擷取層                                │
│  CWA API ──┬── O-A0002-001（雨量觀測）─▶ data/raw/rainfall_*.csv│
│            ├── F-D0047-XXX（鄉鎮預報）─▶ data/raw/forecast_*.csv│
│            └── W-C0034-001（颱風警報）─▶ is_typhoon_period flag │
└──────────────────────────┬──────────────────────────────────────┘
                           │ data_fetcher.py
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                       特徵工程層                                 │
│  feature_engineering.py                                          │
│  · rain_1h / 3h / 6h / 12h / 24h / 2days / 3days               │
│  · rain_intensity_max · forecast_pop_avg_24h/48h                │
│  · is_typhoon_period · altitude                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                        模型推論層                                │
│  model_train.py — 地理切分（北部訓練 / 南部測試）               │
│  LogisticReg AUC=0.975 | DecisionTree AUC=0.750                 │
│  RandomForest AUC=0.997 F1=0.667  ← BEST                        │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                       風險分級層                                 │
│  [0.00–0.20) 低風險 🟢  [0.20–0.50) 中風險 🟡                   │
│  [0.50–0.75) 高風險 🟠  [0.75–1.00] 極高風險 🔴                  │
└────────────────┬────────────────────────┬───────────────────────┘
                 ▼                        ▼
     告警引擎                     視覺化 Dashboard
  alert_dispatcher.py          FastAPI + Leaflet.js
  vulnerable_registry          地圖標記 + 彈窗 + 側欄
  → alert_report.json
```

---

## 快速開始

```bash
git clone https://github.com/YOUR_USERNAME/EcoPredict.git
cd EcoPredict
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # 填入 CWA_API_TOKEN

# 完整 pipeline
python src/data_fetcher.py
python src/feature_engineering.py
python src/model_train.py
python src/backtest.py
python src/alert_dispatcher.py

# 啟動 Dashboard
uvicorn api.api_server:app --port 8000        # 後端
cd frontend && python3 -m http.server 8080    # 前端
# 開啟 http://127.0.0.1:8080
```

---

## 模型評估結果

| 模型 | Precision | Recall | F1 | AUC-ROC |
|------|-----------|--------|----|---------|
| Logistic Regression | 0.158 | 0.750 | 0.261 | 0.975 |
| Decision Tree | 1.000 | 0.500 | 0.667 | 0.750 |
| **Random Forest ✓** | **1.000** | **0.500** | **0.667** | **0.997** |

**回測（颱風白海豚 2026-08-09）**：Precision = **1.0** ／ AUC-ROC = **0.947**
> 所有預測為「高風險」的站點，在颱風後 3 天累積雨量均確認 ≥ 100mm（**零假陽性**）。

---

## 部署架構

| 層 | 平台 | URL |
|----|------|-----|
| 前端 Dashboard | Cloudflare Pages | `https://ecopredict.pages.dev` |
| 後端 API | Render Free Tier | `https://ecopredict-api.onrender.com` |

詳細步驟 → [DEPLOY.md](DEPLOY.md)

---

## 技術限制

| 限制 | 說明 | 改善方向 |
|------|------|---------|
| 標籤為規則式合成 | 無 NCDR 歷史 API，以 NCDR 閾值規則取代 | 替換為真實警戒歷史紀錄 |
| 單快照訓練資料 | 1,000 站單一時間點，以地理切分取代時序 | 持續蒐集後改為時序訓練 |
| Recall = 0.50 | 模型保守，無假陽性但有遺漏 | 積累更多時序資料後改善 |
| 關懷名單為模擬 | 所有個資均為虛構，僅供 Demo | 正式串接民政系統 |

---

## License

[MIT License](LICENSE) © 2026 EcoPredict Team — ReverieHacks 2026 Datathon
