"""
api_server.py
=============
EcoPredict FastAPI 後端（純 API，不服務前端頁面）

Endpoints:
  GET /healthz                  → 健康檢查（Render 宣告上線用）
  GET /api/risk-map             → GeoJSON，含所有測站風險等級與座標
  GET /api/community/{id}       → 單一測站詳情（雨量歷史 + 48h 預報折線 + 關懷人數）
  GET /api/alerts               → 高風險以上的告警清單

啟動方式（本地開發）：
  cd EcoPredict
  .venv/bin/uvicorn api.api_server:app --host 0.0.0.0 --port 8000 --reload

啟動方式（Render 生產）：
  gunicorn -w 2 -k uvicorn.workers.UvicornWorker api.api_server:app
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# 確保 src/ 模組可被 import
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from risk_engine import classify_dataframe, is_alert_threshold  # noqa: E402
from vulnerable_registry import VulnerableRegistry              # noqa: E402

BASE          = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE / "data" / "processed"
RAW_DIR       = BASE / "data" / "raw"
MODELS_DIR    = BASE / "models"

# 允許跨域的 Origins（從環境變數讀取，方便不同環境不同設定）
# 格式：CSV，例如 https://ecopredict.pages.dev,http://localhost:8080
_CORS_ORIGINS_ENV = os.getenv("CORS_ALLOW_ORIGINS", "*")
CORS_ORIGINS: list[str] = (
    ["*"] if _CORS_ORIGINS_ENV == "*"
    else [o.strip() for o in _CORS_ORIGINS_ENV.split(",") if o.strip()]
)

FEATURE_COLS = [
    "rain_1h", "rain_3h", "rain_6h", "rain_12h", "rain_24h",
    "rain_2days", "rain_3days", "rain_intensity_max",
    "forecast_pop_avg_24h", "forecast_pop_avg_48h",
    "is_typhoon_period", "altitude",
]

# ── 應用程式狀態 ─────────────────────────────────────────────────────────────

import pypinyin

COUNTY_MAP = {
    "臺北市": "Taipei City", "台北市": "Taipei City",
    "新北市": "New Taipei City",
    "基隆市": "Keelung City",
    "桃園市": "Taoyuan City",
    "新竹縣": "Hsinchu County", "新竹市": "Hsinchu City",
    "苗栗縣": "Miaoli County",
    "臺中市": "Taichung City", "台中市": "Taichung City",
    "彰化縣": "Changhua County",
    "南投縣": "Nantou County",
    "雲林縣": "Yunlin County",
    "嘉義縣": "Chiayi County", "嘉義市": "Chiayi City",
    "臺南市": "Tainan City", "台南市": "Tainan City",
    "高雄市": "Kaohsiung City",
    "屏東縣": "Pingtung County",
    "宜蘭縣": "Yilan County",
    "花蓮縣": "Hualien County",
    "臺東縣": "Taitung County", "台東縣": "Taitung County",
    "澎湖縣": "Penghu County",
    "金門縣": "Kinmen County",
    "連江縣": "Lienchiang County"
}

def to_en(text: str) -> str:
    if not text: return ""
    text = str(text)
    if text in COUNTY_MAP: return COUNTY_MAP[text]
    
    # Handle suffixes
    orig = text
    if text.endswith("區"): text = text[:-1] + " Dist."
    elif text.endswith("鄉"): text = text[:-1] + " Township"
    elif text.endswith("鎮"): text = text[:-1] + " Township"
    elif text.endswith("市") and text not in COUNTY_MAP: text = text[:-1] + " City"
    
    # Translate the chinese characters using pypinyin
    # Only translate if there are chinese characters (simplified heuristic)
    if any('\u4e00' <= c <= '\u9fff' for c in text):
        parts = []
        for word in pypinyin.pinyin(text, style=pypinyin.NORMAL):
            parts.append(word[0].capitalize())
        return "".join(parts).replace("Dist.", " Dist.").replace("Township", " Township").replace("City", " City").replace("  ", " ").strip()
    return orig

class AppState:
    features_df:  pd.DataFrame | None = None
    forecast_df:  pd.DataFrame | None = None
    registry:     VulnerableRegistry | None = None
    geojson_cache: dict | None = None


_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """啟動時載入資料與模型。"""
    _load_all()
    yield


def _load_all() -> None:
    """載入特徵表、模型、預報、關懷名單。"""
    # 特徵表 + 模型預測
    feat_path = PROCESSED_DIR / "features.csv"
    if not feat_path.exists():
        raise RuntimeError("features.csv 不存在，請先執行 feature_engineering.py")
    df = pd.read_csv(feat_path)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)

    model_path = MODELS_DIR / "best_model.pkl"
    if model_path.exists():
        bundle = joblib.load(model_path)
        model = bundle["model"]
        df["risk_score"] = model.predict_proba(df[FEATURE_COLS])[:, 1]
    else:
        df["risk_score"] = 0.0

    df = classify_dataframe(df, score_col="risk_score")
    _state.features_df = df

    # 預報資料（最新的 CSV）
    fc_files = sorted(RAW_DIR.glob("township_forecast_*.csv"))
    if fc_files:
        fc = pd.read_csv(fc_files[-1])
        fc["start_time"] = pd.to_datetime(fc["start_time"], utc=True)
        fc["end_time"]   = pd.to_datetime(fc["end_time"],   utc=True)
        _state.forecast_df = fc

    # 關懷名單
    try:
        _state.registry = VulnerableRegistry()
    except FileNotFoundError:
        _state.registry = None

    print(f"[EcoPredict] 載入完成：{len(df)} 站，預報 {len(fc_files)} 份，"
          f"關懷名單 {'OK' if _state.registry else 'N/A'}")


# ── FastAPI 初始化 ────────────────────────────────────────────────────────────
app = FastAPI(
    title="EcoPredict API",
    description="社區級洪災/土石流風險預測系統",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,     # 由環境變數 CORS_ALLOW_ORIGINS 控制
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=False,
)

# 移除前端静態檔案掛載（前後端分離後不需要）


# ── 工具函式 ──────────────────────────────────────────────────────────────────
def _haversine_km(lat1, lon1, lat2: float, lon2: float):
    """向量化 Haversine 距離（公里）。lat1/lon1 可為 numpy array。"""
    import numpy as np
    R = 6371.0
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    a = (np.sin(np.radians((lat2 - lat1) / 2)) ** 2 +
         np.cos(lat1r) * np.cos(lat2r) *
         np.sin(np.radians((lon2 - lon1) / 2)) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))


def _nearest_township_forecast(lat: float, lon: float) -> list[dict]:
    """回傳最近鄉鎮的未來 PoP 時間序列（最多 16 個 3hr 時段）。"""
    if _state.forecast_df is None:
        return []
    import numpy as np
    fc = _state.forecast_df
    towns = fc[["township", "lat", "lon"]].drop_duplicates("township")
    dists = _haversine_km(
        towns["lat"].values, towns["lon"].values, lat, lon
    )
    nearest = towns.iloc[int(np.argmin(dists))]["township"]
    t_fc = fc[fc["township"] == nearest].sort_values("start_time").head(16)
    return [
        {
            "start": row["start_time"].isoformat(),
            "end":   row["end_time"].isoformat(),
            "pop":   int(row["pop_3h"]) if pd.notna(row["pop_3h"]) else 0,
            "label": row["start_time"].strftime("%m/%d %H:%M"),
        }
        for _, row in t_fc.iterrows()
    ]


# ── API Endpoints ─────────────────────────────────────────────────────
@app.get("/healthz")
async def health_check() -> dict:
    """Render 健康檢查 endpoint（常時回傳 200）。"""
    return {
        "status": "ok",
        "stations_loaded": len(_state.features_df) if _state.features_df is not None else 0,
        "cors_origins": CORS_ORIGINS,
    }


@app.get("/api/risk-map")
async def get_risk_map() -> dict:
    """
    回傳所有測站的 GeoJSON FeatureCollection。
    每個 Feature 包含：座標、測站資訊、風險等級。
    """
    df = _state.features_df
    if df is None:
        raise HTTPException(503, "資料尚未載入")

    features = []
    for _, row in df.iterrows():
        lat = row.get("lat")
        lon = row.get("lon")
        if pd.isna(lat) or pd.isna(lon):
            continue
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(lon), float(lat)],
            },
            "properties": {
                "station_id":   str(row.get("station_id", "")),
                "station_name": str(row.get("station_name", "")),
                "county":       to_en(str(row.get("county", ""))),
                "town":         to_en(str(row.get("town", ""))),
                "risk_level":   str(row.get("risk_level", "LOW")),
                "risk_label":   str(row.get("risk_label", "低風險")),
                "risk_score":   round(float(row.get("risk_score", 0)), 4),
                "rain_1h":      float(row.get("rain_1h", 0)),
                "rain_24h":     float(row.get("rain_24h", 0)),
                "rain_3days":   float(row.get("rain_3days", 0)),
                "altitude":     float(row.get("altitude", 0)) if pd.notna(row.get("altitude")) else 0,
                "forecast_pop_24h": float(row.get("forecast_pop_avg_24h", 0)) if pd.notna(row.get("forecast_pop_avg_24h")) else 0,
            },
        })

    return {
        "type": "FeatureCollection",
        "features": features,
        "meta": {
            "total_stations": len(features),
            "risk_counts": {
                lvl: int((df["risk_level"] == lvl).sum())
                for lvl in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
            },
        },
    }


@app.get("/api/community/{station_id}")
async def get_community(station_id: str) -> dict:
    """
    回傳單一測站詳細資訊：
      - 雨量歷史（rolling windows 轉成時間序列）
      - 未來 48h 降雨機率預報
      - 同縣市關懷人數
    """
    df = _state.features_df
    if df is None:
        raise HTTPException(503, "資料尚未載入")

    mask = df["station_id"].astype(str) == str(station_id)
    if not mask.any():
        raise HTTPException(404, f"找不到測站 {station_id}")

    row = df[mask].iloc[0]
    lat = float(row.get("lat", 0))
    lon = float(row.get("lon", 0))

    # 雨量歷史：把 rolling window 轉成「T-Xh ago」時間序列
    rainfall_history = [
        {"label": "現在",   "rain_mm": float(row.get("rain_1h",    0))},
        {"label": "-1h",   "rain_mm": float(row.get("rain_1h",    0))},
        {"label": "-3h",   "rain_mm": float(row.get("rain_3h",    0))},
        {"label": "-6h",   "rain_mm": float(row.get("rain_6h",    0))},
        {"label": "-12h",  "rain_mm": float(row.get("rain_12h",   0))},
        {"label": "-24h",  "rain_mm": float(row.get("rain_24h",   0))},
        {"label": "-48h",  "rain_mm": float(row.get("rain_2days", 0))},
        {"label": "-72h",  "rain_mm": float(row.get("rain_3days", 0))},
    ]

    # 未來 48h 降雨機率預報
    forecast = _nearest_township_forecast(lat, lon)

    # 關懷名單統計
    county = to_en(str(row.get("county", "")))
    vulnerable_count = 0
    priority_count = 0
    if _state.registry and county:
        persons = _state.registry.query(county=county)
        vulnerable_count = len(persons)
        priority_count = sum(1 for p in persons if p.is_priority)

    return {
        "station_id":    str(row.get("station_id", "")),
        "station_name":  to_en(str(row.get("station_name", ""))),
        "county":        county,
        "town":          to_en(str(row.get("town", ""))),
        "lat":           lat,
        "lon":           lon,
        "altitude_m":    float(row.get("altitude", 0)) if pd.notna(row.get("altitude")) else 0,
        "risk": {
            "level":     str(row.get("risk_level", "LOW")),
            "label":     str(row.get("risk_label", "低風險")),
            "score":     round(float(row.get("risk_score", 0)), 4),
        },
        "rainfall_history": rainfall_history,
        "forecast_48h":     forecast,
        "vulnerable": {
            "county_total":    vulnerable_count,
            "priority_count":  priority_count,
        },
    }


@app.get("/api/alerts")
async def get_alerts() -> dict:
    """
    回傳所有達到高風險以上的告警，依風險分數降序排列。
    """
    df = _state.features_df
    if df is None:
        raise HTTPException(503, "資料尚未載入")

    alerted = df[df["risk_level"].apply(is_alert_threshold)].copy()
    alerted = alerted.sort_values("risk_score", ascending=False)

    alerts: list[dict] = []
    for _, row in alerted.iterrows():
        alerts.append({
            "station_id":    str(row.get("station_id", "")),
            "station_name":  to_en(str(row.get("station_name", ""))),
            "county":        to_en(str(row.get("county", ""))),
            "town":          to_en(str(row.get("town", ""))),
            "lat":           float(row.get("lat", 0)),
            "lon":           float(row.get("lon", 0)),
            "risk_level":    str(row.get("risk_level", "")),
            "risk_label":    str(row.get("risk_label", "")),
            "risk_score":    round(float(row.get("risk_score", 0)), 4),
            "rain_24h":      float(row.get("rain_24h", 0)),
            "rain_3days":    float(row.get("rain_3days", 0)),
            "forecast_pop_24h": float(row.get("forecast_pop_avg_24h", 0)) if pd.notna(row.get("forecast_pop_avg_24h")) else 0,
        })

    return {
        "total":  len(alerts),
        "alerts": alerts,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
