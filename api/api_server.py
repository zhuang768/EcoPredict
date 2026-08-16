""" """

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


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from risk_engine import classify_dataframe, is_alert_threshold  
from vulnerable_registry import VulnerableRegistry              

BASE          = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE / "data" / "processed"
RAW_DIR       = BASE / "data" / "raw"
MODELS_DIR    = BASE / "models"



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



import pypinyin

COUNTY_MAP = {
    "\u81fa\u5317\u5e02": "Taipei City", "\u53f0\u5317\u5e02": "Taipei City", "\u65b0\u5317\u5e02": "New Taipei City",
    "\u57fa\u9686\u5e02": "Keelung City", "\u6843\u5712\u5e02": "Taoyuan City", "\u65b0\u7af9\u7e23": "Hsinchu County", "\u65b0\u7af9\u5e02": "Hsinchu City",
    "\u82d7\u6817\u7e23": "Miaoli County", "\u81fa\u4e2d\u5e02": "Taichung City", "\u53f0\u4e2d\u5e02": "Taichung City",
    "\u5f70\u5316\u7e23": "Changhua County", "\u5357\u6295\u7e23": "Nantou County", "\u96f2\u6797\u7e23": "Yunlin County",
    "\u5609\u7fa9\u7e23": "Chiayi County", "\u5609\u7fa9\u5e02": "Chiayi City", "\u81fa\u5357\u5e02": "Tainan City", "\u53f0\u5357\u5e02": "Tainan City",
    "\u9ad8\u96c4\u5e02": "Kaohsiung City", "\u5c4f\u6771\u7e23": "Pingtung County", "\u5b9c\u862d\u7e23": "Yilan County",
    "\u82b1\u84ee\u7e23": "Hualien County", "\u81fa\u6771\u7e23": "Taitung County", "\u53f0\u6771\u7e23": "Taitung County",
    "\u6f8e\u6e56\u7e23": "Penghu County", "\u91d1\u9580\u7e23": "Kinmen County", "\u9023\u6c5f\u7e23": "Lienchiang County"
}

STATION_MAP = {
    "\u81fa\u7063\u5927\u5b78": "NTU (National Taiwan Univ.)",
    "\u53f0\u5927": "NTU",
    "\u6771\u6e56\u570b\u5c0f": "Donghu Elem. School",
    "\u6c11\u751f\u570b\u4e2d": "Minsheng Jr. High",
    "\u535a\u5609\u570b\u5c0f": "Bojia Elem. School",
    "\u6587\u5c71": "Wenshan",
    "\u6953\u6e2f": "Fenggang",
    "\u4e5d\u4efd\u4e8c\u5c71": "Jiufenershan",
    "\u4e00N013K": "Sta. N013K",
    "\u4e09S010K": "Sta. S010K",
    "\u6c50\u6b62": "Xizhi",
    "\u4fe1\u7fa9": "Xinyi",
    "\u5927\u5b89": "Da'an",
    "\u5167\u6e56": "Neihu",
    "\u677e\u5c71": "Songshan"
}

def to_en(text: str) -> str:
    if not text: return ""
    text = str(text).strip()
    if text in COUNTY_MAP: return COUNTY_MAP[text]
    if text in STATION_MAP: return STATION_MAP[text]
    
    # Handle specific replacements
    for k, v in STATION_MAP.items():
        if k in text:
            text = text.replace(k, v)
            
    # Handle generic suffixes
    replacements = {
        "\u570b\u5c0f": " Elem.", "\u570b\u4e2d": " Jr. High", "\u9ad8\u4e2d": " High School", 
        "\u5927\u5b78": " Univ.", "\u5927\u6a4b": " Bridge", "\u6a4b": " Bridge", 
        "\u8eca\u7ad9": " Station", "\u5340": " Dist.", "\u9109": " Township", 
        "\u93ae": " Township", "\u5e02": " City", "\u7e23": " County"
    }
    for k, v in replacements.items():
        if text.endswith(k):
            text = text[:-len(k)] + v
            
    # Use pypinyin for remaining Chinese characters
    if any('\u4e00' <= c <= '\u9fff' for c in text):
        res = ""
        for word in pypinyin.pinyin(text, style=pypinyin.NORMAL):
            # Only capitalize the first letter of each pinyin word, but preserve English spaces/cases
            res += word[0].capitalize()
        # Clean up spacing around known english suffixes
        for v in replacements.values():
            res = res.replace(v.replace(" ", ""), v)
        return res.replace("  ", " ").strip()
    return text

class AppState:
    features_df:  pd.DataFrame | None = None
    forecast_df:  pd.DataFrame | None = None
    registry:     VulnerableRegistry | None = None
    geojson_cache: dict | None = None


_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """ """
    _load_all()
    yield


def _load_all() -> None:
    """ """
    
    feat_path = PROCESSED_DIR / "features.csv"
    if not feat_path.exists():
        raise RuntimeError("features.csv \u4e0d\u5b58\u5728，\u8acb\u5148\u57f7\u884c feature_engineering.py")
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

    
    fc_files = sorted(RAW_DIR.glob("township_forecast_*.csv"))
    if fc_files:
        fc = pd.read_csv(fc_files[-1])
        fc["start_time"] = pd.to_datetime(fc["start_time"], utc=True)
        fc["end_time"]   = pd.to_datetime(fc["end_time"],   utc=True)
        _state.forecast_df = fc

    
    try:
        _state.registry = VulnerableRegistry()
    except FileNotFoundError:
        _state.registry = None

    print(f"[EcoPredict] \u8f09\u5165\u5b8c\u6210：{len(df)} \u7ad9，\u9810\u5831 {len(fc_files)} \u4efd，"
          f"\u95dc\u61f7\u540d\u55ae {'OK' if _state.registry else 'N/A'}")



app = FastAPI(
    title="EcoPredict API",
    description="\u793e\u5340\u7d1a\u6d2a\u707d/\u571f\u77f3\u6d41\u98a8\u96aa\u9810\u6e2c\u7cfb\u7d71",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,     
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=False,
)





def _haversine_km(lat1, lon1, lat2: float, lon2: float):
    """ """
    import numpy as np
    R = 6371.0
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    a = (np.sin(np.radians((lat2 - lat1) / 2)) ** 2 +
         np.cos(lat1r) * np.cos(lat2r) *
         np.sin(np.radians((lon2 - lon1) / 2)) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))


def _nearest_township_forecast(lat: float, lon: float) -> list[dict]:
    """ """
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



@app.get("/healthz")
async def health_check() -> dict:
    """ """
    return {
        "status": "ok",
        "stations_loaded": len(_state.features_df) if _state.features_df is not None else 0,
        "cors_origins": CORS_ORIGINS,
    }


@app.get("/api/risk-map")
async def get_risk_map() -> dict:
    """ """
    df = _state.features_df
    if df is None:
        raise HTTPException(503, "\u8cc7\u6599\u5c1a\u672a\u8f09\u5165")

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
                "risk_label":   str(row.get("risk_label", "\u4f4e\u98a8\u96aa")),
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
    """ """
    df = _state.features_df
    if df is None:
        raise HTTPException(503, "\u8cc7\u6599\u5c1a\u672a\u8f09\u5165")

    mask = df["station_id"].astype(str) == str(station_id)
    if not mask.any():
        raise HTTPException(404, f"\u627e\u4e0d\u5230\u6e2c\u7ad9 {station_id}")

    row = df[mask].iloc[0]
    lat = float(row.get("lat", 0))
    lon = float(row.get("lon", 0))

    
    rainfall_history = [
        {"label": "\u73fe\u5728",   "rain_mm": float(row.get("rain_1h",    0))},
        {"label": "-1h",   "rain_mm": float(row.get("rain_1h",    0))},
        {"label": "-3h",   "rain_mm": float(row.get("rain_3h",    0))},
        {"label": "-6h",   "rain_mm": float(row.get("rain_6h",    0))},
        {"label": "-12h",  "rain_mm": float(row.get("rain_12h",   0))},
        {"label": "-24h",  "rain_mm": float(row.get("rain_24h",   0))},
        {"label": "-48h",  "rain_mm": float(row.get("rain_2days", 0))},
        {"label": "-72h",  "rain_mm": float(row.get("rain_3days", 0))},
    ]

    
    forecast = _nearest_township_forecast(lat, lon)

    
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
            "label":     str(row.get("risk_label", "\u4f4e\u98a8\u96aa")),
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
    """ """
    df = _state.features_df
    if df is None:
        raise HTTPException(503, "\u8cc7\u6599\u5c1a\u672a\u8f09\u5165")

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
