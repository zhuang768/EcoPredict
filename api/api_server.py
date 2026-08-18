from __future__ import annotations
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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
    "rain_1h",
    "rain_24h",
    "rain_3days",
    "forecast_pop_avg_24h",
    "forecast_pop_avg_48h",
]

class AppState:
    features_df:  pd.DataFrame | None = None
    forecast_df:  pd.DataFrame | None = None
    registry:     VulnerableRegistry | None = None
    geojson_cache: dict | None = None

_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_all()
    yield

def _load_all() -> None:
    feat_path = PROCESSED_DIR / "features.csv"
    if not feat_path.exists():
        raise RuntimeError("features.csv not found. Run feature_engineering.py")
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

    fc_path = RAW_DIR / "county_forecast.csv"
    if fc_path.exists():
        fc = pd.read_csv(fc_path)
        fc["fcst_time"] = pd.to_datetime(fc["fcst_time"], utc=True)
        _state.forecast_df = fc
    
    try:
        _state.registry = VulnerableRegistry()
    except FileNotFoundError:
        _state.registry = None

    print(f"[EcoPredict Global] Loaded: {len(df)} stations, Forecast {'OK' if _state.forecast_df is not None else 'N/A'}, Registry {'OK' if _state.registry else 'N/A'}")

app = FastAPI(
    title="EcoPredict Global API",
    description="Global Flood & Debris Flow Risk Prediction System",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,     
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=False,
)

def _get_station_forecast(county: str) -> list[dict]:
    if _state.forecast_df is None:
        return []
    fc = _state.forecast_df
    t_fc = fc[fc["county"] == county].sort_values("fcst_time").head(48)
    return [
        {
            "start": row["fcst_time"].isoformat(),
            "pop":   int(row["pop_percent"]) if pd.notna(row["pop_percent"]) else 0,
            "label": row["fcst_time"].strftime("%m/%d %H:%M"),
        }
        for _, row in t_fc.iterrows()
    ]

@app.get("/healthz")
async def health_check() -> dict:
    return {
        "status": "ok",
        "stations_loaded": len(_state.features_df) if _state.features_df is not None else 0,
        "cors_origins": CORS_ORIGINS,
    }

@app.get("/api/risk-map")
async def get_risk_map() -> dict:
    df = _state.features_df
    if df is None:
        raise HTTPException(503, "Data not loaded")

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
                "county":       str(row.get("county", "")),
                "town":         str(row.get("town", "")),
                "risk_level":   str(row.get("risk_level", "LOW")),
                "risk_label":   str(row.get("risk_label", "Low Risk")),
                "risk_score":   round(float(row.get("risk_score", 0)), 4),
                "rain_1h":      float(row.get("rain_1h", 0)),
                "rain_24h":     float(row.get("rain_24h", 0)),
                "rain_3days":   float(row.get("rain_3days", 0)),
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
    df = _state.features_df
    if df is None:
        raise HTTPException(503, "Data not loaded")

    mask = df["station_id"].astype(str) == str(station_id)
    if not mask.any():
        raise HTTPException(404, f"Station {station_id} not found")

    row = df[mask].iloc[0]
    lat = float(row.get("lat", 0))
    lon = float(row.get("lon", 0))
    county = str(row.get("county", ""))

    rainfall_history = [
        {"label": "Current", "rain_mm": float(row.get("rain_1h", 0))},
        {"label": "-24h",  "rain_mm": float(row.get("rain_24h", 0))},
        {"label": "-72h",  "rain_mm": float(row.get("rain_3days", 0))},
    ]

    forecast = _get_station_forecast(county)

    vulnerable_count = 0
    priority_count = 0
    if _state.registry and county:
        persons = _state.registry.query(county=county)
        vulnerable_count = len(persons)
        priority_count = sum(1 for p in persons if p.is_priority)

    return {
        "station_id":    str(row.get("station_id", "")),
        "station_name":  str(row.get("station_name", "")),
        "county":        county,
        "town":          str(row.get("town", "")),
        "lat":           lat,
        "lon":           lon,
        "risk": {
            "level":     str(row.get("risk_level", "LOW")),
            "label":     str(row.get("risk_label", "Low Risk")),
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
    df = _state.features_df
    if df is None:
        raise HTTPException(503, "Data not loaded")

    alerted = df[df["risk_level"].apply(is_alert_threshold)].copy()
    alerted = alerted.sort_values("risk_score", ascending=False)

    alerts: list[dict] = []
    for _, row in alerted.iterrows():
        alerts.append({
            "station_id":    str(row.get("station_id", "")),
            "station_name":  str(row.get("station_name", "")),
            "county":        str(row.get("county", "")),
            "town":          str(row.get("town", "")),
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
