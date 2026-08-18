import logging
import os
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _load_csv(filename: str) -> pd.DataFrame:
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Cannot find data/raw/{filename}. Run data_fetcher.py first.")
    logger.info("Loading %s", path.name)
    return pd.read_csv(path)

def _make_label(row: pd.Series) -> int:
    # A simple deterministic rule to generate labels for our ML model to train on
    # In reality this would be historical disaster labels.
    r24 = row.get("rain_24hr", 0) or 0
    r3d = row.get("rain_3days", 0) or 0
    r1  = row.get("rain_1h", 0) or 0
    pop48 = row.get("forecast_pop_avg_48h", 0) or 0

    if r24 >= 150: return 1
    if r3d >= 200: return 1
    if r1 >= 30: return 1
    # Adding a slight synthetic threshold to get some positive labels globally
    if r24 >= 50 and pop48 >= 60: return 1
    return 0

def build_features(save: bool = True) -> pd.DataFrame:
    obs_df = _load_csv("current_obs.csv")
    forecast_df = _load_csv("county_forecast.csv")

    obs_df["obs_time"] = pd.to_datetime(obs_df["obs_time"], utc=True)
    forecast_df["fcst_time"] = pd.to_datetime(forecast_df["fcst_time"], utc=True)

    now_utc = obs_df["obs_time"].max()
    logger.info("Observation time baseline (UTC): %s", now_utc)

    # Calculate forecast PoP averages
    future_fc = forecast_df[forecast_df["fcst_time"] >= now_utc]
    
    # Group by county to get average PoP
    h24 = future_fc[future_fc["fcst_time"] < now_utc + pd.Timedelta(hours=24)]
    h48 = future_fc[future_fc["fcst_time"] < now_utc + pd.Timedelta(hours=48)]

    pop24_map = h24.groupby("county")["pop_percent"].mean().to_dict()
    pop48_map = h48.groupby("county")["pop_percent"].mean().to_dict()

    rows = []
    for _, station in obs_df.iterrows():
        lat = station.get("lat")
        lon = station.get("lon")
        if pd.isna(lat) or pd.isna(lon):
            continue

        c_name = station["county"]
        row = {
            "station_id":   station["station_id"],
            "station_name": station["station_name"],
            "county":       c_name,
            "town":         station.get("town"),
            "lat":          lat,
            "lon":          lon,
            "obs_time":     now_utc,

            "rain_1h":            station.get("rain_1h", 0),
            "rain_24h":           station.get("rain_24hr", 0),
            "rain_3days":         station.get("rain_3days", 0),

            "forecast_pop_avg_24h": round(pop24_map.get(c_name, 0.0), 1),
            "forecast_pop_avg_48h": round(pop48_map.get(c_name, 0.0), 1),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.fillna(0.0, inplace=True)
    df["label"] = df.apply(_make_label, axis=1)

    pos = df["label"].sum()
    neg = len(df) - pos
    logger.info(
        "Feature table complete: %d rows (Positive label=1: %d / %.1f%%, Negative: %d)",
        len(df), pos, 100 * pos / len(df), neg,
    )

    if save:
        out = PROCESSED_DIR / "features.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        logger.info("Saved -> %s", out)

    return df

if __name__ == "__main__":
    df = build_features(save=True)
    print("\n[Feature Table Summary]")
    print(df[["station_name", "rain_24h", "rain_3days", "forecast_pop_avg_48h", "label"]].head(10).to_string(index=False))
    print(f"\nTotal stations: {len(df)}")
    print(f"Positive (label=1): {df['label'].sum()} ({100*df['label'].mean():.1f}%)")
