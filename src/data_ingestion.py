""" """

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

CWA_BASE = "https://opendata.cwa.gov.tw/api/v1/rest/datastore"
DEMO_TOKEN = "rdec-key-123-45678-011121314"
TOKEN = os.getenv("CWA_API_TOKEN", DEMO_TOKEN)

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json"})


def _get(endpoint: str, params: dict[str, Any] | None = None) -> dict:
    """ """
    url = f"{CWA_BASE}/{endpoint}"
    default_params = {"Authorization": TOKEN, "format": "JSON"}
    if params:
        default_params.update(params)

    try:
        resp = SESSION.get(url, params=default_params, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        logger.error("HTTP error %s | %s", exc.response.status_code, url)
        raise
    except requests.exceptions.ConnectionError:
        logger.error("Failed to connect to CWA API: %s", url)
        raise

    data = resp.json()
    if data.get("success") != "true":
        raise ValueError(f"CWA API returned failure: {data}")
    return data


def fetch_rainfall_obs(limit: int = 1000) -> pd.DataFrame:
    """ """
    logger.info("Fetching rainfall station observation data (O-A0002-001)...")
    raw = _get("O-A0002-001", {"limit": limit})
    stations: list[dict] = raw["records"]["Station"]

    rows = []
    for s in stations:
        geo = s.get("GeoInfo", {})
        coords = geo.get("Coordinates", [])
        wgs84 = next((c for c in coords if c["CoordinateName"] == "WGS84"), coords[0] if coords else {})

        rain = s.get("RainfallElement", {})

        def _prec(key: str) -> float | None:
            val = rain.get(key, {}).get("Precipitation")
            try:
                return float(val) if val not in (None, "-") else None
            except (ValueError, TypeError):
                return None

        rows.append(
            {
                "station_id": s.get("StationId"),
                "station_name": s.get("StationName"),
                "obs_time": s.get("ObsTime", {}).get("DateTime"),
                "lat": float(wgs84.get("StationLatitude", "nan") or "nan"),
                "lon": float(wgs84.get("StationLongitude", "nan") or "nan"),
                "altitude": float(geo.get("StationAltitude", "nan") or "nan"),
                "county": geo.get("CountyName"),
                "town": geo.get("TownName"),
                "precip_now": _prec("Now"),
                "precip_10min": _prec("Past10Min"),
                "precip_1hr": _prec("Past1hr"),
                "precip_3hr": _prec("Past3hr"),
                "precip_6hr": _prec("Past6Hr"),
                "precip_12hr": _prec("Past12hr"),
                "precip_24hr": _prec("Past24hr"),
                "precip_2days": _prec("Past2days"),
                "precip_3days": _prec("Past3days"),
            }
        )

    df = pd.DataFrame(rows)
    df["obs_time"] = pd.to_datetime(df["obs_time"], utc=True)
    logger.info("Retrieved %d rainfall observations", len(df))
    return df


def fetch_county_forecast() -> pd.DataFrame:
    """ """
    logger.info("Fetching county weather forecast (F-C0032-001)...")
    raw = _get("F-C0032-001")
    locations: list[dict] = raw["records"]["location"]

    rows = []
    for loc in locations:
        county = loc["locationName"]
        for elem in loc["weatherElement"]:
            if elem["elementName"] != "PoP":
                continue
            for t in elem["time"]:
                try:
                    pop = int(t["parameter"]["parameterName"])
                except (ValueError, KeyError):
                    pop = None
                rows.append(
                    {
                        "county": county,
                        "start_time": t.get("startTime"),
                        "end_time": t.get("endTime"),
                        "pop_pct": pop,
                    }
                )

    df = pd.DataFrame(rows)
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])
    logger.info("Retrieved %d forecast records (%d counties)", len(df), df["county"].nunique())
    return df


def save_raw(df: pd.DataFrame, name: str) -> Path:
    """ """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RAW_DIR / f"{name}_{ts}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Saved to %s", path)
    return path


def ingest() -> tuple[pd.DataFrame, pd.DataFrame]:
    """ """
    rainfall_df = fetch_rainfall_obs()
    forecast_df = fetch_county_forecast()

    save_raw(rainfall_df, "rainfall_obs")
    save_raw(forecast_df, "county_forecast")

    return rainfall_df, forecast_df


if __name__ == "__main__":
    r_df, f_df = ingest()

    print("\n=== Rainfall Observations (Top 3) ===")
    print(r_df.head(3).to_string(index=False))
    print(f"\nColumns: {list(r_df.columns)}")
    print(f"Total stations: {len(r_df)}")
    print(f"Valid rainfall (precip_24hr not null): {r_df['precip_24hr'].notna().sum()}")

    print("\n=== County Forecast (Top 3) ===")
    print(f_df.head(3).to_string(index=False))
    print(f"\nCounties: {f_df['county'].nunique()}")
