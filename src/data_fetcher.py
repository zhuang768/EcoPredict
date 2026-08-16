""" """

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_DEMO_TOKEN = "rdec-key-123-45678-011121314"
_BASE_URL = "https://opendata.cwa.gov.tw/api/v1/rest/datastore"
_RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
_RAW_DIR.mkdir(parents=True, exist_ok=True)

_COUNTY_FORECAST_ENDPOINTS: list[tuple[str, str]] = [
    ("F-D0047-001", "Yilan County"),
    ("F-D0047-003", "Taoyuan City"),
    ("F-D0047-005", "Hsinchu County"),
    ("F-D0047-007", "Miaoli County"),
    ("F-D0047-009", "Changhua County"),
    ("F-D0047-011", "Nantou County"),
    ("F-D0047-013", "Yunlin County"),
    ("F-D0047-015", "Chiayi County"),
    ("F-D0047-017", "Pingtung County"),
    ("F-D0047-019", "Taitung County"),
    ("F-D0047-021", "Hualien County"),
    ("F-D0047-023", "Penghu County"),
    ("F-D0047-029", "New Taipei City"),
    ("F-D0047-031", "Taichung City"),
    ("F-D0047-033", "Tainan City"),
    ("F-D0047-035", "Kaohsiung City"),
    ("F-D0047-059", "Lienchiang County"),
    ("F-D0047-061", "Taipei City"),
    ("F-D0047-063", "Hsinchu City"),
    ("F-D0047-065", "Chiayi City"),
    ("F-D0047-067", "Keelung City"),
]


def _build_session() -> requests.Session:
    """ """
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        max_retries=requests.adapters.Retry(
            total=3,
            backoff_factor=1.5,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET"],
        )
    )
    session.mount("https://", adapter)
    session.headers.update({"Accept": "application/json"})
    return session


_SESSION = _build_session()


def _get(
    endpoint: str,
    params: Optional[dict] = None,
    timeout: int = 30,
) -> dict:
    """ """
    token = os.getenv("CWA_API_TOKEN", _DEMO_TOKEN)
    url = f"{_BASE_URL}/{endpoint}"
    merged_params = {"Authorization": token, "format": "JSON"}
    if params:
        merged_params.update(params)

    try:
        resp = _SESSION.get(url, params=merged_params, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        logger.error("HTTP %s — %s", exc.response.status_code, url)
        raise
    except requests.exceptions.ConnectionError:
        logger.error("Failed to connect to CWA API: %s", url)
        raise
    except requests.exceptions.Timeout:
        logger.error("Request timeout (%ds): %s", timeout, url)
        raise

    payload = resp.json()
    if str(payload.get("success")).lower() != "true":
        raise ValueError(f"CWA API returned failure: {payload.get('message', payload)}")
    return payload


def _save_csv(df: pd.DataFrame, name: str) -> Path:
    """ """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = _RAW_DIR / f"{name}_{ts}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Saved %d rows -> %s", len(df), path)
    return path


def fetch_rainfall_realtime(save: bool = True) -> pd.DataFrame:
    """ """
    logger.info("Fetching real-time rainfall observation (O-A0002-001)...")

    try:
        raw = _get("O-A0002-001", {"limit": 1000})
    except Exception as exc:
        logger.error("fetch_rainfall_realtime failed: %s", exc)
        raise

    stations: list[dict] = raw["records"]["Station"]
    rows = []

    for s in stations:
        geo = s.get("GeoInfo", {})
        coords = geo.get("Coordinates", [])
        wgs84 = next(
            (c for c in coords if c.get("CoordinateName") == "WGS84"),
            coords[0] if coords else {},
        )

        rain = s.get("RainfallElement", {})

        def _p(key: str) -> Optional[float]:
            """ """
            val = rain.get(key, {}).get("Precipitation")
            if val is None or str(val).strip() in ("-", ""):
                return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None

        rows.append(
            {
                "station_id": s.get("StationId"),
                "station_name": s.get("StationName"),
                "obs_time": s.get("ObsTime", {}).get("DateTime"),
                "lat": _safe_float(wgs84.get("StationLatitude")),
                "lon": _safe_float(wgs84.get("StationLongitude")),
                "altitude": _safe_float(geo.get("StationAltitude")),
                "county": geo.get("CountyName"),
                "town": geo.get("TownName"),
                "rainfall_10min": _p("Past10Min"),
                "rainfall_1hr": _p("Past1hr"),
                "rainfall_3hr": _p("Past3hr"),
                "rainfall_6hr": _p("Past6Hr"),
                "rainfall_12hr": _p("Past12hr"),
                "rainfall_24hr": _p("Past24hr"),
                "rainfall_2days": _p("Past2days"),
                "rainfall_3days": _p("Past3days"),
            }
        )

    df = pd.DataFrame(rows)
    df["obs_time"] = pd.to_datetime(df["obs_time"], utc=True)

    logger.info(
        "Retrieved %d obs; valid rainfall_24hr: %d",
        len(df),
        df["rainfall_24hr"].notna().sum(),
    )

    if save:
        _save_csv(df, "rainfall_realtime")

    return df


def fetch_forecast(save: bool = True) -> pd.DataFrame:
    """ """
    logger.info("Fetching township precipitation probability forecast (%d endpoints)...", len(_COUNTY_FORECAST_ENDPOINTS))

    all_rows: list[dict] = []
    failed: list[str] = []

    for ep_code, county_hint in _COUNTY_FORECAST_ENDPOINTS:
        try:
            raw = _get(ep_code)
        except Exception as exc:
            logger.warning("  ✗ %s (%s) fetch failed: %s", ep_code, county_hint, exc)
            failed.append(ep_code)
            time.sleep(0.5)
            continue

        locations_arr = raw.get("records", {}).get("Locations", [])
        if not locations_arr:
            logger.warning("  ✗ %s - No Locations data", ep_code)
            failed.append(ep_code)
            continue

        outer = locations_arr[0]
        county_name = outer.get("LocationsName", county_hint)
        township_list: list[dict] = outer.get("Location", [])

        for township in township_list:
            t_name = township.get("LocationName")
            lat = _safe_float(township.get("Latitude"))
            lon = _safe_float(township.get("Longitude"))

            for elem in township.get("WeatherElement", []):
                if elem.get("ElementName") != "\u0031\u0032\u5c0f\u6642\u964d\u96e8\u6a5f\u7387":
                    continue
                for slot in elem.get("Time", []):
                    try:
                        pop_val = int(
                            slot["ElementValue"][0]["ProbabilityOfPrecipitation"]
                        )
                    except (KeyError, IndexError, ValueError, TypeError):
                        pop_val = None

                    all_rows.append(
                        {
                            "county": county_name,
                            "township": t_name,
                            "lat": lat,
                            "lon": lon,
                            "start_time": slot.get("StartTime"),
                            "end_time": slot.get("EndTime"),
                            "pop_3h": pop_val,
                        }
                    )

        logger.info("  ✓ %s - %s (%d townships)", ep_code, county_name, len(township_list))

    if not all_rows:
        raise RuntimeError("All forecast endpoints failed, cannot retrieve forecast data.")

    if failed:
        logger.warning("The following endpoints failed, data may be incomplete: %s", failed)

    df = pd.DataFrame(all_rows)
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    df["end_time"] = pd.to_datetime(df["end_time"], utc=True)

    logger.info(
        "Forecast data complete: %d rows, covers %d counties / %d townships",
        len(df),
        df["county"].nunique(),
        df["township"].nunique(),
    )

    if save:
        _save_csv(df, "township_forecast")

    return df


def _safe_float(val) -> Optional[float]:
    """ """
    if val is None or str(val).strip() in ("", "-"):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("EcoPredict - CWA Data Fetching")
    print("=" * 60)

    rainfall_df = fetch_rainfall_realtime(save=True)
    print("\n[Real-time Rainfall Observation]")
    print(f"  Total stations: {len(rainfall_df)}")
    print(f"  Valid rainfall_10min: {rainfall_df['rainfall_10min'].notna().sum()}")
    print(f"  Valid rainfall_24hr: {rainfall_df['rainfall_24hr'].notna().sum()}")
    print(rainfall_df[["station_id", "station_name", "county", "lat", "lon",
                        "rainfall_10min", "rainfall_1hr", "rainfall_24hr"]].head(5).to_string(index=False))

    print("\n[Township Precipitation Probability Forecast]")
    forecast_df = fetch_forecast(save=True)
    print(f"  Total rows: {len(forecast_df)}")
    print(f"  Counties: {forecast_df['county'].nunique()}")
    print(f"  Townships: {forecast_df['township'].nunique()}")
    print(forecast_df.head(8).to_string(index=False))

    print("\n✅ All data saved to data/raw/")
