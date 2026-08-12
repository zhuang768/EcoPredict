"""
data_ingestion.py
=================
從中央氣象署開放資料平台抓取：
  - O-A0002-001：全台自動雨量站即時觀測（含 Now / Past10Min / Past1hr … Past3days）
  - F-C0032-001：縣市 36 小時天氣預報（含降雨機率 PoP）

公開 Demo Token：rdec-key-123-45678-011121314
（使用者申請正式 Token 後，改設定 .env CWA_API_TOKEN 即可）
"""

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

# ── 設定 ──────────────────────────────────────────────────────────────────────
CWA_BASE = "https://opendata.cwa.gov.tw/api/v1/rest/datastore"
DEMO_TOKEN = "rdec-key-123-45678-011121314"
TOKEN = os.getenv("CWA_API_TOKEN", DEMO_TOKEN)

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json"})


# ── 核心 HTTP 工具 ─────────────────────────────────────────────────────────────
def _get(endpoint: str, params: dict[str, Any] | None = None) -> dict:
    """對 CWA API 發送 GET 請求，回傳解析後的 JSON dict。"""
    url = f"{CWA_BASE}/{endpoint}"
    default_params = {"Authorization": TOKEN, "format": "JSON"}
    if params:
        default_params.update(params)

    try:
        resp = SESSION.get(url, params=default_params, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        logger.error("HTTP 錯誤 %s | %s", exc.response.status_code, url)
        raise
    except requests.exceptions.ConnectionError:
        logger.error("無法連線至 CWA API：%s", url)
        raise

    data = resp.json()
    if data.get("success") != "true":
        raise ValueError(f"CWA API 回傳失敗：{data}")
    return data


# ── 雨量站即時觀測 (O-A0002-001) ──────────────────────────────────────────────
def fetch_rainfall_obs(limit: int = 1000) -> pd.DataFrame:
    """
    抓取全台自動雨量站即時觀測資料。

    回傳欄位：
        station_id, station_name, obs_time,
        lat, lon, altitude, county, town,
        precip_now, precip_10min, precip_1hr, precip_3hr,
        precip_6hr, precip_12hr, precip_24hr, precip_2days, precip_3days
    """
    logger.info("抓取雨量站觀測資料 (O-A0002-001) ...")
    raw = _get("O-A0002-001", {"limit": limit})
    stations: list[dict] = raw["records"]["Station"]

    rows = []
    for s in stations:
        geo = s.get("GeoInfo", {})
        coords = geo.get("Coordinates", [])
        # 優先取 WGS84 座標
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
    logger.info("取得 %d 筆雨量站觀測", len(df))
    return df


# ── 縣市天氣預報 (F-C0032-001) ────────────────────────────────────────────────
def fetch_county_forecast() -> pd.DataFrame:
    """
    抓取縣市 36 小時天氣預報（降雨機率 PoP）。

    回傳欄位：
        county, start_time, end_time, pop_pct (降雨機率 0-100)
    """
    logger.info("抓取縣市天氣預報 (F-C0032-001) ...")
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
    logger.info("取得 %d 筆預報記錄（%d 個縣市）", len(df), df["county"].nunique())
    return df


# ── 儲存為 CSV ─────────────────────────────────────────────────────────────────
def save_raw(df: pd.DataFrame, name: str) -> Path:
    """以時間戳記命名並儲存至 data/raw/。"""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RAW_DIR / f"{name}_{ts}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("儲存至 %s", path)
    return path


# ── 對外公開函式 ───────────────────────────────────────────────────────────────
def ingest() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    執行完整資料擷取流程。

    Returns:
        (rainfall_df, forecast_df)
    """
    rainfall_df = fetch_rainfall_obs()
    forecast_df = fetch_county_forecast()

    save_raw(rainfall_df, "rainfall_obs")
    save_raw(forecast_df, "county_forecast")

    return rainfall_df, forecast_df


if __name__ == "__main__":
    r_df, f_df = ingest()

    print("\n=== 雨量站觀測 (前 3 筆) ===")
    print(r_df.head(3).to_string(index=False))
    print(f"\n欄位：{list(r_df.columns)}")
    print(f"總站數：{len(r_df)}")
    print(f"有效雨量 (precip_24hr not null)：{r_df['precip_24hr'].notna().sum()}")

    print("\n=== 縣市預報 (前 3 筆) ===")
    print(f_df.head(3).to_string(index=False))
    print(f"\n縣市數：{f_df['county'].nunique()}")
