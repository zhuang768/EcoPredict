"""
data_fetcher.py
===============
從中央氣象署開放資料平台抓取即時雨量觀測與鄉鎮天氣預報。

資料來源：
  - O-A0002-001：自動雨量站即時觀測（每 10 分鐘更新）
  - F-D0047-XXX：各縣市鄉鎮未來 4 天每 3 小時降雨機率預報

使用方式：
  python src/data_fetcher.py

授權：
  API Token 設定於 .env 的 CWA_API_TOKEN。
  未設定時自動使用官方公開 Demo Token（rdec-key-123-45678-011121314）。
"""

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

# ── 日誌設定 ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── 常數 ────────────────────────────────────────────────────────────────────────
_DEMO_TOKEN = "rdec-key-123-45678-011121314"
_BASE_URL = "https://opendata.cwa.gov.tw/api/v1/rest/datastore"
_RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
_RAW_DIR.mkdir(parents=True, exist_ok=True)

# 全台 22 縣市鄉鎮天氣預報 endpoint 代碼（已實際驗證可用）
# 每兩個代碼屬於同一縣市（奇偶對應不同批次）
_COUNTY_FORECAST_ENDPOINTS: list[tuple[str, str]] = [
    ("F-D0047-001", "宜蘭縣"),
    ("F-D0047-003", "桃園市"),
    ("F-D0047-005", "新竹縣"),
    ("F-D0047-007", "苗栗縣"),
    ("F-D0047-009", "彰化縣"),
    ("F-D0047-011", "南投縣"),
    ("F-D0047-013", "雲林縣"),
    ("F-D0047-015", "嘉義縣"),
    ("F-D0047-017", "屏東縣"),
    ("F-D0047-019", "台東縣"),
    ("F-D0047-021", "花蓮縣"),
    ("F-D0047-023", "澎湖縣"),
    ("F-D0047-029", "新北市"),
    ("F-D0047-031", "台中市"),
    ("F-D0047-033", "台南市"),
    ("F-D0047-035", "高雄市"),
    ("F-D0047-059", "連江縣"),
    ("F-D0047-061", "臺北市"),
    ("F-D0047-063", "新竹市"),
    ("F-D0047-065", "嘉義市"),
    ("F-D0047-067", "基隆市"),
]


# ── HTTP 工具 ────────────────────────────────────────────────────────────────────
def _build_session() -> requests.Session:
    """建立帶有重試策略的 Session。"""
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
    """
    對 CWA API 發送 GET，回傳解析後的 JSON dict。

    Raises:
        requests.HTTPError: HTTP 狀態碼非 2xx。
        requests.ConnectionError: 無法連線。
        ValueError: API 回傳 success != true。
    """
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
        logger.error("無法連線至 CWA API：%s", url)
        raise
    except requests.exceptions.Timeout:
        logger.error("請求逾時（%ds）：%s", timeout, url)
        raise

    payload = resp.json()
    if str(payload.get("success")).lower() != "true":
        raise ValueError(f"CWA API 回傳失敗：{payload.get('message', payload)}")
    return payload


# ── 儲存工具 ────────────────────────────────────────────────────────────────────
def _save_csv(df: pd.DataFrame, name: str) -> Path:
    """以 UTC 時間戳記命名，儲存至 data/raw/。"""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = _RAW_DIR / f"{name}_{ts}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("已儲存 %d 列 → %s", len(df), path)
    return path


# ── 雨量站即時觀測 (O-A0002-001) ────────────────────────────────────────────────
def fetch_rainfall_realtime(save: bool = True) -> pd.DataFrame:
    """
    抓取全台自動雨量站即時觀測資料（每 10 分鐘更新）。

    Returns:
        DataFrame，欄位：
            station_id, station_name, obs_time (tz-aware UTC),
            lat, lon, altitude, county, town,
            rainfall_10min, rainfall_1hr, rainfall_3hr,
            rainfall_6hr, rainfall_12hr, rainfall_24hr,
            rainfall_2days, rainfall_3days

    Raises:
        requests.RequestException: 網路或 HTTP 錯誤。
    """
    logger.info("抓取即時雨量觀測 (O-A0002-001) …")

    try:
        raw = _get("O-A0002-001", {"limit": 1000})
    except Exception as exc:
        logger.error("fetch_rainfall_realtime 失敗：%s", exc)
        raise

    stations: list[dict] = raw["records"]["Station"]
    rows = []

    for s in stations:
        geo = s.get("GeoInfo", {})
        coords = geo.get("Coordinates", [])
        # 優先取 WGS84；無 WGS84 時取第一組座標
        wgs84 = next(
            (c for c in coords if c.get("CoordinateName") == "WGS84"),
            coords[0] if coords else {},
        )

        rain = s.get("RainfallElement", {})

        def _p(key: str) -> Optional[float]:
            """安全解析雨量值，'-' 或空值回傳 None。"""
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
        "取得 %d 站觀測；有效 rainfall_24hr：%d 站",
        len(df),
        df["rainfall_24hr"].notna().sum(),
    )

    if save:
        _save_csv(df, "rainfall_realtime")

    return df


# ── 鄉鎮天氣預報 (F-D0047-XXX) ──────────────────────────────────────────────────
def fetch_forecast(save: bool = True) -> pd.DataFrame:
    """
    抓取全台各縣市鄉鎮未來約 4 天的每 3 小時降雨機率預報（PoP）。

    注意：CWA 將各縣市拆成獨立 endpoint（F-D0047-001 ~ F-D0047-067），
    本函式會逐一呼叫並合併為單一 DataFrame。

    Returns:
        DataFrame，欄位：
            county, township, lat, lon,
            start_time, end_time, pop_3h (降雨機率 0-100，%)

    Raises:
        RuntimeError: 若所有 endpoint 均失敗。
    """
    logger.info("抓取鄉鎮降雨機率預報（共 %d 個縣市 endpoint）…", len(_COUNTY_FORECAST_ENDPOINTS))

    all_rows: list[dict] = []
    failed: list[str] = []

    for ep_code, county_hint in _COUNTY_FORECAST_ENDPOINTS:
        try:
            raw = _get(ep_code)
        except Exception as exc:
            logger.warning("  ✗ %s (%s) 抓取失敗：%s", ep_code, county_hint, exc)
            failed.append(ep_code)
            time.sleep(0.5)  # 政府 API 有速率限制，失敗後稍等
            continue

        locations_arr = raw.get("records", {}).get("Locations", [])
        if not locations_arr:
            logger.warning("  ✗ %s — 無 Locations 資料", ep_code)
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
                if elem.get("ElementName") != "3小時降雨機率":
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

        logger.info("  ✓ %s — %s (%d 鄉鎮)", ep_code, county_name, len(township_list))

    if not all_rows:
        raise RuntimeError("所有 forecast endpoint 均失敗，無法取得預報資料。")

    if failed:
        logger.warning("以下 endpoint 取得失敗，資料可能不完整：%s", failed)

    df = pd.DataFrame(all_rows)
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    df["end_time"] = pd.to_datetime(df["end_time"], utc=True)

    logger.info(
        "預報資料完成：%d 列，涵蓋 %d 縣市 / %d 鄉鎮",
        len(df),
        df["county"].nunique(),
        df["township"].nunique(),
    )

    if save:
        _save_csv(df, "township_forecast")

    return df


# ── 工具函式 ────────────────────────────────────────────────────────────────────
def _safe_float(val) -> Optional[float]:
    """安全轉換為 float，無法轉換時回傳 None。"""
    if val is None or str(val).strip() in ("", "-"):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ── CLI 進入點 ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("EcoPredict — CWA 資料擷取")
    print("=" * 60)

    # --- 即時雨量 ---
    rainfall_df = fetch_rainfall_realtime(save=True)
    print("\n【即時雨量觀測】")
    print(f"  總站數：{len(rainfall_df)}")
    print(f"  有效 rainfall_10min：{rainfall_df['rainfall_10min'].notna().sum()}")
    print(f"  有效 rainfall_24hr ：{rainfall_df['rainfall_24hr'].notna().sum()}")
    print(rainfall_df[["station_id", "station_name", "county", "lat", "lon",
                        "rainfall_10min", "rainfall_1hr", "rainfall_24hr"]].head(5).to_string(index=False))

    # --- 預報 ---
    print("\n【鄉鎮降雨機率預報】")
    forecast_df = fetch_forecast(save=True)
    print(f"  總列數：{len(forecast_df)}")
    print(f"  縣市數：{forecast_df['county'].nunique()}")
    print(f"  鄉鎮數：{forecast_df['township'].nunique()}")
    print(forecast_df.head(8).to_string(index=False))

    print("\n✅ 所有資料已儲存至 data/raw/")
