"""
feature_engineering.py
======================
從 data/raw/ 的 CSV 建立特徵表，輸出至 data/processed/features.csv。

設計說明（重要限制）：
  1. CWA O-A0002-001 回傳的是「已聚合的累積值」而非個別 10 分鐘原始讀值。
     因此 rain_intensity_max 使用 rainfall_10min（當前最新 10 分鐘讀值）代替，
     無法重建過去 1 小時的 6 個獨立讀值，已在 README 中標注此限制。
  2. CWA 預報 API 提供的是「降雨機率 PoP (0-100%)」而非預測毫米數，
     因此 forecast 特徵為未來 24h / 48h 的平均 PoP。
  3. 標籤（label）以台灣 NCDR 官方警戒閾值規則合成（無自動化歷史 NCDR API），
     適合黑客松示範用途，正式應用需以真實警戒紀錄取代。
  4. 「時間序列切分」：目前資料為單一時間快照，訓練/測試以緯度地理切分
     （北台灣訓練、南台灣測試），符合颱風路徑南北差異的物理特性。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

_DEMO_TOKEN = "rdec-key-123-45678-011121314"
_CWA_BASE = "https://opendata.cwa.gov.tw/api/v1/rest/datastore"


# ── 工具函式 ────────────────────────────────────────────────────────────────────
def _load_latest_csv(pattern: str) -> pd.DataFrame:
    """從 data/raw/ 載入最新的符合 pattern 的 CSV。"""
    files = sorted(RAW_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"找不到 data/raw/{pattern}。請先執行 data_fetcher.py。"
        )
    path = files[-1]
    logger.info("載入 %s", path.name)
    return pd.read_csv(path)


def _haversine_km(lat1: np.ndarray, lon1: np.ndarray,
                   lat2: float, lon2: float) -> np.ndarray:
    """向量化 Haversine 距離（公里）。"""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


# ── 颱風警報狀態 ────────────────────────────────────────────────────────────────
def get_typhoon_status() -> bool:
    """
    呼叫 CWA W-C0034-001 確認當前是否有效颱風警報。

    Returns:
        True  — 有效颱風警報（Immediate / Expected）
        False — 無警報或已解除（Past / Cancel）
    """
    token = os.getenv("CWA_API_TOKEN", _DEMO_TOKEN)
    try:
        resp = requests.get(
            f"{_CWA_BASE}/W-C0034-001",
            params={"Authorization": token, "format": "JSON"},
            timeout=10,
        )
        resp.raise_for_status()
        records = resp.json().get("records", {}).get("info", [])
        for item in records:
            urgency = item.get("urgency", "Past")
            if urgency in ("Immediate", "Expected"):
                logger.info("偵測到有效颱風警報：urgency=%s", urgency)
                return True
        logger.info("無有效颱風警報（最近紀錄 urgency=Past 或無資料）")
        return False
    except Exception as exc:
        logger.warning("颱風警報 API 呼叫失敗（%s），預設 is_typhoon_period=False", exc)
        return False


# ── 最近鄉鎮預報映射 ────────────────────────────────────────────────────────────
def _compute_forecast_features(
    station_lat: float,
    station_lon: float,
    forecast_df: pd.DataFrame,
    now_utc: pd.Timestamp,
) -> dict:
    """
    找出最近鄉鎮，計算未來 24h / 48h 的平均 PoP。

    Returns:
        dict with keys: forecast_pop_avg_24h, forecast_pop_avg_48h
    """
    townships = forecast_df[["township", "lat", "lon"]].drop_duplicates("township")
    dists = _haversine_km(
        townships["lat"].values, townships["lon"].values,
        station_lat, station_lon,
    )
    nearest_name = townships.iloc[np.argmin(dists)]["township"]
    t_fc = forecast_df[forecast_df["township"] == nearest_name].copy()
    t_fc["start_time"] = pd.to_datetime(t_fc["start_time"], utc=True)

    future_fc = t_fc[t_fc["start_time"] >= now_utc]

    h24 = future_fc[future_fc["start_time"] < now_utc + pd.Timedelta(hours=24)]
    h48 = future_fc[future_fc["start_time"] < now_utc + pd.Timedelta(hours=48)]

    return {
        "forecast_pop_avg_24h": round(h24["pop_3h"].mean(), 1) if len(h24) else np.nan,
        "forecast_pop_avg_48h": round(h48["pop_3h"].mean(), 1) if len(h48) else np.nan,
        "nearest_forecast_township": nearest_name,
    }


# ── 標籤合成（NCDR 閾值規則） ────────────────────────────────────────────────────
def _make_label(row: pd.Series) -> int:
    """
    根據台灣 NCDR / 農委會水保局警戒閾值規則合成 binary label。

    分類標準（任一條件成立 → label=1）：
      - 24小時累積雨量 ≥ 200mm（豪雨，水利署淹水警戒基準）
      - 3小時累積雨量  ≥ 80mm（短延時強降雨警戒）
      - 過去3天累積    ≥ 130mm 且海拔 ≥ 300m（土石流複合警戒）
      - 1小時雨量      ≥ 40mm（極短時強降雨）

    注意：此為規則式合成標籤，用於黑客松示範。
          正式應用應以 NCDR 警戒歷史紀錄取代。
    """
    r24 = row.get("rain_24h", 0) or 0
    r3  = row.get("rain_3h", 0) or 0
    r3d = row.get("rain_3days", 0) or 0
    r1  = row.get("rain_1h", 0) or 0
    alt = row.get("altitude", 0) or 0

    if r24 >= 200:
        return 1
    if r3 >= 80:
        return 1
    if r3d >= 130 and alt >= 300:
        return 1
    if r1 >= 40:
        return 1
    return 0


# ── 主要特徵工程函式 ────────────────────────────────────────────────────────────
def build_features(save: bool = True) -> pd.DataFrame:
    """
    載入最新 raw CSVs，計算所有特徵，輸出特徵表。

    Returns:
        feature DataFrame，含所有特徵欄位與 label。
    """
    # 1. 載入資料
    rainfall_df = _load_latest_csv("rainfall_realtime_*.csv")
    forecast_df = _load_latest_csv("township_forecast_*.csv")

    # 2. 時間基準
    rainfall_df["obs_time"] = pd.to_datetime(rainfall_df["obs_time"], utc=True)
    now_utc = rainfall_df["obs_time"].max()
    logger.info("觀測時間基準（UTC）：%s", now_utc)

    # 3. 颱風警報狀態（全局，單一值）
    is_typhoon = int(get_typhoon_status())
    logger.info("is_typhoon_period = %s", bool(is_typhoon))

    # 4. 逐站計算特徵
    rows = []
    total = len(rainfall_df)
    for i, station in rainfall_df.iterrows():
        lat = station.get("lat")
        lon = station.get("lon")

        # 缺少座標的站跳過（無法做空間映射）
        if pd.isna(lat) or pd.isna(lon):
            continue

        # 預報特徵
        try:
            fc_feats = _compute_forecast_features(lat, lon, forecast_df, now_utc)
        except Exception:
            fc_feats = {
                "forecast_pop_avg_24h": np.nan,
                "forecast_pop_avg_48h": np.nan,
                "nearest_forecast_township": None,
            }

        row = {
            # 識別資訊
            "station_id":   station["station_id"],
            "station_name": station["station_name"],
            "county":       station.get("county"),
            "town":         station.get("town"),
            "lat":          lat,
            "lon":          lon,
            "altitude":     station.get("altitude"),
            "obs_time":     now_utc,

            # 雨量特徵（直接來自 CWA 累積欄位）
            "rain_1h":            station.get("rainfall_1hr"),
            "rain_3h":            station.get("rainfall_3hr"),
            "rain_6h":            station.get("rainfall_6hr"),
            "rain_12h":           station.get("rainfall_12hr"),
            "rain_24h":           station.get("rainfall_24hr"),
            "rain_2days":         station.get("rainfall_2days"),
            "rain_3days":         station.get("rainfall_3days"),

            # 短時強降雨（僅有最新 10min 讀值，非 6 筆獨立值的最大值）
            "rain_intensity_max": station.get("rainfall_10min"),

            # 預報特徵（PoP 百分比，非毫米數）
            "forecast_pop_avg_24h": fc_feats["forecast_pop_avg_24h"],
            "forecast_pop_avg_48h": fc_feats["forecast_pop_avg_48h"],
            "nearest_forecast_township": fc_feats["nearest_forecast_township"],

            # 颱風警報期間
            "is_typhoon_period": is_typhoon,

            # 地質特徵佔位（需 geopandas + shapefile，此版本留空）
            "distance_to_debris_flow_km": np.nan,
            "flood_potential_level":      np.nan,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # 5. 填補極少數 NaN（雨量感測器故障等）
    rain_cols = ["rain_1h", "rain_3h", "rain_6h", "rain_12h",
                 "rain_24h", "rain_2days", "rain_3days", "rain_intensity_max"]
    df[rain_cols] = df[rain_cols].fillna(0.0)

    # 6. 合成 label
    df["label"] = df.apply(_make_label, axis=1)

    # 7. 統計摘要
    pos = df["label"].sum()
    neg = len(df) - pos
    logger.info(
        "特徵表完成：%d 筆（正例 label=1：%d / %.1f%%，負例：%d）",
        len(df), pos, 100 * pos / len(df), neg,
    )
    logger.info(
        "rain_3days 分布：min=%.1f, q25=%.1f, q50=%.1f, q75=%.1f, max=%.1f",
        df["rain_3days"].min(), df["rain_3days"].quantile(0.25),
        df["rain_3days"].quantile(0.50), df["rain_3days"].quantile(0.75),
        df["rain_3days"].max(),
    )

    if save:
        out = PROCESSED_DIR / "features.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        logger.info("已儲存 → %s", out)

    return df


if __name__ == "__main__":
    df = build_features(save=True)
    print("\n【特徵表摘要】")
    print(df[["station_name", "county", "rain_24h", "rain_3days",
              "forecast_pop_avg_24h", "is_typhoon_period", "label"]].head(10).to_string(index=False))
    print(f"\n總站數：{len(df)}")
    print(f"正例（label=1）：{df['label'].sum()}（{100*df['label'].mean():.1f}%）")
    print(f"特徵欄位：{[c for c in df.columns if c not in ['station_id','station_name','county','town','obs_time','nearest_forecast_township']]}")
