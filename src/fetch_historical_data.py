import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STATIONS_FILE = Path("data/global_stations.json")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def load_stations():
    with open(STATIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def fetch_history_for_station(lat: float, lon: float, station_name: str) -> pd.DataFrame:
    # Fetch past 6 months to get a good spread of weather without overwhelming the free API
    # 6 months of hourly data = ~4300 rows per station. 30 stations = 130,000 rows.
    # Open-Meteo limit is 10,000 API calls per day.
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date=2023-01-01&end_date=2023-06-30"
        f"&hourly=precipitation"
        f"&timezone=UTC"
    )
    
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        logger.warning(f"Failed to fetch {station_name}: {resp.status_code}")
        return pd.DataFrame()
        
    data = resp.json()
    hourly = data.get("hourly", {})
    if not hourly:
        return pd.DataFrame()
        
    df = pd.DataFrame({
        "time": pd.to_datetime(hourly["time"]),
        "precipitation": hourly["precipitation"]
    })
    
    # Fill NAs
    df["precipitation"] = df["precipitation"].fillna(0.0)
    
    # Calculate rolling features
    df["rain_1h"] = df["precipitation"]
    df["rain_24h"] = df["precipitation"].rolling(window=24, min_periods=1).sum()
    df["rain_3days"] = df["precipitation"].rolling(window=72, min_periods=1).sum()
    
    # Calculate future rainfall to act as a proxy for historical "forecasts" and labels
    # shift(-24) means look ahead 24 hours. We sum the next 24h.
    # rolling backwards is tricky in pandas, easier to reverse, roll, reverse.
    future_24h = df["precipitation"][::-1].rolling(window=24, min_periods=1).sum()[::-1]
    future_48h = df["precipitation"][::-1].rolling(window=48, min_periods=1).sum()[::-1]
    
    # Proxy: If it actually rained 20mm in the next 24h, we assume the PoP was 100%.
    df["forecast_pop_avg_24h"] = (future_24h * 5).clip(upper=100)
    df["forecast_pop_avg_48h"] = (future_48h * 5).clip(upper=100)
    
    # Label: 1 if severe rain happened (e.g. > 100mm in 24h or > 150mm in 3days)
    df["label"] = ((df["rain_24h"] >= 100) | (df["rain_3days"] >= 150)).astype(int)
    
    # Drop edge cases with NaN from rolling
    df = df.dropna()
    
    # To keep dataset manageable, sample every 12 hours
    df = df.iloc[::12].copy()
    df["station_name"] = station_name
    
    return df

def build_historical_dataset():
    stations = load_stations()
    all_dfs = []
    
    logger.info("Fetching historical data for 30 cities...")
    for st in tqdm(stations):
        df = fetch_history_for_station(st["lat"], st["lon"], st["name"])
        if not df.empty:
            all_dfs.append(df)
        time.sleep(0.5) # respect API rate limits
        
    final_df = pd.concat(all_dfs, ignore_index=True)
    out_path = PROCESSED_DIR / "historical_features.csv"
    
    # Reorder columns to match expected
    cols = ["station_name", "time", "rain_1h", "rain_24h", "rain_3days", 
            "forecast_pop_avg_24h", "forecast_pop_avg_48h", "label"]
    final_df = final_df[cols]
    
    final_df.to_csv(out_path, index=False)
    
    pos = final_df["label"].sum()
    logger.info(f"Historical dataset complete! Total rows: {len(final_df)}")
    logger.info(f"Real disaster events (label=1): {pos} / {len(final_df)} ({(pos/len(final_df))*100:.1f}%)")
    logger.info(f"Saved to {out_path}")

if __name__ == "__main__":
    build_historical_dataset()
