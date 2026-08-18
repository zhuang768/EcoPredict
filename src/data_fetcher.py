import json
import logging
import time
from pathlib import Path
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
STATIONS_FILE = Path("data/global_stations.json")
TIMEOUT = 30

def load_stations():
    with open(STATIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def fetch_global_weather():
    stations = load_stations()
    if not stations:
        raise ValueError("No global stations found.")

    lats = ",".join(str(s["lat"]) for s in stations)
    lons = ",".join(str(s["lon"]) for s in stations)

    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lats}&longitude={lons}"
        f"&current=precipitation"
        f"&hourly=precipitation_probability,precipitation"
        f"&past_days=3"
        f"&timezone=UTC"
    )

    logger.info("Fetching global weather data from Open-Meteo (batch)...")
    try:
        resp = requests.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.error("Failed to connect to Open-Meteo API: %s", exc)
        raise

    if not isinstance(data, list):
        data = [data] # Handle single station case just in case

    current_obs = []
    forecasts = []

    for idx, station in enumerate(stations):
        station_data = data[idx]
        
        # Current Observation
        current_precip = station_data.get("current", {}).get("precipitation", 0.0)
        
        # Hourly array contains past 3 days (72h) + 7 days forecast (168h) = 240h
        hourly = station_data.get("hourly", {})
        times = hourly.get("time", [])
        precips = hourly.get("precipitation", [])
        pops = hourly.get("precipitation_probability", [])

        # Calculate 3-day accumulation (first 72 hours)
        past_72_precip = sum(precips[:72]) if len(precips) >= 72 else 0.0

        current_obs.append({
            "station_id": station["id"],
            "station_name": station["name"],
            "county": station["name"],
            "town": station["country"],
            "lat": station["lat"],
            "lon": station["lon"],
            "rain_1h": current_precip,
            "rain_24hr": sum(precips[48:72]) if len(precips) >= 72 else past_72_precip, # Last 24h
            "rain_3days": past_72_precip,
            "obs_time": station_data.get("current", {}).get("time", "")
        })

        # Forecast (next 48 hours)
        # The 72nd index is the current hour
        if len(times) > 72:
            future_times = times[72:72+48]
            future_pops = pops[72:72+48]
            for t, p in zip(future_times, future_pops):
                forecasts.append({
                    "county": station["name"],
                    "township": station["country"],
                    "fcst_time": t,
                    "pop_percent": p
                })

    # Save to disk
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    obs_df = pd.DataFrame(current_obs)
    obs_path = RAW_DIR / "current_obs.csv"
    obs_df.to_csv(obs_path, index=False)
    logger.info("Saved %d global observations -> %s", len(obs_df), obs_path)

    fcst_df = pd.DataFrame(forecasts)
    fcst_path = RAW_DIR / "county_forecast.csv"
    fcst_df.to_csv(fcst_path, index=False)
    logger.info("Saved %d forecast rows -> %s", len(fcst_df), fcst_path)

if __name__ == "__main__":
    print("EcoPredict - Global Data Fetching (Open-Meteo)")
    fetch_global_weather()
    print("\n✅ All global data saved to data/raw/")
