import json
import logging
import time
from pathlib import Path

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STATIONS_FILE = Path("data/global_stations.json")
REGISTRY_FILE = Path("data/care_list/vulnerable_registry.json")

def load_stations():
    with open(STATIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def fetch_infrastructure(lat, lon, county):
    # Overpass API requires a user-agent to avoid blocking
    headers = {"User-Agent": "EcoPredict/2.0 (Hackathon Project)"}
    # Query for hospitals, clinics, and social facilities within 10km
    query = f"""
    [out:json][timeout:25];
    (
      node["amenity"="hospital"](around:3000, {lat}, {lon});
      node["amenity"="clinic"](around:3000, {lat}, {lon});
      node["amenity"="social_facility"](around:3000, {lat}, {lon});
    );
    out 5;
    """
    
    url = "https://overpass-api.de/api/interpreter"
    try:
        resp = requests.post(url, data={"data": query}, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        records = []
        for elem in data.get("elements", []):
            tags = elem.get("tags", {})
            name = tags.get("name") or tags.get("name:en")
            if not name:
                continue
                
            amenity = tags.get("amenity", "Facility")
            # Format to match our new schema
            records.append({
                "id": f"INFRA-{elem['id']}",
                "code_name": name,
                "county": county,
                "town": tags.get("addr:city", county),
                "community": amenity.capitalize(),
                "is_priority": True if amenity == "hospital" else False,
                "emergency_contact": tags.get("phone", "N/A"),
                "notes": f"Real {amenity} fetched from OpenStreetMap"
            })
        return records
    except Exception as e:
        logger.error(f"Error fetching infrastructure for {county}: {e}")
        return []

def build_infrastructure_registry():
    stations = load_stations()
    all_records = []
    
    logger.info("Fetching real infrastructure for 30 global cities from OpenStreetMap...")
    for st in tqdm(stations):
        records = fetch_infrastructure(st["lat"], st["lon"], st["name"])
        all_records.extend(records)
        time.sleep(1.5) # Overpass is very strict on rate limits
        
    registry = {
        "_metadata": {
            "description": "EcoPredict Global Critical Infrastructure Registry",
            "source": "OpenStreetMap Overpass API",
            "version": "3.0"
        },
        "records": all_records
    }
    
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
        
    logger.info(f"Registry complete! Saved {len(all_records)} real facilities to {REGISTRY_FILE}")

if __name__ == "__main__":
    build_infrastructure_registry()
