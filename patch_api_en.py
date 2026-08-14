import json
import re

# 1. Update requirements.txt
with open('requirements.txt', 'a') as f:
    f.write('\npypinyin>=0.55.0\n')

# 2. Update api/api_server.py
with open('api/api_server.py', 'r') as f:
    api_code = f.read()

translation_code = """
import pypinyin

COUNTY_MAP = {
    "臺北市": "Taipei City", "台北市": "Taipei City",
    "新北市": "New Taipei City",
    "基隆市": "Keelung City",
    "桃園市": "Taoyuan City",
    "新竹縣": "Hsinchu County", "新竹市": "Hsinchu City",
    "苗栗縣": "Miaoli County",
    "臺中市": "Taichung City", "台中市": "Taichung City",
    "彰化縣": "Changhua County",
    "南投縣": "Nantou County",
    "雲林縣": "Yunlin County",
    "嘉義縣": "Chiayi County", "嘉義市": "Chiayi City",
    "臺南市": "Tainan City", "台南市": "Tainan City",
    "高雄市": "Kaohsiung City",
    "屏東縣": "Pingtung County",
    "宜蘭縣": "Yilan County",
    "花蓮縣": "Hualien County",
    "臺東縣": "Taitung County", "台東縣": "Taitung County",
    "澎湖縣": "Penghu County",
    "金門縣": "Kinmen County",
    "連江縣": "Lienchiang County"
}

def to_en(text: str) -> str:
    if not text: return ""
    text = str(text)
    if text in COUNTY_MAP: return COUNTY_MAP[text]
    
    # Handle suffixes
    orig = text
    if text.endswith("區"): text = text[:-1] + " Dist."
    elif text.endswith("鄉"): text = text[:-1] + " Township"
    elif text.endswith("鎮"): text = text[:-1] + " Township"
    elif text.endswith("市") and text not in COUNTY_MAP: text = text[:-1] + " City"
    
    # Translate the chinese characters using pypinyin
    # Only translate if there are chinese characters (simplified heuristic)
    if any('\\u4e00' <= c <= '\\u9fff' for c in text):
        parts = []
        for word in pypinyin.pinyin(text, style=pypinyin.NORMAL):
            parts.append(word[0].capitalize())
        return "".join(parts).replace("Dist.", " Dist.").replace("Township", " Township").replace("City", " City").replace("  ", " ").strip()
    return orig
"""

# Insert translation code after AppState
api_code = api_code.replace('class AppState:', translation_code + '\nclass AppState:')

# Replace the field extractions in /api/risk-map
api_code = api_code.replace(
    '"station_name":   str(row.get("station_name", "")),',
    '"station_name":   to_en(str(row.get("station_name", ""))),'
)
api_code = api_code.replace(
    '"county":       str(row.get("county", "")),',
    '"county":       to_en(str(row.get("county", ""))),'
)
api_code = api_code.replace(
    '"town":         str(row.get("town", "")),',
    '"town":         to_en(str(row.get("town", ""))),'
)

# Replace the field extractions in /api/community/{station_id}
api_code = api_code.replace(
    '"station_name":  str(row.get("station_name", "")),',
    '"station_name":  to_en(str(row.get("station_name", ""))),'
)
api_code = api_code.replace(
    '"town":          str(row.get("town", "")),',
    '"town":          to_en(str(row.get("town", ""))),'
)
api_code = api_code.replace(
    'county = str(row.get("county", ""))',
    'county = to_en(str(row.get("county", "")))'
)

# Replace the field extractions in /api/alerts
api_code = api_code.replace(
    '"station_name":  str(row.get("station_name", "")),',
    '"station_name":  to_en(str(row.get("station_name", ""))),'
)
api_code = api_code.replace(
    '"county":        str(row.get("county", "")),',
    '"county":        to_en(str(row.get("county", ""))),'
)
api_code = api_code.replace(
    '"town":          str(row.get("town", "")),',
    '"town":          to_en(str(row.get("town", ""))),'
)

with open('api/api_server.py', 'w') as f:
    f.write(api_code)

# 3. Translate frontend/sim-data/*.json
# We import pypinyin locally to translate JSONs immediately
import pypinyin
COUNTY_MAP = {
    "臺北市": "Taipei City", "台北市": "Taipei City", "新北市": "New Taipei City",
    "基隆市": "Keelung City", "桃園市": "Taoyuan City", "新竹縣": "Hsinchu County", "新竹市": "Hsinchu City",
    "苗栗縣": "Miaoli County", "臺中市": "Taichung City", "台中市": "Taichung City",
    "彰化縣": "Changhua County", "南投縣": "Nantou County", "雲林縣": "Yunlin County",
    "嘉義縣": "Chiayi County", "嘉義市": "Chiayi City", "臺南市": "Tainan City", "台南市": "Tainan City",
    "高雄市": "Kaohsiung City", "屏東縣": "Pingtung County", "宜蘭縣": "Yilan County",
    "花蓮縣": "Hualien County", "臺東縣": "Taitung County", "台東縣": "Taitung County",
    "澎湖縣": "Penghu County", "金門縣": "Kinmen County", "連江縣": "Lienchiang County"
}
def to_en(text):
    if not text: return ""
    text = str(text)
    if text in COUNTY_MAP: return COUNTY_MAP[text]
    orig = text
    if text.endswith("區"): text = text[:-1] + " Dist."
    elif text.endswith("鄉"): text = text[:-1] + " Township"
    elif text.endswith("鎮"): text = text[:-1] + " Township"
    elif text.endswith("市") and text not in COUNTY_MAP: text = text[:-1] + " City"
    if any('\\u4e00' <= c <= '\\u9fff' for c in text):
        parts = []
        for word in pypinyin.pinyin(text, style=pypinyin.NORMAL):
            parts.append(word[0].capitalize())
        return "".join(parts).replace("Dist.", " Dist.").replace("Township", " Township").replace("City", " City").replace("  ", " ").strip()
    return orig

for filename in ['frontend/sim-data/risk-map.json', 'frontend/sim-data/alerts.json']:
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        
        if 'features' in data:
            for feat in data['features']:
                if 'properties' in feat:
                    feat['properties']['station_name'] = to_en(feat['properties'].get('station_name', ''))
                    feat['properties']['county'] = to_en(feat['properties'].get('county', ''))
                    feat['properties']['town'] = to_en(feat['properties'].get('town', ''))
        elif 'alerts' in data:
            for alert in data['alerts']:
                alert['station_name'] = to_en(alert.get('station_name', ''))
                alert['county'] = to_en(alert.get('county', ''))
                alert['town'] = to_en(alert.get('town', ''))
                
        with open(filename, 'w') as f:
            json.dump(data, f, ensure_ascii=False)

if os.path.exists('frontend/sim-data/community.json'):
    with open('frontend/sim-data/community.json', 'r') as f:
        data = json.load(f)
    data['station_name'] = to_en(data.get('station_name', ''))
    data['county'] = to_en(data.get('county', ''))
    data['town'] = to_en(data.get('town', ''))
    with open('frontend/sim-data/community.json', 'w') as f:
        json.dump(data, f, ensure_ascii=False)
