# EcoPredict: Global Flood & Debris Flow Risk Prediction System

![EcoPredict Demo](https://via.placeholder.com/1200x600?text=EcoPredict+Global+Dashboard)

**EcoPredict** is an open-source, global AI early warning system built for **ReverieHacks 2026**.
It aggregates real-time weather and forecast data from Open-Meteo across dozens of high-risk cities worldwide, and uses Machine Learning to predict hyper-local flood and debris flow risks.

## Features

- **Global Open-Meteo Integration:** Automatically fetches real-time precipitation and 48-hour PoP forecasts for 30+ major global cities.
- **Machine Learning Risk Engine:** Trains RandomForest/LogisticRegression on rainfall accumulation to determine 4-tier risk levels (Low, Medium, High, Critical).
- **Vulnerable Population Registry:** Simulates a care list of vulnerable individuals in high-risk zones globally, prioritizing those with mobility or health issues.
- **Interactive Dark Mode Map:** Beautiful Leaflet UI with dark-topo basemap and animated pulsing nodes for extreme risks.

## Setup

1. Install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Fetch Global Data:
   ```bash
   python3 src/data_fetcher.py
   ```
3. Train ML Model:
   ```bash
   python3 src/feature_engineering.py
   python3 src/model_train.py
   ```
4. Start Backend Server:
   ```bash
   python3 api/api_server.py
   ```
5. Deploy Frontend:
   ```bash
   npx wrangler pages deploy frontend --project-name ecopredict
   ```

## License
MIT License.
