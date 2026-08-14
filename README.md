# EcoPredict 🌊

![EcoPredict](https://img.shields.io/badge/Status-Hackathon_Ready-success)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111.1-009688)

**Community Flood & Debris Flow Risk Monitor** — A predictive dispatch system designed for local crisis managers and social workers. Built for **Reverie Hacks 2026**.

👉 **[Live Demo](https://ecopredict.pages.dev)**

---

## 🎯 The Problem: Data without Action
When a severe typhoon approaches Taiwan, the Central Weather Administration (CWA) provides incredibly detailed meteorological data—radar maps, rainfall accumulations, and wind speeds. However, for **first responders, social workers, and local village chiefs**, raw weather data is not enough.

They don't just need to know *where* it's raining; they need to know **WHO** is in danger and **WHAT** to do about it. Currently, the process of matching extreme weather alerts with the registry of vulnerable populations is dangerously manual and slow.

## 💡 The Solution: EcoPredict
EcoPredict is not another weather app for the general public. It is a **predictive dispatch system for community disaster management**.

While standard weather apps stop at delivering the forecast, EcoPredict bridges the gap between **Meteorology and Social Care**. We designed it specifically for local crisis managers:

### 1. AI-Driven Risk Scoring
Instead of just looking at raw rainfall (mm), EcoPredict uses a Machine Learning model (Logistic Regression) to calculate a comprehensive "Risk Score" for every local station. It analyzes current 1h intensity, 3-day accumulated saturation, and 48h Probability of Precipitation (PoP) to predict the likelihood of localized flooding and debris flows.

### 2. The "Care Registry" Action Panel
When a community hits `EXTREME` risk, EcoPredict doesn't just flash a red light. It directly integrates with a **Vulnerable Care Registry**. Social workers can immediately see how many vulnerable individuals are in the danger zone, view their specific needs, and dispatch automated Evacuation SMS warnings with a single click.

### 3. Hyper-Localized Dispatch Log UX
The UI is stripped of unnecessary B2C weather animations. It uses a dispatch-log format with WMO (World Meteorological Organization) standard classifications (`MINOR`, `MODERATE`, `SEVERE`, `EXTREME`), designed for rapid decision-making in high-stress environments.

---

## 🚀 How to Demo
When presenting or judging EcoPredict, follow these steps to see its full capability:

1. **LIVE Mode**: Open the app to see real-time data from CWA. The backend automatically calculates risk scores for over 100 stations across Taiwan.
2. **SIMULATION Mode (The Typhoon Gaemi Scenario)**: Click `[ SIMULATION: OFF ]` in the top header. The system will instantly load our simulation of the 2024 Gaemi Typhoon peak. Watch the map light up with `EXTREME` and `SEVERE` alerts.
3. **Dispatch Action**: In the left sidebar's Live Alerts, click `[ VIEW CARE REGISTRY ]` on any red station. The Care Registry modal will open. Click `DISPATCH EVACUATION SMS` to see the automated response system in action.
4. **Radar Overlay**: Click `[ RADAR: OFF ]` in the header to overlay real-time rainfall radar data directly onto the map.

---

## 🛠️ Tech Stack & Architecture
EcoPredict uses a decoupled architecture for maximum scalability and reliability during a crisis.

* **Frontend**: HTML5, Vanilla JS, CSS3, Leaflet.js, Chart.js. Hosted on **Cloudflare Pages** for global edge caching and instant static delivery.
* **Backend**: **FastAPI** (Python). Hosted on **Render**. Provides endpoints for the risk map GeoJSON and community details.
* **Machine Learning**: `scikit-learn` (Logistic Regression), `pandas`, `joblib`.
* **Data Sources**: Central Weather Administration (CWA) Open API (Rainfall, Forecasts).

### Repository Structure
* `api/`: FastAPI server endpoints and configuration.
* `src/`: Data fetching, feature engineering, ML model training, and business logic.
* `frontend/`: The static Cloudflare Pages website.
* `models/`: Pickled ML models and scalers.
* `data/`: Raw and processed dataset storage.
* `.github/workflows/`: CI/CD and Keep-Alive cron jobs.

---

## 💻 Local Development

### Backend Setup
```bash
# 1. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the FastAPI server
uvicorn api.api_server:app --reload
```

### Frontend Setup
Since the frontend is completely static, you can serve it with any local HTTP server:
```bash
cd frontend
python3 -m http.server 8080
```
Open `http://localhost:8080` in your browser. The frontend will automatically detect `localhost` and point to the local `8000` API port.
