import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# We use global features now
FEATURE_COLS = [
    "rain_1h",
    "rain_24h",
    "rain_3days",
    "forecast_pop_avg_24h",
    "forecast_pop_avg_48h",
]

def train_models():
    features_path = PROCESSED_DIR / "features.csv"
    if not features_path.exists():
        raise FileNotFoundError("features.csv does not exist, run feature_engineering.py first.")

    df = pd.read_csv(features_path)
    logger.info("Loaded features.csv: %d rows", len(df))

    # In a real scenario we split by time. Here we split randomly for demo.
    X = df[FEATURE_COLS]
    y = df["label"]

    # Since it's a small dataset (30 cities), and potentially no positive samples depending on weather:
    if y.sum() == 0:
        logger.warning("No positive samples! Injecting synthetic samples to prevent training crash.")
        # Inject synthetic positive sample
        syn_X = pd.DataFrame([{
            "rain_1h": 50, "rain_24h": 200, "rain_3days": 300, 
            "forecast_pop_avg_24h": 90, "forecast_pop_avg_48h": 90
        }] * 5) # Create 5 positive samples
        syn_y = pd.Series([1] * 5)
        X = pd.concat([X, syn_X], ignore_index=True)
        y = pd.concat([y, syn_y], ignore_index=True)
        
        syn_X2 = pd.DataFrame([{
            "rain_1h": 0, "rain_24h": 0, "rain_3days": 0, 
            "forecast_pop_avg_24h": 0, "forecast_pop_avg_48h": 0
        }] * 5) # Create 5 negative samples
        syn_y2 = pd.Series([0] * 5)
        X = pd.concat([X, syn_X2], ignore_index=True)
        y = pd.concat([y, syn_y2], ignore_index=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(class_weight="balanced", max_depth=5),
        "RandomForest": RandomForestClassifier(n_estimators=100, class_weight="balanced")
    }

    results = []
    best_auc = -1
    best_name = None
    best_model = None

    for name, clf in models.items():
        logger.info("Training %s...", name)
        clf.fit(X_train, y_train)
        
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else preds
        
        auc = roc_auc_score(y_test, probs) if len(y_test.unique()) > 1 else 0.5
        
        results.append({"model": name, "auc": auc})
        
        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_model = clf

    logger.info("Best model: %s (AUC=%.4f)", best_name, best_auc)

    model_out = MODELS_DIR / "best_model.pkl"
    with open(model_out, "wb") as f:
        pickle.dump({
            "model_name": best_name,
            "features": FEATURE_COLS,
            "model": best_model
        }, f)
    
    logger.info("Best model saved -> %s", model_out)

if __name__ == "__main__":
    train_models()
