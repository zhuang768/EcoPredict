""" """

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

BASE         = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE / "data" / "processed"
MODELS_DIR   = BASE / "models"
REPORTS_DIR  = BASE / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TYPHOON_EVENT = {
    "name":       "DOLPHIN",
    "warning_issued":  "2026-08-08T23:30:00+08:00",
    "warning_lifted":  "2026-08-09T23:40:00+08:00",
    "simulate_at":     "T-48h (48 hours before event)",
}

GROUND_TRUTH_THRESHOLD_MM = 100.0


def load_model_and_features() -> tuple[object, list[str], pd.DataFrame]:
    """ """
    model_path = MODELS_DIR / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError("Cannot find models/best_model.pkl, run model_train.py first.")

    bundle = joblib.load(model_path)
    model       = bundle["model"]
    model_name  = bundle["model_name"]
    feature_cols = bundle["feature_cols"]
    logger.info("Loading model: %s", model_name)

    features_path = PROCESSED_DIR / "features.csv"
    if not features_path.exists():
        raise FileNotFoundError("Cannot find features.csv, run feature_engineering.py first.")
    df = pd.read_csv(features_path)

    return model, model_name, feature_cols, df


def simulate_pre_event_features(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """ """
    extra_cols = ["station_name", "county", "lat", "lon"]
    all_cols = list(dict.fromkeys(feature_cols + extra_cols))
    if "rain_3days" not in all_cols:
        all_cols.append("rain_3days")

    sim = df[all_cols].copy()

    y_true = (sim["rain_3days"] >= GROUND_TRUTH_THRESHOLD_MM).astype(int)

    for col in feature_cols:
        if col in sim.columns:
            sim[col] = sim[col].fillna(0.0)

    for col in ["rain_1h", "rain_3h", "rain_6h", "rain_12h", "rain_24h",
                "rain_intensity_max"]:
        if col in sim.columns:
            sim[col] = 0.0

    sim["forecast_pop_avg_24h"] = 70.0
    sim["forecast_pop_avg_48h"] = 65.0
    sim["is_typhoon_period"] = 1

    logger.info(
        "Simulated features built: %d rows (Ground truth high risk: %d / %.1f%%)",
        len(sim), y_true.sum(), 100 * y_true.mean(),
    )

    X_sim = sim[feature_cols].copy()
    return X_sim, y_true, sim


def run_backtest() -> None:
    model, model_name, feature_cols, df = load_model_and_features()
    X_sim, y_true, sim_df = simulate_pre_event_features(df, feature_cols)

    y_pred = model.predict(X_sim)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_sim)[:, 1]
    else:
        y_score = model.decision_function(X_sim)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")

    sim_df["predicted_risk"] = y_pred
    sim_df["risk_score"]     = y_score
    sim_df["ground_truth"]   = y_true

    high_risk_stations = sim_df[sim_df["predicted_risk"] == 1].sort_values(
        "risk_score", ascending=False
    )

    report_path = REPORTS_DIR / "backtest_report.csv"
    sim_df[["station_name", "county", "lat", "lon", "rain_3days",
            "predicted_risk", "risk_score", "ground_truth"]].to_csv(
        report_path, index=False, encoding="utf-8-sig"
    )

    print("\n" + "=" * 65)
    print("    EcoPredict - Backtest Report")
    print("=" * 65)
    print(f"  Typhoon Event: {TYPHOON_EVENT['name']}")
    print(f"  Warning Issued: {TYPHOON_EVENT['warning_issued']}")
    print(f"  Warning Lifted: {TYPHOON_EVENT['warning_lifted']}")
    print(f"  Simulate At: {TYPHOON_EVENT['simulate_at']}")
    print(f"  Using Model: {model_name}")
    print(f"  Ground Truth Baseline: rain_3days >= {GROUND_TRUTH_THRESHOLD_MM}mm")
    print("-" * 65)
    print(f"  Test stations: {len(y_true)}")
    print(f"  Actual high risk: {y_true.sum()} stations ({100*y_true.mean():.1f}%)")
    print(f"  Model predicted high risk: {y_pred.sum()} stations")
    print("-" * 65)
    print(f"  Precision  : {prec:.4f}")
    print(f"  Recall     : {rec:.4f}")
    print(f"  F1-score   : {f1:.4f}")
    print(f"  AUC-ROC    : {auc:.4f}" if not np.isnan(auc) else "  AUC-ROC   : N/A (Single class)")
    print("-" * 65)

    if len(high_risk_stations) > 0:
        print(f"\n  [Top 10 Predicted High Risk Stations]")
        top10 = high_risk_stations[["station_name", "county", "lat", "lon",
                                     "rain_3days", "risk_score", "ground_truth"]].head(10)
        print(top10.to_string(index=False))
    else:
        print("\n  Model predicted 0 high risk stations (needs typhoon training data)")

    print(f"\n  Detailed backtest report saved -> {report_path}")

    print("\n  [Full Classification Report]")
    print(classification_report(y_true, y_pred,
                                  target_names=["Low Risk", "High Risk"],
                                  zero_division=0))

    print("\n  ⚠️ Methodology Note:")
    print("  This backtest simulates pre-event features with a single time snapshot,")
    print("  evaluating predictive capability using post-typhoon rain_3days as ground truth.")
    print("  Official application should collect historical data across multiple timestamps.")


if __name__ == "__main__":
    run_backtest()
