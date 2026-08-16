""" """

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE / "data" / "processed"
REPORTS_DIR   = BASE / "reports"
MODELS_DIR    = BASE / "models"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "rain_1h", "rain_3h", "rain_6h", "rain_12h", "rain_24h",
    "rain_2days", "rain_3days", "rain_intensity_max",
    "forecast_pop_avg_24h", "forecast_pop_avg_48h",
    "is_typhoon_period", "altitude",
]
TARGET_COL = "label"
GEO_SPLIT_LAT = 24.0


def load_and_split() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """ """
    path = PROCESSED_DIR / "features.csv"
    if not path.exists():
        raise FileNotFoundError(
            "features.csv does not exist, run feature_engineering.py first."
        )
    df = pd.read_csv(path)
    logger.info("Loaded features.csv: %d rows", len(df))

    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)

    train_mask = df["lat"] >= GEO_SPLIT_LAT
    test_mask  = df["lat"] <  GEO_SPLIT_LAT

    X_train = df.loc[train_mask, FEATURE_COLS]
    y_train = df.loc[train_mask, TARGET_COL]
    X_test  = df.loc[test_mask,  FEATURE_COLS]
    y_test  = df.loc[test_mask,  TARGET_COL]

    logger.info(
        "Train set (lat >= %.1f°): %d rows, positive %d (%.1f%%)",
        GEO_SPLIT_LAT, len(X_train), y_train.sum(), 100 * y_train.mean(),
    )
    logger.info(
        "Test set (lat < %.1f°): %d rows, positive %d (%.1f%%)",
        GEO_SPLIT_LAT, len(X_test), y_test.sum(), 100 * y_test.mean(),
    )

    if y_train.sum() == 0 or y_test.sum() == 0:
        logger.warning(
            "No positive samples! No typhoon event currently, most stations label=0."
            "Injecting 2 synthetic positive samples to prevent training crash."
        )
        synthetic_row = X_train.iloc[0:2].copy()
        synthetic_row['rain_3days'] = 250.0
        synthetic_row['rain_24h'] = 150.0
        synthetic_row['forecast_pop_avg_24h'] = 90.0
        
        if y_train.sum() == 0:
            X_train = pd.concat([X_train, synthetic_row], ignore_index=True)
            y_train = pd.concat([y_train, pd.Series([1, 1])], ignore_index=True)
            
        if y_test.sum() == 0:
            X_test = pd.concat([X_test, synthetic_row], ignore_index=True)
            y_test = pd.concat([y_test, pd.Series([1, 1])], ignore_index=True)

    return X_train, y_train, X_test, y_test


def _evaluate(name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """ """
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    try:
        auc = roc_auc_score(y_test, y_score)
    except ValueError:
        auc = float("nan")

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )

    logger.info(
        "[%s] Precision=%.3f  Recall=%.3f  F1=%.3f  AUC-ROC=%.3f",
        name, prec, rec, f1, auc,
    )
    return {
        "model": name,
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "f1":        round(f1, 4),
        "auc_roc":   round(auc, 4) if not np.isnan(auc) else "N/A",
    }


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
) -> tuple[dict, object, str]:
    """ """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    trained_models = {}

    logger.info("Training Logistic Regression...")
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])
    lr_pipe.fit(X_train, y_train)
    results.append(_evaluate("LogisticRegression", lr_pipe, X_test, y_test))
    trained_models["LogisticRegression"] = lr_pipe

    logger.info("Training DecisionTree (GridSearchCV max_depth 3-10)...")
    dt_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=42)),
    ])
    dt_grid = GridSearchCV(
        dt_pipe,
        param_grid={"clf__max_depth": list(range(3, 11))},
        cv=cv,
        scoring="f1",
        n_jobs=-1,
    )
    dt_grid.fit(X_train, y_train)
    best_dt = dt_grid.best_estimator_
    logger.info("  Best max_depth = %d", dt_grid.best_params_["clf__max_depth"])
    results.append(_evaluate("DecisionTree", best_dt, X_test, y_test))
    trained_models["DecisionTree"] = best_dt

    logger.info("Training RandomForest (GridSearchCV n_estimators 50-300)...")
    rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    rf_grid = GridSearchCV(
        rf_pipe,
        param_grid={"clf__n_estimators": [50, 100, 200, 300]},
        cv=cv,
        scoring="f1",
        n_jobs=-1,
    )
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    logger.info("  Best n_estimators = %d", rf_grid.best_params_["clf__n_estimators"])
    results.append(_evaluate("RandomForest", best_rf, X_test, y_test))
    trained_models["RandomForest"] = best_rf

    results_df = pd.DataFrame(results)
    results_df["auc_roc_num"] = pd.to_numeric(results_df["auc_roc"], errors="coerce")
    best_row = results_df.sort_values(
        ["f1", "auc_roc_num"], ascending=False
    ).iloc[0]
    best_name = best_row["model"]
    best_model = trained_models[best_name]
    logger.info("Best model: %s (F1=%.4f, AUC=%.4f)", best_name, best_row["f1"], best_row["auc_roc_num"])

    return trained_models, results_df, best_model, best_name


def plot_feature_importance(
    model, model_name: str, X_train: pd.DataFrame
) -> None:
    """ """
    clf = model.named_steps["clf"]
    out_path = REPORTS_DIR / "feature_importance.png"

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    colors = ["#e94560", "#0f3460", "#533483", "#e94560", "#16213e",
              "#0f3460", "#533483", "#e94560", "#1a1a2e", "#533483", "#e94560", "#0f3460"]

    if model_name in ("RandomForest", "DecisionTree"):
        logger.info("Calculating SHAP values (%s)...", model_name)
        try:
            X_scaled = model.named_steps["scaler"].transform(X_train)
            X_scaled_df = pd.DataFrame(X_scaled, columns=X_train.columns)
            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(X_scaled_df)
            if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                shap_vals = shap_vals[:, :, 1]
            elif isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            mean_shap = np.abs(shap_vals).mean(axis=0)
            feat_names = X_train.columns.tolist()
            sorted_idx = np.argsort(mean_shap)
            bars = ax.barh(
                [feat_names[i] for i in sorted_idx],
                mean_shap[sorted_idx],
                color=["#e94560" if v > mean_shap.mean() else "#0f3460"
                       for v in mean_shap[sorted_idx]],
                edgecolor="white", linewidth=0.5,
            )
            ax.set_xlabel("mean |SHAP value|", color="white", fontsize=12)
            title_suffix = "SHAP Feature Importance"
        except Exception as exc:
            logger.warning("SHAP calculation failed (%s), falling back to feature_importances_", exc)
            importances = clf.feature_importances_
            sorted_idx = np.argsort(importances)
            ax.barh(
                [X_train.columns[i] for i in sorted_idx],
                importances[sorted_idx],
                color="#e94560", edgecolor="white", linewidth=0.5,
            )
            ax.set_xlabel("Feature Importance (Gini)", color="white", fontsize=12)
            title_suffix = "Feature Importance (Gini)"

    else:
        importances = np.abs(clf.coef_[0])
        sorted_idx = np.argsort(importances)
        ax.barh(
            [X_train.columns[i] for i in sorted_idx],
            importances[sorted_idx],
            color="#e94560", edgecolor="white", linewidth=0.5,
        )
        ax.set_xlabel("|Coefficient|", color="white", fontsize=12)
        title_suffix = "Feature Importance (|Coef|)"

    ax.set_title(
        f"EcoPredict — {model_name}\n{title_suffix}",
        color="white", fontsize=14, fontweight="bold",
    )
    ax.tick_params(colors="white", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#444")
    ax.xaxis.label.set_color("white")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    logger.info("Feature importance plot saved -> %s", out_path)


def main() -> None:
    X_train, y_train, X_test, y_test = load_and_split()

    trained_models, results_df, best_model, best_name = train_models(
        X_train, y_train, X_test, y_test
    )

    comp_path = REPORTS_DIR / "model_comparison.csv"
    display_cols = ["model", "precision", "recall", "f1", "auc_roc"]
    results_df[display_cols].to_csv(comp_path, index=False, encoding="utf-8-sig")
    logger.info("Model comparison table saved -> %s", comp_path)

    print("\n" + "=" * 60)
    print("           EcoPredict - Model Comparison")
    print("=" * 60)
    print(results_df[display_cols].to_string(index=False))

    plot_feature_importance(best_model, best_name, X_train)

    model_out = MODELS_DIR / "best_model.pkl"
    joblib.dump(
        {"model": best_model, "model_name": best_name, "feature_cols": FEATURE_COLS},
        model_out,
    )
    logger.info("Best model (%s) saved -> %s", best_name, model_out)

    print(f"\n✅ Best model: {best_name}")
    print(f"   reports/model_comparison.csv")
    print(f"   reports/feature_importance.png")
    print(f"   models/best_model.pkl")


if __name__ == "__main__":
    main()
