"""
backtest.py
===========
歷史事件回測：颱風「白海豚」（2026-08-09）

方法：
  CWA O-A0002-001 提供多個累積窗格欄位（Past24hr, Past2days, Past3days）。
  颱風白海豚於 2026-08-09 穿越台灣，警報於 2026-08-09T23:40 解除。

  回測邏輯：
    - rain_3days ≈ 颱風事件期間 + 事後總累積（T-72h 到現在）
    - rain_2days ≈ 颱風主體通過後的 2 天累積（T-48h 到現在）
    - 差值 rain_3days - rain_2days ≈ 第 3 天前（颱風前日的降雨）

  「模擬事件前 48 小時」的方式：
    1. 把 rain_24h 清零（颱風尚未到達時的狀態）
    2. 把 rain_3h, rain_6h, rain_12h 清零
    3. 保留 rain_3days 作為「長期累積背景水分」特徵（土壤飽和度）
    4. forecast_pop_avg_24h 設為 70（颱風前預報通常為 60-90%）
    5. is_typhoon_period 設為 1
    
  預測目標：
    識別「真正在颱風事件中受到高雨量的測站」
    以 rain_3days ≥ 100mm 為「實際高風險」基準（ground truth）
    計算 Precision / Recall / F1
"""

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

# 颱風回測設定
TYPHOON_EVENT = {
    "name":       "白海豚 (DOLPHIN)",
    "warning_issued":  "2026-08-08T23:30:00+08:00",
    "warning_lifted":  "2026-08-09T23:40:00+08:00",
    "simulate_at":     "T-48h（事件發生前 48 小時）",
}

# 「高風險」ground truth 閾值：3天累積雨量 ≥ 100mm
GROUND_TRUTH_THRESHOLD_MM = 100.0


def load_model_and_features() -> tuple[object, list[str], pd.DataFrame]:
    """載入最佳模型與特徵表。"""
    model_path = MODELS_DIR / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError("找不到 models/best_model.pkl，請先執行 model_train.py。")

    bundle = joblib.load(model_path)
    model       = bundle["model"]
    model_name  = bundle["model_name"]
    feature_cols = bundle["feature_cols"]
    logger.info("載入模型：%s", model_name)

    features_path = PROCESSED_DIR / "features.csv"
    if not features_path.exists():
        raise FileNotFoundError("找不到 features.csv，請先執行 feature_engineering.py。")
    df = pd.read_csv(features_path)

    return model, model_name, feature_cols, df


def simulate_pre_event_features(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    從當前快照產生「事件前 48 小時」的特徵矩陣與 ground truth。
    """
    # 取出所需欄位，用 dict.fromkeys 去除重複（保持順序）
    extra_cols = ["station_name", "county", "lat", "lon"]
    all_cols = list(dict.fromkeys(feature_cols + extra_cols))
    # 確保 rain_3days 在列中（用於 ground truth）
    if "rain_3days" not in all_cols:
        all_cols.append("rain_3days")

    sim = df[all_cols].copy()

    # Ground truth 必須在屬性清零前記錄（基於當前真實累積）
    y_true = (sim["rain_3days"] >= GROUND_TRUTH_THRESHOLD_MM).astype(int)

    # 逐欄 fillna（避免重複欄位問題）
    for col in feature_cols:
        if col in sim.columns:
            sim[col] = sim[col].fillna(0.0)

    # 清除短期雨量（模擬颶風尚未抵達）
    for col in ["rain_1h", "rain_3h", "rain_6h", "rain_12h", "rain_24h",
                "rain_intensity_max"]:
        if col in sim.columns:
            sim[col] = 0.0

    # 設定颶風前的高預報 PoP
    sim["forecast_pop_avg_24h"] = 70.0
    sim["forecast_pop_avg_48h"] = 65.0
    sim["is_typhoon_period"] = 1

    logger.info(
        "模擬特徵建立完成：%d 筆（Ground truth 高風険站點：%d / %.1f%%）",
        len(sim), y_true.sum(), 100 * y_true.mean(),
    )

    X_sim = sim[feature_cols].copy()
    return X_sim, y_true, sim


def run_backtest() -> None:
    model, model_name, feature_cols, df = load_model_and_features()
    X_sim, y_true, sim_df = simulate_pre_event_features(df, feature_cols)

    # 預測
    y_pred = model.predict(X_sim)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_sim)[:, 1]
    else:
        y_score = model.decision_function(X_sim)

    # 評估指標
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")

    # 找出模型預測為高風險的站點
    sim_df["predicted_risk"] = y_pred
    sim_df["risk_score"]     = y_score
    sim_df["ground_truth"]   = y_true

    high_risk_stations = sim_df[sim_df["predicted_risk"] == 1].sort_values(
        "risk_score", ascending=False
    )

    # 儲存回測報告
    report_path = REPORTS_DIR / "backtest_report.csv"
    sim_df[["station_name", "county", "lat", "lon", "rain_3days",
            "predicted_risk", "risk_score", "ground_truth"]].to_csv(
        report_path, index=False, encoding="utf-8-sig"
    )

    # 輸出
    print("\n" + "=" * 65)
    print("    EcoPredict — 回測報告")
    print("=" * 65)
    print(f"  颱風事件  ：{TYPHOON_EVENT['name']}")
    print(f"  警報發布  ：{TYPHOON_EVENT['warning_issued']}")
    print(f"  警報解除  ：{TYPHOON_EVENT['warning_lifted']}")
    print(f"  模擬時間點：{TYPHOON_EVENT['simulate_at']}")
    print(f"  使用模型  ：{model_name}")
    print(f"  Ground Truth 基準：rain_3days ≥ {GROUND_TRUTH_THRESHOLD_MM}mm")
    print("-" * 65)
    print(f"  測試站數  ：{len(y_true)}")
    print(f"  實際高風險：{y_true.sum()} 站（{100*y_true.mean():.1f}%）")
    print(f"  模型預測高風險：{y_pred.sum()} 站")
    print("-" * 65)
    print(f"  Precision  : {prec:.4f}")
    print(f"  Recall     : {rec:.4f}")
    print(f"  F1-score   : {f1:.4f}")
    print(f"  AUC-ROC    : {auc:.4f}" if not np.isnan(auc) else "  AUC-ROC   : N/A（單一類別）")
    print("-" * 65)

    if len(high_risk_stations) > 0:
        print(f"\n  【預測高風險站點 TOP 10】")
        top10 = high_risk_stations[["station_name", "county", "lat", "lon",
                                     "rain_3days", "risk_score", "ground_truth"]].head(10)
        print(top10.to_string(index=False))
    else:
        print("\n  模型未預測任何高風險站點（模型可能需要颱風期間的真實訓練資料）")

    print(f"\n  回測詳細報告已儲存 → {report_path}")

    # 打印分類報告
    print("\n  【完整分類報告】")
    print(classification_report(y_true, y_pred,
                                  target_names=["低風險", "高風險"],
                                  zero_division=0))

    print("\n  ⚠️  方法論說明：")
    print("  本回測以單一時間快照模擬「事件前」特徵，清零短期雨量後，")
    print("  以颱風後的 rain_3days 實測值作為 ground truth 評估預測能力。")
    print("  正式應用應蒐集多個時間點的歷史資料建立真正的時間序列訓練集。")


if __name__ == "__main__":
    run_backtest()
