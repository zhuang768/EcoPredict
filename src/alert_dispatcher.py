"""
alert_dispatcher.py
===================
告警調度引擎：將風險分類結果對應到關懷名單，產生結構化告警報告。

流程：
  1. 載入最佳模型 → 對 features.csv 進行預測
  2. risk_engine 將機率轉換成風險等級
  3. 篩選達到「高風險（HIGH）」以上的測站
  4. 依縣市查詢 VulnerableRegistry，找出受影響的關懷對象
  5. 輸出告警報告（JSON 至 reports/alert_report.json + console 摘要）

正式部署說明：
  本模組負責產生結構化告警資料。實際通知（LINE Bot、簡訊、電話）
  由下游整合層處理，只需訂閱 alert_report.json 或呼叫 dispatch() 函式。
  串接範例：LINE Messaging API、Twilio SMS、FCM Push Notification。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from risk_engine import classify_dataframe, is_alert_threshold
from vulnerable_registry import VulnerableRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

BASE          = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE / "data" / "processed"
MODELS_DIR    = BASE / "models"
REPORTS_DIR   = BASE / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "rain_1h", "rain_3h", "rain_6h", "rain_12h", "rain_24h",
    "rain_2days", "rain_3days", "rain_intensity_max",
    "forecast_pop_avg_24h", "forecast_pop_avg_48h",
    "is_typhoon_period", "altitude",
]


# ── 核心調度函式 ──────────────────────────────────────────────────────────────
def dispatch(
    features_path: Path | None = None,
    model_path: Path | None = None,
    registry: VulnerableRegistry | None = None,
    alert_threshold: str = "HIGH",
) -> dict:
    """
    執行完整告警調度流程。

    Args:
        features_path:   features.csv 路徑（None → 使用預設路徑）。
        model_path:      best_model.pkl 路徑（None → 使用預設路徑）。
        registry:        VulnerableRegistry 實例（None → 自動載入）。
        alert_threshold: 觸發告警的最低風險等級（"HIGH" 或 "CRITICAL"）。

    Returns:
        結構化告警報告 dict（同時寫入 reports/alert_report.json）。
    """
    features_path = features_path or PROCESSED_DIR / "features.csv"
    model_path    = model_path    or MODELS_DIR / "best_model.pkl"
    registry      = registry      or VulnerableRegistry()

    # 1. 載入模型與特徵
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型：{model_path}，請先執行 model_train.py。")
    bundle     = joblib.load(model_path)
    model      = bundle["model"]
    model_name = bundle["model_name"]
    logger.info("載入模型：%s", model_name)

    if not features_path.exists():
        raise FileNotFoundError(f"找不到特徵表：{features_path}，請先執行 feature_engineering.py。")
    df = pd.read_csv(features_path)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)
    logger.info("載入特徵表：%d 筆", len(df))

    # 2. 模型預測
    X = df[FEATURE_COLS]
    df["risk_score"] = (
        model.predict_proba(X)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X)
    )

    # 3. 風險分類
    df = classify_dataframe(df, score_col="risk_score")
    logger.info(
        "風險分布 — LOW:%d  MEDIUM:%d  HIGH:%d  CRITICAL:%d",
        (df["risk_level"] == "LOW").sum(),
        (df["risk_level"] == "MEDIUM").sum(),
        (df["risk_level"] == "HIGH").sum(),
        (df["risk_level"] == "CRITICAL").sum(),
    )

    # 4. 篩選達到告警門檻的站點
    alert_stations = df[df["risk_level"].apply(is_alert_threshold)].copy()
    logger.info("觸發告警站點：%d 個（門檻：%s+）", len(alert_stations), alert_threshold)

    # 5. 依縣市彙整告警 + 查詢關懷對象
    alert_entries: list[dict] = []
    counties_alerted = alert_stations["county"].dropna().unique()

    for county in sorted(counties_alerted):
        county_stations = alert_stations[alert_stations["county"] == county]
        max_level = "HIGH"
        if (county_stations["risk_level"] == "CRITICAL").any():
            max_level = "CRITICAL"

        # 查詢關懷名單（同縣市）
        persons = registry.query(county=county)
        priority_persons = [p for p in persons if p.is_priority]

        station_details = [
            {
                "station_id":   row["station_id"],
                "station_name": row["station_name"],
                "town":         row.get("town"),
                "risk_level":   row["risk_level"],
                "risk_label":   row["risk_label"],
                "risk_score":   round(float(row["risk_score"]), 4),
                "rain_24h_mm":  float(row.get("rain_24h", 0)),
                "rain_3days_mm":float(row.get("rain_3days", 0)),
                "forecast_pop_avg_24h": row.get("forecast_pop_avg_24h"),
            }
            for _, row in county_stations.iterrows()
        ]

        entry = {
            "county":          county,
            "max_risk_level":  max_level,
            "alert_stations":  len(county_stations),
            "station_details": station_details,
            "vulnerable_persons": {
                "total_in_county": len(persons),
                "priority_count":  len(priority_persons),
                "priority_list":   [p.to_dict() for p in priority_persons],
                "all_list":        [p.to_dict() for p in persons],
            },
            "suggested_actions": _get_actions(max_level),
        }
        alert_entries.append(entry)

    # 6. 組裝最終報告
    report = {
        "report_generated_at": datetime.now(timezone.utc).isoformat(),
        "model_used": model_name,
        "alert_threshold": alert_threshold,
        "summary": {
            "total_stations_analyzed": len(df),
            "stations_triggered_alert": len(alert_stations),
            "counties_alerted": len(alert_entries),
            "total_vulnerable_persons_affected": sum(
                e["vulnerable_persons"]["total_in_county"] for e in alert_entries
            ),
            "priority_persons_to_notify": sum(
                e["vulnerable_persons"]["priority_count"] for e in alert_entries
            ),
        },
        "risk_distribution": {
            "LOW":      int((df["risk_level"] == "LOW").sum()),
            "MEDIUM":   int((df["risk_level"] == "MEDIUM").sum()),
            "HIGH":     int((df["risk_level"] == "HIGH").sum()),
            "CRITICAL": int((df["risk_level"] == "CRITICAL").sum()),
        },
        "alerts": alert_entries,
        "deployment_note": (
            "本報告由 EcoPredict 自動產生。"
            "正式部署時，此 JSON 將作為 LINE Bot Webhook / Twilio SMS / FCM 的輸入，"
            "自動通知各縣市社工師與緊急聯絡人。"
        ),
    }

    # 7. 寫入檔案
    out_path = REPORTS_DIR / "alert_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("告警報告已儲存 → %s", out_path)

    return report


def _get_actions(level: str) -> list[str]:
    """依風險等級回傳建議行動清單。"""
    base = [
        "通知里長辦公室與村里關懷據點",
        "更新社區防災告示板",
    ]
    if level == "HIGH":
        return base + [
            "電話通知優先關懷對象（獨居長者、行動不便者）",
            "確認避難所開設狀態",
            "準備緊急物資（飲水、急救包）",
        ]
    if level == "CRITICAL":
        return base + [
            "立即啟動撤離程序，安排接送服務",
            "通知消防局、警察局待命",
            "開設緊急避難所並廣播公告",
            "對需氧氣機等特殊需求者優先安置",
            "通知鄉鎮市公所啟動一級應變",
        ]
    return base


# ── 主程式（console 輸出） ────────────────────────────────────────────────────
def _print_report(report: dict) -> None:
    s = report["summary"]
    sep = "=" * 65

    print(f"\n{sep}")
    print(f"  EcoPredict 告警報告  |  {report['report_generated_at'][:19]}Z")
    print(sep)
    print(f"  分析站點數    ：{s['total_stations_analyzed']}")
    print(f"  觸發告警站點  ：{s['stations_triggered_alert']}")
    print(f"  受影響縣市    ：{s['counties_alerted']}")
    print(f"  受影響關懷對象：{s['total_vulnerable_persons_affected']} 人")
    print(f"  ⚠ 優先通知人數 ：{s['priority_persons_to_notify']} 人（獨居/行動不便）")

    dist = report["risk_distribution"]
    print(f"\n  風險分布：LOW={dist['LOW']} | MEDIUM={dist['MEDIUM']} | "
          f"HIGH={dist['HIGH']} | CRITICAL={dist['CRITICAL']}")

    for alert in report["alerts"]:
        print(f"\n{'─'*65}")
        level_icon = "🔴" if alert["max_risk_level"] == "CRITICAL" else "🟠"
        print(f"  {level_icon} {alert['county']} — {alert['max_risk_level']} "
              f"（{alert['alert_stations']} 站觸發）")

        for st in alert["station_details"]:
            print(f"    • {st['station_name']}({st['town']}) "
                  f"score={st['risk_score']:.3f} 24h={st['rain_24h_mm']}mm "
                  f"3d={st['rain_3days_mm']}mm")

        vp = alert["vulnerable_persons"]
        print(f"\n  關懷對象（同縣市）：{vp['total_in_county']} 人，"
              f"優先通知 {vp['priority_count']} 人")
        for p in vp["priority_list"]:
            flags = []
            if p["is_living_alone"]:    flags.append("獨居")
            if p["is_mobility_impaired"]: flags.append("行動不便")
            print(f"    ★ [{p['id']}] {p['code_name']} — {p['town']} {p['community']} "
                  f"[{', '.join(flags)}]")
            print(f"      緊急聯絡：{p['emergency_contact']}")
            if p["notes"]:
                print(f"      備注：{p['notes']}")

        print(f"\n  建議行動：")
        for action in alert["suggested_actions"]:
            print(f"    → {action}")

    print(f"\n{sep}")
    print(f"  📄 完整報告已儲存 → reports/alert_report.json")
    print(f"  ℹ  正式部署時此 JSON 將作為 LINE Bot / SMS 的通知輸入")
    print(sep)


if __name__ == "__main__":
    report = dispatch()
    _print_report(report)
