""" """

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


def dispatch(
    features_path: Path | None = None,
    model_path: Path | None = None,
    registry: VulnerableRegistry | None = None,
    alert_threshold: str = "HIGH",
) -> dict:
    """ """
    features_path = features_path or PROCESSED_DIR / "features.csv"
    model_path    = model_path    or MODELS_DIR / "best_model.pkl"
    registry      = registry      or VulnerableRegistry()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}, run model_train.py first.")
    bundle     = joblib.load(model_path)
    model      = bundle["model"]
    model_name = bundle["model_name"]
    logger.info("Loading model: %s", model_name)

    if not features_path.exists():
        raise FileNotFoundError(f"Feature table not found: {features_path}, run feature_engineering.py first.")
    df = pd.read_csv(features_path)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)
    logger.info("Loaded feature table: %d rows", len(df))

    X = df[FEATURE_COLS]
    df["risk_score"] = (
        model.predict_proba(X)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X)
    )

    df = classify_dataframe(df, score_col="risk_score")
    logger.info(
        "Risk Dist - LOW:%d MEDIUM:%d HIGH:%d CRITICAL:%d",
        (df["risk_level"] == "LOW").sum(),
        (df["risk_level"] == "MEDIUM").sum(),
        (df["risk_level"] == "HIGH").sum(),
        (df["risk_level"] == "CRITICAL").sum(),
    )

    alert_stations = df[df["risk_level"].apply(is_alert_threshold)].copy()
    logger.info("Triggered alert stations: %d (threshold: %s+)", len(alert_stations), alert_threshold)

    alert_entries: list[dict] = []
    counties_alerted = alert_stations["county"].dropna().unique()

    for county in sorted(counties_alerted):
        county_stations = alert_stations[alert_stations["county"] == county]
        max_level = "HIGH"
        if (county_stations["risk_level"] == "CRITICAL").any():
            max_level = "CRITICAL"

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
            "This report is auto-generated by EcoPredict."
            "In production, this JSON will be used for LINE Bot/Twilio SMS/FCM,"
            "automatically notifying social workers and emergency contacts."
        ),
    }

    out_path = REPORTS_DIR / "alert_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("Alert report saved -> %s", out_path)

    return report


def _get_actions(level: str) -> list[str]:
    """ """
    base = [
        "Notify village chief and care centers",
        "Update community disaster prevention board",
    ]
    if level == "HIGH":
        return base + [
            "Call priority vulnerable targets (elderly/disabled)",
            "Confirm shelter status",
            "Prepare emergency supplies (water, first aid)",
        ]
    if level == "CRITICAL":
        return base + [
            "Initiate immediate evacuation, arrange transport",
            "Notify fire/police depts to standby",
            "Open emergency shelters and broadcast",
            "Prioritize people needing oxygen machines",
            "Notify township office to start level 1 response",
        ]
    return base


def _print_report(report: dict) -> None:
    s = report["summary"]
    sep = "=" * 65

    print(f"\n{sep}")
    print(f"  EcoPredict Alert Report | {report['report_generated_at'][:19]}Z")
    print(sep)
    print(f"  Analyzed Stations: {s['total_stations_analyzed']}")
    print(f"  Triggered Alerts: {s['stations_triggered_alert']}")
    print(f"  Affected Counties: {s['counties_alerted']}")
    print(f"  Affected Vulnerable Persons: {s['total_vulnerable_persons_affected']}")
    print(f"  ⚠ Priority Notifications: {s['priority_persons_to_notify']} (Living alone/Disabled)")

    dist = report["risk_distribution"]
    print(f"\n  Risk Distribution: LOW={dist['LOW']} | MEDIUM={dist['MEDIUM']} | "
          f"HIGH={dist['HIGH']} | CRITICAL={dist['CRITICAL']}")

    for alert in report["alerts"]:
        print(f"\n{'─'*65}")
        level_icon = "🔴" if alert["max_risk_level"] == "CRITICAL" else "🟠"
        print(f"  {level_icon} {alert['county']} — {alert['max_risk_level']} "
              f"({alert['alert_stations']} stations triggered)")

        for st in alert["station_details"]:
            print(f"    • {st['station_name']}({st['town']}) "
                  f"score={st['risk_score']:.3f} 24h={st['rain_24h_mm']}mm "
                  f"3d={st['rain_3days_mm']}mm")

        vp = alert["vulnerable_persons"]
        print(f"\n  Vulnerable Persons (same county): {vp['total_in_county']}, "
              f"Priority notify {vp['priority_count']} people")
        for p in vp["priority_list"]:
            flags = []
            if p["is_living_alone"]:    flags.append("Living alone")
            if p["is_mobility_impaired"]: flags.append("Mobility impaired")
            print(f"    ★ [{p['id']}] {p['code_name']} — {p['town']} {p['community']} "
                  f"[{', '.join(flags)}]")
            print(f"      Emergency Contact: {p['emergency_contact']}")
            if p["notes"]:
                print(f"      Notes: {p['notes']}")

        print(f"\n  Recommended Actions:")
        for action in alert["suggested_actions"]:
            print(f"    → {action}")

    print(f"\n{sep}")
    print(f"  📄 Full report saved -> reports/alert_report.json")
    print(f"  ℹ In production, this JSON will feed into LINE Bot/SMS notifications")
    print(sep)


if __name__ == "__main__":
    report = dispatch()
    _print_report(report)
