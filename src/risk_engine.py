""" """

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

RiskLevelStr = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]

RISK_LEVELS: list[dict] = [
    {"level": "LOW",      "label": "Low Risk",   "color": "green",  "min": 0.00, "max": 0.20},
    {"level": "MEDIUM",   "label": "Medium Risk",   "color": "yellow", "min": 0.20, "max": 0.50},
    {"level": "HIGH",     "label": "High Risk",   "color": "orange", "min": 0.50, "max": 0.75},
    {"level": "CRITICAL", "label": "Critical Risk", "color": "red",    "min": 0.75, "max": 1.01},
]


@dataclass(frozen=True)
class RiskClassification:
    level: RiskLevelStr
    label: str
    color: str
    score: float


def classify_score(score: float) -> RiskClassification:
    """ """
    if not (0.0 <= score <= 1.0):
        raise ValueError(f"score must be in [0, 1] range, received {score!r}")

    for r in RISK_LEVELS:
        if r["min"] <= score < r["max"]:
            return RiskClassification(
                level=r["level"],
                label=r["label"],
                color=r["color"],
                score=round(score, 4),
            )
    return RiskClassification(level="CRITICAL", label="Critical Risk",
                              color="red", score=1.0)


def classify_dataframe(
    df: pd.DataFrame,
    score_col: str = "risk_score",
) -> pd.DataFrame:
    """ """
    if score_col not in df.columns:
        raise KeyError(f"Column '{score_col}' does not exist in DataFrame.")

    out = df.copy()
    classifications = out[score_col].apply(classify_score)
    out["risk_level"] = classifications.apply(lambda c: c.level)
    out["risk_label"] = classifications.apply(lambda c: c.label)
    out["risk_color"] = classifications.apply(lambda c: c.color)
    return out


def is_alert_threshold(level: RiskLevelStr) -> bool:
    """ """
    return level in ("HIGH", "CRITICAL")


if __name__ == "__main__":
    tests = [0.0, 0.1, 0.2, 0.35, 0.5, 0.6, 0.75, 0.9, 1.0]
    print(f"{'Score':>6}  {'Level':<10}  {'Label':<6}  {'Alert?'}")
    print("-" * 42)
    for s in tests:
        c = classify_score(s)
        print(f"{s:>6.2f}  {c.level:<10}  {c.label:<6}  {is_alert_threshold(c.level)}")
