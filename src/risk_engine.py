"""
risk_engine.py
==============
將模型輸出的機率值（0–1）轉換成標準化風險等級。

風險分級標準（對應 NCDR 災害警戒色碼）：
  [0.00 – 0.20) → 低風險   (綠) LOW
  [0.20 – 0.50) → 中風險   (黃) MEDIUM
  [0.50 – 0.75) → 高風險   (橙) HIGH
  [0.75 – 1.00] → 極高風險 (紅) CRITICAL
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

# ── 型別定義 ────────────────────────────────────────────────────────────────────
RiskLevelStr = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]

RISK_LEVELS: list[dict] = [
    {"level": "LOW",      "label": "低風險",   "color": "green",  "min": 0.00, "max": 0.20},
    {"level": "MEDIUM",   "label": "中風險",   "color": "yellow", "min": 0.20, "max": 0.50},
    {"level": "HIGH",     "label": "高風險",   "color": "orange", "min": 0.50, "max": 0.75},
    {"level": "CRITICAL", "label": "極高風險", "color": "red",    "min": 0.75, "max": 1.01},
]


@dataclass(frozen=True)
class RiskClassification:
    level: RiskLevelStr
    label: str
    color: str
    score: float


def classify_score(score: float) -> RiskClassification:
    """
    將單一機率值轉換成 RiskClassification。

    Args:
        score: 模型輸出機率值，範圍 [0, 1]。

    Returns:
        RiskClassification dataclass。

    Raises:
        ValueError: score 超出 [0, 1] 範圍。
    """
    if not (0.0 <= score <= 1.0):
        raise ValueError(f"score 必須在 [0, 1] 範圍內，收到 {score!r}")

    for r in RISK_LEVELS:
        if r["min"] <= score < r["max"]:
            return RiskClassification(
                level=r["level"],
                label=r["label"],
                color=r["color"],
                score=round(score, 4),
            )
    # 邊界值 1.0 → CRITICAL
    return RiskClassification(level="CRITICAL", label="極高風險",
                              color="red", score=1.0)


def classify_dataframe(
    df: pd.DataFrame,
    score_col: str = "risk_score",
) -> pd.DataFrame:
    """
    對 DataFrame 中的機率欄位批次分類，加入 risk_level / risk_label / risk_color 欄位。

    Args:
        df:        包含 score_col 欄位的 DataFrame（不修改原始物件）。
        score_col: 機率值欄位名稱，預設為 "risk_score"。

    Returns:
        新 DataFrame（含原有欄位 + risk_level / risk_label / risk_color）。
    """
    if score_col not in df.columns:
        raise KeyError(f"欄位 '{score_col}' 不存在於 DataFrame 中。")

    out = df.copy()
    classifications = out[score_col].apply(classify_score)
    out["risk_level"] = classifications.apply(lambda c: c.level)
    out["risk_label"] = classifications.apply(lambda c: c.label)
    out["risk_color"] = classifications.apply(lambda c: c.color)
    return out


def is_alert_threshold(level: RiskLevelStr) -> bool:
    """回傳該風險等級是否達到需要觸發告警的門檻（HIGH 或 CRITICAL）。"""
    return level in ("HIGH", "CRITICAL")


if __name__ == "__main__":
    # 簡易自我測試
    tests = [0.0, 0.1, 0.2, 0.35, 0.5, 0.6, 0.75, 0.9, 1.0]
    print(f"{'Score':>6}  {'Level':<10}  {'Label':<6}  {'Alert?'}")
    print("-" * 42)
    for s in tests:
        c = classify_score(s)
        print(f"{s:>6.2f}  {c.level:<10}  {c.label:<6}  {is_alert_threshold(c.level)}")
