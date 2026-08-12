"""
vulnerable_registry.py
======================
模擬關懷名單資料庫的查詢介面。

資料來源：data/care_list/vulnerable_registry.json
（全部為虛構假資料，不含任何真實個資）

提供依縣市、鄉鎮、社區的查詢功能，
以及「需優先通知」的篩選邏輯（獨居 OR 行動不便）。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

REGISTRY_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "care_list" / "vulnerable_registry.json"
)


@dataclass
class VulnerablePerson:
    id: str
    code_name: str
    county: str
    town: str
    community: str
    is_living_alone: bool
    is_mobility_impaired: bool
    age_group: str
    emergency_contact: str
    notes: str

    @property
    def is_priority(self) -> bool:
        """獨居或行動不便者視為優先通知對象。"""
        return self.is_living_alone or self.is_mobility_impaired

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "code_name": self.code_name,
            "county": self.county,
            "town": self.town,
            "community": self.community,
            "is_living_alone": self.is_living_alone,
            "is_mobility_impaired": self.is_mobility_impaired,
            "age_group": self.age_group,
            "emergency_contact": self.emergency_contact,
            "notes": self.notes,
            "is_priority": self.is_priority,
        }


class VulnerableRegistry:
    """關懷名單資料庫查詢介面。"""

    def __init__(self, path: Path = REGISTRY_PATH) -> None:
        if not path.exists():
            raise FileNotFoundError(f"關懷名單不存在：{path}")
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        self._records: list[VulnerablePerson] = [
            VulnerablePerson(**r) for r in raw["records"]
        ]
        logger.info("載入關懷名單：%d 筆", len(self._records))

    def query(
        self,
        county: Optional[str] = None,
        town: Optional[str] = None,
        priority_only: bool = False,
    ) -> list[VulnerablePerson]:
        """
        依縣市、鄉鎮篩選關懷對象。

        Args:
            county:        縣市名稱（None = 不篩選）。
            town:          鄉鎮區名稱（None = 不篩選）。
            priority_only: 若 True，只回傳獨居或行動不便者。

        Returns:
            符合條件的 VulnerablePerson 列表。
        """
        results = self._records
        if county:
            results = [p for p in results if p.county == county]
        if town:
            results = [p for p in results if p.town == town]
        if priority_only:
            results = [p for p in results if p.is_priority]
        return results

    def query_by_station_area(
        self, county: str, town: str
    ) -> list[VulnerablePerson]:
        """依氣象站所在縣市/鄉鎮查詢，回傳同縣市的所有關懷對象。"""
        return self.query(county=county)

    @property
    def all_records(self) -> list[VulnerablePerson]:
        return list(self._records)

    def summary(self) -> dict:
        return {
            "total": len(self._records),
            "living_alone": sum(1 for p in self._records if p.is_living_alone),
            "mobility_impaired": sum(1 for p in self._records if p.is_mobility_impaired),
            "priority": sum(1 for p in self._records if p.is_priority),
            "counties": sorted({p.county for p in self._records}),
        }


if __name__ == "__main__":
    reg = VulnerableRegistry()
    print("【關懷名單統計】")
    s = reg.summary()
    for k, v in s.items():
        print(f"  {k}: {v}")

    print("\n【臺北市查詢結果】")
    for p in reg.query(county="臺北市", priority_only=True):
        print(f"  [{p.id}] {p.code_name} — {p.town} {p.community} | 優先:{p.is_priority}")
