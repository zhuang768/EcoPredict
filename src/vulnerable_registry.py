""" """

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
        """ """
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
    """ """

    def __init__(self, path: Path = REGISTRY_PATH) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Care registry not found: {path}")
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        self._records: list[VulnerablePerson] = [
            VulnerablePerson(**r) for r in raw["records"]
        ]
        logger.info("Loaded care registry: %d records", len(self._records))

    def query(
        self,
        county: Optional[str] = None,
        town: Optional[str] = None,
        priority_only: bool = False,
    ) -> list[VulnerablePerson]:
        """ """
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
        """ """
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
    print("[Care Registry Statistics]")
    s = reg.summary()
    for k, v in s.items():
        print(f"  {k}: {v}")

    print("\n[Taipei City Query Results]")
    for p in reg.query(county="Taipei City", priority_only=True):
        print(f"  [{p.id}] {p.code_name} - {p.town} {p.community} | Priority:{p.is_priority}")
