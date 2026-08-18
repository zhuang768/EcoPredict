import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "data" / "care_list" / "vulnerable_registry.json"

@dataclass
class Facility:
    id: str
    code_name: str
    county: str
    town: str
    community: str
    is_priority: bool
    emergency_contact: str
    notes: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "code_name": self.code_name,
            "county": self.county,
            "town": self.town,
            "community": self.community,
            "is_priority": self.is_priority,
            "emergency_contact": self.emergency_contact,
            "notes": self.notes,
        }

class VulnerableRegistry:
    def __init__(self, path: Path = REGISTRY_PATH) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Registry not found: {path}")
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        self._records: list[Facility] = [Facility(**r) for r in raw["records"]]
        logger.info("Loaded infrastructure registry: %d records", len(self._records))

    def query(self, county: Optional[str] = None, town: Optional[str] = None, priority_only: bool = False) -> list[Facility]:
        results = self._records
        if county:
            results = [p for p in results if p.county == county]
        if town:
            results = [p for p in results if p.town == town]
        if priority_only:
            results = [p for p in results if p.is_priority]
        return results

    @property
    def all_records(self) -> list[Facility]:
        return list(self._records)

if __name__ == "__main__":
    reg = VulnerableRegistry()
    print("[Registry Statistics]")
    print(f"Total Facilities: {len(reg.all_records)}")
