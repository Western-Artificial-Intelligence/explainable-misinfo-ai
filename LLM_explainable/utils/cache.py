from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _key(s: str) -> str:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return h


class DiskCache:
    def __init__(self, cache_dir: str):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        path = self.dir / f"{key}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def set_json(self, key: str, value: Dict[str, Any]) -> None:
        path = self.dir / f"{key}.json"
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def key_for_claim(self, claim: str) -> str:
        return _key(claim.strip())
