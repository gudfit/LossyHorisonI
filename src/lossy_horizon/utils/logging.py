from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from typing import Any, Dict, Optional


def git_hash(cwd: Optional[str] = None) -> Optional[str]:
    try:
        out = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
            .decode()
            .strip()
        )
        return out
    except Exception:
        return None


class JSONLLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def write(self, record: Dict[str, Any]) -> None:
        rec = {"timestamp": datetime.utcnow().isoformat() + "Z", **record}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
