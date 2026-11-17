"""Bounded official downloads with immutable content hashes and response provenance."""
from pathlib import Path
from urllib.request import urlopen, Request
from datetime import datetime, timezone
import hashlib
import json
import time

from config import OUTPUT_ROOT
BASE = OUTPUT_ROOT / "acs_housing_sources"


def download(url, relative, attempts=3):
    path = BASE / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    side = path.with_suffix(path.suffix + ".provenance.json")
    if path.exists() and side.exists():
        return path, json.loads(side.read_text(encoding="utf-8"))
    for trial in range(attempts):
        try:
            with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=100) as response:
                data = response.read()
                expected = response.headers.get("Content-Length")
                if expected and len(data) != int(expected):
                    raise ValueError("incomplete download")
                info = dict(file=str(path.relative_to(BASE)), url=url, final_url=response.url,
                            retrieved_utc=datetime.now(timezone.utc).isoformat(), bytes=len(data),
                            sha256=hashlib.sha256(data).hexdigest(),
                            headers=dict(response.headers))
            path.write_bytes(data)
            side.write_text(json.dumps(info, indent=2), encoding="utf-8")
            return path, info
        except Exception:
            if trial == attempts - 1:
                raise
            time.sleep(trial + 1)
