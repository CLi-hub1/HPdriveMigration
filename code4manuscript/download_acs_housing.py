"""Official ACS 5-year housing tables, all relevant county estimates and MOEs."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from download_utils import download
import pandas as pd
import json

from config import OUTPUT_ROOT
BASE = OUTPUT_ROOT / "acs_housing_sources"
OUT = BASE / "processed"
support = pd.read_csv(OUT / "frozen_county_support.csv", dtype={"GEOID": str})
TABLES = ["B25064", "B25070", "B25077"]


def table(job):
    year, name = job
    prefix = "prototype" if year <= 2020 else "table-based-SF/data"
    url = f"https://www2.census.gov/programs-surveys/acs/summary_file/{year}/{prefix}/5YRData/acsdt5y{year}-{name.lower()}.dat"
    path, info = download(url, f"raw/acs/acsdt5y{year}-{name.lower()}.dat")
    parts = []
    for chunk in pd.read_csv(path, sep="|", dtype=str, chunksize=50000):
        chunk = chunk.rename(columns={"#GEO_ID": "GEO_ID"})
        # The earliest prototype used comma separated values.
        if "GEO_ID" not in chunk:
            break
        g = chunk[chunk.GEO_ID.str.fullmatch(r"0500000US\d{5}", na=False)].copy()
        if len(g):
            parts.append(g)
    if not parts:
        raw = pd.read_csv(path, dtype=str)
        geocol = next(c for c in raw if c.upper() in ["GEO_ID", "GEOID"])
        raw = raw.rename(columns={geocol: "GEO_ID"})
        parts = [raw[raw.GEO_ID.str.fullmatch(r"0500000US\d{5}", na=False)].copy()]
    full = pd.concat(parts, ignore_index=True)
    # 2018 prototype E/M suffix moved to the front in 2019.
    full = full.rename(columns={f"{name}_{i:03d}{suffix}": f"{name}_{suffix}{i:03d}"
                                for i in range(1, 12) for suffix in ["E", "M"]})
    assert not full.GEO_ID.duplicated().any()
    full["GEOID"] = full.GEO_ID.str[-5:]
    full["Year"] = year
    selected = full[full.GEOID.isin(set(support.GEOID))]
    selected.to_csv(OUT / f"acs5_{year}_{name}_county_source.csv", index=False)
    info.update(year=year, table=name, raw_counties=len(full), selected_counties=len(selected),
                period_start=year - 4, period_end=year)
    print("ACS housing", year, name, len(selected), flush=True)
    return info


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(table, [(year, name) for year in range(2018, 2024) for name in TABLES]))
    (BASE / "acs_housing_download_manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
