"""Download official ACS five-year county estimates without replacing missing data."""
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
import hashlib
import json
import math
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from config import OUTPUT_ROOT
HERE = OUTPUT_ROOT / "acs_demographic_download"
OUT = HERE / "latest_data" / "acs"
OUT.mkdir(parents=True, exist_ok=True)
YEARS = range(2021, 2025)
RACE = [f"B02001_{i:03d}E" for i in range(2, 9)]
AGED = [f"B01001_{i:03d}E" for i in [20, 21, 22, 23, 24, 25, 44, 45, 46, 47, 48, 49]]
DEGREE = [f"B15003_{i:03d}E" for i in range(22, 26)]
VARIABLES = ["B01003_001E", "B01001_001E", "B19013_001E", "B25077_001E",
             "B02001_001E", "B15003_001E", *RACE, *AGED, *DEGREE]
assert len(VARIABLES) == len(set(VARIABLES)) and len(VARIABLES) < 50


def get_json(url, path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    with urlopen(url, timeout=60) as response:
        data = response.read()
    parsed = json.loads(data)
    path.write_bytes(data)
    return parsed


def proportion(numerator, denominator):
    return numerator / denominator.where(denominator > 0)


def download_table(task):
    year, table = task
    url = (f"https://www2.census.gov/programs-surveys/acs/summary_file/{year}/"
           f"table-based-SF/data/5YRData/acsdt5y{year}-{table.lower()}.dat")
    path = OUT / f"acsdt5y{year}-{table.lower()}.dat"
    if not path.exists():
        tmp = path.with_suffix(".partial")
        with urlopen(url, timeout=60) as response, tmp.open("wb") as dest:
            expected = response.headers.get("Content-Length")
            while chunk := response.read(1024 * 1024):
                dest.write(chunk)
        if expected:
            assert tmp.stat().st_size == int(expected), (year, table, "incomplete")
        tmp.replace(path)
    columns = {}
    for var in VARIABLES:
        if var.startswith(table + "_"):
            columns[f"{table}_E{var.split('_')[1][:-1]}"] = var
    frames = []
    for chunk in pd.read_csv(path, sep="|", dtype=str, chunksize=30000):
        county = chunk[chunk["GEO_ID"].str.fullmatch(r"0500000US\d{5}", na=False)].copy()
        if len(county):
            frames.append(county)
    full = pd.concat(frames, ignore_index=True)
    assert not full["GEO_ID"].duplicated().any()
    assert len(full) > 3000
    full.to_csv(OUT / f"acs5_{year}_{table}_county_estimates_moe.csv", index=False, encoding="utf-8-sig")
    selected = full[["GEO_ID", *columns]].rename(columns=columns)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    info = {"year": year, "period_start": year - 4, "period_end": year, "kind": table,
            "url": url, "file": path.name, "retrieved_date": "2026-10-20",
            "sha256": digest.hexdigest(), "bytes": path.stat().st_size, "county_rows": len(full)}
    print(f"ACS {year} {table}: {len(full)} counties; estimates and margins of error retained.", flush=True)
    return selected, info


tables = sorted({var.split("_")[0] for var in VARIABLES})
with ThreadPoolExecutor(max_workers=4) as pool:
    downloaded = list(pool.map(download_table, [(year, table) for year in YEARS for table in tables]))
by_year = {}
for data, info in downloaded:
    by_year.setdefault(info["year"], []).append((data, info))

manifest, frames = [], []
for year in YEARS:
    base = f"https://api.census.gov/data/{year}/acs/acs5"
    meta_path = OUT / f"acs5_{year}_variables.json"
    metadata = get_json(base + "/variables.json", meta_path)["variables"]
    for var in VARIABLES:
        assert var in metadata, (year, var)
    assert "White alone" in metadata[RACE[0]]["label"]
    assert "two or more races" in metadata[RACE[-1]]["label"].lower()
    assert "Median household income" in metadata["B19013_001E"]["label"]
    assert "65 and 66 years" in metadata[AGED[0]]["label"]
    assert "Bachelor" in metadata[DEGREE[0]]["label"]
    df = by_year[year][0][0]
    manifest.append(by_year[year][0][1])
    for data, info in by_year[year][1:]:
        assert set(df["GEO_ID"]) == set(data["GEO_ID"]), (year, info["kind"], "county mismatch")
        df = df.merge(data, on="GEO_ID", validate="one_to_one")
        manifest.append(info)
    df["state"] = df["GEO_ID"].str[-5:-3]
    df["county"] = df["GEO_ID"].str[-3:]
    if True:
        for col in VARIABLES:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] < 0, col] = float("nan")
        result = pd.DataFrame({"Year": year, "GEOID": df["state"] + df["county"],
                               "ACS_period_start": year - 4,
                               "ACS_period_end": year, "population": df["B01003_001E"],
                               "income": df["B19013_001E"],
                               "home_value": df["B25077_001E"]})
        result["ln_income"] = result["income"].apply(lambda x: math.log(x) if x > 0 else float("nan"))
        result["aging_percent"] = 100 * proportion(df[AGED].sum(axis=1, min_count=len(AGED)), df["B01001_001E"])
        result["education_percent"] = 100 * proportion(df[DEGREE].sum(axis=1, min_count=len(DEGREE)), df["B15003_001E"])
        race_shares = df[RACE].div(df["B02001_001E"].where(df["B02001_001E"] > 0), axis=0)
        result["racial_diversity"] = 1 - race_shares.pow(2).sum(axis=1, min_count=len(RACE))
        race_total = df[RACE].sum(axis=1, min_count=len(RACE))
        assert ((race_total - df["B02001_001E"]).dropna().abs() < 1).all()
        assert result["GEOID"].str.fullmatch(r"\d{5}").all()
        assert not result["GEOID"].duplicated().any()
        assert result["racial_diversity"].dropna().between(0, 1).all()
        assert result["aging_percent"].dropna().between(0, 100).all()
        assert result["education_percent"].dropna().between(0, 100).all()
        result["ct_boundary_review"] = result["GEOID"].str.startswith("09")
        result["contiguous_us"] = ~df["state"].isin(["02", "15", "60", "66", "69", "72", "78"])
        frames.append(result)
    (OUT / f"acs5_{year}_selected_metadata.json").write_text(
        json.dumps({v: metadata[v] for v in VARIABLES}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ACS {year}: {len(df)} counties, estimates and margins of error retained.", flush=True)

panel = pd.concat(frames, ignore_index=True)
assert not panel[["Year", "GEOID"]].duplicated().any()
panel.to_csv(OUT / "acs_county_2021_2024.csv", index=False, encoding="utf-8-sig")
(OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
summary = panel.groupby("Year").agg(counties=("GEOID", "size"), population=("population", "sum"))
summary.to_csv(OUT / "coverage.csv", encoding="utf-8-sig")
print(summary.to_string())
