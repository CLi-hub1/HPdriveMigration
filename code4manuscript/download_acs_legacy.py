"""Recover 2016/2017 housing tables from official sequence-specific state ZIPs."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from download_utils import download
import csv
import io
import json
import zipfile
import pandas as pd

from config import OUTPUT_ROOT
BASE = OUTPUT_ROOT / "acs_housing_sources"
OUT = BASE / "processed"
support = pd.read_csv(OUT / "frozen_county_support.csv", dtype={"GEOID": str})
county_set = set(support.GEOID)
state_path, _ = download("https://www2.census.gov/geo/docs/reference/state.txt", "raw/acs/state.txt")
states = pd.read_csv(state_path, sep="|", dtype=str)
states = states[states.STATE.isin(support.GEOID.str[:2])]
LOOKUPS = {}
manifest = []
for year in [2016, 2017]:
    url = (f"https://www2.census.gov/programs-surveys/acs/summary_file/{year}/documentation/"
           "user_tools/ACS_5yr_Seq_Table_Number_Lookup.txt")
    path, info = download(url, f"raw/acs/ACS_{year}_5yr_Seq_Table_Number_Lookup.txt")
    manifest.append(info)
    lookup = list(csv.reader(path.read_text(encoding="cp1252").splitlines()))
    selected = {}
    for table in ["B25064", "B25070", "B25077"]:
        row = next(row for row in lookup if len(row) > 5 and row[1] == table and row[4])
        selected[table] = dict(sequence=row[2], start_1_based=int(row[4]),
                               cells=int(row[5].split()[0]))
    LOOKUPS[year] = selected
print(LOOKUPS, flush=True)


def state_year(job):
    year, state, postal, state_name = job
    folder = state_name.replace(" ", "")
    # Census folder capitalization follows "DistrictOfColumbia".
    if postal == "DC":
        folder = "DistrictOfColumbia"
    root = (f"https://www2.census.gov/programs-surveys/acs/summary_file/{year}/data/"
            f"5_year_seq_by_state/{folder}/All_Geographies_Not_Tracts_Block_Groups/")
    geo_name = f"g{year}5{postal.lower()}.csv"
    path, info = download(root + geo_name, "raw/acs/" + geo_name)
    files = [info]
    rows = csv.reader(path.read_text(encoding="utf-8-sig", errors="replace").splitlines())
    mapping = {}
    for row in rows:
        if row[2] == "050" and row[3] == "00":
            geoid = row[9] + row[10]
            assert row[48].endswith("US" + geoid)
            if geoid in county_set:
                mapping[row[4]] = geoid
    records = {}
    sequences = sorted(set(v["sequence"] for v in LOOKUPS[year].values()))
    for seq in sequences:
        name = f"{year}5{postal.lower()}{seq}000.zip"
        path, info = download(root + name, "raw/acs/" + name)
        files.append(info)
        with zipfile.ZipFile(path) as z:
            for kind in ["e", "m"]:
                member = f"{kind}{year}5{postal.lower()}{seq}000.txt"
                for row in csv.reader(io.TextIOWrapper(z.open(member), encoding="utf-8-sig")):
                    if row[5] not in mapping:
                        continue
                    geoid = mapping[row[5]]
                    rec = records.setdefault(geoid, dict(GEOID=geoid, Year=year))
                    for table, definition in LOOKUPS[year].items():
                        if definition["sequence"] == seq:
                            start = definition["start_1_based"] - 1
                            for i in range(definition["cells"]):
                                rec[f"{table}_{kind.upper()}{i+1:03d}"] = row[start+i]
    assert len(records) == len(mapping), (year, postal, len(mapping), len(records))
    print("ACS legacy", year, postal, len(records), flush=True)
    return list(records.values()), files


jobs = [(year, row.STATE, row.STUSAB, row.STATE_NAME)
        for year in [2016, 2017] for row in states.itertuples(index=False)]
with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(state_year, jobs))
records = [rec for rows, _ in results for rec in rows]
for _, files in results:
    manifest.extend(files)
full = pd.DataFrame(records)
assert len(full) == 216 and not full[["GEOID", "Year"]].duplicated().any()
for year in [2016, 2017]:
    for table in ["B25064", "B25070", "B25077"]:
        cols = ["GEOID", "Year"] + [c for c in full if c.startswith(table + "_")]
        full.loc[full.Year.eq(year), cols].to_csv(OUT / f"acs5_{year}_{table}_county_source.csv", index=False)
(BASE / "acs_legacy_download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
(OUT / "acs_legacy_sequence_mapping.json").write_text(json.dumps(LOOKUPS, indent=2), encoding="utf-8")
print("ACS legacy complete", len(full), flush=True)
