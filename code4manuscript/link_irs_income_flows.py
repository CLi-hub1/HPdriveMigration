"""Estimate paired Results 5 migration economic-composition models from IRS SOI flows."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels import PanelOLS
from scipy.stats import t
from statsmodels.stats.multitest import multipletests

from config import PROJECT_ROOT, OUTPUT_ROOT
SOURCE = PROJECT_ROOT / 'results5'
ROOT = OUTPUT_ROOT / 'results5_linkage'
OUT = ROOT / 'data'
RAW = SOURCE / 'data/raw_irs'
PANEL_PATH = SOURCE / 'data/panel_lag0.csv'
SELECTION = SOURCE / 'data/results3_selection_pair8.csv'
PAIR_FILES = {
    "countyoutflow1718.csv": 2017,
    "countyoutflow1819.csv": 2018,
    "countyoutflow1920.csv": 2019,
    "countyoutflow2021.csv": 2020,
    "countyoutflow2122.csv": 2021,
    "countyoutflow2223.csv": 2022,
}
NATIVE_TYPES = [
    "APTechnologicalInfrastructural", "APInstitutional", "APBehavioralCultural", "APNatureBased",
    "MPTechnologicalInfrastructural", "MPInstitutional", "MPBehavioralCultural", "MPNatureBased",
]
OD_TYPES = NATIVE_TYPES + [x + "_dest" for x in NATIVE_TYPES]
KEY = ["GEOID", "GEOID_destination", "Year"]
OUTCOMES = {
    "ln_person_flow_rate": "Log IRS individuals moving / origin population",
    "ln_return_flow_rate": "Log IRS tax returns moving / origin population",
    "asinh_mean_agi_10k": "Asinh(mean AGI per moving return / $10,000)",
    "asinh_agi_intensity_10k": "Asinh(migrant AGI per 1,000 origin residents / $10,000)",
}
POLICY_LABELS = {
    "APTechnologicalInfrastructural": "AP technological / infrastructural (origin)",
    "APBehavioralCultural": "AP behavioral / cultural (origin)",
    "APNatureBased": "AP nature-based (origin)",
    "MPBehavioralCultural": "MP behavioral / cultural (origin)",
    "MPNatureBased": "MP nature-based (origin)",
    "APInstitutional_dest": "AP institutional (destination)",
    "MPInstitutional_dest": "MP institutional (destination)",
    "MPBehavioralCultural_dest": "MP behavioral / cultural (destination)",
}
ORIGIN_COLOR = "#FA853E"
DEST_COLOR = "#1B6CA8"
plt.rcParams.update({
    "font.family": "Arial", "font.size": 14, "axes.labelsize": 15,
    "xtick.labelsize": 13.5, "ytick.labelsize": 13.5, "axes.linewidth": 1.05,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "legend.frameon": False,
})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_irs_flows():
    pieces = []
    source_info = []
    for name, start in PAIR_FILES.items():
        path = RAW / name
        if not path.exists():
            raise FileNotFoundError(f"Missing official IRS source file: {path}")
        df = pd.read_csv(path, dtype=str, encoding="cp1252")
        df.columns = df.columns.str.lower()
        required = {"y1_statefips", "y1_countyfips", "y2_statefips", "y2_countyfips", "n1", "n2", "agi"}
        if not required.issubset(df.columns):
            raise ValueError(f"Unexpected IRS schema in {name}: {sorted(df.columns)}")
        for col in ("y1_statefips", "y2_statefips"):
            df[col] = df[col].str.strip().str.zfill(2)
        for col in ("y1_countyfips", "y2_countyfips"):
            df[col] = df[col].str.strip().str.zfill(3)
        for col in ("n1", "n2", "agi"):
            df[col] = pd.to_numeric(df[col], errors="raise")
        df["origin_fips"] = df.y1_statefips + df.y1_countyfips
        df["destination_fips"] = df.y2_statefips + df.y2_countyfips
        valid = (
            df.y1_statefips.between("01", "56") & df.y2_statefips.between("01", "56")
            & df.y1_countyfips.ne("000") & df.y2_countyfips.ne("000")
            & df.origin_fips.ne(df.destination_fips)
        )
        flow = df.loc[valid, ["origin_fips", "destination_fips", "n1", "n2", "agi"]].copy()
        flow["Year"] = start
        flow["flow_start_year"] = start
        flow["flow_end_year"] = start + 1
        flow["irs_method_break"] = start == 2022
        flow["n1_suppressed"] = flow.n1.eq(-1)
        flow["n2_suppressed"] = flow.n2.eq(-1)
        flow["agi_suppressed"] = flow.agi.eq(-1)
        for col in ("n1", "n2", "agi"):
            flow.loc[flow[col].eq(-1), col] = np.nan
        if flow.duplicated(["origin_fips", "destination_fips", "Year"]).any():
            raise ValueError(f"Duplicate county-to-county IRS rows in {name}")
        pieces.append(flow)
        source_info.append({"file": name, "sha256": sha(path), "raw_rows": len(df),
                            "county_to_county_rows": len(flow), "flow_start_year": start})
    all_flows = pd.concat(pieces, ignore_index=True)
    all_flows["GEOID"] = all_flows.origin_fips.astype(int)
    all_flows["GEOID_destination"] = all_flows.destination_fips.astype(int)
    all_flows = all_flows.drop(columns=["origin_fips", "destination_fips", "flow_start_year", "flow_end_year"])
    if all_flows.duplicated(KEY).any():
        raise ValueError("County-to-county IRS files have duplicate keys across periods")
    return all_flows, source_info


def make_aligned_panel():
    panel = pd.read_csv(PANEL_PATH)
    for col in KEY + ["flow_start_year", "flow_end_year", "n2"]:
        panel[col] = panel[col].astype(int)
    panel["GEOID"] = panel.GEOID.astype(int)
    irs, source_info = load_irs_flows()
    aligned = panel.merge(irs, on=KEY, how="left", validate="one_to_one", suffixes=("_panel", "_irs"))
    aligned["irs_flow_matched"] = aligned.n1.notna() | aligned.agi.notna() | aligned.n2_irs.notna()
    aligned["n2_source_matches_panel"] = aligned.n2_irs.eq(aligned.n2_panel)
    mismatch = aligned.irs_flow_matched & ~aligned.n2_source_matches_panel
    mismatch_count = int(mismatch.sum())
    if mismatch_count:
        examples = aligned.loc[mismatch, KEY + ["n2_panel", "n2_irs"]].head(10).to_dict("records")
        raise ValueError(f"Official IRS N2 does not reproduce the frozen panel in {mismatch_count} rows: {examples}")
    # Output measures. IRS AGI is reported in $1,000; negative AGI is valid.
    aligned["population"] = pd.to_numeric(aligned.population, errors="coerce")
    aligned["agi_usd"] = aligned.agi * 1000
    aligned["mean_agi_per_return_usd"] = aligned.agi_usd / aligned.n1.where(aligned.n1.gt(0))
    aligned["agi_per_1000_residents_10k"] = aligned.agi * 100 / aligned.population
    aligned["ln_person_flow_rate"] = np.log(aligned.n2_irs / aligned.population)
    aligned.loc[~aligned.n2_irs.gt(0), "ln_person_flow_rate"] = np.nan
    aligned["ln_return_flow_rate"] = np.log(aligned.n1 / aligned.population)
    aligned.loc[~aligned.n1.gt(0), "ln_return_flow_rate"] = np.nan
    aligned["asinh_mean_agi_10k"] = np.arcsinh(aligned.mean_agi_per_return_usd / 10000)
    aligned["asinh_agi_intensity_10k"] = np.arcsinh(aligned.agi_per_1000_residents_10k)
    # Retain the manuscript's panel outcome for a direct replication audit.
    aligned["ln_person_flow_rate_panel"] = aligned.ln_outflow_rate
    aligned["ln_person_flow_rate_difference"] = aligned.ln_person_flow_rate - aligned.ln_person_flow_rate_panel
    coverage = aligned.groupby("Year").agg(
        panel_OD_rows=("GEOID", "size"), matched_IRS_rows=("irs_flow_matched", "sum"),
        n1_observed=("n1", "count"), n2_observed=("n2_irs", "count"), agi_observed=("agi", "count"),
        n2_match=("n2_source_matches_panel", "sum"),
    ).reset_index()
    complete_four = aligned.assign(all_four_outcomes=aligned[list(OUTCOMES)].notna().all(axis=1)) \
        .groupby("Year").all_four_outcomes.sum().rename("all_four_outcomes").reset_index()
    coverage = coverage.merge(complete_four, on="Year", validate="one_to_one")
    checks = {
        "panel_rows": len(panel), "unique_OD_year_keys": int(panel[KEY].drop_duplicates().shape[0]),
        "irs_valid_county_flow_rows": len(irs), "matched_panel_rows": int(aligned.irs_flow_matched.sum()),
        "unmatched_panel_rows": int((~aligned.irs_flow_matched).sum()), "n2_mismatch_rows": mismatch_count,
        "max_abs_ln_person_rate_difference": float(aligned.ln_person_flow_rate_difference.abs().max()),
        "n1_suppressed_in_matched_panel": int((aligned.irs_flow_matched & aligned.n1.isna()).sum()),
        "agi_suppressed_in_matched_panel": int((aligned.irs_flow_matched & aligned.agi.isna()).sum()),
        "negative_valid_AGI_rows": int((aligned.agi < 0).sum()),
        "common_four_outcome_rows": int(aligned[list(OUTCOMES)].notna().all(axis=1).sum()),
        "source_files": source_info,
    }
    return aligned, coverage, checks



def main():
    """Rebuild the aligned panel only, without fitting the superseded pooled models."""
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "tables").mkdir(exist_ok=True)
    aligned, coverage, checks = make_aligned_panel()
    aligned.to_csv(OUT / "results5_economiccomposition_aligned_panel.csv", index=False)
    coverage.to_csv(ROOT / "tables/results5_irs_coverage_by_year.csv", index=False)
    (OUT / "provenance").mkdir(exist_ok=True)
    (OUT / "provenance/panel_rebuild_audit.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(f"Built {len(aligned)} aligned rows; mismatches={checks['n2_mismatch_rows']}")

if __name__ == "__main__":
    main()
