"""Reproduce Results 3 sensitivity models for the eight pair-clustered Results 2 terms.

The model formulas, sample transformations and standard-error conventions follow
analysis/scripts/run_analysis.py. Inputs are read from this project’s frozen
reproducibility bundle; outputs stay in this directory.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels import PanelOLS, RandomEffects
from scipy.stats import t
from statsmodels.stats.multitest import multipletests


from config import PROJECT_ROOT, OUTPUT_ROOT
PACKAGE = PROJECT_ROOT / 'results3'
ANALYSIS = PACKAGE / "data"
INPUT = PACKAGE / "data"
OUT = OUTPUT_ROOT / "results3" / "tables"
TYPES = [p + c + s for s in ["", "_dest"] for p in ["AP", "MP"]
         for c in ["TechnologicalInfrastructural", "Institutional", "BehavioralCultural", "NatureBased"]]
KEY = ["GEOID", "GEOID_destination", "Year"]
EXPECTED_SELECTED = {
    "APTechnologicalInfrastructural", "APBehavioralCultural", "APNatureBased",
    "MPBehavioralCultural", "MPNatureBased", "APInstitutional_dest",
    "MPInstitutional_dest", "MPBehavioralCultural_dest",
}
SPECS = [
    ("baseline", "Main specification", "main"),
    ("RE", "Random effects", "re"),
    ("winsor1", "1% winsorizing", "sensitivity"),
    ("trim5", "5% trimming", "sensitivity"),
    ("sample80", "Random 80% sample", "sensitivity"),
    ("exclude2020", "Excluding 2020", "sensitivity"),
    ("lag1", "One-year lag", "lag"),
    ("lag2", "Two-year lag", "lag"),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_panel(path, lag):
    d = pd.read_csv(path)
    required = KEY + ["flow_start_year", "ln_outflow_rate"] + TYPES
    assert set(required) <= set(d), set(required) - set(d)
    for col in KEY + ["flow_start_year"]:
        d[col] = d[col].astype(int)
    assert not d.duplicated(KEY).any(), f"Duplicate model keys in {path}"
    assert np.isfinite(d[required].to_numpy(float)).all()
    assert (d.flow_start_year - d.Year == lag).all(), "Calendar lag mismatch"
    assert not d.GEOID.floordiv(1000).isin([2, 9, 15]).any()
    assert not d.GEOID_destination.floordiv(1000).isin([2, 9, 15]).any()
    return d.sort_values(KEY).reset_index(drop=True)


def fit(frame, kind):
    d = frame.copy()
    d["county_pair"] = d.GEOID.astype(str) + "_" + d.GEOID_destination.astype(str)
    d = d.set_index(["county_pair", "Year"])
    if kind == "RE":
        model = RandomEffects(d.ln_outflow_rate, d[TYPES])
    else:
        model = PanelOLS(d.ln_outflow_rate, d[TYPES], entity_effects=True, time_effects=True)
    return model.fit(cov_type="clustered", cluster_entity=True)


def result_rows(model, specification, frame):
    p = model.pvalues
    family_p = p.reindex(TYPES).fillna(1).to_numpy()
    bh = multipletests(family_p, method="fdr_bh")[1]
    holm = multipletests(family_p, method="holm")[1]
    df = float(model.df_resid)
    rows = []
    for i, variable in enumerate(TYPES):
        coefficient = float(model.params[variable])
        se = float(model.std_errors[variable])
        pv = float(p[variable])
        row = {
            "Variable": variable, "Coefficient": coefficient, "Std. Error": se,
            "P-value": pv, "BH_q": float(bh[i]), "Holm_p": float(holm[i]),
            "Model": "RE" if specification == "baseline_RE" else "TWFE",
            "Covariance": "pair", "Specification": specification,
            "Observations": int(model.nobs), "df_resid": df,
            "multiplicity_family_size": 16, "family_estimable_p_count": int(np.isfinite(family_p).sum()),
            "nominal_p_lt_0_1": pv < .1, "nominal_p_lt_0_05": pv < .05,
            "N_frame": len(frame),
        }
        for level in [90, 95]:
            z = t.ppf(.5 + level / 200, df)
            row[f"CI{level}_lower"] = coefficient - z * se
            row[f"CI{level}_upper"] = coefficient + z * se
        rows.append(row)
    return rows


def select_pair_results2():
    p = ANALYSIS / "all_aggregate_and_type_models.csv"
    d = pd.read_csv(p)
    base = d[(d.Specification == "baseline_types16") & (d.Model == "TWFE") & (d.Covariance == "pair")]
    assert len(base) == 16
    selected = base[np.isfinite(base["P-value"]) & (base["P-value"] < .1)].copy()
    assert set(selected.Variable) == EXPECTED_SELECTED, set(selected.Variable) ^ EXPECTED_SELECTED
    selected["selection_rule"] = "Results 2 pair covariance; two-sided nominal p < 0.1"
    selected["selection_source_sha256"] = sha(p)
    return selected, p


def transform_specs(panels):
    d = panels[0]
    specs = {"baseline": d, "exclude2020": d[d.Year != 2020], "lag1": panels[1], "lag2": panels[2]}
    w = d.copy()
    w.ln_outflow_rate = w.ln_outflow_rate.clip(*w.ln_outflow_rate.quantile([.01, .99]))
    specs["winsor1"] = w
    lo, hi = d.ln_outflow_rate.quantile([.05, .95])
    specs["trim5"] = d[d.ln_outflow_rate.between(lo, hi)]
    specs["sample80"] = d.sample(frac=.8, random_state=42)
    return specs


def plot(selected, data):
    from html import escape
    labels = {
        "TechnologicalInfrastructural": "Technological / infrastructural",
        "Institutional": "Institutional", "BehavioralCultural": "Behavioral / cultural",
        "NatureBased": "Nature-based",
    }
    def display(v):
        side = "destination" if v.endswith("_dest") else "origin"
        base = v.removesuffix("_dest")
        return f"{labels[base[2:]]} {base[:2]} ({side})"

    colors = {"main": "#808080", "re": "#111111", "sensitivity": "#1B6CA8", "lag": "#FA853E"}
    width, panel_h, gap_x, gap_y = 1400, 310, 48, 52
    left, plot_w = 235, 405
    height = 90 + 4 * panel_h + 3 * gap_y
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#222} .small{font-size:12px} .axis{stroke:#999;stroke-width:1}</style>',
             '<text x="35" y="38" font-size="22" font-weight="bold">Results 3 sensitivity analysis (pair-clustered)</text>',
             '<text x="35" y="61" font-size="13">Thin/thick intervals: 95%/90%; grey: TWFE baseline; black: RE; blue: sample/outcome sensitivities; orange: calendar lags. Numbers show nominal p.</text>']
    for i, variable in enumerate(selected):
        col, row = i % 2, i // 2
        x0 = 35 + col * (left + plot_w + gap_x)
        y0 = 95 + row * (panel_h + gap_y)
        dd = data[data.Variable == variable].copy()
        lo = min(float(dd.CI95_lower.min()), 0.0); hi = max(float(dd.CI95_upper.max()), 0.0)
        span = hi - lo
        lo -= .12 * span; hi += .28 * span
        def xpos(v): return x0 + left + (float(v) - lo) / (hi - lo) * plot_w
        parts.append(f'<text x="{x0}" y="{y0+16}" font-size="15" font-weight="bold">{chr(97+i)}. {escape(display(variable))}</text>')
        grid_y0 = y0 + 40
        for tick_i in range(5):
            val = lo + (hi - lo) * tick_i / 4
            x = xpos(val)
            parts.append(f'<line x1="{x:.1f}" y1="{grid_y0}" x2="{x:.1f}" y2="{grid_y0+218}" class="axis" opacity=".4"/>')
            parts.append(f'<text x="{x:.1f}" y="{grid_y0+238}" class="small" text-anchor="middle">{val:.3f}</text>')
        zero_x = xpos(0)
        parts.append(f'<line x1="{zero_x:.1f}" y1="{grid_y0}" x2="{zero_x:.1f}" y2="{grid_y0+218}" stroke="#666" stroke-dasharray="4,3"/>')
        for j, (spec, label, family) in enumerate(SPECS):
            model = "RE" if spec == "RE" else "TWFE"
            actual = "baseline_RE" if spec == "RE" else spec
            r = dd[(dd.Specification == actual) & (dd.Model == model)].iloc[0]
            y = grid_y0 + j * 30 + 12
            color = colors[family]
            parts.append(f'<text x="{x0+left-10}" y="{y+4}" class="small" text-anchor="end">{escape(label)}</text>')
            parts.append(f'<line x1="{xpos(r.CI95_lower):.1f}" y1="{y}" x2="{xpos(r.CI95_upper):.1f}" y2="{y}" stroke="{color}" stroke-width="1.2" opacity=".55"/>')
            parts.append(f'<line x1="{xpos(r.CI90_lower):.1f}" y1="{y}" x2="{xpos(r.CI90_upper):.1f}" y2="{y}" stroke="{color}" stroke-width="2.2"/>')
            fill = color if family == "main" else "white"
            parts.append(f'<circle cx="{xpos(r.Coefficient):.1f}" cy="{y}" r="4.5" fill="{fill}" stroke="{color}" stroke-width="1.4"/>')
            ptext = "&lt;.001" if r["P-value"] < .001 else f"{r['P-value']:.3f}"
            parts.append(f'<text x="{xpos(r.Coefficient)+7:.1f}" y="{y-5}" class="small">{ptext}</text>')
    parts.append('</svg>')
    (OUT / "Fig3_SelectedSensitivity_pair.svg").write_text("\n".join(parts), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    selected, selection_source = select_pair_results2()
    panels = {lag: read_panel(INPUT / f"panel_lag{lag}.csv", lag) for lag in [0, 1, 2]}
    specs = transform_specs(panels)
    records = []
    sample_records = []
    for spec, frame in specs.items():
        sample_records.append({
            "Specification": spec, "N": len(frame), "directed_pairs": len(frame[KEY[:2]].drop_duplicates()),
            "origin_counties": frame.GEOID.nunique(), "destination_counties": frame.GEOID_destination.nunique(),
            "policy_years": ",".join(map(str, sorted(frame.Year.unique()))),
            "flow_start_years": ",".join(map(str, sorted(frame.flow_start_year.unique()))),
        })
        records.extend(result_rows(fit(frame, "TWFE"), spec, frame))
        if spec == "baseline":
            records.extend(result_rows(fit(frame, "RE"), "baseline_RE", frame))
    full = pd.DataFrame(records)
    full["selected_for_results3"] = full.Variable.isin(selected.Variable)
    full["selection_rule"] = "Results 2 pair covariance; two-sided nominal p < 0.1"
    full["selection_source_sha256"] = sha(selection_source)
    selected_results = full[full.selected_for_results3].copy()
    assert len(selected_results) == 8 * 8
    assert set(selected_results.Variable) == EXPECTED_SELECTED
    assert selected_results.groupby("Specification").Variable.nunique().eq(8).all()
    full.to_csv(OUT / "results3_sensitivity_full16.csv", index=False)
    selected_results.to_csv(OUT / "results3_sensitivity_selected8.csv", index=False)
    selected.to_csv(OUT / "results3_selection_pair8.csv", index=False)
    pd.DataFrame(sample_records).to_csv(OUT / "sample_manifest.csv", index=False)
    # Final accepted figure is generated by plot_results3_original_style.py.
    # The legacy schematic plot() function is retained for provenance, not called.
    manifest = {
        "status": "COMPLETE", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "Results 2 pair-clustered contemporaneous TWFE; finite two-sided nominal p < 0.1",
        "selected_count": 8, "selected_variables": selected.Variable.tolist(),
        "selection_source": selection_source.name, "selection_source_sha256": sha(selection_source),
        "inputs": {str(lag): {"path": f"panel_lag{lag}.csv",
                              "sha256": sha(INPUT / f"panel_lag{lag}.csv")} for lag in panels},
        "models": "All 16 policy type main effects jointly; county-pair and year fixed effects; clustered by county pair. The baseline RE comparator has no explicit intercept or year effects, matching the existing script.",
        "specifications": [x[0] for x in SPECS], "random_seed": 42,
        "p_values": "Two-sided t tests using model residual degrees of freedom, matching the existing analysis.",
        "followup_reselection": False,
    }
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Results 3 sensitivity analysis\n\n"
        "This directory contains a path-adjusted reproduction of the existing Results 3 sensitivity specifications, "
        "using the eight Results 2 pair-clustered effects with nominal p < 0.1. All 16 policy-type main effects "
        "remain in every fitted model. The eight displayed effects are not reselected by their sensitivity p values.\n\n"
        "Specifications: baseline TWFE; original baseline RE comparator; 1% outcome winsorization; 5% symmetric "
        "trimming; seeded random 80% rows (seed 42); excluding 2020; and exact-calendar lag 1 and lag 2. "
        "Pair-clustered inference is used throughout. BH and Holm columns are retained as audit diagnostics; "
        "the Results 3 selection and displayed gate use the original p values.\n\n"
        "`results3_sensitivity_selected8.csv` is the 8 selected terms × 8 specifications. "
        "`results3_sensitivity_full16.csv` preserves all model coefficients for audit. "
        "The figure is provided as `Fig3_SelectedSensitivity_pair.svg`.\n", encoding="utf-8")
    print(f"COMPLETE {OUT} selected={len(selected.Variable)} rows={len(selected_results)}")


if __name__ == "__main__":
    main()
