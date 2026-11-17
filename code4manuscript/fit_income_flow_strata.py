"""Re-estimate paired Results 5 models within outcome-defined quintiles."""
from pathlib import Path
from datetime import datetime, timezone
import json
import warnings
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels import PanelOLS
from scipy.stats import t
from statsmodels.stats.multitest import multipletests
from config import PROJECT_ROOT, OUTPUT_ROOT
ROOT = PROJECT_ROOT / 'results5'
OUT = OUTPUT_ROOT / 'results5'
INPUT = ROOT / 'data/results5_economiccomposition_aligned_panel.csv'
NATIVE_TYPES = [
    "APTechnologicalInfrastructural", "APInstitutional", "APBehavioralCultural", "APNatureBased",
    "MPTechnologicalInfrastructural", "MPInstitutional", "MPBehavioralCultural", "MPNatureBased",
]
OD_TYPES = NATIVE_TYPES + [f"{name}_dest" for name in NATIVE_TYPES]
OUTCOMES = {
    "ln_person_flow_rate": "Log IRS individuals moving / origin population",
    "ln_return_flow_rate": "Log IRS tax returns moving / origin population",
    "asinh_mean_agi_10k": "Asinh(mean AGI per moving return / $10,000)",
    "asinh_agi_intensity_10k": "Asinh(migrant AGI per 1,000 origin residents / $10,000)",
}
RANGE_LABELS = {
    "ln_person_flow_rate": "IRS individuals moving per 1,000 origin residents",
    "ln_return_flow_rate": "IRS tax returns moving per 1,000 origin residents",
    "asinh_mean_agi_10k": "Mean AGI per moving tax return (USD)",
    "asinh_agi_intensity_10k": "Migrant AGI per 1,000 origin residents (USD)",
}
PRIMARY_OUTCOMES = ["asinh_mean_agi_10k", "asinh_agi_intensity_10k"]
SUPPLEMENT_OUTCOMES = ["ln_person_flow_rate", "ln_return_flow_rate"]
SELECTED = [
    "APTechnologicalInfrastructural", "APBehavioralCultural", "APNatureBased",
    "MPBehavioralCultural", "MPNatureBased", "APInstitutional_dest",
    "MPInstitutional_dest", "MPBehavioralCultural_dest",
]
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
BLUE_COLOR = "#1B6CA8"
ORANGE_COLOR = "#FA853E"
plt.rcParams.update({
    "font.family": "Arial", "font.size": 16, "axes.labelsize": 17,
    "xtick.labelsize": 16, "ytick.labelsize": 16, "axes.linewidth": 1.15,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "legend.frameon": False,
})


def load_and_quintile():
    data = pd.read_csv(INPUT)
    required = {"GEOID", "GEOID_destination", "Year", "flow_start_year", *OD_TYPES, *OUTCOMES}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Aligned Results 5 panel is missing required columns: {sorted(missing)}")
    data["GEOID"] = pd.to_numeric(data.GEOID, errors="raise").astype(int)
    data["GEOID_destination"] = pd.to_numeric(data.GEOID_destination, errors="raise").astype(int)
    data["Year"] = pd.to_numeric(data.Year, errors="raise").astype(int)
    data["flow_start_year"] = pd.to_numeric(data.flow_start_year, errors="raise").astype(int)
    if data.duplicated(["GEOID", "GEOID_destination", "Year"]).any():
        raise ValueError("Duplicate county-pair-year keys in aligned panel")

    cutpoints = []
    bin_counts = []
    for outcome in OUTCOMES:
        values = data[outcome]
        if values.isna().any():
            raise ValueError(f"Unexpected missing values in {outcome}: {int(values.isna().sum())}")
        labels = [1, 2, 3, 4, 5]
        bins, edges = pd.qcut(values, q=5, labels=labels, retbins=True, duplicates="raise")
        qcol = f"quintile_{outcome}"
        data[qcol] = bins.astype(int)
        for q in labels:
            member = data.loc[data[qcol].eq(q), outcome]
            cutpoints.append({
                "outcome": outcome, "outcome_label": OUTCOMES[outcome], "quintile": q,
                "lower_edge": float(edges[q - 1]), "upper_edge": float(edges[q]),
                "observed_min": float(member.min()), "observed_max": float(member.max()),
                "n_rows": int(len(member)),
            })
            bin_counts.append({"outcome": outcome, "quintile": q, "n_rows": int(len(member))})
    return data, pd.DataFrame(cutpoints), pd.DataFrame(bin_counts)


def fit_model(data, outcome, quintile, sample):
    d = data.loc[data[f"quintile_{outcome}"].eq(quintile)].copy()
    if sample == "exclude_2022_2023":
        d = d.loc[d.flow_start_year.ne(2022)].copy()
    d = d.dropna(subset=[outcome] + OD_TYPES)
    if d.empty:
        raise ValueError(f"No rows for {sample}, {outcome}, Q{quintile}")
    d["county_pair"] = d.GEOID.astype(str) + "_" + d.GEOID_destination.astype(str)
    panel = d.set_index(["county_pair", "Year"])
    design = panel[OD_TYPES].to_numpy(dtype=float)
    rank = int(np.linalg.matrix_rank(design))
    rank_with_ones = int(np.linalg.matrix_rank(np.column_stack([np.ones(len(design)), design])))
    if rank != len(OD_TYPES) or rank_with_ones != rank + 1:
        raise ValueError(f"Unexpected rank or constant in {sample}/{outcome}/Q{quintile}: {rank}, {rank_with_ones}")
    model = PanelOLS(
        panel[outcome], panel[OD_TYPES], entity_effects=True, time_effects=True,
        drop_absorbed=True, check_rank=True,
    )
    if model.has_constant:
        raise ValueError("Policy-only design incorrectly classified as containing a constant")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit(cov_type="clustered", cluster_entity=True)
    warning_text = " | ".join(str(w.message) for w in caught)
    dof = float(result.df_resid)
    q95 = t.ppf(.975, dof)
    q90 = t.ppf(.95, dof)
    rows = []
    for policy in OD_TYPES:
        estimable = policy in result.params.index
        coefficient = float(result.params[policy]) if estimable else np.nan
        se = float(result.std_errors[policy]) if estimable else np.nan
        pvalue = float(result.pvalues[policy]) if estimable else np.nan
        rows.append({
            "sample": sample, "outcome": outcome, "outcome_label": OUTCOMES[outcome],
            "quintile": int(quintile), "policy": policy,
            "coefficient": coefficient, "std_error": se, "p_value": pvalue,
            "ci95_low": coefficient - q95 * se if estimable else np.nan,
            "ci95_high": coefficient + q95 * se if estimable else np.nan,
            "ci90_low": coefficient - q90 * se if estimable else np.nan,
            "ci90_high": coefficient + q90 * se if estimable else np.nan,
            "estimable": bool(estimable), "n_od": int(result.nobs),
            "n_pairs": int(d.county_pair.nunique()),
            "n_origin_counties": int(d.GEOID.nunique()),
            "n_destination_counties": int(d.GEOID_destination.nunique()),
            "year_min": int(d.Year.min()), "year_max": int(d.Year.max()),
            "df_resid": dof, "r2_within": float(result.rsquared_within),
            "r2": float(result.rsquared),
            "residual_RSS": float(np.sum(result.resids**2)),
            "rmse": float(np.sqrt(np.mean(result.resids**2))),
            "F_poolability_conventional": float(result.f_pooled.stat),
            "F_poolability_p_conventional": float(result.f_pooled.pval),
            "design_rank": rank, "rank_with_ones": rank_with_ones,
            "model_has_constant": bool(model.has_constant), "check_rank": True,
            "cluster": "county-pair", "fixed_effects": "county-pair and year",
            "model_warnings": warning_text,
        })
    return pd.DataFrame(rows)


def adjust_selected(results):
    focus = results.loc[results.policy.isin(SELECTED)].copy()
    for col in (
        "bh_q_within_outcome_quintile", "holm_p_within_outcome_quintile",
        "bh_q_within_outcome", "holm_p_within_outcome",
        "bh_q_global_160", "holm_p_global_160",
    ):
        focus[col] = np.nan
    primary = focus.loc[focus["sample"].eq("all_available")]
    for _, idx in primary.groupby(["outcome", "quintile"]).groups.items():
        valid = focus.loc[idx, "p_value"].notna()
        vi = focus.loc[idx].index[valid]
        if len(vi):
            ps = focus.loc[vi, "p_value"].to_numpy()
            focus.loc[vi, "bh_q_within_outcome_quintile"] = multipletests(ps, method="fdr_bh")[1]
            focus.loc[vi, "holm_p_within_outcome_quintile"] = multipletests(ps, method="holm")[1]
    for _, idx in primary.groupby("outcome").groups.items():
        valid = focus.loc[idx, "p_value"].notna()
        vi = focus.loc[idx].index[valid]
        if len(vi):
            ps = focus.loc[vi, "p_value"].to_numpy()
            focus.loc[vi, "bh_q_within_outcome"] = multipletests(ps, method="fdr_bh")[1]
            focus.loc[vi, "holm_p_within_outcome"] = multipletests(ps, method="holm")[1]
    gi = primary.index[focus.loc[primary.index, "p_value"].notna()]
    if len(gi):
        ps = focus.loc[gi, "p_value"].to_numpy()
        focus.loc[gi, "bh_q_global_160"] = multipletests(ps, method="fdr_bh")[1]
        focus.loc[gi, "holm_p_global_160"] = multipletests(ps, method="holm")[1]
    return focus


def metric_xlim(estimates, outcome):
    d = estimates.loc[estimates.outcome.eq(outcome)]
    low = min(float(d.ci95_low.min()), 0.0)
    high = max(float(d.ci95_high.max()), 0.0)
    span = max(high - low, 0.001)
    return low - .07 * span, high + .07 * span


def plot_figure(estimates, outcomes, stem, sample):
    d_all = estimates.loc[estimates["sample"].eq(sample) & estimates.policy.isin(SELECTED)].copy()
    xlims = {outcome: metric_xlim(d_all, outcome) for outcome in outcomes}
    variable_colors = [BLUE_COLOR, ORANGE_COLOR]
    fig = plt.figure(figsize=(19, 19), facecolor="white")
    grid = fig.add_gridspec(4, 2, wspace=.15, hspace=.22)
    plot_axes=[];plot_ids=[]
    for idx, policy in enumerate(SELECTED):
        outer_row, outer_col = divmod(idx, 2)
        nested = grid[idx].subgridspec(2, 2, height_ratios=[.14, .86], wspace=.08, hspace=.02)
        heading = fig.add_subplot(nested[0, :])
        heading.axis("off")
        heading.text(0, .44, chr(ord("a") + idx), transform=heading.transAxes,
                     ha="left", va="center", fontsize=29, fontweight="bold")
        heading.text(.06, .44, POLICY_LABELS[policy], transform=heading.transAxes,
                     ha="left", va="center", fontsize=19, fontweight="bold")
        for metric_idx, outcome in enumerate(outcomes):
            ax = fig.add_subplot(nested[1, metric_idx])
            plot_axes.append(ax);plot_ids.append(f'{chr(97+idx)}{metric_idx+1}')
            color = variable_colors[metric_idx]
            d = d_all.loc[d_all.policy.eq(policy) & d_all.outcome.eq(outcome)].sort_values("quintile")
            # Link the five quintile-specific point estimates, following their Q1-to-Q5 order.
            ax.plot(d.coefficient, d.quintile, color=color, lw=1.8,
                    linestyle=(0, (4, 3)), alpha=.9, zorder=1.6)
            for row in d.itertuples():
                y = int(row.quintile)
                ax.hlines(y, row.ci95_low, row.ci95_high, color=color, lw=1.15, alpha=.42,
                          capstyle="round", zorder=2)
                ax.hlines(y, row.ci90_low, row.ci90_high, color=color, lw=2.0,
                          capstyle="round", zorder=3)
                ax.scatter(row.coefficient, y, color=color, s=58, zorder=4)
            ax.axvline(0, color="#777777", lw=.9, ls="--", zorder=1)
            ax.set_xlim(*xlims[outcome])
            ax.set_ylim(.5, 5.5)
            ax.set_yticks([1, 2, 3, 4, 5], ["Q1", "Q2", "Q3", "Q4", "Q5"])
            show_ylabels = outer_col == 0 and metric_idx == 0
            ax.tick_params(axis="y", labelsize=17, length=0, pad=5, labelleft=show_ylabels)
            ax.tick_params(axis="x", labelsize=16.5, pad=5)
            ax.set_xlabel("")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="x", color="#dddddd", lw=.65, zorder=0)
    fig.subplots_adjust(left=.07, right=.99, top=.99, bottom=.095, wspace=.15, hspace=.22)
    figure_dir = OUT / "figures"
    source_dir = OUT / "data/figure_source_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    qa_dir=OUT/'qa/rankfix_20261025';qa_dir.mkdir(parents=True,exist_ok=True)
    fig.canvas.draw()
    d_all.loc[d_all.outcome.isin(outcomes)].to_csv(source_dir / f"{stem}.csv", index=False)
    fig.savefig(figure_dir / f"{stem}.png", dpi=600, facecolor="white", bbox_inches="tight")
    fig.savefig(figure_dir / f"{stem}.svg", facecolor="white", bbox_inches="tight")
    fig.savefig(figure_dir / f"{stem}.pdf", facecolor="white", bbox_inches="tight")
    fig.savefig(qa_dir/f'{stem}.preview.png',dpi=300,facecolor='white',bbox_inches='tight')
    plt.close(fig)


def plot_quintile_figure(estimates, outcomes, stem):
    plot_figure(estimates, outcomes, stem, "all_available")


def plot_sensitivity_figure(estimates, outcomes, stem):
    plot_figure(estimates, outcomes, stem, "exclude_2022_2023")


def format_dollars(value):
    value = float(value)
    sign = "−" if value < 0 else ""
    return f"{sign}${abs(value):,.0f}"


def quintile_ranges(data, outcome, sample):
    d = data.copy()
    if sample == "exclude_2022_2023":
        d = d.loc[d.flow_start_year.ne(2022)]
    values = d[outcome]
    if outcome.startswith("ln_"):
        display_values = np.exp(values) * 1000
        formatter = lambda x: f"{float(x):,.3g}"
    else:
        display_values = np.sinh(values) * 10000
        formatter = format_dollars
    work = pd.DataFrame({"quintile": d[f"quintile_{outcome}"], "value": display_values})
    intervals = []
    for q in range(1, 6):
        vals = work.loc[work.quintile.eq(q), "value"]
        intervals.append(f"Q{q} {formatter(vals.min())} to {formatter(vals.max())}")
    return f"{RANGE_LABELS[outcome]}: " + "; ".join(intervals)


def caption(data, stem, outcomes, sample):
    if sample == "all_available":
        sample_text = "The primary estimates use all 16,709 county-pair-year observations."
    else:
        sample_text = "The sensitivity analysis excludes the 2022–2023 IRS flow cohort; full-sample quintile cut points are retained."
    policy_text = "; ".join(f"{chr(97+i)}, {POLICY_LABELS[p]}" for i, p in enumerate(SELECTED))
    range_text = " Quintile value ranges in the plotted sample: " + " | ".join(
        quintile_ranges(data, outcome, sample) for outcome in outcomes
    ) + "."
    return (
        f"{stem.replace('_', ' ')}. Panels a–h show {policy_text}. "
        f"Within each policy panel, the left plot is {OUTCOMES[outcomes[0]]} and the right plot is {OUTCOMES[outcomes[1]]}. "
        f"Q1–Q5 run from the lowest to the highest county-pair-year value of that outcome. Blue denotes the left-hand variable; orange denotes the right-hand variable. "
        f"Points are coefficients from paired two-way fixed-effects regressions that include all 16 origin- and destination-level policy terms, county-pair fixed effects, and year fixed effects; standard errors are clustered by county pair. Dashed lines connect the coefficient estimates across Q1–Q5 within each variable. "
        f"Thick lines show 90% confidence intervals (CIs); thin lines show 95% CIs. {sample_text}{range_text} "
        "These are associations within outcome-defined migration-flow strata, not estimates for individual income or demographic groups."
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    data, cutpoints, bin_counts = load_and_quintile()
    if len(data) != 16709:
        print(f"NOTE: panel contains {len(data)} rows (historical reference panel had 16,709)", flush=True)
    rows = []
    for sample in ("all_available", "exclude_2022_2023"):
        for outcome in OUTCOMES:
            for quintile in range(1, 6):
                fitted = fit_model(data, outcome, quintile, sample)
                rows.append(fitted)
                first = fitted.iloc[0]
                print(f"{sample} / {outcome} / Q{quintile}: N={first.n_od}, pairs={first.n_pairs}", flush=True)
    results = pd.concat(rows, ignore_index=True)
    selected = adjust_selected(results)
    cutpoints.to_csv(OUT / "tables" / "results5_quintile_cutpoints.csv", index=False)
    bin_counts.to_csv(OUT / "tables" / "results5_quintile_bin_counts.csv", index=False)
    results.to_csv(OUT / "tables" / "results5_quintile_all_models.csv", index=False)
    selected.to_csv(OUT / "tables" / "results5_quintile_selected_effects.csv", index=False)
    metrics=results[['sample','outcome','quintile','n_od','n_pairs','df_resid','r2','r2_within',
        'residual_RSS','rmse','F_poolability_conventional','F_poolability_p_conventional',
        'design_rank','rank_with_ones','model_has_constant','check_rank']].drop_duplicates()
    metrics=metrics.rename(columns={'n_od':'N','n_pairs':'directed_pairs','r2':'R2',
        'r2_within':'R2_within','rmse':'RMSE'})
    metrics['pair_FE']=True;metrics['year_FE']=True;metrics['cluster']='directed county pair'
    metrics.to_csv(OUT/'tables/results5_publication_fit_metrics.csv',index=False)

    plot_quintile_figure(selected, PRIMARY_OUTCOMES, "Fig5_EconomicComposition_quintile_pair")
    plot_quintile_figure(selected, SUPPLEMENT_OUTCOMES, "FigS5_EconomicCompositionFlow_quintile_pair")
    plot_sensitivity_figure(selected, PRIMARY_OUTCOMES, "FigS6_EconomicComposition_quintile_excluding_2022_2023_pair")
    plot_sensitivity_figure(selected, SUPPLEMENT_OUTCOMES, "FigS7_EconomicCompositionFlow_quintile_excluding_2022_2023_pair")

    captions = {
        "Fig5_EconomicComposition_quintile_pair": (PRIMARY_OUTCOMES, "all_available"),
        "FigS5_EconomicCompositionFlow_quintile_pair": (SUPPLEMENT_OUTCOMES, "all_available"),
        "FigS6_EconomicComposition_quintile_excluding_2022_2023_pair": (PRIMARY_OUTCOMES, "exclude_2022_2023"),
        "FigS7_EconomicCompositionFlow_quintile_excluding_2022_2023_pair": (SUPPLEMENT_OUTCOMES, "exclude_2022_2023"),
    }
    for stem, (outcomes, sample) in captions.items():
        (OUT / "figures" / f"{stem}_caption.txt").write_text(caption(data, stem, outcomes, sample) + "\n", encoding="utf-8")

    primary = selected.loc[selected["sample"].eq("all_available")]
    correction_summary = {}
    for outcome in OUTCOMES:
        d = primary.loc[primary.outcome.eq(outcome)]
        correction_summary[outcome] = {
            "n_selected_tests": int(d.p_value.notna().sum()),
            "raw_p_lt_0_05": int((d.p_value < .05).sum()),
            "bh_q_lt_0_05_within_outcome_40_tests": int((d.bh_q_within_outcome < .05).sum()),
            "holm_p_lt_0_05_within_outcome_40_tests": int((d.holm_p_within_outcome < .05).sum()),
        }
    audit = {
        "input": INPUT.name, "rows": int(len(data)),
        "unique_county_pairs": int(data[["GEOID", "GEOID_destination"]].drop_duplicates().shape[0]),
        "year_min": int(data.Year.min()), "year_max": int(data.Year.max()),
        "outcomes_and_bin_counts": bin_counts.to_dict("records"),
        "cutpoints": cutpoints.to_dict("records"),
        "models_fitted": int(results[["sample", "outcome", "quintile"]].drop_duplicates().shape[0]),
        "coefficient_rows_all_16_terms": int(len(results)),
        "not_estimable_coefficients": int((~results.estimable).sum()),
        "check_rank": True, "constant_identified_in_any_model": bool(metrics.model_has_constant.any()),
        "design_ranks": sorted(metrics.design_rank.unique().tolist()),
        "rank_with_ones": sorted(metrics.rank_with_ones.unique().tolist()),
        "estimator_revision": "Rank-checked PanelOLS; corrected erroneous constant detection from check_rank=False in linearmodels 6.1",
        "missing_policy_estimates": results.loc[~results.estimable, ["sample", "outcome", "quintile", "policy", "model_warnings"]].to_dict("records"),
        "correction_summary_primary_eight_selected_effects": correction_summary,
        "multiple_testing": "Raw p-values and CIs are reported. BH/Holm are applied to 8 selected policies x 5 quintiles within each outcome (40 tests/outcome), and globally across 160 selected tests.",
        "design": "Each outcome is separately divided into pooled row-level quintiles using its own full-sample value. The same outcome is then regressed within each stratum. Sensitivity results omit 2022-2023 and retain full-sample cut points. All models include the original 16 paired policy regressors, county-pair FE, year FE, and county-pair clustered SE.",
        "interpretation": "Outcome-defined flow-observation strata are exploratory and do not identify individual migrant income or demographic groups.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "tables" / "results5_quintile_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    # Detailed project documentation is maintained in the three top-level README files.
    print(f"COMPLETE {OUT}; models={audit['models_fitted']}; coefficient_rows={len(results)}; not_estimable={audit['not_estimable_coefficients']}", flush=True)


if __name__ == "__main__":
    main()
