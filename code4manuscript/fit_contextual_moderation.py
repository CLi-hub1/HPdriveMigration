"""Expanded paired Results 4 moderation models and style-matched figures."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import textwrap
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels import PanelOLS
from scipy.stats import t
from statsmodels.stats.multitest import multipletests

from config import PROJECT_ROOT, OUTPUT_ROOT
PACKAGE = PROJECT_ROOT / 'results4'
OUT = OUTPUT_ROOT / 'results4'
TABLES = OUT / 'tables'
INPUT = PACKAGE / "data/panel_lag0.csv"
MED = PACKAGE / "data/results5_mediator_dataset_2016_2023.csv"
SELECTION = PACKAGE / "data/results3_selection_pair8.csv"
POLICIES = [
    "APTechnologicalInfrastructural", "APBehavioralCultural", "APNatureBased",
    "MPBehavioralCultural", "MPNatureBased", "APInstitutional_dest",
    "MPInstitutional_dest", "MPBehavioralCultural_dest",
]
KEY = ["GEOID", "GEOID_destination", "Year"]
PANEL_MODS = {
    "tree": ("tree_canopy_share_total_county_area", "Tree canopy coverage"),
    "hpi": ("ln_hpi", "Ln (FHFA house-price index)"),
    "rent_burden": ("rent_burden_30plus", "Rent burden (30%+ share)"),
    "gross_rent": ("ln_gross_rent", "Ln (gross rent)"),
    "cooling": ("cooling_public_indoor_area_share_total_county_area", "Public indoor facility area share"),
    "renewable_generation": ("renewable_generation_share", "Renewable generation share"),
}
MAIN_MODS = [
    ("income", "Ln (income)"), ("aging", "Ageing rate"),
    ("RD", "Racial diversity"), ("tree", "Tree canopy coverage"),
]
SUPP_MODS = [
    ("EA", "Educational attainment rate"), ("hpi", "Ln (FHFA house-price index)"),
    ("rent_burden", "Rent burden (30%+ share)"), ("gross_rent", "Ln (gross rent)"),
    ("cooling", "Public indoor facility area share"),
    ("renewable_generation", "Renewable generation share"),
]
ORANGE, BLUE = "#FA853E", "#1B6CA8"
plt.rcParams.update({
    "font.family": "Arial", "font.size": 6.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.3, "ytick.labelsize": 6.3, "axes.linewidth": .7,
    "xtick.major.width": .7, "ytick.major.width": .7, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.unicode_minus": True, "legend.frameon": False,
})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def indexed(df):
    d = df.copy()
    d["county_pair"] = d.GEOID.astype(str) + "_" + d.GEOID_destination.astype(str)
    return d.set_index(["county_pair", "Year"])


def fit(df, variables):
    d = indexed(df)
    x = sm.add_constant(d[variables], has_constant="add")
    model = PanelOLS(d.ln_outflow_rate, x, entity_effects=True, time_effects=True)
    return model.fit(cov_type="clustered", cluster_entity=True)


def attach_moderators(data):
    med = pd.read_csv(MED)
    med["GEOID"] = med.GEOID.astype(int)
    base = data.copy()
    for key, (source, _) in PANEL_MODS.items():
        sub = med[["GEOID", "Year", source]].copy()
        for suffix, county_col in [("origin", "GEOID"), ("dest", "GEOID_destination")]:
            rhs = sub.rename(columns={"GEOID": county_col, source: f"{key}_{suffix}"})
            base = base.merge(rhs, on=[county_col, "Year"], how="left", validate="many_to_one")
    return base


def load_inputs():
    data = pd.read_csv(INPUT)
    selected = pd.read_csv(SELECTION).Variable.tolist()
    assert selected == POLICIES, (selected, POLICIES)
    data = attach_moderators(data)
    mods = ["income", "aging", "RD", "EA", *PANEL_MODS.keys()]
    required = KEY + ["ln_outflow_rate"] + POLICIES + [f"{m}_{s}" for m in mods for s in ["origin", "dest"]]
    for col in KEY:
        data[col] = data[col].astype(int)
    assert not data.duplicated(KEY).any()
    miss = {c: int(data[c].isna().sum()) for c in required if data[c].isna().any()}
    return data, selected, miss


def model_rows(model, spec, selected, focal_interactions):
    result = pd.DataFrame({
        "Variable": model.params.index, "Coefficient": model.params.values,
        "Std. Error": model.std_errors.values, "P-value": model.pvalues.values,
        "Specification": spec, "Covariance": "pair", "Observations": int(model.nobs),
    })
    df = float(model.df_resid)
    for level in (90, 95):
        q = t.ppf(.5 + level / 200, df)
        result[f"CI{level}_lower"] = result.Coefficient - q * result["Std. Error"]
        result[f"CI{level}_upper"] = result.Coefficient + q * result["Std. Error"]
    result["is_policy_interaction"] = result.Variable.isin(focal_interactions)
    result["BH_q"] = np.nan
    result["Holm_p"] = np.nan
    mask = result.Variable.isin(focal_interactions)
    if mask.any():
        ps = result.loc[mask, "P-value"].to_numpy(float)
        result.loc[mask, "BH_q"] = multipletests(ps, method="fdr_bh")[1]
        result.loc[mask, "Holm_p"] = multipletests(ps, method="holm")[1]
    result["selection_sha256"] = sha(SELECTION)
    return result


def run_models(data, selected):
    mods = ["income", "aging", "RD", "EA", *PANEL_MODS.keys()]
    labels = dict(MAIN_MODS + SUPP_MODS[:1] + SUPP_MODS[1:])
    complete_n_by_moderator = {}
    coefs, curves, tests, metrics, supports, covs = [], [], [], [], [], []
    for degree in (1, 2):
        for moderator in mods:
            endpoint_columns = [f"{moderator}_{s}" for s in ["origin", "dest"]]
            d = data.dropna(subset=endpoint_columns).copy()
            complete_n_by_moderator[moderator] = len(d)
            xvars = selected.copy()
            interaction_terms = []
            for side in ("origin", "dest"):
                col = f"{moderator}_{side}"
                xvars.append(col)
                if degree == 2:
                    d[col + "_sq"] = d[col] ** 2
                    xvars.append(col + "_sq")
                supports.append({
                    "degree": degree, "Moderator": col, "N": len(d),
                    "min": d[col].min(), "q01": d[col].quantile(.01),
                    "q99": d[col].quantile(.99), "max": d[col].max(),
                    "missing_before_complete_case": int(data[col].isna().sum()),
                })
            for policy in selected:
                col = f"{moderator}_{'dest' if policy.endswith('_dest') else 'origin'}"
                for power in range(1, degree + 1):
                    term = policy + ":" + col + ("_sq" if power == 2 else "")
                    d[term] = d[policy] * d[col] ** power
                    xvars.append(term)
                    interaction_terms.append(term)
            model = fit(d, xvars)
            spec = f"degree{degree}_{moderator}"
            coefs.append(model_rows(model, spec, selected, interaction_terms))
            for policy in selected:
                col = f"{moderator}_{'dest' if policy.endswith('_dest') else 'origin'}"
                terms = [policy, policy + ":" + col]
                if degree == 2:
                    terms.append(policy + ":" + col + "_sq")
                b = model.params.loc[terms].to_numpy(float)
                vcv = model.cov.loc[terms, terms].to_numpy(float)
                for value in np.linspace(d[col].quantile(.01), d[col].quantile(.99), 100):
                    v = np.array([1.0, value] + ([value ** 2] if degree == 2 else []))
                    effect, variance = float(v @ b), float(v @ vcv @ v)
                    se = np.sqrt(variance) if variance >= 0 else np.nan
                    q = t.ppf(.975, model.df_resid)
                    curves.append({
                        "Specification": spec, "degree": degree, "Covariance": "pair",
                        "Policy": policy, "Moderator": col, "Moderator_value": value,
                        "Effect": effect, "CI95_lower": effect-q*se, "CI95_upper": effect+q*se,
                        "variance": variance, "N": len(d), "selection_sha256": sha(SELECTION),
                    })
                inter = [policy + ":" + col] + ([policy + ":" + col + "_sq"] if degree == 2 else [])
                ix = model.params.index.get_indexer(inter)
                wald = model.wald_test(np.eye(len(model.params))[ix])
                tests.append({
                    "Specification": spec, "Policy": policy, "Moderator": col,
                    "degree": degree, "joint_interaction_stat": float(wald.stat),
                    "joint_interaction_p": float(wald.pval), "N": int(model.nobs),
                })
            for r in model.params.index:
                for c in model.params.index:
                    covs.append({"Specification": spec, "Row_term": r, "Column_term": c,
                                 "Covariance_value": float(model.cov.loc[r, c])})
            resid = np.asarray(model.resids)
            metrics.append({"Specification": spec, "N": int(model.nobs), "R2": float(model.rsquared),
                            "R2_within": float(model.rsquared_within), "RMSE": float(np.sqrt(np.mean(resid**2)))})
            print(f"{spec}: N={int(model.nobs)}", flush=True)
    return (pd.concat(coefs, ignore_index=True), pd.DataFrame(curves), pd.DataFrame(tests),
            pd.DataFrame(metrics), pd.DataFrame(supports), pd.DataFrame(covs), complete_n_by_moderator)


def policy_name(policy):
    side = "destination" if policy.endswith("_dest") else "origin"
    base = policy.removesuffix("_dest")
    prefix, cat = base[:2], base[2:]
    cat = {"TechnologicalInfrastructural": "Technological / infrastructural",
           "Institutional": "Institutional", "BehavioralCultural": "Behavioral / cultural",
           "NatureBased": "Nature-based"}[cat]
    return f"{cat} {prefix} ({side})"


def plot_grid(curves, selected, moderators, labels, stem, ncols, legend):
    d = curves[(curves.degree == 2) & (curves.Covariance == "pair")].copy()
    assert set(d.Policy) == set(selected)
    fig, axes = plt.subplots(2, ncols, figsize=(183/25.4, 126/25.4), squeeze=False)
    fig.subplots_adjust(left=.091, right=.779, bottom=.112, top=.962,
                        wspace=.39 if ncols == 4 else .30, hspace=.23)
    colors = plt.get_cmap("tab10").colors
    for row, side in enumerate(["origin", "dest"]):
        policies = [p for p in selected if p.endswith("_dest") == (side == "dest")]
        for col, moderator in enumerate(moderators):
            ax = axes[row, col]
            low, high = [0], [0]
            if policies:
                for i, policy in enumerate(policies):
                    column = f"{moderator}_{side}"
                    c = d[(d.Policy == policy) & (d.Moderator == column)].sort_values("Moderator_value")
                    assert len(c) == 100 and np.all(np.diff(c.Moderator_value) > 0)
                    ax.plot(c.Moderator_value, c.Effect, color=colors[i % 10], lw=.85, zorder=3)
                    ax.fill_between(c.Moderator_value, c.CI95_lower, c.CI95_upper,
                                    color=colors[i % 10], alpha=.2, lw=0, zorder=2)
                    low.extend([c.Effect.min(), c.CI95_lower.min()])
                    high.extend([c.Effect.max(), c.CI95_upper.max()])
                lo, hi = float(np.nanmin(low)), float(np.nanmax(high))
                span = hi-lo
                ax.set_ylim(lo-.04*span, hi+.05*span)
            else:
                ax.text(.5, .5, "No selected\n" + side + " type", ha="center", va="center",
                        transform=ax.transAxes, fontsize=6)
            ax.axhline(0, color="#777777", lw=.5, ls="--", zorder=1)
            label=labels[moderator]
            if len(label)>19:label='\n'.join(textwrap.wrap(label,19))
            ax.set_xlabel(label, fontsize=8.1, labelpad=3)
            ax.set_ylabel("migration impact" if col==0 else "", fontsize=8.1, labelpad=4)
            ax.tick_params(labelsize=7.3,pad=2)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            if col == 0:
                ax.annotate("ab"[row], (0, 1), xycoords="axes fraction",
                            xytext=(-31, 1), textcoords="offset points",
                            fontsize=13.5, weight="bold", va="top")
        if legend and policies:
            handles = [Line2D([], [], color=colors[i % 10], lw=.8,
                              label="\n".join(textwrap.wrap(policy_name(p), 23)))
                       for i, p in enumerate(policies)]
            axes[row, -1].legend(handles=handles, loc="center left", bbox_to_anchor=(1.08, .5),
                                 fontsize=7.1, handlelength=1.7, labelspacing=.85,handletextpad=.65)
    fig.canvas.draw()
    qa=OUT/'qa/2026-10-25';qa.mkdir(parents=True,exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    (OUT / "data" / "figure_source_data").mkdir(parents=True, exist_ok=True)
    source = d[d.Moderator.apply(lambda x: any(x.startswith(m + "_") for m in moderators))]
    source.to_csv(
        OUT / "data" / "figure_source_data" / f"{stem}.csv", index=False)
    fig.savefig(OUT / 'figures' / f'{stem}.png',dpi=600,facecolor='white')
    fig.savefig(OUT / 'figures' / f'{stem}.svg',facecolor='white')
    fig.savefig(OUT / 'figures' / f'{stem}.pdf',facecolor='white')
    fig.savefig(qa/f'{stem}.preview.png',dpi=300,facecolor='white')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    data, selected, missing = load_inputs()
    coef, curves, tests, metrics, support, covs, complete_n_by_moderator = run_models(data, selected)
    coef.to_csv(TABLES / "results4_moderation_models.csv", index=False)
    curves.to_csv(TABLES / "results4_marginal_effects.csv", index=False)
    tests.to_csv(TABLES / "results4_joint_interaction_tests.csv", index=False)
    metrics.to_csv(TABLES / "results4_fit_metrics.csv", index=False)
    support.to_csv(TABLES / "results4_moderator_support.csv", index=False)
    covs.to_csv(TABLES / "results4_covariance_matrices.csv", index=False)
    main_keys, main_labels = zip(*MAIN_MODS)
    supp_groups = [
        (SUPP_MODS[:3], "FigS4a_Results4_SupplementaryModerators_pair"),
        (SUPP_MODS[3:], "FigS4b_Results4_SupplementaryModerators_pair"),
    ]
    plot_grid(curves, selected, list(main_keys), dict(MAIN_MODS), "Fig4_DynamicModeration_pair", 4, True)
    for group, stem in supp_groups:
        plot_grid(curves, selected, [key for key, _ in group], dict(group), stem, 3, True)
    manifest = {
        "status": "COMPLETE", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_source": SELECTION.name, "selection_sha256": sha(SELECTION),
        "input_od_panel": INPUT.name, "input_od_sha256": sha(INPUT),
        "moderator_panel": MED.name, "moderator_panel_sha256": sha(MED),
        "selected_policy_effects": selected, "selected_effect_count": len(selected),
        "main_moderators": dict(MAIN_MODS), "supplementary_moderators": dict(SUPP_MODS),
        "supplementary_figures": [stem for _, stem in supp_groups],
        "missing_before_complete_case": missing, "complete_case_N_by_moderator": complete_n_by_moderator,
        "model_count": int(len(metrics)), "inference": "pair-clustered standard errors; county-pair and year fixed effects",
        "figure_style": "Original paired Fig.4 plotting style; main panel order replaces education with tree canopy; six additional moderators in matching two-row panel style",
    }
    (TABLES / "analysis_manifest_rerun.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    sample_note = "Moderator-specific complete-case N: " + "; ".join(
        f"{name}={n:,}" for name, n in complete_n_by_moderator.items()) + ". "
    (TABLES / "README_generated_run.md").write_text(
        "# Expanded Results 4 moderation analysis\n\n"
        "Uses the fixed eight-effect paired Results 3 set. Main Fig. 4 moderators: log income, ageing share, racial diversity, and tree canopy. "
        "Educational attainment, log FHFA HPI, rent burden 30%+, log gross rent, OSM-mapped indoor facility area share, and renewable generation share are reported in two supplementary figures, each preserving the two-row endpoint layout. "
        "For each moderator, linear and quadratic two-endpoint specifications are estimated with pair and year fixed effects and pair-clustered standard errors. "
        + sample_note
        + "Figures preserve the original Fig. 4 visual conventions; curves are policy marginal associations with pointwise 95% intervals over the 1st–99th percentile support. "
        "This is a post-selection exploratory analysis, not causal effect modification evidence.\n", encoding="utf-8")
    main_n = {k: v for k, v in complete_n_by_moderator.items() if k in dict(MAIN_MODS)}
    print(f"COMPLETE {OUT}; selected={len(selected)}; main_N_by_moderator={main_n}; models={len(metrics)}", flush=True)


if __name__ == "__main__":
    main()
