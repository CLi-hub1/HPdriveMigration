"""Create the updated eight-term Fig. 3 with the project's original plot style."""
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
import numpy as np
import pandas as pd


from config import PROJECT_ROOT, OUTPUT_ROOT
TABLES = PROJECT_ROOT / 'results3/tables'
OUT = OUTPUT_ROOT / 'figures_type_sensitivity'
OUT.mkdir(parents=True, exist_ok=True)
ORANGE = "#FA853E"
BLUE = "#1B6CA8"
SPECS = [
    ("baseline", "Main specification", "main"),
    ("RE", "Random effects", "re"),
    ("winsor1", "1% winsorizing", "sensitivity"),
    ("trim5", "5% trimming", "sensitivity"),
    ("sample80", "Random 80% samples", "sensitivity"),
    ("exclude2020", "Excluding 2020 samples", "sensitivity"),
    ("lag1", "One-year lag", "lag"),
    ("lag2", "Two-year lag", "lag"),
]
plt.rcParams.update({
    "font.family": "Arial", "font.size": 6.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.3, "ytick.labelsize": 6.3, "axes.linewidth": .7,
    "xtick.major.width": .7, "ytick.major.width": .7, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.unicode_minus": True, "legend.frameon": False,
})


def number(x, _):
    return f"{0 if abs(x) < 5e-8 else x:.3f}"


def name(variable):
    side = "destination" if variable.endswith("_dest") else "origin"
    base = variable.removesuffix("_dest")
    prefix, category = base[:2], base[2:]
    category = {
        "TechnologicalInfrastructural": "Technological / infrastructural",
        "Institutional": "Institutional", "BehavioralCultural": "Behavioral / cultural",
        "NatureBased": "Nature-based",
    }[category]
    return f"{category} {prefix} ({side})"


def p_label(p):
    return "<0.001" if p < .001 else f"{p:.3f}" if np.isfinite(p) else "NA"


def main():
    data = pd.read_csv(TABLES / "results3_sensitivity_selected8.csv")
    variables = list(dict.fromkeys(data.Variable))
    parts = []
    for order, (spec, label, kind) in enumerate(SPECS):
        model = "RE" if spec == "RE" else "TWFE"
        actual = "baseline_RE" if spec == "RE" else spec
        d = data[(data.Specification == actual) & (data.Model == model) & data.Variable.isin(variables)].copy()
        assert set(d.Variable) == set(variables), (spec, set(d.Variable) ^ set(variables))
        d["row_order_bottom_to_top"] = order
        d["display_specification"] = label
        d["display_family"] = kind
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)

    ncol = 2 if len(variables) > 1 else 1
    nrow = max(1, int(np.ceil(len(variables) / ncol)))
    fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                             figsize=(183 / 25.4, 61 * nrow / 25.4))
    fig.subplots_adjust(left=.162, right=.983, bottom=.093 if nrow == 2 else .12,
                        top=.972, wspace=.285, hspace=.42)
    for i, ax in enumerate(axes.flat):
        if i >= len(variables):
            ax.set_visible(False)
            continue
        variable = variables[i]
        dd = d[d.Variable == variable].sort_values("row_order_bottom_to_top")
        ax.axhspan(1.5, 5.5, facecolor="#EDF6F9", zorder=0)
        ax.axhspan(5.5, 7.5, facecolor="#FFF4ED", zorder=0)
        for yy in [1.5, 5.5, 7.5]:
            ax.axhline(yy, color="#BDBDBD", lw=.35, zorder=1)
        main_effect = float(dd.loc[dd.display_family == "main", "Coefficient"].iloc[0])
        ax.axvline(0, color="#888888", ls=(0, (2.5, 2)), lw=.55, zorder=1)
        ax.axvline(main_effect, color="#BDBDBD", ls=":", lw=.45, zorder=1)
        annotations = []
        for _, r in dd.iterrows():
            color = {"main": "#808080", "re": "#000000", "sensitivity": BLUE, "lag": ORANGE}[r.display_family]
            yy = r.row_order_bottom_to_top
            ax.hlines(yy, r.CI95_lower, r.CI95_upper, color=color, lw=.55, alpha=.52, zorder=2)
            ax.hlines(yy, r.CI90_lower, r.CI90_upper, color=color, lw=.9, zorder=3)
            ax.plot(r.Coefficient, yy, "o", ms=3.7, color=color, mew=.65,
                    mfc=color if r.display_family == "main" else "white", zorder=4)
            annotations.append(ax.annotate(
                p_label(r["P-value"]), (r.Coefficient, yy), xytext=(4.5, 1),
                textcoords="offset points", ha="left", va="bottom", fontsize=5.5,
                annotation_clip=False, zorder=5))
        lo, hi = min(dd.CI95_lower.min(), 0), max(dd.CI95_upper.max(), 0)
        span = hi - lo
        ax.set_xlim(lo - .12 * span, hi + .24 * span)
        ax.set_ylim(-1, 8)
        fig.canvas.draw()
        refs = [ax.transData.transform((v, 0))[0] for v in [0, main_effect]]
        for annotation in annotations:
            box = annotation.get_window_extent(fig.canvas.get_renderer())
            if any(box.x0 - 1 < x < box.x1 + 1 for x in refs):
                annotation.set_position((-4.5, 1))
                annotation.set_ha("right")
                box = annotation.get_window_extent(fig.canvas.get_renderer())
                if any(box.x0 - 1 < x < box.x1 + 1 for x in refs):
                    annotation.xy = (max(annotation.xy[0], 0, main_effect), annotation.xy[1])
                    annotation.set_position((4.5, 1))
                    annotation.set_ha("left")
        ax.set_yticks(range(8))
        ax.set_yticklabels([s[1] for s in SPECS] if i % ncol == 0 else [])
        ax.set_xlabel("\n".join(textwrap.wrap(name(variable), width=37)), fontsize=6.8, labelpad=5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.xaxis.set_major_formatter(FuncFormatter(number))
        ax.spines[["top", "right"]].set_visible(False)
        ax.annotate(chr(97 + i), (0, 1), xycoords="axes fraction",
                    xytext=(-78 if i % ncol == 0 else -18, 0), textcoords="offset points",
                    fontsize=11, weight="bold", ha="left", va="top", annotation_clip=False)

    stem = OUT / "Fig3_SelectedSensitivity_pair"
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    print(f"Saved original-style Fig. 3 to {stem}.[png|svg|pdf]")


if __name__ == "__main__":
    main()
