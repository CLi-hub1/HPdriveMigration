"""Restore three search-series ED Fig. 2 from completed, traceable grouped CV."""
from pathlib import Path
import hashlib
import json
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from scipy.stats import t

from config import PROJECT_ROOT, OUTPUT_ROOT
SEARCH = PROJECT_ROOT / "results1/data/classifier_search"
OUT = OUTPUT_ROOT / "figures_classifier"
FIG, DATA, QA = [OUT / p for p in ["figures", "source_data", "qa"]]
for p in [FIG, DATA, QA]:
    p.mkdir(parents=True, exist_ok=True)

NAMES = ["Loss", "Accuracy", "Precision", "Recall", "F1", "ROC AUC"]
STRATEGIES = ["Grid search", "Random search", "Bayesian optimization"]
COLORS = ["#FA873E", "#2472AC", "#7FC79B"]
CAPTION = (
    "**Extended Data Fig. 2 | Equal-budget hyperparameter searches using grouped internal cross-validation.** "
    "Orange, blue and green diamonds denote the five validation-fold scores of the best candidate "
    "selected by grid search, random search and Bayesian optimization (Optuna TPE), respectively. "
    "Each strategy evaluates 12 candidates on the same five folds of 964 unique previously labelled "
    "texts in 95 reporting-account/text-connected groups. Every candidate/fold starts from fresh "
    "public pretrained DistilBERT weights; neither fine-tuned weights nor checkpoints cross folds. "
    "Learning rates range from 10⁻⁵ to 5×10⁻⁵, training lasts 3 or 5 epochs, and batch size is 8 or 16. "
    "The selection objective is the unweighted mean fold micro-F1. Loss is mean four-label binary "
    "cross-entropy; accuracy is exact-match subset accuracy; precision, recall and F1 use micro "
    "averaging; ROC AUC is the mean of four label-specific AUCs. Classification uses sigmoid "
    "probability >0.5. Grey marks show the fold mean and Student-t 95% interval. Training folds "
    "overlap, and the same fold scores selected the candidate: these intervals describe fold "
    "variation and the scores are selection-optimistic internal CV, not nested or recent-year "
    "external validation. The fixed-parameter model used to generate all current regression labels "
    "is unchanged (10⁻⁵ learning rate, 5 epochs, batch size 8); its fixed-parameter grouped-CV reference "
    "is provided separately in Source Data. No new search winner is silently substituted into the "
    "policy measures. One seeded run per candidate is compared; this does not establish a general "
    "ranking of search algorithms. Source Data include all 36 candidates, 180 fitted folds and selected OOF predictions."
)


def main():
    assert json.loads((SEARCH / "completion.json").read_text())["status"] == "COMPLETE"
    df = pd.read_csv(SEARCH / "selected_candidate_fold_metrics.csv")
    assert len(df) == 15 and set(df.strategy) == set(STRATEGIES)
    assert df.groupby("strategy").fold.nunique().eq(5).all()
    assert np.isfinite(df[NAMES].to_numpy()).all()
    assert df.groupby("strategy").n_validation.sum().eq(964).all()
    df.to_csv(DATA / "ExtendedDataFig2_selected_fold_metrics.csv", index=False)
    reference = pd.read_csv(PROJECT_ROOT / "results1/data/figure_source_data/classifier_search/Frozen_regression_classifier_reference_folds.csv")
    reference["role"] = "Fixed-parameter classifier actually used for the current policy labels; not an HPO winner"
    reference.to_csv(DATA / "Frozen_regression_classifier_reference_folds.csv", index=False)

    plt.rcParams.update({"font.family": "Arial", "font.size": 12, "axes.labelsize": 15,
                         "xtick.labelsize": 15, "ytick.labelsize": 12, "axes.linewidth": 1.8,
                         "xtick.major.width": 1.8, "ytick.major.width": 1.8,
                         "xtick.major.size": 4, "ytick.major.size": 4,
                         "ytick.minor.size": 2, "ytick.minor.width": 1,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(183/25.4, (183*840/1259)/25.4))
    fig.subplots_adjust(left=.112, right=.976, bottom=.115, top=.928)
    summary = []
    for metric_i, metric in enumerate(NAMES):
        if metric_i % 2 == 0:
            ax.axvspan(metric_i-.5, metric_i+.5, color="#ECECEC", ec="#A8A8A8", lw=.35, zorder=0)
        for strategy_i, strategy in enumerate(STRATEGIES):
            values = df.loc[df.strategy.eq(strategy)].sort_values("fold")[metric].to_numpy()
            mean = values.mean()
            half = t.ppf(.975, 4) * values.std(ddof=1) / np.sqrt(5)
            x = metric_i + [-.32, 0, .32][strategy_i]
            # Fixed deterministic offsets retain all five fold points.
            ax.scatter(x + np.linspace(-.032, .032, 5), values, marker="D",
                       s=15, color=COLORS[strategy_i], edgecolors="none", zorder=2)
            ax.vlines(x, mean-half, mean+half, color="#555555", lw=.9, zorder=3)
            ax.hlines([mean-half, mean+half], x-.07, x+.07,
                      color="#555555", lw=.9, zorder=3)
            ax.hlines(mean, x-.14, x+.14, color="#555555", lw=1.2, zorder=3)
            summary.append({"strategy": strategy, "metric": metric, "n_folds": 5,
                            "mean": mean, "CI95_lower": mean-half, "CI95_upper": mean+half})
    stats = pd.DataFrame(summary)
    stats.to_csv(DATA / "ExtendedDataFig2_fold_mean_intervals.csv", index=False)
    assert stats.CI95_lower.min() >= -.1 and stats.CI95_upper.max() <= 1.1
    ax.set(xlim=(-.5, 5.5), ylim=(-.1, 1.1))
    ax.set_xticks(range(6), ["Loss", "Accuracy", "Precision", "Recall", "F1 score", "ROC AUC"])
    ax.set_ylabel("Range")
    ax.yaxis.set_major_locator(MultipleLocator(.2))
    ax.yaxis.set_minor_locator(MultipleLocator(.1))
    ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([], [], marker="D", color=c, linestyle="none", markersize=6.4)
               for c in COLORS]
    fig.legend(handles, ["Grid search", "Random optimization", "Bayesian optimization"],
               frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.55, .996),
               fontsize=12, columnspacing=.8, handletextpad=.3)
    fig.canvas.draw()
    fig.savefig(FIG / "ExtendedDataFig2.pdf", dpi=600, facecolor="white")
    fig.savefig(FIG / "ExtendedDataFig2.svg", dpi=600, facecolor="white")
    fig.savefig(FIG / "ExtendedDataFig2.png", dpi=600, facecolor="white")
    plt.close(fig)
    (DATA / "figure_caption.md").write_text(CAPTION + "\n", encoding="utf-8")
    checks = {
        "selected_candidates": 3, "selected_folds": 15, "plotted_metric_points": 90,
        "source_sha256": hashlib.sha256((SEARCH / "selected_candidate_fold_metrics.csv").read_bytes()).hexdigest(),
        "original_aspect_ratio": "1259:840", "updated_width_mm": 183,
        "updated_height_mm": 183*840/1259,
        "original_series_order": STRATEGIES, "original_representative_colors": COLORS,
        "all_data_and_intervals_within_axes": True,
        "new_search_used_to_generate_existing_policy_labels": False,
        "main_classifier_reference_separately_retained": True,
        "interpretation": "Selected internal CV is not unbiased external validation.",
    }
    (QA / "numeric_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
