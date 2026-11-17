"""Reconstruct supplementary Figures S1, S4 and S5 from saved plot data.

The historical plot layout code is unavailable. This implementation preserves
the saved observations, fitted curves and statistics, but uses a new layout.
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from config import PROJECT_ROOT, OUTPUT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('figure', choices=['S1', 'S4', 'S5'])
    fig_id = parser.parse_args().figure
    root = PROJECT_ROOT / 'data4results/02_Supplementary_figures'
    output = OUTPUT_ROOT / 'supplementary_descriptive_figures'
    output.mkdir(parents=True, exist_ok=True)
    if fig_id == 'S1':
        data = pd.read_csv(root / 'Figure_S01/FigS1_global_US_raw_reporting_rows.csv')
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        for ax, kind in zip(axes, ['adaptation', 'mitigation']):
            part = data[data.kind.eq(kind)].sort_values('year')
            ax.bar(part.year, part.US, label='United States')
            ax.bar(part.year, part.Others, bottom=part.US, label='Other countries')
            ax.set(title=kind.capitalize(), xlabel='Reporting year', ylabel='Reported rows')
        axes[1].legend(frameon=False)
    else:
        folder = root / f'Figure_S{int(fig_id[1:]):02d}'
        filename = 'FigS4_origin_county_year.csv' if fig_id == 'S4' else 'FigS5_origin_county_year_destination_partner_means.csv'
        points = pd.read_csv(folder / filename)
        curves = pd.read_csv(folder / 'fitted_lines_and_95CI.csv')
        statistics = pd.read_csv(folder / 'correlation_and_regression_statistics.csv')
        fig, axes = plt.subplots(4, 2, figsize=(10, 13), constrained_layout=True)
        for ax, row in zip(axes.flat, statistics.itertuples()):
            curve = curves[curves.Variable.eq(row.Variable)].sort_values('X')
            ax.scatter(points[row.Variable], points.sample_migration_rate, s=8, alpha=.5)
            ax.plot(curve.X, curve.Fitted, color='0.3')
            ax.fill_between(curve.X.to_numpy(), curve.CI_low.to_numpy(), curve.CI_high.to_numpy(), color='0.8', alpha=.6)
            ax.set(xlabel=row.Variable, ylabel='Sample outflow rate')
            ax.text(.97, .97, f'Pearson r = {row.Pearson_r:.3f}\nSpearman rho = {row.Spearman_rho:.3f}\nOLS slope = {row.Slope_beta:.4g}',
                    ha='right', va='top', transform=ax.transAxes, fontsize=8)
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(output / f'Figure_{fig_id}.{ext}', dpi=300, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
