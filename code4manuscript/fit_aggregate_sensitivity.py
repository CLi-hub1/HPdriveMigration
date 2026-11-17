"""Estimate the four aggregate AP/MP terms across the Fig. 2 specifications."""
from pathlib import Path
import argparse

import pandas as pd

from config import OUTPUT_ROOT
from model_helpers import TOTAL, fit, rows
from fit_type_sensitivity import INPUT, read_panel, transform_specs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_ROOT / 'aggregate_sensitivity')
    output = parser.parse_args().output_dir
    if output.exists() and any(output.iterdir()):
        raise SystemExit('Select an empty output directory.')
    output.mkdir(parents=True, exist_ok=True)
    panels = {lag: read_panel(INPUT / f'panel_lag{lag}.csv', lag) for lag in (0, 1, 2)}
    specifications = transform_specs(panels)
    # Keep the original paired fixed effects and random-effects comparator.
    result = []
    for name in ('baseline', 'winsor1', 'trim5', 'sample80', 'exclude2020', 'lag1', 'lag2'):
        frame = specifications[name]
        model = fit(frame, TOTAL, 'pair', kind='TWFE')
        result.append(rows(model, 'TWFE', 'pair', name, TOTAL))
        if name == 'baseline':
            model = fit(frame, TOTAL, 'pair', kind='RE')
            result.append(rows(model, 'RE', 'pair', 'baseline_RE', TOTAL))
    result = pd.concat(result, ignore_index=True)
    assert len(result) == 32
    result.to_csv(output / 'aggregate_sensitivity_full_precision.csv', index=False)
    print(f'Wrote {len(result)} coefficient rows.')


if __name__ == '__main__':
    main()
