"""Compare selected reproduced models with the frozen manuscript results."""
import numpy as np
import pandas as pd

from config import PROJECT_ROOT
from model_helpers import TOTAL, TYPES, fit, load_panel
import fit_income_flow_strata as income
import fit_type_sensitivity as sensitivity


def main():
    # Re-estimate the aggregate and joint 16-term contemporaneous models.
    panel = load_panel(PROJECT_ROOT / 'results2/data/frozen_inputs/panel_lag0.csv', 0)
    reference = pd.read_csv(PROJECT_ROOT / 'results2/data/full_model_archive/all_aggregate_and_type_models.csv')
    for variables, family in ((TOTAL, 'aggregate'), (TYPES, 'types16')):
        model = fit(panel, variables, 'pair')
        expected = reference[(reference.Specification == 'baseline_' + family)
                             & (reference.Model == 'TWFE') & (reference.Covariance == 'pair')].set_index('Variable')
        error = max(abs(model.params[v] - expected.loc[v, 'Coefficient']) for v in variables)
        assert error < 1e-10, (family, error)
        print(f'PASS baseline {family}: n={model.nobs}, max coefficient difference={error:.3g}')

    # Re-estimate the sensitivity specifications and compare all 128 terms.
    panels = {lag: sensitivity.read_panel(sensitivity.INPUT / f'panel_lag{lag}.csv', lag) for lag in (0, 1, 2)}
    frames = sensitivity.transform_specs(panels)
    actual = []
    for spec, frame in frames.items():
        actual.extend(sensitivity.result_rows(sensitivity.fit(frame, 'TWFE'), spec, frame))
        if spec == 'baseline':
            actual.extend(sensitivity.result_rows(sensitivity.fit(frame, 'RE'), 'baseline_RE', frame))
    actual = pd.DataFrame(actual)
    expected = pd.read_csv(PROJECT_ROOT / 'results3/tables/results3_sensitivity_full16.csv')
    both = actual.merge(expected, on=['Variable', 'Specification', 'Model'],
                        suffixes=('_new', '_reference'), validate='one_to_one')
    assert len(both) == 128
    for column in ('Coefficient', 'Std. Error', 'P-value'):
        error = np.max(np.abs(both[column + '_new'] - both[column + '_reference']))
        assert error < 1e-9, (column, error)
    print('PASS type sensitivity: 128 coefficients, standard errors and P values')

    # Check one outcome-stratum model, including the corrected rank setting.
    data, _, _ = income.load_and_quintile()
    result = income.fit_model(data, 'asinh_agi_intensity_10k', 5, 'all_available').set_index('policy')
    expected = pd.read_csv(PROJECT_ROOT / 'results5/tables/results5_quintile_all_models.csv')
    expected = expected[(expected['sample'] == 'all_available')
                        & (expected['outcome'] == 'asinh_agi_intensity_10k')
                        & (expected['quintile'] == 5)].set_index('policy')
    error = (result.coefficient - expected.coefficient).abs().max()
    assert error < 1e-10, error
    print(f'PASS AGI-intensity Q5: n={result.n_od.iloc[0]}, max coefficient difference={error:.3g}')


if __name__ == '__main__':
    main()
