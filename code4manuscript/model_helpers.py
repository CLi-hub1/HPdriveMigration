"""Results 2 -> one selection manifest -> Results 3/4. Never rewrites earlier results."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import shutil
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t
from statsmodels.stats.multitest import multipletests
from linearmodels import PanelOLS, RandomEffects

CATEGORIES = ['TechnologicalInfrastructural', 'Institutional', 'BehavioralCultural', 'NatureBased']
TYPES = [p+c+s for s in ['', '_dest'] for p in ['AP', 'MP'] for c in CATEGORIES]
TOTAL = ['APtotal', 'MPtotal', 'APtotal_dest', 'MPtotal_dest']
MODS = ['income', 'aging', 'RD', 'EA']
KEY = ['GEOID', 'GEOID_destination', 'Year']
PRIMARY = 'pair'
THRESHOLD = .1

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def indexed(df):
    d = df.copy()
    d['county_pair'] = d.GEOID.astype(str) + '_' + d.GEOID_destination.astype(str)
    return d.set_index(['county_pair', 'Year'])

def fit(df, variables, cluster, kind='TWFE', constant=False):
    d = indexed(df)
    x = sm.add_constant(d[variables], has_constant='add') if constant else d[variables]
    if kind == 'PooledOLS':
        return sm.OLS(d.ln_outflow_rate, sm.add_constant(x, has_constant='add')).fit()
    model = PanelOLS(d.ln_outflow_rate, x, entity_effects=True, time_effects=True) if kind == 'TWFE' else RandomEffects(d.ln_outflow_rate, x)
    opts = {'cluster_entity': True} if cluster == 'pair' else {'clusters': d[['GEOID', 'GEOID_destination']]}
    return model.fit(cov_type='clustered', **opts)

def rows(model, kind, cluster, specification, family):
    panel = hasattr(model, 'std_errors')
    se = model.std_errors if panel else model.bse
    p = model.pvalues
    df = float(model.df_resid)
    result = pd.DataFrame({'Variable': model.params.index, 'Coefficient': model.params.values,
                           'Std. Error': se.values, 'P-value': p.values,
                           'Model': kind, 'Covariance': cluster, 'Specification': specification,
                           'Observations': int(model.nobs), 'df_resid': df})
    for level in [90, 95]:
        z = t.ppf(.5+level/200, df)
        result[f'CI{level}_lower'] = result.Coefficient-z*result['Std. Error']
        result[f'CI{level}_upper'] = result.Coefficient+z*result['Std. Error']
    result['BH_q'] = np.nan
    result['Holm_p'] = np.nan
    family_mask = result.Variable.isin(family)
    use = family_mask & np.isfinite(result['P-value'])
    if use.any():
        family_p = result.loc[family_mask, 'P-value'].fillna(1).to_numpy()
        for column,method in [('BH_q','fdr_bh'),('Holm_p','holm')]:
            adjusted = pd.Series(multipletests(family_p,method=method)[1],index=result.index[family_mask])
            result.loc[use,column] = adjusted.loc[use[use].index]
    result['multiplicity_family_size'] = len(family)
    result['family_estimable_p_count'] = int(use.sum())
    result['nominal_p_lt_0_1'] = result['P-value'] < .1
    result['nominal_p_lt_0_05'] = result['P-value'] < .05
    return result

def fit_metrics(model, frame, kind, cluster, spec):
    residuals = np.asarray(model.resids if hasattr(model, 'resids') else model.resid)
    rss = float(np.square(residuals).sum())
    n, k = int(model.nobs), len(model.params)
    # The archived paper used a slope-only information-criterion convention.
    # Keep it labelled and also expose actual residual degrees of freedom.
    record = dict(Model=kind, Covariance=cluster, Specification=spec, N=n, slope_parameter_count=k,
                  df_resid=float(model.df_resid), residual_RSS=rss, RMSE=np.sqrt(rss/n),
                  AIC_slope_only=n*np.log(rss/n)+2*k,
                  BIC_slope_only=n*np.log(rss/n)+k*np.log(n),
                  R2=float(model.rsquared),
                  directed_pairs=frame[KEY[:2]].drop_duplicates().shape[0],
                  origin_counties=frame.GEOID.nunique(), destination_counties=frame.GEOID_destination.nunique(),
                  policy_years=','.join(map(str, sorted(frame.Year.unique()))),
                  flow_start_years=','.join(map(str, sorted(frame.flow_start_year.unique()))))
    if hasattr(model, 'rsquared_within'):
        record['R2_within'] = float(model.rsquared_within)
    if hasattr(model, 'f_pooled'):
        record['F_poolability_conventional'] = float(model.f_pooled.stat)
        record['F_poolability_p_conventional'] = float(model.f_pooled.pval)
    return record

def load_panel(path, lag):
    d = pd.read_csv(path)
    needed = KEY + ['flow_start_year', 'ln_outflow_rate'] + TYPES + TOTAL
    assert set(needed) <= set(d), set(needed)-set(d)
    for col in KEY + ['flow_start_year']:
        d[col] = d[col].astype(int)
    assert not d.duplicated(KEY).any(), f'Duplicate model keys: {path}'
    assert np.isfinite(d[needed].to_numpy(float)).all()
    assert (d[TYPES+TOTAL] >= 0).all().all()
    assert (d.flow_start_year-d.Year == lag).all(), 'Calendar lag mismatch'
    assert not d.GEOID.floordiv(1000).isin([2,9,15]).any()
    assert not d.GEOID_destination.floordiv(1000).isin([2,9,15]).any()
    for side in ['', '_dest']:
        for p in ['AP', 'MP']:
            assert np.allclose(d[p+'total'+side], d[[p+c+side for c in CATEGORIES]].sum(axis=1))
    if {'n2', 'population'} <= set(d):
        assert (d.n2 > 0).all() and (d.population > 0).all()
        assert np.allclose(d.ln_outflow_rate, np.log(d.n2/d.population), atol=1e-12, rtol=0)
    return d.sort_values(KEY).reset_index(drop=True)

