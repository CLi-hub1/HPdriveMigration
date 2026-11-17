"""Recreate county-year and annual descriptive data from the packaged frozen inputs.

This implements the data aggregation in describe_and_plot.py, without its missing
historical plotting dependencies. It does not refit a policy classifier or panel model.
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from scipy.stats import linregress,pearsonr,spearmanr,ttest_rel
from config import PROJECT_ROOT, OUTPUT_ROOT
BASE=PROJECT_ROOT/'results1'
CATS=['TechnologicalInfrastructural','Institutional','BehavioralCultural','NatureBased']
TYPES=[p+c+s for s in ['', '_dest'] for p in ['AP','MP'] for c in CATS]
TOTAL=['APtotal','MPtotal','APtotal_dest','MPtotal_dest']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,default=OUTPUT_ROOT/'descriptive_tables');a=ap.parse_args()
    out=a.output_dir.resolve()
    if out.exists() and any(out.iterdir()):raise SystemExit('Choose an empty output directory.')
    out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(BASE/'data/frozen_inputs/panel_lag0.csv')
    pops=pd.read_csv(BASE/'data/frozen_inputs/county_population.csv')
    policy=pd.read_csv(BASE/'data/frozen_inputs/county_policy_2017_2023.csv')
    summary=[]
    for side,key in [('origin','GEOID'),('dest','GEOID_destination')]:
        cols=[v for v in TOTAL+TYPES if v.endswith('_dest')==(side=='dest')]
        cov=[m+'_'+side for m in ['income','HV','RD','aging','EA']]
        c=d[['Year',key]+cols+cov].drop_duplicates()
        if c.duplicated(['Year',key]).any():raise ValueError('Inconsistent endpoint exposures')
        c=c.merge(d.groupby(['Year',key]).n2.agg(observed_sample_n2='sum',observed_partner_count='count').reset_index(),on=['Year',key]).rename(columns={key:'GEOID'})
        c=c.merge(pops,on=['Year','GEOID'],validate='many_to_one')
        c['sample_migration_rate']=c.observed_sample_n2/c.population;c['Role']=side
        c.to_csv(out/f'descriptive_uniform_classifier_label_counts_{side}_county_year.csv',index=False)
        s=c[cols+cov+['observed_sample_n2','sample_migration_rate']].describe().T.reset_index(names='Variable')
        s['Role']=side;summary.append(s)
    pd.concat(summary).to_csv(out/'descriptive_county_year_statistics.csv',index=False)
    agg={c:'mean' for c in d if c.startswith(('AP','MP'))}|{'n2':'sum','population':'first'}
    c=d.groupby(['Year','GEOID']).agg(agg).reset_index();c['outflow_rate']=c.n2/c.population;c['HPtotal']=c.APtotal+c.MPtotal
    c.to_csv(out/'Fig1_origin_county_year_with_destination_partner_means.csv',index=False)
    stats=[]
    for panel,x in zip('cdef',TOTAL):
        lm=linregress(c[x],c.outflow_rate);pr=pearsonr(c[x],c.outflow_rate);sr=spearmanr(c[x],c.outflow_rate)
        stats.append(dict(panel=panel,x=x,n=len(c),Pearson_r=pr.statistic,Pearson_p=pr.pvalue,Spearman_beta=sr.statistic,Spearman_p=sr.pvalue,OLS_rho=lm.slope,OLS_p=lm.pvalue))
    pd.DataFrame(stats).to_csv(out/'Fig1_scatter_statistics.csv',index=False)
    tests=[dict(kind=k,n=len(c),paired_t_p_nominal=ttest_rel(c[k+'total'],c[k+'total_dest']).pvalue,whisker_lower_percentile=2.5,whisker_upper_percentile=97.5) for k in ['AP','MP']]
    pd.DataFrame(tests).to_csv(out/'Fig1_box_paired_tests.csv',index=False)
    annual=[]
    for year,g in policy.groupby('Year'):
        for kind in ['AP','MP']:
            for typ in CATS:
                annual.append(dict(Year=year,Policy_class=kind,Type=typ,Predicted_label_count=g[kind+typ].sum(),Class_label_total=g[kind+'total'].sum(),Class_deduplicated_action_groups=g[kind+'_action_count'].sum(),Reporting_counties=g.GEOID.nunique(),Included_in_primary_regression=year in set(d.Year)))
    pd.DataFrame(annual).to_csv(out/'annual_policy_types_2017_2023.csv',index=False)
    print(out)
if __name__=='__main__':main()
