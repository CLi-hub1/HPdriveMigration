"""Portable Results 2 replay from frozen panels; estimates the final PAIR models.

Outputs go to a new folder. Historical master scripts selected two-way effects;
do not run their main() to recreate the current project decision.
"""
from pathlib import Path
import argparse, sys, json, hashlib
import pandas as pd

from config import PROJECT_ROOT, OUTPUT_ROOT
BASE=PROJECT_ROOT/'results2'
from model_helpers import load_panel,fit,rows,fit_metrics,TYPES,TOTAL

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-dir',type=Path,default=OUTPUT_ROOT/'results2')
    a=ap.parse_args();out=a.output_dir.resolve()
    if out.exists() and any(out.iterdir()):raise SystemExit('Choose an empty output directory; frozen files are preserved.')
    out.mkdir(parents=True,exist_ok=True)
    fits=[];metrics=[]
    for lag in [0,1,2]:
        d=load_panel(BASE/'data/frozen_inputs'/f'panel_lag{lag}.csv',lag)
        spec='baseline' if lag==0 else f'lag{lag}'
        for family,variables in [('types16',TYPES),('aggregate',TOTAL)]:
            m=fit(d,variables,'pair',kind='TWFE')
            fits.append(rows(m,'TWFE','pair',spec+'_'+family,variables))
            metrics.append(fit_metrics(m,d,'TWFE','pair',spec+'_'+family))
    f=pd.concat(fits,ignore_index=True)
    f.to_csv(out/'results2_pair_models.csv',index=False)
    pd.DataFrame(metrics).to_csv(out/'model_fit_metrics.csv',index=False)
    b=f[f.Specification.eq('baseline_types16')]
    chosen=b[b['P-value']<.1]
    chosen.to_csv(out/'results2_selected_pair8.csv',index=False)
    (out/'selection_pair8.json').write_text(json.dumps({'rule':'two-sided nominal p < 0.1 in the contemporaneous joint 16-type TWFE model, directed county-pair clustered covariance','variables':chosen.Variable.tolist(),'primary_covariance':'pair','model_N':16709,'reselection_in_results3_5':False,'source_panel_sha256':hashlib.sha256((BASE/'data/frozen_inputs/panel_lag0.csv').read_bytes()).hexdigest()},indent=2),encoding='utf-8')
    print(f'Saved {len(f)} coefficient rows and {len(chosen)} selected terms to {out}')

if __name__=='__main__':main()
