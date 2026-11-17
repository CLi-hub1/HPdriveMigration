"""Aggregate independently re-inferred policy groups on audited reporter support."""
from pathlib import Path
import argparse,json,hashlib
import pandas as pd

from config import POLICY_WORK
B=POLICY_WORK/'policy_inputs';D=B/'data';W=POLICY_WORK
parser=argparse.ArgumentParser()
parser.add_argument('--input',default='policy_action_groups_2017_2023_classified.csv')
parser.add_argument('--tag',default='')
args=parser.parse_args()
source=D/args.input
inference=json.loads(source.with_suffix('.inference.json').read_text(encoding='utf-8'))
assert hashlib.sha256(source.read_bytes()).hexdigest()==inference['output_sha256']
input_source=D/Path(inference['input']).name
assert hashlib.sha256(input_source.read_bytes()).hexdigest()==inference['input_sha256'], 'Inference source has changed'
d=pd.read_csv(source,dtype={'Account':str,'GEOID':str},keep_default_na=False)
assert d.selected.all()
roster=pd.read_csv(D/'action_reporting_roster_2017_2023.csv',dtype={'Account':str,'GEOID':str},keep_default_na=False)
eligible=roster[roster.eligible_both & roster.Year.between(2017,2023)][['Year','Account','GEOID']]
strict_count=eligible.groupby('Account').Year.nunique()
strict=set(strict_count[strict_count.eq(7)].index)
# The title-only sensitivity can remove all usable MP text for a reporter.
# Missing measurement is not zero policy: omit its whole county/year total.
mp_keys=d[d.Type.eq('MP')][['Year','Account','GEOID']].drop_duplicates()
check=eligible.merge(mp_keys,on=['Year','Account','GEOID'],how='left',indicator=True,validate='one_to_one')
missing_support=check[check._merge.eq('left_only')].drop(columns='_merge')
missing_support.to_csv(D/f'missing_policy_measurement_support{args.tag}.csv',index=False,encoding='utf-8-sig')
bad_county_years=missing_support[['Year','GEOID']].drop_duplicates()
eligible=eligible.merge(bad_county_years.assign(missing_measurement=True),on=['Year','GEOID'],how='left',validate='many_to_one')
eligible=eligible[eligible.missing_measurement.ne(True)].drop(columns='missing_measurement')
d=d.merge(eligible,on=['Year','Account','GEOID'],validate='many_to_one')
assert d.mapping_strict_ok.all()
classes={'Technological/Infrastructural':'TechnologicalInfrastructural','Institutional':'Institutional','Behavioral/Cultural':'BehavioralCultural','Nature-Based':'NatureBased'}
for label in classes: assert d['pred_'+label].isin([0,1]).all()
assert d.model_sha256.nunique()==1
metadata=[]
for tag,accounts in [('',eligible),('_strict_repeat_reporters',eligible[eligible.Account.isin(strict)])]:
    sub=d.merge(accounts,on=['Year','Account','GEOID'],validate='many_to_one')
    result=accounts.groupby(['Year','GEOID']).agg(reporting_accounts=('Account','nunique'),Accounts=('Account',lambda s:';'.join(sorted(s)))).reset_index()
    for kind in ['AP','MP']:
        g=sub[sub.Type.eq(kind)].copy();group=g.groupby(['Year','GEOID'])
        frame=group[['pred_'+c for c in classes]].sum().rename(columns={'pred_'+k:kind+v for k,v in classes.items()})
        frame[kind+'total']=frame.sum(axis=1)
        frame[kind+'_action_count']=group.size()
        frame[kind+'_low_confidence_count']=group.low_confidence_review.sum()
        result=result.merge(frame.reset_index(),on=['Year','GEOID'],how='left',validate='one_to_one')
    valuecols=[c for c in result if c not in ['Year','GEOID','Accounts','reporting_accounts']]
    result[valuecols]=result[valuecols].fillna(0).astype(int)
    assert not result.duplicated(['Year','GEOID']).any()
    for k in ['AP','MP']: assert (result[k+'total']==result[[k+v for v in classes.values()]].sum(axis=1)).all()
    p=D/f'county_year_classified{args.tag}{tag}_2017_2023.csv'
    result.to_csv(p,index=False,encoding='utf-8-sig')
    metadata.append(dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),rows=len(result),counties=result.GEOID.nunique(),county_years=result.groupby('Year').size().to_dict(),eligible_accounts=accounts.Account.nunique(),year_sums=result.groupby('Year')[['APtotal','MPtotal','AP_action_count','MP_action_count']].sum().to_dict('index')))
    if not tag and not args.tag:
        old=pd.read_csv(W/'latest_data/processed/county_year_classified_2017_2022.csv',dtype={'GEOID':str},keep_default_na=False)
        z=result[result.Year.le(2022)].merge(old,on=['Year','GEOID'],how='outer',indicator=True,suffixes=('_new','_previous'),validate='one_to_one')
        z['changed']=(z._merge!='both')
        for c in [v for v in old if v not in ['Year','GEOID']]: z['changed']|=z[c+'_new'].ne(z[c+'_previous'])
        z.to_csv(D/'county_year_previous_version_comparison.csv',index=False,encoding='utf-8-sig')
        print('Changed 2017–2022 county-years:',int(z.changed.sum()))
    print(p.name,result.shape,flush=True)
out=dict(input_file=source.name,input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),model_sha256=d.model_sha256.iloc[0],strict_repeat_accounts_2017_2023=sorted(strict),n_strict_repeat_accounts=len(strict),missing_measurement_accounts=missing_support.to_dict('records'),excluded_missing_measurement_county_years=bad_county_years.to_dict('records'),results=metadata,definitions={'APtotal_MPtotal':'sum of four binary class counts; multilevel assignments can count one action more than once','AP_MP_action_count':'one per normalized action text group within account/report year/type','geography':'historical account/TIGER2019 audit; flagged accounts excluded; CT retained in source table, excluded by downstream OD boundary rule','year':'CDP reporting year, not independently verified implementation year','support':'main roster account-years with meaningful AP and MP responses; if sensitivity removes all MP text for any account, omit that entire county-year rather than fill zero; AP zero only when response exists but no heat/hot/drought target; strict-repeat membership defined from main reporting roster'})
(D/f'aggregation{args.tag}_qc.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
