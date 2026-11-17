"""Import the user-supplied 2017–2023 CDP archive with explicit source reconciliation.

No mutation of earlier outputs. Classifier reuse is keyed to the exact input string.
"""
from pathlib import Path
from collections import Counter
import hashlib, json, re
import pandas as pd

from config import POLICY_WORK
B=POLICY_WORK/"policy_inputs"
W=POLICY_WORK
D=B/'data'
P=W/'latest_data/processed'
S=W/'raw_sources'
D.mkdir(parents=True, exist_ok=True)

def save(df,name):
    df.to_csv(D/name,index=False,encoding='utf-8-sig')

def txt(v):
    s='' if pd.isna(v) else str(v).strip()
    return '' if s in {'','nan','None','Question not applicable','Response not provided','Not applicable','N/A','n/a'} else s

def norm(v): return re.sub(r'\s+',' ',txt(v))
def account(v): return str(int(float(v))) if txt(v) else ''
def text_join(a,d,t=''):
    parts=[str(x).strip() for x in (a,d) if str(x).strip()]
    return '. '.join(parts) if parts else str(t).strip()

def coord(v):
    v=txt(v)
    m=re.fullmatch(r'POINT\s*\(\s*([-+.\d]+)\s+([-+.\d]+)\s*\)',v,re.I)
    if m:return f'{float(m[1]):.8f},{float(m[2]):.8f}'
    m=re.search(r'\(\s*([-+.\d]+)\s*,\s*([-+.\d]+)\s*\)',v)
    if m:return f'{float(m[2]):.8f},{float(m[1]):.8f}' # Socrata location export: latitude, longitude
    return ''

def read(p): return pd.read_csv(p,dtype=str,keep_default_na=False)
def digest(df,cols):
    return Counter(tuple(norm(x) for x in row) for row in df[cols].itertuples(index=False,name=None))

raw=[]; global_comparison=[]; inventory=[]; field_map=[]
for f in sorted((B/'raw_user_policy').rglob('*.csv')):
    y=int(f.name[:4]); k='AP' if 'adaptation policies' in f.parts else 'MP'
    a=read(f)
    cf=next(c for c in a if c.lower() in ['country','country/area'])
    us=a[a[cf].isin(['USA','United States of America','United States'])]
    inventory.append(dict(Year=y,Type=k,global_rows=len(a),US_rows=len(us),source_file=str(f.relative_to(B))))
    # Full historical global exports: compare all common fields, retaining row multiplicity.
    if y<=2020:
        oldfile=next((S/('CAA' if k=='AP' else 'CMA')).glob(f'{y}*.csv'))
        prior=read(oldfile)
        common=sorted(set(a)&set(prior))
        c1,c2=digest(a,common),digest(prior,common)
        global_comparison.append(dict(Year=y,Type=k,old_file=str(oldfile.relative_to(W)),new_file=str(f.relative_to(B)),old_rows=len(prior),new_rows=len(a),columns_equal=set(a)==set(prior),common_columns=len(common),only_new_rows=sum((c1-c2).values()),only_old_rows=sum((c2-c1).values())))
    lower={c.lower():c for c in a}
    def column(*choices): return next((lower[s.lower()] for s in choices if s.lower() in lower),'')
    mapping={
      'Account':column('Account number','Organization Number'),
      'Organization':column('Organization','Organization Name'), 'City':column('City'),
      'Action':column(*(['Adaptation Action','Action'] if k=='AP' else ['Emissions reduction activity','Mitigation action','Action type'])),
      'Description':column('Action description and web link to further information','Action description and web link','Action description and implementation progress','Scope and impact of action','Action description'),
      'Title':column('Action title'),
      'MeansOfImplementation':column('Means of implementation'),
      'Hazard':column('Climate hazards','Climate hazard','Climate hazard(s) that action addresses') if k=='AP' else '',
      'Status':column('Status of action','Implementation status','Status of action in reporting year','Status of action in the reporting year'),
      'StartYear':column('Start year of action'), 'Coordinates':column('City Location')}
    assert mapping['Account'] and mapping['Action'] and mapping['Description']
    for target,source in mapping.items(): field_map.append(dict(Year=y,Type=k,target=target,source_column=source,source_file=str(f.relative_to(B))))
    for i,r in us.iterrows():
        rr={key:txt(r[col]) if col else '' for key,col in mapping.items()}
        rr['Account']=account(rr['Account'])
        rr['SourceTitle']=str(r[mapping['Title']]) if mapping['Title'] else ''
        rr['SourceAction']=rr['Action'];rr['SourceDescription']=rr['Description']
        rr['text_field_rule']='source_action_and_description_with_title_only_fallback'
        if y==2019 and k=='MP':
            assert not rr['Action'] and not rr['Description']
            rr['Action']=rr['Title'];rr['Description']=rr['MeansOfImplementation']
            rr['text_field_rule']='2019_original_author_fallback_title_plus_means_of_implementation'
        rr.update(Year=y,Type=k,source_file=str(f.relative_to(B)),source_row=int(i)+2,source_provenance='user_CDP_archive_20261021')
        raw.append(rr)
raw=pd.DataFrame(raw)
save(raw,'user_US_rows_before_recovery.csv')
save(pd.DataFrame(inventory),'user_inventory.csv')
save(pd.DataFrame(global_comparison),'global_export_comparison_2017_2020.csv')
save(pd.DataFrame(field_map),'field_mapping.csv')

# Retain the empty 2018 topic export; use the exact previously downloaded official Q8.4 cells.
broken=raw[raw.Year.eq(2018)&raw.Type.eq('MP')].copy()
assert len(broken)==136 and broken[['Action','Description','Title']].eq('').all().all()
save(broken,'2018_MP_user_export_all_action_fields_empty.csv')
cells=pd.read_csv(W/'latest_data/cdp/2018_mitigation_US_full_questionnaire_8_4.csv',keep_default_na=False)
assert len(cells)==3888 and not cells.duplicated(['account_number','row_number','column_number']).any()
wide=cells.pivot(index=['account_number','organization','row_number'],columns='column_number',values='response_answer').reset_index()
assert len(wide)==648
meta=broken.set_index('Account')
recovered=[]
for _,r in wide.iterrows():
    ac=account(r.account_number); old=meta.loc[ac]
    recovered.append(dict(Year=2018,Type='MP',Account=ac,Organization=txt(r.organization),City=old.City,Action=txt(r.get(1,'')),Description=txt(r.get(5,'')),SourceAction=txt(r.get(1,'')),SourceDescription=txt(r.get(5,'')),Title='',MeansOfImplementation='',text_field_rule='official_full_questionnaire_8.4_action_and_description',Hazard='',Status=txt(r.get(2,'')),StartYear='',Coordinates=old.Coordinates,source_file='work/latest_data/cdp/2018_mitigation_US_full_questionnaire_8_4.csv',source_row=int(r.row_number),source_provenance='official_full_questionnaire_2018_Q8.4_recovery'))
raw=pd.concat([raw[~(raw.Year.eq(2018)&raw.Type.eq('MP'))],pd.DataFrame(recovered)],ignore_index=True)
raw['heat_match']=raw.Hazard.str.contains('heat|hot|drought',case=False,regex=True)
raw['has_action_text']=(raw.Action.str.len()+raw.Description.str.len()+raw.Title.str.len()).gt(0)
raw['selected']=(raw.Type.eq('MP')|raw.heat_match)&raw.has_action_text
oldraw=pd.read_csv(P/'policy_rows_2017_2023_audit.csv',dtype={'Account':str,'GEOID':str},keep_default_na=False)
comparisons=[]; diffs=[]
for (y,k),g in raw.groupby(['Year','Type']):
    prior=oldraw[oldraw.Year.eq(y)&oldraw.Type.eq(k)]
    core=['Account','Action','Description','Hazard']
    for label,cols in [('core_action_fields',core),('titles_status_startyear',core+['Title','Status','StartYear'])]:
        n,o=digest(g,cols),digest(prior,cols)
        comparisons.append(dict(Year=y,Type=k,comparison=label,new_rows=len(g),old_rows=len(prior),only_new_rows=sum((n-o).values()),only_old_rows=sum((o-n).values()),columns=';'.join(cols)))
        if label=='core_action_fields':
            for source,c in [('new_only',n-o),('previous_only',o-n)]:
                for row,multiplicity in c.items():
                    diffs.append(dict(Year=y,Type=k,side=source,multiplicity=multiplicity,**dict(zip(cols,row))))
save(pd.DataFrame(comparisons),'US_canonical_field_comparison.csv')
save(pd.DataFrame(diffs,columns=['Year','Type','side','multiplicity','Account','Action','Description','Hazard']),'US_core_field_differences.csv')

# Geography reuse is conditional on exact account/year support AND unchanged names/coordinates.
oldmap=pd.read_csv(P/'account_year_geography_audit.csv',dtype={'Account':str,'GEOID':str,'historical_GEOID':str,'coordinate_GEOID':str},keep_default_na=False)
geocheck=[]
for (y,ac),g in raw.groupby(['Year','Account']):
    prior=oldraw[oldraw.Year.eq(y)&oldraw.Account.eq(ac)]
    ncoords={coord(x) for x in g.Coordinates if coord(x)}; ocoords={coord(x) for x in prior.Coordinates if coord(x)}
    nnames={norm(x) for x in g.Organization if txt(x)}; onames={norm(x) for x in prior.Organization if txt(x)}
    ncities={norm(x) for x in g.City if txt(x)}; ocities={norm(x) for x in prior.City if txt(x)}
    geocheck.append(dict(Year=y,Account=ac,previous_account_year_present=len(prior)>0,coordinates_match=ncoords==ocoords,organization_match=nnames==onames,city_match=ncities==ocities,new_coordinates=';'.join(sorted(ncoords)),previous_coordinates=';'.join(sorted(ocoords)),new_organization=';'.join(sorted(nnames)),previous_organization=';'.join(sorted(onames))))
geocheck=pd.DataFrame(geocheck)
geocheck=geocheck.merge(oldmap[['Year','Account','mapping_strict_ok']],on=['Year','Account'],how='left',validate='one_to_one')
geocheck['geography_reuse_approved']=geocheck[['previous_account_year_present','organization_match','city_match']].all(axis=1)&(geocheck.coordinates_match|~geocheck.mapping_strict_ok)
geocheck['reuse_note']=geocheck.coordinates_match.map({True:'same_numeric_coordinates',False:'retain_previous_exclusion_not_rehabilitated'})
save(geocheck,'geography_reuse_verification.csv')
assert geocheck.geography_reuse_approved.all(), 'Changed geography requires explicit new mapping audit'
assert len(geocheck)==len(oldmap)
save(oldmap,'account_year_geography_audit.csv')
raw=raw.merge(oldmap[['Year','Account','GEOID','mapping_source','mapping_flags','mapping_strict_ok']],on=['Year','Account'],validate='many_to_one')
save(raw,'policy_rows_2017_2023_audit.csv')

raw['ActionKey']=raw.Action.map(norm);raw['DescriptionKey']=raw.Description.map(norm)
raw['IdentityTitle']=raw.apply(lambda r:norm(r.Title) if not r.ActionKey and not r.DescriptionKey else '',axis=1)
keys=['Year','Type','Account','ActionKey','DescriptionKey','IdentityTitle']
groups=[]
for _,g in raw[raw.has_action_text].groupby(keys,dropna=False,sort=False):
    r=g.iloc[0].to_dict()
    r.update(raw_rows=len(g),source_rows=';'.join(map(str,g.source_row)),Hazard=' | '.join(sorted(set(g.Hazard)-{''})),Status=' | '.join(sorted(set(g.Status)-{''})),StartYear=' | '.join(sorted(set(g.StartYear)-{''})),Title=' | '.join(sorted(set(g.Title)-{''})),selected=bool(g.selected.any()),heat_match=bool(g.heat_match.any()),varying_fields=';'.join(c for c in ['Hazard','Title','Status','StartYear','Coordinates'] if g[c].nunique()>1))
    r['text']=text_join(r['Action'],r['Description'],r['Title'])
    r['text_sha256']=hashlib.sha256(r['text'].encode()).hexdigest()
    groups.append(r)
actions=pd.DataFrame(groups)
save(actions,'policy_action_groups_2017_2023.csv')
title_sensitivity=actions.copy()
mask=title_sensitivity.Year.eq(2019)&title_sensitivity.Type.eq('MP')
title_sensitivity.loc[mask,'Action']=''
title_sensitivity.loc[mask,'Description']=''
title_sensitivity.loc[mask,'text']=title_sensitivity.loc[mask,'Title']
title_sensitivity.loc[mask,'text_sha256']=title_sensitivity.loc[mask,'text'].map(lambda s:hashlib.sha256(s.encode()).hexdigest())
title_sensitivity.loc[mask,'selected']=title_sensitivity.loc[mask,'Title'].ne('')
title_sensitivity.loc[mask,'text_field_rule']='2019_title_only_sensitivity_keep_original_action_group_identity'
save(title_sensitivity,'policy_action_groups_2017_2023_title_only_sensitivity.csv')
save(actions[actions.raw_rows.gt(1)],'policy_duplicate_groups.csv')
previous=pd.read_csv(W/'ml_rebuild/policy_action_groups_2017_2023_classified.csv',dtype={'Account':str,'GEOID':str},keep_default_na=False)
predcols=[c for c in previous if c.startswith(('pred_','prob_'))]+['valid_text','encoding_warning','low_confidence_review','no_type_predicted','human_review_status','model_sha256']
for c in [x for x in predcols if x.startswith('pred_')]+['model_sha256']:
    assert previous.groupby('text')[c].nunique().max()==1, f'Inconsistent prediction {c}'
# Earlier GPU batches have small probability roundoff differences for identical strings;
# hard labels agree. This pending reuse table is an audit; the parent reruns all groups.
lookup=previous[['text']+predcols].drop_duplicates('text')
selected=actions[actions.selected].merge(lookup,on='text',how='left',validate='many_to_one',indicator='prediction_lookup')
unknown=selected[selected.prediction_lookup.eq('left_only')]
save(unknown,'unknown_texts_for_frozen_classifier.csv')
selected['prediction_source']='frozen_classifier_exact_input_text_reuse'
selected.loc[selected.prediction_lookup.eq('left_only'),'prediction_source']='pending_frozen_model_inference'
save(selected,'policy_action_groups_2017_2023_classified_pending.csv')
report=raw[raw.has_action_text].drop_duplicates(['Year','Account','Type'])
roster=report.groupby(['Year','Account']).Type.agg(lambda x:set(x)).reset_index(name='answered_types')
roster['answered_both']=roster.answered_types.map(lambda x:x=={'AP','MP'})
roster=roster.merge(oldmap,on=['Year','Account'],validate='one_to_one')
roster['eligible_both']=roster.answered_both & roster.mapping_strict_ok
save(roster.drop(columns='answered_types'),'action_reporting_roster_2017_2023.csv')
q=raw.groupby(['Year','Type']).agg(raw_rows=('Account','size'),reporting_accounts=('Account','nunique'),selected_raw_rows=('selected','sum')).reset_index()
q=q.merge(actions.groupby(['Year','Type']).agg(unique_text_action_groups=('Account','size'),selected_action_groups=('selected','sum')).reset_index(),on=['Year','Type'],validate='one_to_one')
q=q.merge(selected.groupby(['Year','Type']).agg(selected_groups_geography_QC=('mapping_strict_ok','sum'),unknown_classifier_text_groups=('prediction_lookup',lambda s:s.eq('left_only').sum())).reset_index(),on=['Year','Type'],validate='one_to_one')
save(q,'policy_year_quality_summary.csv')
qc=dict(global_exports_2017_2020=pd.DataFrame(global_comparison).to_dict('records'),canonical_comparison=comparisons,geography_reused_verified_account_years=len(geocheck),selected_action_groups=len(selected),unknown_action_groups=len(unknown),unknown_unique_texts=unknown.text.nunique(),title_only_new_groups=int((selected.Action.eq('')&selected.Description.eq('')).sum()),selected_groups_by_year_type=q.to_dict('records'))
(D/'import_qc.json').write_text(json.dumps(qc,indent=2,default=lambda v:v.item()),encoding='utf-8')
print(q.to_string(index=False))
print('Unknown action groups:',len(unknown),'unique texts:',unknown.text.nunique())
