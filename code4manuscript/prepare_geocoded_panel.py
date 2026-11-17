"""Auditable policy action counts and IRS OD rows, with no inferred class labels."""
from pathlib import Path
import sys, re, json, csv
import pandas as pd
import shapefile
from shapely.geometry import shape, Point
from shapely.strtree import STRtree

from config import POLICY_WORK
B=POLICY_WORK
S=B/"raw_sources"
D=B/"latest_data"
O=D/"processed"
O.mkdir(parents=True, exist_ok=True)

def save(d,name):
    d.to_csv(O/name,index=False,encoding="utf-8-sig")

def state_map():
    abbreviations="AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split()
    fips="01 02 04 05 06 08 09 10 11 12 13 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 44 45 46 47 48 49 50 51 53 54 55 56".split()
    return dict(zip(abbreviations,fips))
ST=state_map()
CONUS=set(ST.values())-{"02","15"}

# Do not convert empty/unknown questionnaire responses into a zero action.
nulls={"", "nan", "None", "Question not applicable", "Response not provided", "Not applicable"}
def txt(v):
    s="" if pd.isna(v) else str(v).strip()
    return "" if s in nulls else s
def account(v):
    return str(int(float(v))) if txt(v) else ""
def norm_text(v):
    return re.sub(r"\s+"," ",txt(v))

raw=[]
for kind,fname in [("AP","2017-2020CAA.xlsx"),("MP","2017-2020CMA.xlsx")]:
    d=pd.read_excel(S/"climatepolicy-induced migration"/fname).fillna("")
    d=d[d.Country.eq("USA")].copy()
    for i,r in d.iterrows():
        raw.append(dict(Year=int(r.Year),Type=kind,Account=account(r["Account number"]),Organization=txt(r.Organization),
                        City=txt(r.City),Action=txt(r.Action),Description=txt(r["Action description"]),Title="",
                        Hazard=txt(r.get("Climate hazards","")),Status="",StartYear="",Coordinates=txt(r["City Location"]),
                        source_file=fname,source_row=int(i)+2))
for year in [2021,2022,2023]:
    for kind,longkind in [("AP","adaptation"),("MP","mitigation")]:
        filename=f"{year}_{longkind}_US_raw.csv"
        d=pd.read_csv(D/"cdp"/filename,keep_default_na=False)
        for i,r in d.iterrows():
            if year==2021:
                key,org="account_number","organization"
                actioncol="adaptation_action" if kind=="AP" else "mitigation_action"
                desccol="action_description_and"
                hazardcol="climate_hazard"
                statuscol="status_of_action" if kind=="AP" else "implementation_status"
            else:
                key,org="organization_number","organization_name"
                actioncol="action" if kind=="AP" else "action_type"
                desccol="action_description_and_web"
                hazardcol="climate_hazard_s_that_action"
                statuscol="status_of_action_in_reporting" if kind=="AP" else "status_of_action_in_the"
            raw.append(dict(Year=year,Type=kind,Account=account(r[key]),Organization=txt(r[org]),City=txt(r.get("city","")),
                            Action=txt(r[actioncol]),Description=txt(r[desccol]),Title=txt(r.get("action_title","")),
                            Hazard=txt(r.get(hazardcol,"")),Status=txt(r.get(statuscol,"")),StartYear=txt(r.get("start_year_of_action","")),
                            Coordinates=txt(r.get("city_location","")),source_file=filename,source_row=i+2))
raw=pd.DataFrame(raw)
# The archived 2018 mitigation topic export contains only empty placeholder actions.
# Recover question 8.4 cells from CDP's full questionnaire; retain the broken export.
broken2018=raw[raw.Year.eq(2018)&raw.Type.eq("MP")].copy()
assert broken2018.Action.eq("").all() and broken2018.Description.eq("").all()
save(broken2018,"2018_mitigation_archived_empty_export.csv")
cells=pd.read_csv(D/"cdp/2018_mitigation_US_full_questionnaire_8_4.csv",keep_default_na=False)
assert len(cells)==3888
assert not cells.duplicated(["account_number","row_number","column_number"]).any()
wide=cells.pivot(index=["account_number","organization","row_number"],columns="column_number",values="response_answer").reset_index()
assert len(wide)==648
prior2018=broken2018.set_index("Account")
recovered=[]
for _,r in wide.iterrows():
    acc=account(r.account_number)
    oldrow=prior2018.loc[acc] if acc in prior2018.index else None
    recovered.append(dict(Year=2018,Type="MP",Account=acc,Organization=txt(r.organization),
                          City=txt(oldrow.City) if oldrow is not None else "",Action=txt(r.get(1,"")),
                          Description=txt(r.get(5,"")),Title="",Hazard="",Status=txt(r.get(2,"")),StartYear="",
                          Coordinates=txt(oldrow.Coordinates) if oldrow is not None else "",
                          source_file="2018_mitigation_US_full_questionnaire_8_4.csv",source_row=int(r.row_number)))
raw=pd.concat([raw[~(raw.Year.eq(2018)&raw.Type.eq("MP"))],pd.DataFrame(recovered)],ignore_index=True)
raw["heat_match"]=raw.Hazard.str.contains("heat|hot|drought",case=False,regex=True)
raw["has_action_text"]=(raw.Action.str.len()+raw.Description.str.len()+raw.Title.str.len()).gt(0)
raw["selected"]=((raw.Type.eq("MP"))|raw.heat_match)&raw.has_action_text

# The old account-to-county mapping is retained and compared to coordinates.
old=pd.concat([pd.read_excel(S/"Data"/(k+".xlsx")).assign(Type=k) for k in ["AP","MP"]]).fillna("")
old["Account"]=old["Account number"].map(account)
old["GEOID"]=old.GEOID.map(lambda v:f"{int(v):05d}" if txt(v) else "")
oldmap=old.groupby("Account").GEOID.agg(lambda s:sorted(set(s)-{""})).to_dict()
save(old[["Account","Year","Type","City","City Location","GEOID","County"]].drop_duplicates(),"historical_account_geography.csv")

sf=shapefile.Reader(str(S/"climatepolicy-induced migration/tl_2019_us_county(includeFIPScode)/tl_2019_us_county.shp"),encoding="utf-8")
geoms=[]; gids=[]; names=[]
for sr in sf.iterShapeRecords():
    rec=sr.record.as_dict()
    geoms.append(shape(sr.shape.__geo_interface__))
    gids.append(str(rec["GEOID"]).zfill(5)); names.append(rec["NAMELSAD"])
tree=STRtree(geoms)
def geocode(wkt):
    vals=re.fullmatch(r"POINT\s*\(\s*([-+.\d]+)\s+([-+.\d]+)\s*\)",wkt,re.I)
    if not vals:return []
    p=Point(float(vals[1]),float(vals[2]))
    return sorted({gids[int(i)] for i in tree.query(p,predicate="within")})
assert geocode("POINT (-118.405 33.8889)")==["06037"]
assert geocode("POINT (-121.741 38.5449)")==["06113"]

maprows=[]
for (year,acc),g in raw.groupby(["Year","Account"]):
    locations=sorted(set(g.Coordinates)-{""})
    spatial=sorted({geoid for loc in locations for geoid in geocode(loc)})
    prior=oldmap.get(acc,[])
    # shortcut: point assignment reproduces the study; administrative area weighting requires jurisdiction polygons.
    chosen=prior[0] if len(prior)==1 else (spatial[0] if len(spatial)==1 else "")
    source="historical_account" if len(prior)==1 else ("TIGER2019_point" if len(spatial)==1 else "unmatched")
    orgs=sorted(set(g.Organization)-{""})
    expected=set()
    for org in orgs:
        match=re.search(r",\s*([A-Z]{2})\s*$",org)
        if match and match[1] in ST:expected.add(ST[match[1]])
    flags=[]
    if len(prior)>1:flags.append("historical_multiple_counties")
    if len(spatial)>1:flags.append("current_multiple_counties")
    if len(prior)==1 and len(spatial)==1 and prior!=spatial:flags.append("historical_vs_current_coordinate")
    if chosen and expected and chosen[:2] not in expected:flags.append("organization_state_vs_assigned_county")
    if spatial and expected and any(x[:2] not in expected for x in spatial):flags.append("organization_state_vs_coordinate")
    if locations and not spatial:flags.append("coordinate_outside_TIGER_counties")
    if acc=="55799":flags.append("known_Arlington_VA_point_in_Northampton")
    if any(re.search(r"regional council|metropolitan council|mayors caucus|planning council",o,re.I) for o in orgs):
        flags.append("possibly_multicounty_jurisdiction")
    if not chosen:flags.append("unmatched")
    if chosen and chosen[:2] not in CONUS:flags.append("outside_contiguous_US")
    maprows.append(dict(Year=year,Account=acc,GEOID=chosen,mapping_source=source,
                        historical_GEOID=";".join(prior),coordinate_GEOID=";".join(spatial),
                        Organization=";".join(orgs),City=";".join(sorted(set(g.City)-{""})),
                        Coordinates=";".join(locations),mapping_flags=";".join(flags),mapping_strict_ok=not flags))
mapping=pd.DataFrame(maprows)
save(mapping,"account_year_geography_audit.csv")
raw=raw.merge(mapping[["Year","Account","GEOID","mapping_source","mapping_flags","mapping_strict_ok"]],on=["Year","Account"],validate="many_to_one")
save(raw,"policy_rows_2017_2023_audit.csv")

# Collapse exact normalized action+description within account/year/type; cross-year repeats remain.
# This defines text-identical action groups, not independently verified legal instruments.
raw["ActionKey"]=raw.Action.map(norm_text)
raw["DescriptionKey"]=raw.Description.map(norm_text)
raw["IdentityTitle"]=raw.apply(lambda r:norm_text(r.Title) if not r.ActionKey and not r.DescriptionKey else "",axis=1)
keys=["Year","Type","Account","ActionKey","DescriptionKey","IdentityTitle"]
groups=[]
for key,g in raw[raw.has_action_text].groupby(keys,dropna=False,sort=False):
    r=g.iloc[0].to_dict()
    r.update(raw_rows=len(g),source_rows=";".join(map(str,g.source_row)),Hazard=" | ".join(sorted(set(g.Hazard)-{""})),
             Status=" | ".join(sorted(set(g.Status)-{""})),StartYear=" | ".join(sorted(set(g.StartYear)-{""})),
             Title=" | ".join(sorted(set(g.Title)-{""})),selected=bool(g.selected.any()),
             heat_match=bool(g.heat_match.any()),varying_fields=";".join(c for c in ["Hazard","Title","Status","StartYear","Coordinates"] if g[c].nunique()>1))
    groups.append(r)
actions=pd.DataFrame(groups)
save(actions,"policy_action_groups_2017_2023.csv")
save(actions[actions.raw_rows.gt(1)],"policy_duplicate_groups.csv")
sel=actions[actions.selected & actions.mapping_strict_ok & actions.GEOID.str[:2].isin(CONUS)].copy()
save(sel[sel.Year.isin([2021,2022])][["Year","Type","Account","Organization","City","Action","Description","Title","Hazard","Status","StartYear","Coordinates","GEOID","mapping_source","raw_rows"]],"new_policy_text_for_classification_2021_2022.csv")

# A zero AP count is permitted only for an account with a nonempty adaptation answer in that year.
report=raw[raw.has_action_text].drop_duplicates(["Year","Account","Type"])
roster=report.groupby(["Year","Account"]).Type.agg(lambda x:set(x)).reset_index(name="answered_types")
roster["answered_both"]=roster.answered_types.map(lambda s:s=={"AP","MP"})
roster=roster.merge(mapping,on=["Year","Account"],validate="one_to_one")
roster["eligible_both"]=roster.answered_both & roster.mapping_strict_ok
save(roster.drop(columns="answered_types"),"action_reporting_roster_2017_2023.csv")
counts=sel.groupby(["Year","Account","GEOID","Type"]).size().unstack(fill_value=0).reset_index().rename(columns={"AP":"AP_action_count","MP":"MP_action_count"})
counts=roster[["Year","Account","GEOID","eligible_both","answered_both","mapping_strict_ok"]].merge(counts,on=["Year","Account","GEOID"],how="left",validate="one_to_one")
for col in ["AP_action_count","MP_action_count"]:
    counts.loc[counts.eligible_both,col]=counts.loc[counts.eligible_both,col].fillna(0)
save(counts,"account_year_counts_deduplicated.csv")

eligible=counts[counts.eligible_both & counts.Year.between(2017,2022)].copy()
strict_accounts=eligible.groupby("Account").Year.nunique()
strict_accounts=set(strict_accounts[strict_accounts.eq(6)].index)
for label,sub in [("all_mapped_responders",eligible),("strict_repeat_reporters",eligible[eligible.Account.isin(strict_accounts)])]:
    county=sub.groupby(["Year","GEOID"]).agg(AP_action_count=("AP_action_count","sum"),MP_action_count=("MP_action_count","sum"),
            reporting_accounts=("Account","nunique"),Accounts=("Account",lambda s:";".join(sorted(s)))).reset_index()
    save(county,f"county_year_{label}_2017_2022.csv")

# Also preserve simple original-row counts as an audit/sensitivity, never mix with deduplicated years.
original_selected=raw[raw.selected & raw.GEOID.str[:2].isin(CONUS)]
rawcounts=original_selected.groupby(["Year","GEOID","Type"]).size().unstack(fill_value=0).reset_index()
save(rawcounts,"county_year_raw_rows_unfiltered_mapping_2017_2023.csv")

irsstats=[]
for start,end in [(2020,2021),(2021,2022),(2022,2023)]:
    suffix=f"{start%100:02d}{end%100:02d}"
    f=D/"irs"/f"countyoutflow{suffix}.csv"
    df=pd.read_csv(f,dtype=str,encoding="cp1252");df.columns=df.columns.str.lower()
    for c in ["y1_statefips","y2_statefips"]:df[c]=df[c].str.zfill(2)
    for c in ["y1_countyfips","y2_countyfips"]:df[c]=df[c].str.zfill(3)
    for c in ["n1","n2","agi"]:df[c]=pd.to_numeric(df[c],errors="raise")
    df["origin_fips"]=df.y1_statefips+df.y1_countyfips
    df["destination_fips"]=df.y2_statefips+df.y2_countyfips
    valid=df.y1_statefips.isin(CONUS)&df.y2_statefips.isin(CONUS)&df.y1_countyfips.ne("000")&df.y2_countyfips.ne("000")
    valid&=df.origin_fips.ne(df.destination_fips)
    out=df[valid].copy()
    out["flow_start_year"]=start;out["flow_end_year"]=end;out["policy_year_for_legacy_alignment"]=start
    out["n1_suppressed"]=out.n1.eq(-1);out["n2_suppressed"]=out.n2.eq(-1);out["agi_suppressed"]=out.agi.eq(-1)
    out["irs_method_break"]=start>=2022
    out["connecticut_boundary_change"]=(start>=2022)&(out.y1_statefips.eq("09")|out.y2_statefips.eq("09"))
    assert not out.duplicated(["origin_fips","destination_fips"]).any()
    assert out.n2.ge(0).any()
    save(out,f"irs_OD_{start}_{end}_contiguous_US.csv")
    irsstats.append(dict(period=f"{start}-{end}",raw_rows=len(df),OD_rows=len(out),n2_suppressed=int(out.n2_suppressed.sum()),n2_nonnegative_total=int(out.loc[~out.n2_suppressed,"n2"].sum()),origins=out.origin_fips.nunique(),destinations=out.destination_fips.nunique(),connecticut_rows=int(out.connecticut_boundary_change.sum())))

summary=raw.groupby(["Year","Type"]).agg(raw_rows=("Account","size"),reporting_accounts=("Account","nunique"),selected_raw_rows=("selected","sum")).reset_index()
summary=summary.merge(actions.groupby(["Year","Type"]).agg(unique_text_action_groups=("Account","size"),selected_action_groups=("selected","sum")).reset_index(),on=["Year","Type"],how="left")
summary=summary.merge(sel.groupby(["Year","Type"]).size().rename("selected_groups_geography_QC").reset_index(),on=["Year","Type"],how="left").fillna(0)
save(summary,"policy_year_quality_summary.csv")
(O/"core_update_qc.json").write_text(json.dumps(dict(strict_repeat_accounts_2017_2022=sorted(strict_accounts),n_strict_repeat_accounts=len(strict_accounts),irs=irsstats,policy=summary.to_dict("records")),indent=2),encoding="utf-8")
assert len(raw)==sum(summary.raw_rows)
assert sel.mapping_strict_ok.all()
assert set(eligible[eligible.Account.isin(strict_accounts)].groupby("Account").Year.nunique()) in [{6},set()]
print(summary.to_string(index=False))
print(json.dumps(dict(strict_repeat_accounts=len(strict_accounts),irs=irsstats)))
