"""Download and build the Results5 county-year mediator dataset (2016-2023).

All acquired source files and API payloads are retained under new_data4results5/raw.
The final panel is written under analysis/results5_moderationeffects.
"""
from __future__ import annotations

import csv
import concurrent.futures
from datetime import datetime, timezone
import io
import json
import math
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

from config import OUTPUT_ROOT
ROOT = OUTPUT_ROOT / "resource_build"
BASE = ROOT / "new_data4results5"
RAW = BASE / "raw"
OUT = ROOT / "analysis" / "results5_moderationeffects"
VENDOR = BASE / "vendor"
import shapefile  # noqa: E402
import openpyxl  # noqa: E402

YEARS = list(range(2016, 2024))
STATE_FIPS_ABBR = {
    "01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT","10":"DE","11":"DC","12":"FL","13":"GA","15":"HI","16":"ID","17":"IL","18":"IN","19":"IA","20":"KS","21":"KY","22":"LA","23":"ME","24":"MD","25":"MA","26":"MI","27":"MN","28":"MS","29":"MO","30":"MT","31":"NE","32":"NV","33":"NH","34":"NJ","35":"NM","36":"NY","37":"NC","38":"ND","39":"OH","40":"OK","41":"OR","42":"PA","44":"RI","45":"SC","46":"SD","47":"TN","48":"TX","49":"UT","50":"VT","51":"VA","53":"WA","54":"WV","55":"WI","56":"WY","60":"AS","66":"GU","69":"MP","72":"PR","78":"VI"
}
RENEWABLE_CODES = {"WND", "SUN", "WAT", "GEO", "BIO", "WDS", "WAS", "LFG", "MSW", "OBS", "OBG", "OBL", "AB", "SLW", "HPS", "WH"}
COOLING_FILTER = "(amenity=library or amenity=community_centre or amenity=social_facility or amenity=place_of_worship or shop=mall) and geometry:polygon"
OHSOME_AREA = "https://api.ohsome.org/v1/elements/area"
TCC_SERVICE = "https://imagery.geoplatform.gov/iipp/rest/services/Vegetation/USFS_EDW_NLCD_TCC_CONUS/ImageServer"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent":"Results5 research data preparation/1.0 (academic use)"})


def mkdirs():
    for p in [RAW/"boundaries", RAW/"osm"/"area_responses", RAW/"tree_canopy"/"statistics_responses", RAW/"eia860", RAW/"eia923", RAW/"acs_industry", BASE/"processed", OUT]:
        p.mkdir(parents=True, exist_ok=True)


def get(url: str, *, params=None, data=None, timeout=180, attempts=4):
    last = None
    for attempt in range(attempts):
        try:
            resp = SESSION.post(url, data=data, params=params, timeout=timeout) if data is not None else SESSION.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(2 ** attempt)
    raise last


def load_target_panel():
    p = ROOT / "mechanisms" / "source_data" / "analysis_mediators.csv"
    df = pd.read_csv(p, dtype={"GEOID":str})
    df["GEOID"] = df["GEOID"].str.zfill(5)
    df["Year"] = df["Year"].astype(int)
    df = df[df.Year.isin(YEARS)].copy()
    if df.duplicated(["GEOID", "Year"]).any():
        raise ValueError("Existing mediator panel contains duplicate county-year rows")
    if len(df) != 864:
        raise ValueError(f"Expected the existing 108 x 8 support (864 rows), found {len(df)}")
    # Recheck from source rather than assume the saved support count.
    support = df[["GEOID","State","County","in_origin_support","in_destination_support"]].drop_duplicates("GEOID")
    if len(support) != 108:
        raise ValueError(f"Expected 108 counties, found {len(support)}")
    support.to_csv(BASE / "processed" / "target_counties.csv", index=False, encoding="utf-8-sig")
    return df, support


def fetch_tiger(support):
    d = RAW / "boundaries"
    zpath = d / "tl_2020_us_county.zip"
    if not zpath.exists():
        url = "https://www2.census.gov/geo/tiger/TIGER2020/COUNTY/tl_2020_us_county.zip"
        zpath.write_bytes(get(url).content)
    extract = d / "tl_2020_us_county"
    if not (extract / "tl_2020_us_county.shp").exists():
        extract.mkdir(exist_ok=True)
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(extract)
    reader = shapefile.Reader(str(extract / "tl_2020_us_county.shp"), encoding="utf-8")
    names = [f[0] for f in reader.fields[1:]]
    records = []
    features = []
    wanted = set(support.GEOID)
    for sr in reader.iterShapeRecords():
        row = dict(zip(names, sr.record))
        geoid = str(row["GEOID"]).zfill(5)
        if geoid not in wanted:
            continue
        statefp, countyfp = geoid[:2], geoid[2:]
        land = int(row["ALAND"])
        water = int(row["AWATER"])
        rec = {"GEOID":geoid,"StateFP":statefp,"CountyFP":countyfp,"County_name_TIGER":row["NAME"],"AREALAND_m2":land,"AREAWATER_m2":water,"AREATOTAL_m2":land+water,"StateAbbr":STATE_FIPS_ABBR.get(statefp)}
        records.append(rec)
        features.append({"type":"Feature","properties":{"GEOID":geoid,"NAME":row["NAME"],"AREALAND":land,"AREAWATER":water,"AREATOTAL":land+water},"geometry":sr.shape.__geo_interface__})
    areas = pd.DataFrame(records)
    missing = wanted - set(areas.GEOID)
    if missing:
        raise ValueError(f"TIGER polygons missing target counties: {sorted(missing)}")
    areas = support.merge(areas, on="GEOID", how="left", validate="one_to_one")
    areas.to_csv(BASE / "processed" / "county_area_denominators.csv", index=False, encoding="utf-8-sig")
    (d / "target_county_boundaries.geojson").write_text(json.dumps({"type":"FeatureCollection","features":features},separators=(",",":")),encoding="utf-8")
    return areas, features


def osm_time_series(areas, features):
    path = RAW / "osm" / "osm_county_year_areas.csv"
    raw=RAW/"osm"/"osm_all_counties_2016_2023.json"
    if raw.exists(): payload=json.loads(raw.read_text(encoding="utf-8"))
    else:
        bpolys={"type":"FeatureCollection","features":features}
        response=get(OHSOME_AREA+"/groupBy/boundary",data={"bpolys":json.dumps(bpolys,separators=(",",":")),"time":"2016-01-01/2024-01-01/P1Y","filter":COOLING_FILTER,"format":"json"},timeout=900)
        raw.write_bytes(response.content);payload=response.json()
    grouped=payload.get("groupByResult")
    if grouped is None: raise ValueError(f"Unexpected ohsome boundary-group response: {list(payload)[:5]}")
    # ohsome labels collection members feature1...featureN in submitted feature order.
    geoids=[str(f["properties"]["GEOID"]).zfill(5) for f in features]
    rows=[]
    for group in grouped:
        m=re.search(r"(\d+)$",str(group.get("groupByObject","")))
        if not m: continue
        ix=int(m.group(1))-1
        if ix<0 or ix>=len(geoids): continue
        geoid=geoids[ix]
        for result in group.get("result",[]):
            match=re.search(r"(20\d{2})",result.get("timestamp",""))
            if not match: continue
            year=int(match.group(1))
            if year in YEARS:
                rows.append({"GEOID":geoid,"Year":year,"osm_indoor_facility_polygon_area_m2":float(result.get("value",0) or 0),"osm_response_timestamp":result.get("timestamp",""),"osm_filter":COOLING_FILTER,"osm_group_by_object":group.get("groupByObject")})
    out=pd.DataFrame(rows).drop_duplicates(["GEOID","Year"],keep="last")
    out.to_csv(path,index=False,encoding="utf-8-sig")
    return out


def download_eia():
    # 2015 provides the opening stock needed to compute 2016 net capacity change.
    for y in range(2015,2024):
        p=RAW/"eia860"/f"eia860{y}.zip"
        if not p.exists():
            url=f"https://www.eia.gov/electricity/data/eia860/archive/xls/eia860{y}.zip"
            p.write_bytes(get(url).content)
        if y>=2016:
            p2=RAW/"eia923"/f"f923_{y}.zip"
            if not p2.exists():
                url=f"https://www.eia.gov/electricity/data/eia923/archive/xls/f923_{y}.zip"
                p2.write_bytes(get(url).content)


def download_cbp():
    d=RAW/"cbp";d.mkdir(parents=True,exist_ok=True)
    for y in YEARS:
        short=str(y)[2:]
        p=d/f"cbp{short}co.zip"
        if not p.exists(): p.write_bytes(get(f"https://www2.census.gov/programs-surveys/cbp/datasets/{y}/cbp{short}co.zip").content)


def cbp_industry_composition(areas):
    wanted=set(areas.GEOID.astype(str).str.zfill(5))
    rows=[]
    for y in YEARS:
        short=str(y)[2:]
        zpath=RAW/"cbp"/f"cbp{short}co.zip"
        with zipfile.ZipFile(zpath) as zf:
            txt=next(n for n in zf.namelist() if n.lower().endswith(".txt"))
            with zf.open(txt) as stream:
                d=pd.read_csv(stream,dtype={"fipstate":str,"fipscty":str,"naics":str},low_memory=False)
        d["GEOID"]=(d["fipstate"].str.zfill(2)+d["fipscty"].str.zfill(3))
        d=d[d.GEOID.isin(wanted)].copy()
        d["est"]=pd.to_numeric(d["est"],errors="coerce")
        total=d[d.naics.eq("------")].groupby("GEOID",as_index=False).agg(total_establishments=("est","sum"))
        manufacturing=d[d.naics.isin(["31----","32----","33----"])].groupby("GEOID",as_index=False).agg(manufacturing_establishments=("est","sum"))
        joined=total.merge(manufacturing,on="GEOID",how="left")
        joined["manufacturing_establishments"]=joined.manufacturing_establishments.fillna(0)
        joined["Year"]=y
        rows.extend(joined.to_dict("records"))
    out=pd.DataFrame(rows)
    full=pd.MultiIndex.from_product([sorted(wanted),YEARS],names=["GEOID","Year"]).to_frame(index=False)
    out=full.merge(out,on=["GEOID","Year"],how="left")
    out.to_csv(BASE/"processed"/"cbp_county_establishments_2016_2023.csv",index=False,encoding="utf-8-sig")
    return out


def workbook_table(zip_path: Path, name_predicate, key_col="Plant Code", sheet_name=None):
    temp=RAW/"eia_workbooks"
    temp.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members=[n for n in zf.namelist() if name_predicate(n)]
        if not members:
            raise FileNotFoundError(f"No matching workbook in {zip_path.name}: {zf.namelist()[:5]}")
        target=temp/Path(members[0]).name
        if not target.exists():
            with zf.open(members[0]) as src, target.open("wb") as dst:
                dst.write(src.read())
    wb=openpyxl.load_workbook(target,read_only=True,data_only=True)
    # Do not depend on Excel's active-tab metadata for EIA generator status.
    # An archive update could change the active tab to Proposed or Retired.
    if sheet_name is not None and sheet_name not in wb.sheetnames:
        raise ValueError(f"Expected worksheet {sheet_name!r} in {target.name}; found {wb.sheetnames}")
    ws=wb[sheet_name] if sheet_name is not None else wb.active
    rows=ws.iter_rows(values_only=True)
    headers=None
    for vals in rows:
        clean=[str(v).replace("\n"," ").strip() if v is not None else "" for v in vals]
        if key_col in clean:
            headers=clean
            break
    if headers is None:
        raise ValueError(f"Could not locate header row in {target.name}")
    data=[]
    for vals in rows:
        if not any(v is not None for v in vals): continue
        data.append(dict(zip(headers,vals)))
    wb.close()
    return pd.DataFrame(data)


def standard_county_name(value):
    s=re.sub(r"[^A-Z0-9]","",str(value).upper())
    for suffix in ["COUNTY","PARISH","BOROUGH","CENSUSAREA","MUNICIPALITY","CITYANDCOUNTY","CITY"]:
        if s.endswith(suffix): s=s[:-len(suffix)]
    return s


def eia_capacity(areas):
    rows=[]
    targets=areas.copy()
    targets["_name_norm"]=targets.County_name_TIGER.map(standard_county_name)
    for y in range(2015,2024):
        zip_path=RAW/"eia860"/f"eia860{y}.zip"
        g=workbook_table(zip_path,lambda n: f"Generator_Y{y}" in n and n.lower().endswith(".xlsx"),key_col="Plant Code",sheet_name="Operable")
        required=["State","County","Energy Source 1","Nameplate Capacity (MW)"]
        miss=[x for x in required if x not in g.columns]
        if miss: raise ValueError(f"EIA860 {y}: missing {miss}; fields={list(g.columns)[:20]}")
        g["_statefp"]=g.State.map(lambda x: next((f for f,a in STATE_FIPS_ABBR.items() if a==str(x).strip().upper()),None))
        g["_name_norm"]=g.County.map(standard_county_name)
        g["_mw"]=pd.to_numeric(g["Nameplate Capacity (MW)"],errors="coerce").fillna(0).clip(lower=0)
        g["_renewable"]=g["Energy Source 1"].astype(str).str.strip().str.upper().isin(RENEWABLE_CODES)
        m=g.merge(targets[["GEOID","StateFP","_name_norm"]],left_on=["_statefp","_name_norm"],right_on=["StateFP","_name_norm"],how="inner")
        agg=m.groupby("GEOID",as_index=False).agg(total_capacity_mw=("_mw","sum"))
        ren=m.loc[m["_renewable"]].groupby("GEOID",as_index=False).agg(renewable_capacity_mw=("_mw","sum"))
        agg=agg.merge(ren,on="GEOID",how="left")
        agg["renewable_capacity_mw"]=agg["renewable_capacity_mw"].fillna(0.0)
        for _,r in agg.iterrows(): rows.append({"GEOID":r.GEOID,"Year":y,**r.drop(labels=["GEOID"]).to_dict()})
    out=pd.DataFrame(rows)
    full=pd.MultiIndex.from_product([areas.GEOID.astype(str).str.zfill(5).tolist(),range(2015,2024)],names=["GEOID","Year"]).to_frame(index=False)
    out=full.merge(out,on=["GEOID","Year"],how="left")
    out[["total_capacity_mw","renewable_capacity_mw"]]=out[["total_capacity_mw","renewable_capacity_mw"]].fillna(0.0)
    out.to_csv(BASE/"processed"/"eia860_county_capacity_2015_2023.csv",index=False,encoding="utf-8-sig")
    return out


def eia_generation(areas):
    targets=areas.copy()
    targets["_name_norm"]=targets.County_name_TIGER.map(standard_county_name)
    # Plant-to-county crosswalk is taken from EIA-860 generator records, then applied to EIA-923 plant records.
    rows=[]
    for y in YEARS:
        zip_path=RAW/"eia923"/f"f923_{y}.zip"
        temp=RAW/"eia_workbooks"; temp.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            members=[n for n in zf.namelist() if "Schedules_2_3_4_5" in n and n.lower().endswith(".xlsx")]
            if not members: raise FileNotFoundError(f"No generation workbook in {zip_path.name}")
            target=temp/Path(members[0]).name
            if not target.exists():
                with zf.open(members[0]) as src,target.open("wb") as dst: dst.write(src.read())
        wb=openpyxl.load_workbook(target,read_only=True,data_only=True)
        ws=wb["Page 1 Generation and Fuel Data"]
        iterator=ws.iter_rows(values_only=True)
        headers=None
        for vals in iterator:
            clean=[str(v).replace("\n"," ").strip() if v is not None else "" for v in vals]
            if "Plant Id" in clean and "Net Generation (Megawatthours)" in clean:
                headers=clean;break
        if headers is None: raise ValueError(f"EIA923 {y}: no expected headers")
        idx={h:i for i,h in enumerate(headers)}
        gtot={}; gren={}
        for vals in iterator:
            if not vals or vals[idx["Plant Id"]] is None: continue
            if "YEAR" in idx and str(vals[idx["YEAR"]]).strip() not in (str(y),"", "None"): continue
            pid=str(vals[idx["Plant Id"]]).strip()
            fuel=str(vals[idx["Reported Fuel Type Code"]]).strip().upper()
            try: mwh=float(str(vals[idx["Net Generation (Megawatthours)"]]).replace(",",""))
            except (ValueError,TypeError): continue
            if not math.isfinite(mwh): continue
            gtot[pid]=gtot.get(pid,0.0)+mwh
            if fuel in RENEWABLE_CODES: gren[pid]=gren.get(pid,0.0)+mwh
        wb.close()
        plant=workbook_table(RAW/"eia860"/f"eia860{y}.zip",lambda n:f"Generator_Y{y}" in n and n.lower().endswith(".xlsx"),key_col="Plant Code",sheet_name="Operable")
        plant["_statefp"]=plant.State.map(lambda x:next((f for f,a in STATE_FIPS_ABBR.items() if a==str(x).strip().upper()),None))
        plant["_name_norm"]=plant.County.map(standard_county_name)
        plant["_pid"]=plant["Plant Code"].astype(str).str.replace(r"\.0$","",regex=True)
        cross=plant[["_pid","_statefp","_name_norm"]].drop_duplicates("_pid").merge(targets[["GEOID","StateFP","_name_norm"]],left_on=["_statefp","_name_norm"],right_on=["StateFP","_name_norm"],how="inner")
        for _,p in cross.iterrows():
            pid=p._pid
            rows.append({"GEOID":p.GEOID,"Year":y,"total_net_generation_mwh":gtot.get(pid,0.0),"renewable_net_generation_mwh":gren.get(pid,0.0)})
    out=pd.DataFrame(rows)
    if not out.empty:
        out=out.groupby(["GEOID","Year"],as_index=False).sum(numeric_only=True)
    full=pd.MultiIndex.from_product([areas.GEOID.astype(str).str.zfill(5).tolist(),YEARS],names=["GEOID","Year"]).to_frame(index=False)
    out=full.merge(out,on=["GEOID","Year"],how="left")
    out[["total_net_generation_mwh","renewable_net_generation_mwh"]]=out[["total_net_generation_mwh","renewable_net_generation_mwh"]].fillna(0.0)
    out.to_csv(BASE/"processed"/"eia923_county_generation_2016_2023.csv",index=False,encoding="utf-8-sig")
    return out


def tcc_statistics(areas,features):
    p=RAW/"tree_canopy"/"tcc_county_year_statistics.csv"
    geom={f["properties"]["GEOID"]:f["geometry"] for f in features}
    def arcgis_polygon(g):
        coords=g["coordinates"]
        rings=coords if g["type"]=="Polygon" else [ring for polygon in coords for ring in polygon]
        return {"rings":rings,"spatialReference":{"wkid":4326}}
    area_map={str(r.GEOID).zfill(5):r for _,r in areas.iterrows()}
    tasks=[(geoid,y) for geoid in area_map for y in YEARS]
    def one_stat(key):
        geoid,y=key
        fpath=RAW/"tree_canopy"/"statistics_responses"/f"{geoid}_{y}.json"
        if fpath.exists(): return geoid,y,json.loads(fpath.read_text(encoding="utf-8"))
        stamp=int(datetime(y,1,1,tzinfo=timezone.utc).timestamp()*1000)
        params={"geometry":json.dumps(arcgis_polygon(geom[geoid]),separators=(",",":")),"geometryType":"esriGeometryPolygon","time":str(stamp),"f":"json"}
        response=get(TCC_SERVICE+"/computeStatisticsHistograms",data=params,timeout=240)
        payload=response.json()
        fpath.write_bytes(response.content)
        return geoid,y,payload
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
        futures={pool.submit(one_stat,key):key for key in tasks}
        for i,future in enumerate(concurrent.futures.as_completed(futures),1):
            results.append(future.result())
            if i%48==0: print(f"USFS TCC statistics {i}/{len(tasks)}",flush=True)
    rows=[]
    for geoid,y,payload in results:
            a=area_map[geoid]
            stats=payload.get("statistics") or []
            stat=stats[0] if stats else {}
            mean=stat.get("mean")
            count=stat.get("count")
            # The published NLCD TCC is percent canopy. Service statistics are over valid raster pixels.
            # Convert mean cover on county land to covered area, then divide by total county area (including water).
            if mean is not None:
                canopy_share=(float(mean)/100.0)*(float(a.AREALAND_m2)/float(a.AREATOTAL_m2))
            else: canopy_share=float("nan")
            rows.append({"GEOID":geoid,"Year":y,"tree_canopy_mean_percent_valid_pixels":mean,"tree_canopy_valid_pixel_count":count,"tree_canopy_share_total_county_area":canopy_share,"tcc_source":"USFS NLCD TCC 30m annual; 2025-6 ImageServer","tcc_response_error":payload.get("error",{}).get("message","")})
    out=pd.DataFrame(rows).drop_duplicates(["GEOID","Year"],keep="last")
    out.to_csv(p,index=False,encoding="utf-8-sig")
    return out


def main():
    mkdirs()
    panel,support=load_target_panel()
    areas,features=fetch_tiger(support)
    print(f"Target support: {len(support)} counties; {len(panel)} county-years",flush=True)
    print("Downloading OSM historical polygon-area series...",flush=True)
    osm=osm_time_series(areas,features)
    print(f"OSM returned {len(osm)} county-years",flush=True)
    print("Downloading EIA-860/EIA-923 annual source files...",flush=True)
    download_eia()
    print("Downloading Census CBP annual establishment files...",flush=True)
    download_cbp()
    print("Processing EIA capacity and generation...",flush=True)
    cap_path=BASE/"processed"/"eia860_county_capacity_2015_2023.csv"
    gen_path=BASE/"processed"/"eia923_county_generation_2016_2023.csv"
    cap=pd.read_csv(cap_path,dtype={"GEOID":str}) if cap_path.exists() else eia_capacity(areas)
    gen=pd.read_csv(gen_path,dtype={"GEOID":str}) if gen_path.exists() else eia_generation(areas)
    cbp_path=BASE/"processed"/"cbp_county_establishments_2016_2023.csv"
    cbp=pd.read_csv(cbp_path,dtype={"GEOID":str}) if cbp_path.exists() else cbp_industry_composition(areas)
    print("Querying annual USFS NLCD tree canopy statistics...",flush=True)
    tcc=tcc_statistics(areas,features)
    out=panel[["GEOID","Year","State","County","in_origin_support","in_destination_support"]].copy()
    out=out.merge(areas[["GEOID","AREALAND_m2","AREAWATER_m2","AREATOTAL_m2"]],on="GEOID",how="left",validate="many_to_one")
    out=out.merge(osm[["GEOID","Year","osm_indoor_facility_polygon_area_m2"]],on=["GEOID","Year"],how="left",validate="one_to_one")
    out["cooling_public_indoor_area_share_total_county_area"]=out.osm_indoor_facility_polygon_area_m2/out.AREATOTAL_m2
    out=out.merge(tcc[["GEOID","Year","tree_canopy_share_total_county_area"]],on=["GEOID","Year"],how="left",validate="one_to_one")
    out=out.merge(cap,on=["GEOID","Year"],how="left",validate="one_to_one")
    prev=cap.assign(Year=cap.Year+1).rename(columns={"renewable_capacity_mw":"prior_year_renewable_capacity_mw","total_capacity_mw":"prior_year_total_capacity_mw"})
    out=out.merge(prev[["GEOID","Year","prior_year_renewable_capacity_mw"]],on=["GEOID","Year"],how="left",validate="one_to_one")
    out["renewable_capacity_share"]=out.renewable_capacity_mw/out.total_capacity_mw.replace({0:float("nan")})
    out.loc[out.total_capacity_mw.eq(0),"renewable_capacity_share"]=0.0
    out=out.merge(gen,on=["GEOID","Year"],how="left",validate="one_to_one")
    out["renewable_generation_share_raw"]=out.renewable_net_generation_mwh/out.total_net_generation_mwh.replace({0:float("nan")})
    out.loc[out.total_net_generation_mwh.eq(0),"renewable_generation_share_raw"]=0.0
    # Four EIA county-years show <0.9% annual reconciliation differences where fuel-specific renewable MWh slightly exceed reported total net MWh.
    # Keep the unbounded ratio for audit and bound the model-ready share to its valid [0,1] range.
    out["renewable_generation_share"]=out.renewable_generation_share_raw.clip(lower=0,upper=1)
    out=out.merge(cbp,on=["GEOID","Year"],how="left",validate="one_to_one")
    out["manufacturing_establishment_share"]=out.manufacturing_establishments/out.total_establishments.replace({0:float("nan")})
    housing=panel[["GEOID","Year","ln_hpi","ln_gross_rent","rent_burden_30plus"]].copy()
    out=out.merge(housing,on=["GEOID","Year"],how="left",validate="one_to_one")
    out=out.sort_values(["GEOID","Year"]).reset_index(drop=True)
    final=OUT/"results5_mediator_dataset_2016_2023.csv"
    out.to_csv(final,index=False,encoding="utf-8-sig")
    # Summarize coverage/missingness without modifying the source panel.
    vars=["cooling_public_indoor_area_share_total_county_area","tree_canopy_share_total_county_area","renewable_capacity_share","renewable_generation_share","manufacturing_establishment_share","ln_hpi","ln_gross_rent","rent_burden_30plus"]
    qa={"rows":len(out),"unique_counties":int(out.GEOID.nunique()),"years":sorted(out.Year.unique().tolist()),"duplicate_county_years":int(out.duplicated(["GEOID","Year"]).sum()),"support_count_origin":int(pd.to_numeric(support.in_origin_support).sum()),"support_count_destination":int(pd.to_numeric(support.in_destination_support).sum()),"missing_by_indicator":{v:int(out[v].isna().sum()) for v in vars},"nonmissing_by_indicator":{v:int(out[v].notna().sum()) for v in vars},"renewable_generation_share_values_capped":int((out.renewable_generation_share_raw>1).sum()),"indicator_columns":vars}
    (OUT/"results5_mediator_dataset_qa.json").write_text(json.dumps(qa,indent=2),encoding="utf-8")
    print(json.dumps(qa,indent=2),flush=True)


if __name__=="__main__":
    main()
