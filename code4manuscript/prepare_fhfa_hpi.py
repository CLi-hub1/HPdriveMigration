"""Extract annual FHFA county HPI values for the observed migration support."""
import argparse
from pathlib import Path

import pandas as pd

from config import PROJECT_ROOT, OUTPUT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hpi-file', type=Path, required=True, help='FHFA county HPI workbook.')
    parser.add_argument('--panel', type=Path, default=PROJECT_ROOT / 'results3/data/panel_lag0.csv')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_ROOT / 'fhfa_hpi')
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit('Select an empty output directory.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    od = pd.read_csv(args.panel, dtype={'GEOID': str, 'GEOID_destination': str})
    origins = set(od.GEOID.str.zfill(5))
    destinations = set(od.GEOID_destination.str.zfill(5))
    counties = sorted(origins | destinations)
    assert len(origins) == len(destinations) == 107 and len(counties) == 108
    support = pd.DataFrame({'GEOID': counties})
    support['in_origin_support'] = support.GEOID.isin(origins)
    support['in_destination_support'] = support.GEOID.isin(destinations)
    support.to_csv(args.output_dir / 'frozen_county_support.csv', index=False)
    base = pd.MultiIndex.from_product([counties, range(2016, 2024)], names=['GEOID', 'Year']).to_frame(index=False)
    base = base.merge(support, on='GEOID', validate='many_to_one')
    hpi = pd.read_excel(args.hpi_file, header=5, dtype=str).rename(columns={
        'FIPS code': 'GEOID', 'Annual Change (%)': 'hpi_yoy_pct',
        'HPI': 'county_hpi_native_base', 'HPI with 1990 base': 'county_hpi_1990_100',
        'HPI with 2000 base': 'county_hpi_2000_100'})
    hpi['Year'] = pd.to_numeric(hpi['Year'], errors='raise')
    hpi['GEOID'] = hpi.GEOID.str.zfill(5)
    hpi = hpi[hpi.GEOID.isin(counties) & hpi.Year.between(2016, 2023)].copy()
    assert not hpi[['GEOID', 'Year']].duplicated().any()
    for col in ('hpi_yoy_pct', 'county_hpi_native_base', 'county_hpi_1990_100', 'county_hpi_2000_100'):
        hpi[col + '_raw'] = hpi[col]
        hpi[col] = pd.to_numeric(hpi[col], errors='coerce')
    hpi['fhfa_row_present'] = True
    hpi = base.merge(hpi, on=['GEOID', 'Year'], how='left', validate='one_to_one')
    hpi['fhfa_row_present'] = hpi.fhfa_row_present.fillna(False).astype(bool)
    hpi['county_hpi'] = hpi.county_hpi_native_base
    hpi.to_csv(args.output_dir / 'fhfa_county_annual_2016_2023.csv', index=False)
    print(f'Wrote {len(hpi)} county-year rows.')


if __name__ == '__main__':
    main()
