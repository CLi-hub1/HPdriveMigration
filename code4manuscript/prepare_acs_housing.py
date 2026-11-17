"""Build ACS county-year housing variables from staged five-year table files."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import OUTPUT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True,
                        help='Directory containing frozen_county_support.csv and annual ACS table CSVs.')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_ROOT / 'acs_housing')
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit('Select an empty output directory.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    support = pd.read_csv(args.input_dir / 'frozen_county_support.csv', dtype={'GEOID': str})
    frames, sentinels = [], []
    for year in range(2016, 2024):
        tables = [pd.read_csv(args.input_dir / f'acs5_{year}_{name}_county_source.csv', dtype={'GEOID': str})
                  for name in ('B25064', 'B25070', 'B25077')]
        data = tables[0].drop(columns='GEO_ID', errors='ignore')
        for table in tables[1:]:
            data = data.merge(table.drop(columns='GEO_ID', errors='ignore'),
                              on=['GEOID', 'Year'], how='outer', validate='one_to_one')
        assert len(data) == 108 and not data.GEOID.duplicated().any()
        for col in [c for c in data if c.startswith('B25')]:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            for row in data.loc[data[col].lt(0), ['GEOID', col]].itertuples(index=False, name=None):
                sentinels.append({'Year': year, 'GEOID': row[0], 'field': col, 'raw_value': row[1]})
            data[col] = data[col].where(data[col] >= 0)
        data['ACS_period_start'], data['ACS_period_end'] = year - 4, year
        data['gross_rent_nominal_usd'] = data.B25064_E001
        data['gross_rent_moe90_usd'] = data.B25064_M001
        data['ln_gross_rent'] = np.log(data.gross_rent_nominal_usd.where(data.gross_rent_nominal_usd > 0))
        data['home_value_nominal_usd'] = data.B25077_E001
        data['home_value_moe90_usd'] = data.B25077_M001
        data['ln_home_value'] = np.log(data.home_value_nominal_usd.where(data.home_value_nominal_usd > 0))
        data['renter_units_total'] = data.B25070_E001
        data['rent_burden_not_computed_units'] = data.B25070_E011
        data['rent_burden_computed_units'] = data.B25070_E001 - data.B25070_E011
        data['rent_burden_30plus_units'] = data[[f'B25070_E{i:03d}' for i in (7, 8, 9, 10)]].sum(axis=1, min_count=4)
        data['rent_burden_50plus_units'] = data.B25070_E010
        denominator = data.rent_burden_computed_units.where(data.rent_burden_computed_units > 0)
        data['rent_burden_30plus'] = data.rent_burden_30plus_units / denominator
        data['rent_burden_50plus'] = data.rent_burden_50plus_units / denominator
        components = data[[f'B25070_E{i:03d}' for i in range(2, 12)]].sum(axis=1, min_count=10)
        assert (components - data.B25070_E001).dropna().eq(0).all()
        assert data.rent_burden_30plus.dropna().between(0, 1).all()
        frames.append(data)
    result = pd.concat(frames, ignore_index=True).merge(support, on='GEOID', validate='many_to_one')
    result.to_csv(args.output_dir / 'acs_housing_county_2016_2023.csv', index=False)
    pd.DataFrame(sentinels, columns=['Year', 'GEOID', 'field', 'raw_value']).to_csv(args.output_dir / 'negative_value_flags.csv', index=False)
    print(f'Wrote {len(result)} county-year rows.')


if __name__ == '__main__':
    main()
