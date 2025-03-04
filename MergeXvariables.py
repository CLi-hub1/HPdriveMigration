import pandas as pd

# Read X variables data
## Economic related data
income = pd.read_excel(r'filepath\income.xlsx')
HV = pd.read_excel(r'filepath\medianhomevalue.xlsx')
## Social related data
POPD = pd.read_excel(r'filepath\pop.xlsx')
ER = pd.read_excel(r'filepath\employmentrate.xlsx')
RD = pd.read_excel(r'filepath\racialdiversity.xlsx')
aging = pd.read_excel(r'filepath\agingrate.xlsx')
EA = pd.read_excel(r'filepath\educationattainment.xlsx')
## Environmental related data
heat = pd.read_excel(r'filepath\heatcontext.xlsx')
PM25 = pd.read_excel(r'filepath\PM2.5mean.xlsx')
BUrate = pd.read_csv(r'filepath\built-upratemean.csv')

# Process data
data_list = [income, HV, POPD, ER, RD, aging, EA]
for data in data_list:
    data['GEO_ID'] = data['GEO_ID'].str.replace('0500000US', '')
    data['GEO_ID'] = data['GEO_ID'].astype(int)
income = income.rename(columns={'GEO_ID': 'GEOID'})
HV = HV.rename(columns={'GEO_ID': 'GEOID'})
POPD = POPD.rename(columns={'GEO_ID': 'GEOID'})
ER = ER.rename(columns={'GEO_ID': 'GEOID'})
RD = RD.rename(columns={'GEO_ID': 'GEOID'})
aging = aging.rename(columns={'GEO_ID': 'GEOID'})
EA = EA.rename(columns={'GEO_ID': 'GEOID'})
heat = heat.rename(columns={'FIPS': 'GEOID'})
PM25['statefips'] = PM25['statefips'].astype(str)
PM25['countyfips'] = PM25['countyfips'].astype(str)
PM25['countyfips'] = PM25['countyfips'].apply(lambda x: x.zfill(3))
PM25['GEOID'] = (PM25['statefips'] + PM25['countyfips']).astype(int)

# Merge data
merged_df = income.copy()
merged_df = merged_df.merge(HV, on='GEOID', how='left')
merged_df = merged_df.merge(POPD, on='GEOID', how='left')
merged_df = merged_df.merge(ER, on='GEOID', how='left')
merged_df = merged_df.merge(RD, on='GEOID', how='left')
merged_df = merged_df.merge(aging, on='GEOID', how='left')
merged_df = merged_df.merge(EA, on='GEOID', how='left')
merged_df = merged_df.merge(heat, on='GEOID', how='left')
merged_df = merged_df.merge(BUrate, on='GEOID', how='left')

## Calculate population density
merged_df['PopD2017'] = merged_df['PopD2017'] / merged_df['ALAND']
merged_df['PopD2018'] = merged_df['PopD2018'] / merged_df['ALAND']
merged_df['PopD2019'] = merged_df['PopD2019'] / merged_df['ALAND']
merged_df['PopD2020'] = merged_df['PopD2020'] / merged_df['ALAND']

for i in range(2017, 2021):
    # i = 2017
    column_name = [col for col in merged_df.columns if str(i) in str(col)]
    new_df = merged_df[['GEOID']+['INTPTLAT']+['INTPTLON'] + column_name].copy() 
    PM25specific = PM25[PM25['year'] == i]
    X = new_df.merge(PM25specific, on='GEOID', how='left')
    X = X.drop(columns=['statefips', 'countyfips', 'year'])
    X = X.rename(columns={'DS_PM_pred': 'PM'+str(i)})
    X.to_csv(r'filepath\AAXcit'+str(i)+'1.csv', index=False)
