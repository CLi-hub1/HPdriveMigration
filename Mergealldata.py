import pandas as pd

#Load latitude and longitude, Xcit, CPcit, and ycit data
APcit = pd.read_excel(r'filepath\AP4types.xlsx')
MPcit = pd.read_excel(r'filepath\MP4types.xlsx')
merge = pd.DataFrame()

#Remove duplicates
APcit = APcit.drop_duplicates(subset=['Year', 'GEOID', 'merge'], keep='first')
MPcit = MPcit.drop_duplicates(subset=['Year', 'GEOID', 'merge'], keep='first')

for i in range(2017, 2021):
    # i=2017
    X = pd.read_csv(r'filepath\AAXcit'+str(i)+'.csv')
    y = pd.read_csv(r'filepath\netflow'+str(i)+'.csv')
    ## Process APcit and MPcit data
    AP = APcit[APcit['Year'] == i]
    AP = AP[['GEOID']+['Technological/Infrastructural'] + ['Institutional'] + ['Behavioral/Cultural'] + ['Nature-Based']]
    aggregated_AP = AP.groupby('GEOID').sum().reset_index()
    aggregated_AP['total'] = aggregated_AP.iloc[:, 1:].sum(axis=1)
    aggregated_AP.columns = aggregated_AP.columns.map(lambda x: 'AP' + str(x))
    MP = MPcit[MPcit['Year'] == i]
    MP = MP[['GEOID']+['Technological/Infrastructural'] + ['Institutional'] + ['Behavioral/Cultural'] + ['Nature-Based']]
    aggregated_MP = MP.groupby('GEOID').sum().reset_index()
    aggregated_MP['total'] = aggregated_MP.iloc[:, 1:].sum(axis=1)
    aggregated_MP.columns = aggregated_MP.columns.map(lambda x: 'MP' + str(x))
    aggregated_AP = aggregated_AP.rename(columns={'APGEOID': 'GEOID'})
    aggregated_MP = aggregated_MP.rename(columns={'MPGEOID': 'GEOID'})
    ## Merge data
    CP = pd.merge(aggregated_AP, aggregated_MP, on="GEOID", how='outer')
    CP.fillna(0, inplace=True)
    df = pd.merge(CP, X, on='GEOID', how='left')
    df = pd.merge(df, y, on='GEOID', how='left')
    new_columns = [col.replace(str(i), '') for col in df.columns]
    df.rename(columns=dict(zip(df.columns, new_columns)), inplace=True)
    df.insert(0, 'Year', str(i))
    merge = pd.concat([merge, df])
merge['HV'] = merge['HV'].astype(int)
merge['income'] = merge['income'].astype(int)
merge['netsame'].fillna(0, inplace=True)
merge['nettotal'] = merge['netsame'] + merge['netdiff']
merge = merge.dropna()
merge.rename(columns={'APTechnological/Infrastructural':'APTechnologicalInfrastructural','APBehavioral/Cultural':'APBehavioralCultural','APNature-Based':'APNatureBased','MPTechnological/Infrastructural':'MPTechnologicalInfrastructural','MPBehavioral/Cultural':'MPBehavioralCultural','MPNature-Based':'MPNatureBased'}, inplace=True)
merge.to_excel(r'filepath\AAAmergenet.xlsx', index=False)



