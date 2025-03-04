import pandas as pd

for i in range(17, 22):
    # Process inflow migrants data
    Flow = pd.read_csv(r'filepath\countyinflow'+str(i)+''+str(i+1)+'.csv')
    Flow['y2_statefips'] = Flow['y2_statefips'].astype(str)
    Flow['y2_countyfips'] = Flow['y2_countyfips'].astype(str)
    Flow['y2_countyfips'] = Flow['y2_countyfips'].apply(lambda x: x.zfill(3))
    Flow['GEOID'] = (Flow['y2_statefips'] + Flow['y2_countyfips']).astype(int)
    Flow['total'] = Flow['n1'] + Flow['n2']

    df = Flow[Flow['y1_countyname'].str.contains('Total Migration-Same State|Total Migration-Different State')]
    df['y1_countyname'] = df['y1_countyname'].str.extract(r'(Same State|Different State)')
    process = df[['GEOID', 'total', 'y1_countyname']]
    process1 = process.pivot(index='GEOID', columns='y1_countyname', values='total').reset_index()
    process1['intotal'] = process1['Different State'] + process1['Same State']
    inflow = process1.rename(columns={'Different State': 'indiff', 'Same State': 'insame'})

    # Process outflow migrants data
    out = pd.read_csv(r'filepath\countyoutflow'+str(i)+''+str(i+1)+'.csv')

    out['y1_statefips'] = out['y1_statefips'].astype(str)
    out['y1_countyfips'] = out['y1_countyfips'].astype(str)
    out['y1_countyfips'] = out['y1_countyfips'].apply(lambda x: x.zfill(3))
    out['GEOID'] = (out['y1_statefips'] + out['y1_countyfips']).astype(int)
    out['total'] = out['n1'] + out['n2']

    dfout = out[out['y2_countyname'].str.contains('Total Migration-Same State|Total Migration-Different State')]
    dfout['y2_countyname'] = dfout['y2_countyname'].str.extract(r'(Same State|Different State)')
    processout = dfout[['GEOID', 'total', 'y2_countyname']]
    process1out = processout.pivot(index='GEOID', columns='y2_countyname', values='total').reset_index()
    process1out['outtotal'] = process1out['Different State'] + process1out['Same State']
    outflow = process1out.rename(columns={'Different State': 'outdiff', 'Same State': 'outsame'})

    # Calculate net migration
    netflow = inflow.merge(outflow, on='GEOID', how='left')
    netflow['netdiff'] = netflow['indiff'] - netflow['outdiff']
    netflow['netsame'] = netflow['insame'] - netflow['outsame']
    netflow['nettotal'] = netflow['intotal'] - netflow['outtotal']
    new = netflow[['GEOID']+['netdiff']+['netsame']+['nettotal']].copy()

    inflow.to_csv(r'filepath\inflow20'+str(i)+'.csv', index=False)
    outflow.to_csv(r'filepath\outflow20'+str(i)+'.csv', index=False)
    new.to_csv(r'filepath\netflow20'+str(i)+'.csv', index=False)
