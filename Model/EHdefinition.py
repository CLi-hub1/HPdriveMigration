import pandas as pd
import numpy as np
from RobustRidgeR_CasualFunction import IVRobustR

heat = pd.read_excel(r'filepath\heatcontext.xlsx')
heat = heat.rename(columns={'FIPS': 'GEOID'})
for i in range(2017,2021):
    var = pd.read_csv(r'filepath\WBGT'+str(i)+'.csv')
    var = var.rename(columns={'StCoFIPS': 'GEOID'})
    heat = heat.merge(var, on='GEOID', how='left')
    heat.rename(columns={'COUNT': 'WBGTCOUNT'+str(i)}, inplace=True)
    heat.rename(columns={'mean6-8': 'WBGTmean6-8'+str(i)}, inplace=True)
for i in range(2017,2021):
    var2 = pd.read_csv(r'filepath\HI'+str(i)+'.csv')
    var2 = var2.rename(columns={'StCoFIPS': 'GEOID'})
    heat = heat.merge(var2, on='GEOID', how='left')
    heat.rename(columns={'COUNT': 'HICOUNT'+str(i)}, inplace=True)
    heat.rename(columns={'mean6-8': 'HImean6-8'+str(i)}, inplace=True)
heat.drop(columns=['COUNT2017', 'COUNT2018','COUNT2019','COUNT2020'], inplace=True)

mer = pd.read_excel(r'filepath\AAAmergenet.xlsx')
conta = pd.DataFrame()

EHD = 'HImean6-8'
for i in range(2017,2021):
    # i = 2017
    mer1 = mer[mer['Year'] == i]
    mer1 = mer1.merge(heat[['GEOID', EHD +str(i)+'']], on='GEOID', how='left')
    columns = list(mer1.columns)
    count_index = columns.index('COUNT')
    wbgt_index = columns.index(EHD+str(i)+'')
    columns[count_index], columns[wbgt_index] = columns[wbgt_index], columns[count_index]
    mer1 = mer1[columns]
    mer1.drop(columns=['COUNT'], inplace=True)
    mer1.rename(columns={EHD+str(i)+'': 'COUNT'}, inplace=True)
    conta = pd.concat([conta, mer1])


ALL = conta.copy()
list8 = ['netdiff','netsame','nettotal']
i = 3
results_df,results_df_TWFE,results_df_simple = IVRobustR(ALL,list8,i)

results_df.to_csv(r'filepath\IV2SR_'+EHD+''+str(i)+'.csv')

