import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

list = ['APTechnologicalInfrastructural','APInstitutional','APBehavioralCultural','APNatureBased','MPTechnologicalInfrastructural','MPInstitutional','MPBehavioralCultural','MPNatureBased']

Amerge = pd.read_excel(r'filepath\AAAmergenet.xlsx')
PER = pd.read_excel(r'filepath\2015_USclimatebrief_countylevel.xlsx')
PER = PER.rename(columns={'County_FIPS': 'GEOID'})

# %% [markdown]public opinions' impacts on heat-related policies
list_Xcon = ['income2017','HV2017','PopD2017','EM2017','RD2017','aging2017','EA2017','COUNT2017','BU2017MEAN','PM2017']
list_Xvar = ['happening','humancaused','scienceconsensus','x78_worried','harmUS','regulatecarbon','supportRES']
merged_df = pd.merge(PER, Amerge, on='GEOID', how='inner')
remaining_df = PER[~PER['GEOID'].isin(merged_df['GEOID'])]

Xcontrol = pd.read_csv(r'filepath\Countydata_Xcit\AAXcit2017.csv')
Xcontrol = Xcontrol.replace('-', np.nan)
Xcontrol['HV2017'] = Xcontrol['HV2017'].astype(float)
Xcontrol['income2017'] = Xcontrol['income2017'].astype(float)
Xcontrol['income2017'] = np.log10(Xcontrol['income2017'])
Xcontrol['HV2017'] = np.log10(Xcontrol['HV2017'])
Xcontrol['PopD2017'] = np.log10(Xcontrol['PopD2017'])
Xcontrol['EM2017'] = Xcontrol['EM2017']/100
Xcontrol['aging2017'] = Xcontrol['aging2017']/100
Xcontrol['EA2017'] = Xcontrol['EA2017']/100
Xcontrol['BU2017MEAN'] = Xcontrol['BU2017MEAN']/100

merged_df1 = pd.merge(merged_df, Xcontrol, on='GEOID', how='inner')
merged_df1 = merged_df1.groupby('GEOID').mean().reset_index()
remaining_df1 = pd.merge(remaining_df, Xcontrol, on='GEOID', how='inner')
remaining_df1[list] = 0

# Filter data
merged_df1_select = merged_df1[['GEOID'] + ['State_FIPS'] + list + list_Xcon + list_Xvar]
remaining_df1_select = remaining_df1[['GEOID'] + ['State_FIPS'] + list + list_Xcon + list_Xvar]
allmerge = pd.concat([merged_df1_select, remaining_df1_select])
ALL = allmerge.dropna()
ALL['happening'] = ALL['happening']/100
ALL['humancaused'] = ALL['humancaused']/100
ALL['scienceconsensus'] = ALL['scienceconsensus']/100
ALL['x78_worried'] = ALL['x78_worried']/100
ALL['harmUS'] = ALL['harmUS']/100
ALL['regulatecarbon'] = ALL['regulatecarbon']/100
ALL['supportRES'] = ALL['supportRES']/100

result = pd.DataFrame(columns=['y', 'x', 'coef', 'p_value', 'StdErr', 'CI_90', 'CI_95', 'AdjR2','observations'])

for y_col in list:
    X = ALL[list_Xvar].astype(float)
    y = ALL[y_col]
    # # Use Random Forest for feature selection
    rf = RandomForestRegressor()
    sfm = SelectFromModel(rf)
    X_selected = sfm.fit_transform(X, y)
    selected_features = X.columns[sfm.get_support()]
    formula = ''+y_col+' ~ income2017+ HV2017+ PopD2017+ EM2017+ RD2017+ aging2017+ EA2017+ COUNT2017+ BU2017MEAN+ PM2017'
    if len(selected_features) > 0:
        formula += ' + ' + ' + '.join(selected_features)
    formula += ' + C(State_FIPS)'
    model = smf.ols(formula=formula, data=ALL).fit()
    adjr2 = model.rsquared_adj

    coef = model.params
    p_values = model.pvalues
    std_err = model.bse
    CI_90 = (model.conf_int(alpha=0.1)[1]- model.conf_int(alpha=0.1)[0]) / 2
    CI_95 = (model.conf_int(alpha=0.05)[1] - model.conf_int(alpha=0.05)[0]) / 2
    observations = model.nobs    
    for x_col, c, p, se, ci90, ci95 in zip(model.params.index, coef, p_values, std_err, CI_90, CI_95):
        if x_col != 'Intercept':
            result = result.append({'y': y_col, 'x': x_col, 'coef': c, 'p_value': p, 'StdErr': se, 'CI_90': ci90, 'CI_95': ci95, 'observations': observations, 'AdjR2': adjr2}, ignore_index=True)


result['mark'] = np.where(result['p_value'] < 0.01, '***', 
                np.where(result['p_value'] < 0.05, '**', 
                    np.where(result['p_value'] < 0.1, '*', '')))

result_con = result[result['x'].isin(list_Xvar)]
result_con.to_excel(r'filepath\Fig4_publicimpactonCP.xlsx')
