import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy
from scipy.stats import chi2
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from scipy.stats import t
from scipy.stats import f
import matplotlib.pyplot as plt
from RobustRidgeR import SecondstageR

# list = ['netdiff','netsame','nettotal']
# i = 3
# ALL = pd.read_excel(r'filepath\AAAmergenet.xlsx')


def IV2SRdecomposition(ALL,list,i):
    ## Load data
    PER = pd.read_excel(r'filepath\2015_USclimatebrief_countylevel.xlsx')
    PER = PER.rename(columns={'County_FIPS': 'GEOID'})
    ## Instrument selection
    meantemp = pd.read_excel(r'filepath\meantem1995-2014.xlsx')
    meanprecipitation = pd.read_excel(r'filepath\meanPRECIPITATION1995-2014.xlsx')
    ALL = pd.merge(ALL, PER, on='GEOID', how='left')
    ALL = pd.merge(ALL, meantemp, on='GEOID', how='left')
    ALL = pd.merge(ALL, meanprecipitation, on='GEOID', how='left')

    ## Preprocessing
    ALL['income'] = np.log10(ALL['income'])
    ALL['HV'] = np.log10(ALL['HV'])
    ALL['PopD'] = np.log10(ALL['PopD'])
    ALL['EM'] = ALL['EM']/100
    ALL['aging'] = ALL['aging']/100
    ALL['EA'] = ALL['EA']/100
    ALL['BUMEAN'] = ALL['BUMEAN']/100
    ALL['netdiff'] = ALL['netdiff']/1000
    ALL['netsame'] = ALL['netsame']/1000
    ALL['nettotal'] = ALL['nettotal']/1000
    AF = ALL.copy()
    AF = AF.dropna()
    AF[['REGION']] = AF[['REGION']].astype(int)

    ## IV2SLS regression
    # First stage regression
    formulaAP = 'APtotal ~ income + HV + PopD + EM + RD + aging + EA + COUNT + BUMEAN + PM + Valuetem + ValuePRE + C(State_FIPS) + C(Year)'
    formulaMP = 'MPtotal ~ income + HV + PopD + EM + RD + aging + EA + COUNT + BUMEAN + PM + Valuetem + ValuePRE + C(State_FIPS) + C(Year)'
    mean_APtotal = AF['APtotal'].mean()
    var_APtotal = AF['APtotal'].var()
    mean_MPtotal = AF['MPtotal'].mean()
    var_MPtotal = AF['MPtotal'].var()

    # Choose the appropriate regression model for the first stage (Poisson or Negative Binomial) based on whether variance is greater than mean
    if var_APtotal > mean_APtotal:
        model_APtotal = smf.glm(formula=formulaAP, data=AF, family=sm.families.NegativeBinomial()).fit()
    else:
        model_APtotal = smf.glm(formula=formulaAP, data=AF, family=sm.families.Poisson()).fit()
    AF['APtotal_hat'] = model_APtotal.fittedvalues
    if var_MPtotal > mean_MPtotal:
        model_MPtotal = smf.glm(formula=formulaMP, data=AF, family=sm.families.NegativeBinomial()).fit()
    else:
        model_MPtotal = smf.glm(formula=formulaMP, data=AF, family=sm.families.Poisson()).fit()
    AF['MPtotal_hat'] = model_MPtotal.fittedvalues
    null_model_AP = smf.glm('APtotal ~ 1', data=AF, family=model_APtotal.family).fit()
    llf_full_AP = model_APtotal.llf
    llf_null_AP = null_model_AP.llf
    lr_stat_AP = -2 * (llf_null_AP - llf_full_AP)
    df_diff_AP = model_APtotal.df_model - null_model_AP.df_model
    Fvalue_AP = chi2.sf(lr_stat_AP, df_diff_AP)
    null_model_MP = smf.glm('MPtotal ~ 1', data=AF, family=model_MPtotal.family).fit()
    llf_full_MP = model_MPtotal.llf
    llf_null_MP = null_model_MP.llf
    lr_stat_MP = -2 * (llf_null_MP - llf_full_MP)
    df_diff_MP = model_MPtotal.df_model - null_model_MP.df_model
    Fvalue_MP = chi2.sf(lr_stat_MP, df_diff_MP)

    # Second stage regression
    formula_seperate = f"{list[i-1]} ~ PROPAPTechnologicalInfrastructural + PROPAPInstitutional + PROPAPBehavioralCultural + PROPAPNatureBased +PROPMPTechnologicalInfrastructural + PROPMPInstitutional + PROPMPBehavioralCultural + PROPMPNatureBased + income + HV + PopD + EM + RD + aging + EA + COUNT + BUMEAN + PM + C(State_FIPS) + C(Year)"
    y, X = patsy.dmatrices(formula_seperate, data=AF, return_type='dataframe')
    AAresults_seperate,residuals_second_stage_seperate= SecondstageR(y,X,AF,Fvalue_AP,Fvalue_MP)

    AF['PROPAPTechnologicalInfrastructural'] = AF['APTechnologicalInfrastructural']/AF['APtotal']*AF['APtotal_hat']
    AF['PROPAPInstitutional'] = AF['APInstitutional']/AF['APtotal']*AF['APtotal_hat']
    AF['PROPAPBehavioralCultural'] = AF['APBehavioralCultural']/AF['APtotal']*AF['APtotal_hat']
    AF['PROPAPNatureBased'] = AF['APNatureBased']/AF['APtotal']*AF['APtotal_hat']
    AF['PROPMPTechnologicalInfrastructural'] = AF['MPTechnologicalInfrastructural']/AF['MPtotal']*AF['MPtotal_hat']
    AF['PROPMPInstitutional'] = AF['MPInstitutional']/AF['MPtotal']*AF['MPtotal_hat']
    AF['PROPMPBehavioralCultural'] = AF['MPBehavioralCultural']/AF['MPtotal']*AF['MPtotal_hat']
    AF['PROPMPNatureBased'] = AF['MPNatureBased']/AF['MPtotal']*AF['MPtotal_hat']

    columns_to_fill = ['PROPAPTechnologicalInfrastructural', 'PROPAPInstitutional', 'PROPAPBehavioralCultural', 'PROPAPNatureBased',
                   'PROPMPTechnologicalInfrastructural', 'PROPMPInstitutional', 'PROPMPBehavioralCultural', 'PROPMPNatureBased']
    AF[columns_to_fill] = AF[columns_to_fill].fillna(0)

    formula_region = ''+list[i-1]+' ~ AP1 + AP2 + AP3 + AP4 + MP1 + MP2 + MP3 + MP4 +income + HV + PopD + EM + RD + aging + EA + COUNT + BUMEAN + PM + C(State_FIPS) + C(Year)'
    y, X = patsy.dmatrices(formula_region, data=AF, return_type='dataframe')
    AAresults_region,residuals_second_stage_region= SecondstageR(y,X,AF,Fvalue_AP,Fvalue_MP)

    results_df_seperate = AAresults_seperate.copy()
    results_df_region = AAresults_region.copy()

    return results_df_seperate,results_df_region

# results_df_seperate.to_csv(r'filepath\Ptypemaxtem5y'+str(i)+'.csv')
# results_df_region.to_csv(r'filepath\Pregionmaxtem5y'+str(i)+'.csv')