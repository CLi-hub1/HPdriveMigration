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

def IVRobustR(ALL,list,i):
    PER = pd.read_excel(r'filepath\2015_USclimatebrief_countylevel.xlsx')
    PER = PER.rename(columns={'County_FIPS': 'GEOID'})
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
    # print(dir(model_APtotal))
    # Test for relevance (weak instrument test). Calculate likelihood ratio test as a substitute for F value, Plrt_AP and Plrt_MP less than 0.05 indicates that the instrument variables pass the weak instrument test, i.e., effective
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
    formula = f"{list[i-1]} ~ APtotal_hat + MPtotal_hat + income + HV + PopD + EM + RD + aging + EA + COUNT + BUMEAN + PM + C(State_FIPS) + C(Year)"
    y, X = patsy.dmatrices(formula, data=AF, return_type='dataframe')
    AAresults,residuals_second_stage= SecondstageR(y,X,AF,Fvalue_AP,Fvalue_MP)

    ## TWFE regression and Endogeneity test
    formula_TWFE = ''+list[i-1]+' ~ APtotal + MPtotal + income + HV + PopD + EM + RD + aging + EA + COUNT + BUMEAN + PM + C(State_FIPS) + C(Year)'
    model_TWFE = smf.ols(formula=formula_TWFE, data=AF).fit()
    params_TWFE = model_TWFE.params
    conf_int_TWFE = model_TWFE.conf_int(alpha=0.05)
    CI_95_TWFE = (conf_int_TWFE[1] - conf_int_TWFE[0])/2
    CI90_TWFE = model_TWFE.conf_int(alpha=0.1)
    CI_90_TWFE = (CI90_TWFE[1] - CI90_TWFE[0])/2
    std_err_TWFE = model_TWFE.bse
    pvalues_TWFE = model_TWFE.pvalues
    adjr2_TWFE = model_TWFE.rsquared_adj
    observations_TWFE = model_TWFE.nobs
    RMSE_TWFE = np.sqrt(model_TWFE.mse_resid)
    results_df_TWFE = pd.DataFrame({'Coef': params_TWFE, 'P-value': pvalues_TWFE, 'StdErr': std_err_TWFE, 'CI_90': CI_90_TWFE, 'CI_95': CI_95_TWFE, 'AdjR2': adjr2_TWFE, 'Observations': observations_TWFE, 'RMSE': RMSE_TWFE})
    residuals_TWFE = model_TWFE.resid
    hausman_test_statistic = np.dot(residuals_second_stage,residuals_TWFE)
    df = len(residuals_second_stage)
    Hausmanp_value = "{:.6f}".format(1 - stats.chi2.cdf(hausman_test_statistic, df))
    print('Endogeneity test:', Hausmanp_value)

    ## Simple OLS regression
    formula_simple = ''+list[i-1]+' ~ APtotal + MPtotal + income + HV + PopD + EM + RD + aging + EA + COUNT + BUMEAN + PM'
    model_simple = smf.ols(formula=formula_simple, data=AF).fit()
    params_simple = model_simple.params
    conf_int_simple = model_simple.conf_int(alpha=0.05)
    CI_95_simple = (conf_int_simple[1] - conf_int_simple[0])/2
    CI90_simple = model_simple.conf_int(alpha=0.1)
    CI_90_simple = (CI90_simple[1] - CI90_simple[0])/2
    std_err_simple = model_simple.bse
    pvalues_simple = model_simple.pvalues
    adjr2_simple = model_simple.rsquared_adj
    observations_simple = model_simple.nobs
    RMSE_simple = np.sqrt(model_simple.mse_resid)
    results_df_simple = pd.DataFrame({'Coef': params_simple, 'P-value': pvalues_simple, 'StdErr': std_err_simple, 'CI_90': CI_90_simple, 'CI_95': CI_95_simple,  'AdjR2': adjr2_simple, 'Observations': observations_simple, 'RMSE': RMSE_simple})

    ## Additional tests, refer to https://www.nature.com/articles/s41893-024-01281-2
    Z = AF[['Valuetem', 'ValuePRE']].values  ## Instrument variables
    X = AF[['income', 'HV', 'PopD', 'EM', 'RD', 'aging', 'EA', 'COUNT', 'BUMEAN', 'PM']].values ## Control variables
    Y = AF[list[i-1]].values.reshape(-1, 1) ## Dependent variable
    APtotal_hat = AF['APtotal_hat'].values.reshape(-1, 1) 
    MPtotal_hat = AF['MPtotal_hat'].values.reshape(-1, 1) 
    np.set_printoptions(precision=4, suppress=False)

    ## Kleibergen-Paap LM test
    model_weak_iv = sm.OLS(np.hstack([APtotal_hat, MPtotal_hat]), np.hstack([X, Z])).fit()
    residuals_weak_iv = model_weak_iv.resid
    LM_stat = np.sum(residuals_weak_iv ** 2)  
    LM_p_value = 1 - chi2.cdf(LM_stat, df=len(Z[0]))

    # Anderson-Rubin test
    model_AR = sm.OLS(Y, Z).fit()
    residuals_AR = model_AR.resid
    AR_stat = np.sum(residuals_AR ** 2) 
    AR_p_value = 1 - chi2.cdf(AR_stat, df=len(Z[0]))

    # Stock-Wright test
    model_SW = sm.OLS(np.hstack([APtotal_hat, MPtotal_hat]), Z).fit() 
    residuals_SW = model_SW.resid
    SW_stat = np.sum(residuals_SW ** 2) 
    SW_p_value = 1 - chi2.cdf(SW_stat, df=len(Z[0]))

    AAresults['KP F-stat'] = LM_stat
    AAresults['KP F-Pvalue'] = LM_p_value
    AAresults['AR Chi2-pvalue'] = AR_p_value
    AAresults['SW S-Pvalue'] = SW_p_value
    AAresults['Hausman-Pvalue'] = Hausmanp_value
    results_df = AAresults

    return results_df,results_df_TWFE,results_df_simple