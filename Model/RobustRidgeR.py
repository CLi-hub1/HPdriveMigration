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


def SecondstageR(y,X,AF,Fvalue_AP,Fvalue_MP):
    # Second stage regression
    # Define robust weights
    def huber_weights(residuals, threshold):
        abs_residuals = np.abs(residuals)
        weights = np.where(abs_residuals <= threshold, 1, threshold / abs_residuals)
        return weights

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define alpha values and thresholds
    alphas = np.logspace(-2, 1, 50)
    thresholds = np.linspace(4, 20, 60)  ## Set to 40 for time lag effect, 20 for others

    best_alpha = None
    best_threshold = None
    best_score = float('inf')

    # Five-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    for alpha in alphas:
        for threshold in thresholds:
            cv_scores = []

            all_residuals = []
            all_weights = []

            for train_index, test_index in kf.split(X_scaled):
                X_train, X_test = X_scaled[train_index], X_scaled[test_index]
                y_train, y_test = y.values.flatten()[train_index], y.values.flatten()[test_index]

                ridge = Ridge(alpha=alpha, fit_intercept=True)
                ridge.fit(X_train, y_train)

                residuals_train = y_train - ridge.predict(X_train)
                weights_train = huber_weights(residuals_train, threshold)

                weighted_ridge = Ridge(alpha=alpha, fit_intercept=True)
                weighted_ridge.fit(X_train * np.sqrt(weights_train[:, np.newaxis]), y_train * np.sqrt(weights_train))

                predictions_test = weighted_ridge.predict(X_test)
                predictions_train = weighted_ridge.predict(X_train)

                all_residuals.extend(residuals_train)
                all_residuals.extend(y_test - predictions_test)

                all_weights.extend(weights_train)
                all_weights.extend(huber_weights(y_test - predictions_test, threshold))

            all_residuals = np.array(all_residuals)
            all_weights = np.array(all_weights)
            weighted_rmse = np.sqrt(mean_squared_error(all_residuals, np.zeros_like(all_residuals)))
            cv_scores.append(weighted_rmse)

            mean_cv_score = np.mean(cv_scores)
            results.append((alpha, threshold, mean_cv_score))

            if mean_cv_score < best_score:
                best_score = mean_cv_score
                best_alpha = alpha
                best_threshold = threshold

    results = np.array(results)

    final_ridge = Ridge(alpha=best_alpha, fit_intercept=True)
    final_ridge.fit(X_scaled, y)

    beta_standardized = final_ridge.coef_
    intercept_standardized = final_ridge.intercept_

    mu_X = scaler.mean_
    sigma_X = scaler.scale_

    beta_original = beta_standardized / sigma_X
    intercept_original = intercept_standardized - np.sum((mu_X * beta_standardized) / sigma_X)

    residuals_second_stage = y.values.flatten() - (X_scaled @ beta_standardized.flatten() + intercept_standardized)
    n, p = X.shape
    rss = (residuals_second_stage ** 2).sum()
    variance = rss / (n - p)
    best_weights = huber_weights(residuals_second_stage, best_threshold)

    ridge_cov_matrix = np.linalg.inv(X_scaled.T @ X_scaled + best_alpha * np.eye(p))
    var_beta_scaled = variance * ridge_cov_matrix.diagonal()

    var_beta_original = var_beta_scaled / (sigma_X ** 2)
    std_err_original = np.sqrt(var_beta_original)

    t_values_original = beta_original / std_err_original
    df = n - p
    p_values_original = 2 * (1 - t.cdf(np.abs(t_values_original), df=df))

    y_actual = y.values.flatten()
    y_pred = final_ridge.predict(X_scaled).flatten()
    weighted_mean = np.sum(best_weights * y_actual) / np.sum(best_weights)
    weighted_sst = np.sum(best_weights * (y_actual - weighted_mean) ** 2)
    weighted_ssr = np.sum(best_weights * (y_actual - y_pred) ** 2)
    weighted_r2 = 1 - (weighted_ssr / weighted_sst)

    MSE = np.mean((y_actual - y_pred) ** 2)
    RMSE = np.sqrt(MSE)

    AF['IV2SLSresid'] = residuals_second_stage
    AFresid = AF.copy()

    params = beta_original.flatten()
    std_err = std_err_original
    conf_int_90 = 1.645 * std_err
    conf_int_95 = 1.96 * std_err
    pvalues = p_values_original.flatten()
    Observation = n

    AAresults= pd.DataFrame({'Feature': X.columns,'Coef': params, 'P-value': pvalues, 'StdErr': std_err, 'CI90': conf_int_90, 'CI95': conf_int_95,'R2': weighted_r2, 'Observations': Observation})
    AAresults['RMSE'] = RMSE
    AAresults['alpha4ridge'] = best_alpha
    AAresults['threshold4robust'] = best_threshold
    AAresults['LRT_AP'] = Fvalue_AP
    AAresults['LRT_MP'] = Fvalue_MP

    return AAresults,residuals_second_stage