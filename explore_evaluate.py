# This is a file that will be used to evalute regression models

#---------------------------Imports----------------------------------------------------------

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

#---------------------------Stats Functions--------------------------------------------------

def correlation_table(train):
    '''Function to show correlation'''
    sp_r = []
    sp_p = []
    pr_r = []
    pr_p = []
    # Turn this into a loop when you have time...
    #Bedrooms:
    r, p_value = spearmanr(train.bedrooms, train.tax_value)
    corr, p = pearsonr(train.bedrooms, train.tax_value)
    sp_r.append(r)
    sp_p.append(p_value)
    pr_r.append(corr)
    pr_p.append(p)
    # Bathrooms:
    r, p_value = spearmanr(train.bathrooms, train.tax_value)
    corr, p = pearsonr(train.bathrooms, train.tax_value)
    sp_r.append(r)
    sp_p.append(p_value)
    pr_r.append(corr)
    pr_p.append(p)
    # Square feet:
    r, p_value = spearmanr(train.sqft, train.tax_value)
    corr, p = pearsonr(train.sqft, train.tax_value)
    sp_r.append(r)
    sp_p.append(p_value)
    pr_r.append(corr)
    pr_p.append(p)
    # Average home price within each zipcode:
    r, p_value = spearmanr(train.zipcode_avg_price, train.tax_value)
    corr, p = pearsonr(train.zipcode_avg_price, train.tax_value)
    sp_r.append(r)
    sp_p.append(p_value)
    pr_r.append(corr)
    pr_p.append(p)
    # Build dataframe:
    df = pd.DataFrame(data={'spearmanr_corr': sp_r,
                        'spearmanr_p': sp_p,
                        'pearsonr_corr': pr_r,
                        'pearsonr_p': pr_p,}, index=['bedrooms', 'bathrooms', 'sqft', 'zipcode_avg_price'])
    return df

#---------------------------Evaluate Functions-----------------------------------------------

def residuals(actual, predicted):
    return actual - predicted

def sse(actual, predicted):
    return (residuals(actual, predicted) **2).sum()

def mse(actual, predicted):
    n = actual.shape[0]
    return sse(actual, predicted) / n

def rmse(actual, predicted):
    return math.sqrt(mse(actual, predicted))

def ess(actual, predicted):
    return ((predicted - actual.mean()) ** 2).sum()

def tss(actual):
    return ((actual - actual.mean()) ** 2).sum()

def r2_score(actual, predicted):
    return ess(actual, predicted) / tss(actual)

#---------------------------Model vs Baseline Comparison Functions-----------------------------

def regression_errors(actual, predicted):
    return pd.Series({
        'sse': sse(actual, predicted),
        'ess': ess(actual, predicted),
        'tss': tss(actual),
        'mse': mse(actual, predicted),
        'rmse': rmse(actual, predicted),
    })

def baseline_mean_errors(actual):
    predicted = actual.mean()
    return {
        'sse': sse(actual, predicted),
        'mse': mse(actual, predicted),
        'rmse': rmse(actual, predicted),
    }

def better_than_baseline(actual, predicted):
    rmse_baseline = rmse(actual, actual.mean())
    rmse_model = rmse(actual, predicted)
    return rmse_model < rmse_baseline

#---------------------------Plotting Functions-----------------------------------------------

def plot_residuals(actual, predicted):
    residuals = actual - predicted
    plt.figure(figsize = (9,6), facecolor="aliceblue")
    sns.set_theme(style="whitegrid")
    plt.hlines(0, actual.min(), actual.max(), ls=':')
    plt.scatter(actual, residuals)
    plt.ylabel('residual ($y - \hat{y}$)')
    plt.xlabel('actual value ($y$)')
    plt.title('Actual vs Residual')
    plt.show()

#---------------------------Feature Functions--------------------------------------------------
# RFE
def rfe_output(X_train, y_train):
    lm = LinearRegression()
    rfe = RFE(lm, 1)
    X_rfe = rfe.fit_transform(X_train,y_train) 
    lm.fit(X_rfe,y_train)
    var_ranks = rfe.ranking_
    var_names = X_train.columns.tolist()
    # Create DF list
    ranking = pd.DataFrame({'Var': var_names, 'Rank': var_ranks})
    return ranking
