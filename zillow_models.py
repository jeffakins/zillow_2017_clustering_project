# This file will be used to store the code for the Zillow models
#------------------------------------------------------------------------------------------------------
#---------------Imports--------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import explained_variance_score
import matplotlib.pyplot as plt
import seaborn as sns

#----------------Function to split Train, Val, Test into X, y------------------------------------------

def zillow_xy(train, validate, test):
    '''Function to split train, validate, and test
    into their X and y components'''
    # create X & y version of train, where y is a series with just the target variable and X are all the features. 

    X_train = train.drop(columns=['logerror'])
    y_train = train.logerror

    X_validate = validate.drop(columns=['logerror'])
    y_validate = validate.logerror

    X_test = test.drop(columns=['logerror'])
    y_test = test.logerror

    return X_train, y_train, X_validate, y_validate, X_test, y_test

#------------------------------------------------------------------------------------------------------
#---------------Models---------------------------------------------------------------------------------

def baseline(y_train, y_validate):
    '''Function that will create a baseline prediction'''
    # Turn y into a dataframe
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    # Predict mean:
        #Train
    pred_mean_train = y_train.logerror.mean()
    y_train['pred_mean'] = pred_mean_train
        #Validate
    pred_mean_validate = y_validate.logerror.mean()
    y_validate['pred_mean'] = pred_mean_validate
    # Find the Root Mean Squared Error
        #Train
    rmse_train = mean_squared_error(y_train.logerror, y_train.pred_mean) ** .5
        #Validate
    rmse_validate = mean_squared_error(y_validate.logerror, y_validate.pred_mean) ** .5

    r_sqr = explained_variance_score(y_validate.logerror, y_validate.pred_mean)

    # print("Baseline RMSE using Mean\nTrain/In-Sample: ", round(rmse_train, 2), 
    #   "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))

    return rmse_train, rmse_validate, r_sqr


#---------------Linear Regression (OLS)---------------------------------------------------------------

def linear_regression(X_train, y_train, X_validate, y_validate):
    '''Function to perform a linear regression on our data'''
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    # Create
    lm = LinearRegression(normalize=True)
    # Fit
    lm.fit(X_train, y_train.logerror)
    # Predict
    y_train['pred_lm'] = lm.predict(X_train)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.logerror, y_train.pred_lm) ** (1/2)

    # predict validate
    y_validate['pred_lm'] = lm.predict(X_validate)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.logerror, y_validate.pred_lm) ** (1/2)

    r_sqr = explained_variance_score(y_validate.logerror, y_validate.pred_lm)

    # print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", round(rmse_train, 2), 
    #     "\nValidation/Out-of-Sample: ", round(rmse_validate, 2))
    return rmse_train, rmse_validate, r_sqr
    

#---------------LassoLars------------------------------------------------------------------------------

def lasso_lars(X_train, y_train, X_validate, y_validate):
    '''LassoLars regression for Zillow'''
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)

    # create the model object
    lars = LassoLars(alpha=1)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series!
    lars.fit(X_train, y_train.logerror)

    # predict train
    y_train['pred_lars'] = lars.predict(X_train)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.logerror, y_train.pred_lars) ** (1/2)

    # predict validate
    y_validate['pred_lars'] = lars.predict(X_validate)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.logerror, y_validate.pred_lars) ** (1/2)

    r_sqr = explained_variance_score(y_validate.logerror, y_validate.pred_lars)

    # print("RMSE for Lasso + Lars\nTraining/In-Sample: ", round(rmse_train, 2), 
    #     "\nValidation/Out-of-Sample: ", round(rmse_validate, 2))
    return rmse_train, rmse_validate, r_sqr


#---------------Tweedie Regressor (GLM)----------------------------------------------------------------

def tweedie_regressor(X_train, y_train, X_validate, y_validate, power=1, alpha=0):
    '''Tweedie regressor for zillow data
    power = 0: Normal Distribution
    power = 1: Poisson Distribution
    power = (1,2): Compound Distribution
    power = 2: Gamma Distribution
    power = 3: Inverse Gaussian Distribution'''

    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)

    # create the model object
    glm = TweedieRegressor(power=power, alpha=alpha)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    glm.fit(X_train, y_train.logerror)

    # predict train
    y_train['pred_glm'] = glm.predict(X_train)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.logerror, y_train.pred_glm) ** (1/2)

    # predict validate
    y_validate['pred_glm'] = glm.predict(X_validate)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.logerror, y_validate.pred_glm) ** (1/2)

    r_sqr = explained_variance_score(y_validate.logerror, y_validate.pred_glm)

    # print("RMSE for GLM using Tweedie, power=", power, " & alpha=", alpha, 
    #     "\nTraining/In-Sample: ", round(rmse_train, 2), 
    #     "\nValidation/Out-of-Sample: ", round(rmse_validate, 2))
    return rmse_train, rmse_validate, r_sqr


#---------------Polynomial Regression------------------------------------------------------------------

def polynomial_regression(X_train, y_train, X_validate, y_validate, degree=2):
    '''Function to model the zillow data using a polynomial linear regression'''
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)

    pf = PolynomialFeatures(degree)

        # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train)
        # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(X_validate)
        # X_test_degree2 = pf.transform(X_test)
        # create the model object
    lm2 = LinearRegression(normalize=True)
        # fit the model to our training data. We must specify the column in y_train, 
        # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2, y_train.logerror)
        # predict train
    y_train['pred_lm2'] = lm2.predict(X_train_degree2)
        # evaluate: rmse
    rmse_train = mean_squared_error(y_train.logerror, y_train.pred_lm2)**(1/2)
        # predict validate
    y_validate['pred_lm2'] = lm2.predict(X_validate_degree2)
        # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.logerror, y_validate.pred_lm2)**(1/2)
        # evaluate r^2
    r_sqr = explained_variance_score(y_validate.logerror, y_validate.pred_lm2)

    # print("RMSE for Polynomial Model, degrees=",degree, "\nTraining/In-Sample: ", rmse_train, 
    #     "\nValidation/Out-of-Sample: ", rmse_validate,
    #     "\nr^2 validate:", r_sqr)

    return rmse_train, rmse_validate, r_sqr

#------------------------------------------------------------------------------------------------------
#---------------Compare Regression Data Frame Function-------------------------------------------------

def make_metric_df(model_name, rmse_train, rmse_validate, r_sqr, metric_df):
    if metric_df.size ==0:
        metric_df = pd.DataFrame(data=[                         # Dictionary
            {
                'model': model_name,                            # Pass Model Name
                'RMSE Train': round(rmse_train, 6),             # Pass Train RMSE Information
                'RMSE Validate': round(rmse_validate, 6),       # Pass Validate RMSE Information
                'r^2 Validate': r_sqr                           # Pass r^2 info
            }])
        return metric_df
    else:
        return metric_df.append(
            {
                'model': model_name,                            # Pass Model Name
                'RMSE Train': round(rmse_train, 6),             # Pass Train RMSE Information
                'RMSE Validate': round(rmse_validate, 6),       # Pass Validate RMSE Information
                'r^2 Validate': r_sqr                           # Pass r^2 info
            }, ignore_index=True)

#---------------Compare Regression---------------------------------------------------------------------
def model_compare(X_train, y_train, X_validate, y_validate):
    '''Function to compare regression models'''
    # create the metric_df as a blank dataframe
    metric_df = pd.DataFrame()

    # Baseline
    rmse_train, rmse_validate, r_sqr = baseline(y_train, y_validate)
        # make our first entry into the metric_df with mean baseline
    metric_df = make_metric_df('Baseline (mean)',
                            rmse_train,
                            rmse_validate,
                            r_sqr,
                            metric_df)
    # OLS
    rmse_train, rmse_validate, r_sqr = linear_regression(X_train, y_train, X_validate, y_validate)
    metric_df = make_metric_df('OLS Regressor',
                            rmse_train,
                            rmse_validate,
                            r_sqr,
                            metric_df)

    # LassoLars
    rmse_train, rmse_validate, r_sqr = lasso_lars(X_train, y_train, X_validate, y_validate)
    metric_df = make_metric_df('Lasso: Alpha 1',
                            rmse_train,
                            rmse_validate,
                            r_sqr,
                            metric_df)

    # Tweedie:
        # power = 0: Normal Distribution
        # power = 1: Poisson Distribution
        # power = (1,2): Compound Distribution
        # power = 2: Gamma Distribution
        # power = 3: Inverse Gaussian Distribution

        # Tweedie regressor - Power 0
    rmse_train, rmse_validate, r_sqr = tweedie_regressor(X_train, y_train, X_validate, y_validate, power=0, alpha=0)
    metric_df = make_metric_df('GLM Normal',
                            rmse_train,
                            rmse_validate,
                            r_sqr,
                            metric_df)
    
        # Tweedie regressor - Power 1
    rmse_train, rmse_validate, r_sqr = tweedie_regressor(X_train, y_train, X_validate, y_validate, power=1, alpha=0)
    metric_df = make_metric_df('GLM Poisson',
                            rmse_train,
                            rmse_validate,
                            r_sqr,
                            metric_df)

        # Tweedie regressor - Power 2
    # y_validate_t2 = tweedie_regressor(X_train, y_train, X_validate, y_validate, power=2, alpha=0)
    # metric_df = make_metric_df(y_validate_t2.logerror,
    #             y_validate_t2.pred_glm,
    #             'GLM Gamma',
    #             metric_df)

    #     # Tweedie regressor - Power 3
    # y_validate_t3 = tweedie_regressor(X_train, y_train, X_validate, y_validate, power=3, alpha=0)
    # metric_df = make_metric_df(y_validate_t3.logerror,
    #             y_validate_t3.pred_glm,
    #             'GLM Inverse Gauss',
    #             metric_df)
    
    # Polynomial deg 2
    rmse_train, rmse_validate, r_sqr = polynomial_regression(X_train, y_train, X_validate, y_validate, degree=2)
    metric_df = make_metric_df('Polynomial 2deg',
                            rmse_train,
                            rmse_validate,
                            r_sqr,
                            metric_df)

    # Polynomial deg 3
    rmse_train, rmse_validate, r_sqr = polynomial_regression(X_train, y_train, X_validate, y_validate, degree=3)
    metric_df = make_metric_df('Polynomial 3deg',
                            rmse_train,
                            rmse_validate,
                            r_sqr,
                            metric_df)

    return metric_df


#------------------------------------------------------------------------------------------------------
#---------------plot Regression------------------------------------------------------------------------
def plot_baseline(y_train, y_validate):
    y_train, y_validate = baseline(y_train, y_validate)
    # plot to visualize actual vs predicted. 
    plt.hist(y_train.logerror, color='blue', alpha=.5, label="Samples")
    plt.hist(y_train.pred_mean, bins=1, color='red', alpha=.5, rwidth=1000, label="Mean")
    #plt.xlabel("Home Value")
    #plt.ylabel("not sure")
    plt.legend()
    plt.show()
    return None

