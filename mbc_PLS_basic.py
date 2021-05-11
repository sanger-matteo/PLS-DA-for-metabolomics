#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ****************************************************************************

# Module contains functions to optimize and cross validate PLS

# ****************************************************************************

import numpy   as np
import pandas  as pd
import sklearn as skl

import matplotlib.pyplot as plt

from sys import stdout

# ****************************************************************************

# COMMON variables: local function variables (NOT global) that are very
# similar in use and content, thus we use similar names
#
#  - M_x     M_xx     M_xxx  : (DataFrame) data to fit; cols are predictors
#  - M_y     M_yy     M_yyy  : (DataFrame) target variable; cols are responses
#  - Ncomp   opt_Nc          : (int) number of latent variables to test


def PLS_CrossVal( M_x, M_y, Ncomp, cv_split):
    # Define PLS object with Ncomp components. Then, using scikit-learn, apply
    # PLSRegression to get cross-validated estimates for each input data point
    #
    # INPUT:
    #   - M_x      - M_y      - Ncomp
    #   - cv_split : (obj or int) cross-validation splitting strategy
    # OUTPUT:
    #   - M_y_cv   : (np.array) estimated probability for predictors

    funPLS = skl.cross_decomposition.PLSRegression( n_components = Ncomp )
    M_y_cv = skl.model_selection.cross_val_predict( funPLS,  M_x,  M_y,  cv= cv_split)

    return M_y_cv


def optimise_PLS_CrossVal( M_xx, M_yy, Ncomp, test_prop, plot_MSE):
    # Map the function "PLS_CrossVal" to test PLS with a range of number of
    # components, going from from 1 to maximum of "Ncomp". Option plot_MSE
    # determines whether to display the MSE and R2 plots and estimated optimal
    # number of components
    #
    # INPUT:
    #   - M_xx     - M_yy     - Ncomp
    #   - test_prop: (int) % size of the k-fold. From this value the function
    #                will create the k-fold groups for the test
    #   - plot_MSE : (bool) choose to plot MSE and R2 charts
    # OUTPUT:
    #   - mse      : mean square estimate
    #   - r2       : coefficient of estimation R-squared

    # Create selection function, for K-fold cross-validation with shuffle
    values = np.array(M_xx.index.values)
    k_fold = round(len(values) / round((len(values) * test_prop )) )
    kf_idx = skl.model_selection.KFold(n_splits=k_fold, shuffle=True, random_state=1)

    # Perform PLS regressions for full range of N components, using k-fold
    # sampling method kf_idx and calculate the scores
    mse = []
    r2  = []
    components = np.arange(1, Ncomp)

    for ii in components:
        M_yy_cv = PLS_CrossVal( M_xx, M_yy, ii, kf_idx)
        # Calculate scores for MSE and R2
        mse.append( skl.metrics.mean_squared_error( M_yy, M_yy_cv) )
        r2.append(  skl.metrics.r2_score( M_yy, M_yy_cv) )
        # Show computation progress and update status on the same line
        #progress = 100*(ii+1)/Ncomp
        #stdout.write("\r%d%% completed" % progress)
        #stdout.flush()

    if plot_MSE is True:
        print("\n")
        plot_metrics(mse, 'MSE', 'min')
        # plot_metrics(r2 , 'R2' , 'max')

    return mse, r2


def plot_metrics(vals, ylabel, objective):
    # Function to plot an np.array "vals" (either MSE or R2 from cross-valid.
    # of PLS) and display the min or max value with a cross
    fig, ax = plt.subplots(figsize=(8, 3))
    xticks = np.arange(1, len(vals)+1)

    plt.plot(xticks, np.array(vals), '-v', color='blue', mfc='blue')

    if objective=='min':
        idx = np.argmin(vals)
    else:
        idx = np.argmax(vals)
    plt.plot(xticks[idx], np.array(vals)[idx], 'P', ms=10, mfc='red')

    ax.set_label('Number of PLS components in regression')
    ax.set_ylabel(ylabel)
    ax.set_xticks( xticks )
    plt.grid(color = "grey", linestyle='--')
    plt.title('PLS cross-val')

    plt.show()


def PLS_fit_model(M_xxx, M_yyy, opt_Nc, response_variable_name):
    # Perform PLS regression and return the transformed training sample scores
    #
    # INPUT:
    #   - M_xxx    - M_yyy    - opt_Nc
    #   - response_variable_name : column name used from M_yyy table
    # OUTPUT:
    #   - DF_PLS_scores : (DataFrame) with the latent var. scores and the
    #                     M_yyy values in the last columns

    # Define PLS object with opt_Nc components. No scaling is necessary
    # (data was already scaled)
    plsr = skl.cross_decomposition.PLSRegression(n_components= opt_Nc , scale=False)

    # Build a PLS-DA model and save the values of opt_N_comp scores in DataFrame
    # together with the YY response table
    plsr.fit( M_xxx, M_yyy)
    scores = plsr.x_scores_
    DF_PLS_scores = pd.DataFrame(data = scores, columns= ["Score_LV_"+str(xx+1) for xx in np.arange(opt_Nc)] )
    DF_PLS_scores[ response_variable_name ] = np.array(M_yyy)

    return DF_PLS_scores


def CrossSelect_TwoTables( M_x, M_y, select_col, categories):
    # The input are two matricex with the same ordered list of observations:
    #  - M_x is the variable matrix
    #  - M_y is the predictors matrix
    # The function use M_y to find the index of the rows where the value(s)
    # "categories" in column "select_col" is True
    #
    # INPUT:
    #   - categories :  is a list of 1+ elements
    #   - select_col :  column name to use for selection
    # OUTPUT:
    #   - redX , redY:  equivalent DataFrames with reduced number of rows

    idx = []
    for cc in categories:
        idx = idx + M_y.index[M_y[select_col] == cc].tolist()
    idx.sort()
    redX = M_x.loc[idx,:]
    redY = M_y.loc[idx,:]

    return redX, redY

def StandardScale_FeatureTable( M_x, idxCol ):
    # Take the variables of predictors DataFrame (M_x) only: idxCol indicate
    # the column separating variables from categorical/informational data.
    # Then, use "StandardScaler" to scale-standardize the features onto a unit
    # scale with mean = 0 and variance = 1    ( z=(x-u)/s )
    X_vars = M_x.iloc[ :, idxCol: ].copy()
    X_desc = M_x.iloc[ :, :idxCol ].copy()
    temp_X = skl.preprocessing.StandardScaler().fit_transform( X_vars )

    # reformat it in proper pd.DataFrame
    rows = X_vars.index.values
    cols = X_vars.columns.values
    X_vars_scaled = pd.DataFrame( temp_X, index=rows , columns= cols)
    X_scaled = pd.concat( [ X_desc , X_vars_scaled], axis=1)

    return X_vars_scaled, X_scaled


# ****************************************************************************
