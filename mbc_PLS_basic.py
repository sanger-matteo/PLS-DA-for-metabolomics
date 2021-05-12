#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ****************************************************************************

# Module contains functions to optimize and cross validate PLS

# ****************************************************************************

import numpy   as np
import pandas  as pd

# ****************************************************************************

# COMMON variables: local function variables (NOT global) that are very
# similar in use and content, thus we use similar names
#
#  - M_X    : (DataFrame) data to fit; cols are predictors
#  - M_Y    : (DataFrame) target variable; cols are responses
#  - Ncomp  : (int) number of latent variables to test

def PLS_ModelPredict( X_train, X_test, Y_train, Y_test, Ncomp):
    # The function creates a PLS model with a train data set. Then test it to
    # create prediction (Y_pred) that can be compared with Y_test to calculate
    # the error and accuracy
    from sklearn.cross_decomposition import PLSRegression
    from sklearn import metrics
    from sklearn import metrics

    plsr = PLSRegression(n_components = Ncomp)
    plsr.fit( X_train, Y_train )
    Y_pred = plsr.predict( X_test )
    Y_pred = Y_pred.astype(float).flatten()

    # Calculate scores for MSE and R2
    mse = metrics.mean_squared_error( Y_test, Y_pred)
    r2  = metrics.r2_score( Y_test, Y_pred)

    return Y_pred, mse, r2



def optimise_PLS_CrossVal( M_X, M_Y, Test_Ncomp, uniqueID_Col,
                           RespVar, min_Resp_cat, max_Resp_cat, P_Threshold ,
                           outLoop, inLoop, outerPropT2T, innerPropT2T,
                           display_plot):
    # Perform a double cross validation of the PLR regression. The major steps
    # of the algorithm are the following:
    #
    # FUNC used:
    # --> RandomSelect_P_TrainTest
    # --> PLS_ModelPredict
    #
    # INPUT:
    #  - M_X          : (pd.df) predictors, num variables only; already scaled
    #  - M_Y          : (pd.df) responses,
    #  - Test_Ncomp   : maximum number of components to test for each model
    #  - uniqueID_Col : (str) M_Y column to create training and test data sets
    #  - RespVar      : (str) M_Y column to use as response variable PLS-DA
    #  - min_Resp_cat : lowest value in RespVar
    #  - min_Resp_cat : highest value in RespVar
    #  - P_Threshold  : threshold at 50%, between min_Resp_cat and max_Resp_cat
    #  - outLoop      : how many outer loops to run
    #  - inLoop       : how many inner loops to run
    #  - outerPropT2T : proportion of data to use as train set in outer loop
    #  - innerPropT2T : proportion of data to use as train set in inner loop
    #
    # OUTPUT:
    #  - accuracy     : (pd.df) 1st row - suggested best number of components
    #                           for each outer loop iteration tested
    #                           2ns row - accuracy calculated in each iteration
    #                           of outer loop
    #  - comparPred   :

    # Create a range of numbers to iterate euqal to Test_Ncomp
    range_comp = np.arange(1, Test_Ncomp+1)

    # Initialize empty DataFrames to store results
    innerMSE = pd.DataFrame( 0 , index= [ "i"+str(xx+1) for xx in range(inLoop)] ,
                                 columns= [ "lv"+str(xx+1) for xx in range(Test_Ncomp)])

    outerMSE = pd.DataFrame( 0 , index= [ "o"+str(xx+1) for xx in range(outLoop)] ,
                                 columns= [ "lv"+str(xx+1) for xx in range(Test_Ncomp)])

    accuracy = pd.DataFrame( 0 , index = ["Suggested_N_comp", "Accuracy"] ,
                                 columns= [ "o"+str(xx+1) for xx in range(outLoop)] )

    # At each iteration of either inner or outer loop create train and test sets
    for ooL in range(outLoop):
        # Split into "outer" training and test sets, based on uniqueID_Col
        oX_train, oX_test ,oY_train, oY_test, _,_ = RandomSelect_P_TrainTest( M_X, M_Y, uniqueID_Col, outerPropT2T )

        for iiL in range(inLoop):
            # Split into "inner" training and test sets, based on uniqueID_Col
            iX_train, iX_test ,iY_train, iY_test, _,_ = RandomSelect_P_TrainTest( oX_train, oY_train, uniqueID_Col, innerPropT2T )
            # Take only the response column "RespVar"
            iY_train = iY_train.loc[:, RespVar ]
            iY_test  = iY_test.loc[ :, RespVar ]

            # Create PLS models for full range of number of components
            for cc in range_comp:
                iY_pred, mse, _ = PLS_ModelPredict( iX_train, iX_test,
                                                    iY_train, iY_test, cc)
                innerMSE.iloc[iiL, cc-1 ] = mse

            # --> Checkpoint Plot 1

        # Store average MSE for iiL-th inner loop, average on each modeled
        # number of components. Then, find the "leftmost" minimum MSE, which is
        # the optimal number of components (opt_Ncomp)
        outerMSE.iloc[ooL, :] = innerMSE.mean(axis=0).tolist()
        opt_Ncomp = np.argmin(outerMSE.iloc[ooL, :])
        # Take only the response column "RespVar"
        oY_train = oY_train.loc[:, RespVar ]
        oY_test  = oY_test.loc[ :, RespVar ]
        # Obtain the predicted scores for the model using the test sets
        oY_pred, mse, _ = PLS_ModelPredict( oX_train, oX_test,
                                            oY_train, oY_test, opt_Ncomp)
        # Transform the oY_pred in categorical values using P_Threshold cutoff
        Y_pred_thres = []
        for yn in range(len(oY_pred)):
            if oY_pred[yn] > P_Threshold:
                Y_pred_thres.append(max_Resp_cat)
            elif oY_pred[yn] <= P_Threshold:
                Y_pred_thres.append(min_Resp_cat)
        #  Calculate accuracy of modelling with opt_Ncomp at this ooL-th iteration
        temp_accu = [ qq == ee for qq,ee in zip( oY_test.tolist(), Y_pred_thres) ]
        accuracy.iloc[0, ooL] = opt_Ncomp
        accuracy.iloc[1, ooL] = sum(temp_accu) / len(temp_accu)
        comparPred = pd.DataFrame( [oY_test.tolist(), oY_pred, Y_pred_thres, temp_accu] ).T
        comparPred.columns = ["oY_test", "oY_pred", "Y_pred_thres", "T-F"]

        # specificity is a ratio (N correct for class 1 / total number of class 1) - Sick
        # selectivity is a ratio (N correct for class 2 / total number of class 2) - Healthy

        # --> Checkpoint Plot 2
        if display_plot:
            print("Outer-loop: ", str(ooL+1))
            plot_metrics( outerMSE.iloc[ooL, :].tolist() , 'MSE', 'min')

    return accuracy, comparPred, outerMSE, innerMSE




def plot_metrics(vals, ylabel, objective):
    # Function to plot an np.array "vals" (either MSE or R2 from cross-valid.
    # of PLS) and display the min or max value with a cross
    # vals      = can be a np.array or a single list. If it is np.array, each
    #             column is an individual line to plot on sequential x positions
    # ylabel    = (str) the name of the plotted y-values (e.g. MSE or R2)
    # objective = (str) highlight minimum or maximum data point ("min" or "max")

    import matplotlib.pyplot as plt

    if isinstance(vals, list):
        x_Npoints = len(vals)
        y_Nlines  = 1
    else:
        x_Npoints = vals.shape[0]
        y_Nlines  = vals.shape[1]

    fig, ax = plt.subplots(figsize=(8, 3))
    xticks = np.arange( 1 , x_Npoints+1)

    # Plot all the line in the dataset vals
    for ii in range(y_Nlines):
        # Set the y_value to plot ii-th line
        if isinstance(vals, list):
            yValues = vals
        else:
            yValues = vals[:,ii]

        plt.plot(xticks, np.array(yValues), '-v', color='blue', mfc='blue')

        # Determine minimum and visualize it
        if objective=='min':
            idx = np.argmin(yValues)
        else:
            idx = np.argmax(yValues)
        plt.plot(xticks[idx], np.array(yValues)[idx], 'P', ms=10, mfc='red')

    # Adjust plot parameters
    ax.set_xlabel('LVs in regression')
    ax.set_ylabel( ylabel )
    ax.set_xticks( xticks )
    plt.grid(color = "grey", linestyle='--')
    plt.title('Error in PLS models')

    plt.show()



def PLS_fit_model(M_X, M_Y, Ncomp, RespVar):
    # Perform PLS regression and return the transformed training sample scores
    #
    # INPUT:
    #   - M_X    - M_Y    - Ncomp
    #   - RespVar : column name used from M_Y table
    # OUTPUT:
    #   - DF_PLS_scores : (DataFrame) with the latent var. scores and the
    #                     M_Y values in the last columns

    # Define PLS object with Ncomp components. No scaling is necessary
    # (data was already scaled)
    from sklearn.cross_decomposition import PLSRegression
    plsr = PLSRegression(n_components= Ncomp , scale=False)

    # Build a PLS-DA model and save the values of opt_N_comp scores in DataFrame
    # together with the YY response table
    plsr.fit( M_X, M_Y)
    scores = plsr.x_scores_
    DF_PLS_scores = pd.DataFrame( data = scores,
                                  columns = ["Score_LV_"+str(xx+1) for xx in np.arange(Ncomp)] )
    DF_PLS_scores[ RespVar ] = np.array(M_Y)

    return DF_PLS_scores



def CrossSelect_TwoTables( M_X, M_Y, RespVar, categories):
    # The input are two matricex with the same ordered list of observations:
    #  - M_X is the variable matrix
    #  - M_Y is the predictors matrix
    # The function use M_Y to find the index of the rows where the value(s)
    # "categories" in column "RespVar" is True
    #
    # INPUT:
    #   - categories :  is a list of 1+ elements
    #   - RespVar :  column name to use for selection
    # OUTPUT:
    #   - redX , redY:  equivalent DataFrames with reduced number of rows

    idx = []
    for cc in categories:
        idx = idx + M_Y.index[M_Y[RespVar] == cc].tolist()
    idx.sort()
    redX = M_X.loc[idx,:]
    redY = M_Y.loc[idx,:]

    return redX, redY



def StandardScale_FeatureTable( M_X, idxCol ):
    # Take the variables of predictors DataFrame (M_X) only: idxCol indicate
    # the column separating variables from categorical/informational data.
    # Then, use "StandardScaler" to scale-standardize the features onto a unit
    # scale with mean = 0 and variance = 1    ( z=(x-u)/s )
    X_vars = M_X.iloc[ :, idxCol: ].copy()
    X_desc = M_X.iloc[ :, :idxCol ].copy()

    from sklearn.preprocessing import StandardScaler
    temp_X = StandardScaler().fit_transform( X_vars )

    # Reformat into proper pd.DataFrame
    rows = X_vars.index.values
    cols = X_vars.columns.values
    X_vars_scaled = pd.DataFrame( temp_X, index=rows , columns= cols)
    X_scaled = pd.concat( [ X_desc , X_vars_scaled], axis=1)

    return X_vars_scaled, X_scaled



def RandomSelect_P_TrainTest( M_X, M_Y, uID_colname, proportion ):
    # Create training and test sets for M_X and M_Y, where all samples from one
    # patient are either training or test. Function assume that the two matrices
    # (M_X and M_Y) are same ordered list of observations
    import random

    # Create a the unique list of identifiers to use to create the two sets
    uID_List  = np.unique( M_Y[ uID_colname ].values )
    random.shuffle(uID_List)

    # Find IDs and row position in dataframe
    test_pID  = random.sample( uID_List.tolist(), round(len(uID_List)*proportion) )
    test_idx  = np.where( M_Y[uID_colname].isin( test_pID ))[0].tolist()

    train_pID = np.setdiff1d(uID_List, test_pID)
    train_idx = np.where( M_Y[uID_colname].isin( train_pID ))[0].tolist()

    X_train = M_X.iloc[ train_idx, : ]
    X_test  = M_X.iloc[ test_idx,  : ]
    Y_train = M_Y.iloc[ train_idx, : ]
    Y_test  = M_Y.iloc[ test_idx,  : ]

    return X_train, X_test ,Y_train, Y_test, train_idx, test_idx


# ****************************************************************************
