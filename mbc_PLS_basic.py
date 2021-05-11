#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ****************************************************************************

# Module contains functions to optimize and cross validate PLS

# ****************************************************************************

import numpy   as np
import pandas  as pd
import sklearn as skl
import random
import matplotlib.pyplot as plt

from sys import stdout

# ****************************************************************************

# COMMON variables: local function variables (NOT global) that are very
# similar in use and content, thus we use similar names
#
#  - M_x     M_xx     M_xxx  : (DataFrame) data to fit; cols are predictors
#  - M_y     M_yy     M_yyy  : (DataFrame) target variable; cols are responses
#  - Ncomp   Ncomp          : (int) number of latent variables to test

def PLS_ModelPredict( X_train, X_test, Y_train, Y_test, Ncomp):
    # The function creates a PLS model with a train data set. Then test it to
    # create prediction (Y_pred) that can be compared with Y_test to calculate
    # the error and accuracy

    plsr = skl.cross_decomposition.PLSRegression(n_components = Ncomp)
    plsr.fit( X_train, Y_train )
    Y_pred = plsr.predict( X_test )
    Y_pred = Y_pred.astype(float).flatten()

    # Calculate scores for MSE and R2
    mse = skl.metrics.mean_squared_error( Y_test, Y_pred)
    r2  = skl.metrics.r2_score( Y_test, Y_pred)

    return Y_pred, mse, r2



def optimise_PLS_CrossVal( M_X, M_Y, Test_Ncomp, uniqueID_Col,
                           RespVar, min_Resp_cat, max_Resp_cat, P_Threshold ,
                           outLoop, inLoop, outerPropT2T, innerPropT2T):
    # Perform a double cross validation of the PLR regression. The major steps
    # of the algorithm are the following:
    #
    # At each iteration of outer loop: Create training and test sets  and then
    # execute the inner loop using all dataset available (M_X and M_Y)
    #
    #   At each iteration of inner loop: Create training and test sets and then
    #   execute the inner loop using only the oX_train and oY_train
    #
    #       Create PLS models for full range of number of components
    #
    #   Store the average MSE from the ii-th inner loop. The smallest, leftmost
    #   MSE is the optimal number of components (opt_Ncomp) to use.
    #   Use the opt_Ncomp to predict values for oX_test and compared with the
    #   true values in oY_test. Calculate accuracy of modelling with opt_Ncomp
    #   at this oo-th iteration
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

    # Create list of components' range to iterate
    range_comp = np.arange(1, Test_Ncomp+1)

    # Initialize empty DataFrames to store results
    innerMSE = pd.DataFrame( 0 , index= [ "i"+str(xx+1) for xx in range(inLoop)] ,
                                 columns= [ "lv"+str(xx+1) for xx in range(Test_Ncomp)])

    outerMSE = pd.DataFrame( 0 , index= [ "o"+str(xx+1) for xx in range(outLoop)] ,
                                 columns= [ "lv"+str(xx+1) for xx in range(Test_Ncomp)])

    accuracy = pd.DataFrame( 0 , index = ["Suggested_N_comp", "Accuracy"] ,
                                 columns= [ "o"+str(xx+1) for xx in range(outLoop)] )

    for ooL in range(outLoop):
        # Outer Split - in training and test sets, based on their patient ID
        oX_train, oX_test ,oY_train, oY_test, _,_ = RandomSelect_P_TrainTest( M_X, M_Y, uniqueID_Col, outerPropT2T )

        for iiL in range(inLoop):
            # Inner Split - in training and test sets, based on their patient ID
            iX_train, iX_test ,iY_train, iY_test, _,_ = RandomSelect_P_TrainTest( oX_train, oY_train, uniqueID_Col, innerPropT2T )
            # Take only the response columne "RespVar"
            iY_train = iY_train.loc[:, RespVar ]
            iY_test  = iY_test.loc[ :, RespVar ]

            # Perform PLS on full range of number of components and save MSEs
            for cc in range_comp:

                iY_pred, mse, _ = PLS_ModelPredict( iX_train, iX_test,
                                                    iY_train, iY_test, cc)
                innerMSE.iloc[iiL, cc-1 ] = mse

            #print("Outer-loop: ", str(ooL+1), " ;    Inner-loop: ", str(iiL+1))
            #plot_metrics( innerMSE.mean(axis=0).tolist() , 'MSE', 'min')

        # Store the average MSE for each tested number of latent variables
        outerMSE.iloc[ooL, :] = innerMSE.mean(axis=0).tolist()
        # Find the "leftmost" minimum number of components, which is optimal to
        # use for modelling and prediction testing
        opt_Ncomp = np.argmin(outerMSE.iloc[ooL, :])
        # Take only the response column "RespVar"
        oY_train = oY_train.loc[:, RespVar ]
        oY_test  = oY_test.loc[ :, RespVar ]
        # Obtain the predicted scores for the model using the test sets
        oY_pred, mse, _ = PLS_ModelPredict( oX_train, oX_test,
                                            oY_train, oY_test, opt_Ncomp)

        # Use P_Threshold to transform the oY_pred in categorical values
        Y_pred_thres = []
        for yn in range(len(oY_pred)):
            if oY_pred[yn] > P_Threshold:
                Y_pred_thres.append(max_Resp_cat)
            elif oY_pred[yn] <= P_Threshold:
                Y_pred_thres.append(min_Resp_cat)
        # Assess accuracy in prediction and accuracy efficiency
        temp_accu = [ qq == ee for qq,ee in zip( oY_test.tolist(), Y_pred_thres) ]
        accuracy.iloc[0, ooL] = opt_Ncomp
        accuracy.iloc[1, ooL] = sum(temp_accu) / len(temp_accu)
        comparPred = pd.DataFrame( [oY_test.tolist(), oY_pred, Y_pred_thres, temp_accu] ).T
        comparPred.columns = ["oY_test", "oY_pred", "Y_pred_thres", "T-F"]

    return accuracy, comparPred




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



def PLS_fit_model(M_xxx, M_yyy, Ncomp, RespVar):
    # Perform PLS regression and return the transformed training sample scores
    #
    # INPUT:
    #   - M_xxx    - M_yyy    - Ncomp
    #   - RespVar : column name used from M_yyy table
    # OUTPUT:
    #   - DF_PLS_scores : (DataFrame) with the latent var. scores and the
    #                     M_yyy values in the last columns

    # Define PLS object with Ncomp components. No scaling is necessary
    # (data was already scaled)
    plsr = skl.cross_decomposition.PLSRegression(n_components= Ncomp , scale=False)

    # Build a PLS-DA model and save the values of opt_N_comp scores in DataFrame
    # together with the YY response table
    plsr.fit( M_xxx, M_yyy)
    scores = plsr.x_scores_
    DF_PLS_scores = pd.DataFrame( data = scores,
                                  columns = ["Score_LV_"+str(xx+1) for xx in np.arange(Ncomp)] )
    DF_PLS_scores[ RespVar ] = np.array(M_yyy)

    return DF_PLS_scores



def CrossSelect_TwoTables( M_x, M_y, RespVar, categories):
    # The input are two matricex with the same ordered list of observations:
    #  - M_x is the variable matrix
    #  - M_y is the predictors matrix
    # The function use M_y to find the index of the rows where the value(s)
    # "categories" in column "RespVar" is True
    #
    # INPUT:
    #   - categories :  is a list of 1+ elements
    #   - RespVar :  column name to use for selection
    # OUTPUT:
    #   - redX , redY:  equivalent DataFrames with reduced number of rows

    idx = []
    for cc in categories:
        idx = idx + M_y.index[M_y[RespVar] == cc].tolist()
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



def RandomSelect_P_TrainTest( M_x, M_y, uID_colname, proportion ):
    # Create training and test sets for M_x and M_y, where all samples from one
    # patient are either training or test. Function assume that the two matrices
    # (M_x and M_y) are same ordered list of observations

    # Create a the unique list of identifiers to use to create the two sets
    uID_List  = np.unique( M_y[ uID_colname ].values )

    test_pID  = random.sample( uID_List.tolist(), round(len(uID_List)*proportion) )
    test_idx  = np.where( M_y[uID_colname].isin( test_pID ))[0].tolist()

    train_pID = np.setdiff1d(uID_List, test_pID)
    train_idx = np.where( M_y[uID_colname].isin( train_pID ))[0].tolist()

    X_train = M_x.iloc[ train_idx, : ]
    X_test  = M_x.iloc[ test_idx,  : ]
    Y_train = M_y.iloc[ train_idx, : ]
    Y_test  = M_y.iloc[ test_idx,  : ]

    return X_train, X_test ,Y_train, Y_test, train_idx, test_idx


# ****************************************************************************
