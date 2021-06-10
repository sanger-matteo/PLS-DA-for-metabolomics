#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ****************************************************************************

# Module contains a collection of functions to:
# - maniulate metabolomics datasets and tables
# - perform, cross validate and optimize PLS
# - plot PLS results

# ****************************************************************************

import numpy   as np
import pandas  as pd

# ****************************************************************************

# COMMON variables: local function variables (NOT global) that are very
# similar in use and content, thus we use similar names
#
#  - M_X    : (DataFrame) data to fit; cols are predictors
#  - M_Y    : (DataFrame) target variable; cols are responses
#  - nLV  : (int) number of latent variables to test

def PLS_ModelPredict( X_train, X_test, Y_train, Y_test, nLV):
    # The function creates a PLS model with a train data set. Then test it to
    # create prediction (Y_pred) that can be compared with Y_test to calculate
    # the error, accuracy, specificity and selectivity
    from sklearn.cross_decomposition import PLSRegression
    from sklearn import metrics
    from sklearn import metrics

    plsr = PLSRegression(n_components = nLV)
    plsr.fit( X_train, Y_train )
    Y_pred = plsr.predict( X_test )
    Y_pred = Y_pred.astype(float).flatten()

    # Calculate scores for MSE and R2
    mse = metrics.mean_squared_error( Y_test, Y_pred)
    r2  = metrics.r2_score( Y_test, Y_pred)

    return Y_pred, mse, r2



def optimise_PLS_CrossVal( M_X, M_Y, Test_nLV, uniqueID_Col,
                           RespVar, Resp_class1, Resp_class2, P_Threshold ,
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
    #  - Test_nLV     : maximum number of latent var. to test for each model
    #  - uniqueID_Col : (str) M_Y column to create training and test data sets
    #  - RespVar      : (str) M_Y column to use as response variable PLS-DA
    #  - Resp_class1  : lowest value in RespVar
    #  - Resp_class2  : highest value in RespVar
    #  - P_Threshold  : threshold at 50%, between Resp_class1 and Resp_class2
    #  - outLoop      : how many outer loops to run
    #  - inLoop       : how many inner loops to run
    #  - outerPropT2T : proportion of data to use as train set in outer loop
    #  - innerPropT2T : proportion of data to use as train set in inner loop
    #
    # OUTPUT:
    #  - EvalResults  : (pd.df) Save results evaluating different parameters
    #         Best_nLV ------ Best number of latent var. found in oo-th cycle
    #         accuracy ------ tot_N True prediction / tot_N of predictions
    #         specificity --- FalsPos / (FalsPos+TrueNeg)    (i.e. test healthy)
    #         sensitivity --- TruePos / (TruePos+FalsNeg)    (i.e. test sick)
    #  - comparPred   :

    # Create a range of numbers to iterate euqal to Test_nLV
    range_nLV = np.arange(1, Test_nLV+1)

    # Initialize empty DataFrames to store results
    innerMSE = pd.DataFrame( 0 , index= [ "i"+str(xx+1) for xx in range(inLoop)] ,
                                 columns= [ "lv"+str(xx+1) for xx in range(Test_nLV)])

    outerMSE = pd.DataFrame( 0 , index= [ "o"+str(xx+1) for xx in range(outLoop)] ,
                                 columns= [ "lv"+str(xx+1) for xx in range(Test_nLV)])


    EvalResults = pd.DataFrame( 0 , index = ["Best_nLV", "Accuracy", "Specificity", "Selectivity"] ,
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
            for cc in range_nLV:
                iY_pred, mse, _ = PLS_ModelPredict( iX_train, iX_test,
                                                    iY_train, iY_test, cc)
                innerMSE.iloc[iiL, cc-1 ] = mse

        # Store average MSE for iiL-th inner loop (mean MSE at model with N
        # latent variables). Then, find the "leftmost" minimum MSE, which is
        # the optimal number of latent variables (opt_nLV)
        outerMSE.iloc[ooL, :] = innerMSE.mean(axis=0).tolist()
        #opt_nLV = np.argmin(outerMSE.iloc[ooL, :])
        from scipy.signal import argrelextrema
        y_line  = outerMSE.iloc[ooL, :].values
        all_min = argrelextrema( y_line , np.less)[0]
        # Handle case in which no local extrem point is found. Also, indexes
        # start at 0, so optimal_nLV must be adjusted by +1
        if all_min.size == 0 :            opt_nLV = len(y_line)
        else:                             opt_nLV = all_min[0] +1

        # Take only the response column "RespVar"
        oY_train = oY_train.loc[:, RespVar ]
        oY_test  = oY_test.loc[ :, RespVar ]
        # Obtain the predicted scores for the model using the test sets
        oY_pred, mse, _ = PLS_ModelPredict( oX_train, oX_test,
                                            oY_train, oY_test, opt_nLV)
        # Transform the oY_pred in categorical values using P_Threshold cutoff
        Y_pred_thres = []
        for yn in range(len(oY_pred)):
            if oY_pred[yn] > P_Threshold:
                Y_pred_thres.append(Resp_class2)
            elif oY_pred[yn] <= P_Threshold:
                Y_pred_thres.append(Resp_class1)
        # Calculate accuracy of modelling with opt_nLV at this ooL-th iteration
        temp_accu = [ qq == ee for qq,ee in zip( oY_test.tolist(), Y_pred_thres) ]
        EvalResults.iloc[0, ooL] = opt_nLV
        EvalResults.iloc[1, ooL] = sum(temp_accu) / len(temp_accu)
        # Find T/F Positive and T/F Negative to calc. Specificity, Sensitivity
        # True Negat.      True Posit.      False Negat.      False Posit.
        TN = 0;            TP = 0;          FN = 0;           FP = 0;
        for ii in range(len(oY_test)):
            if   oY_test.tolist()[ii]==0 and Y_pred_thres[ii]==0 :      TN +=1
            elif oY_test.tolist()[ii]==1 and Y_pred_thres[ii]==1 :      TP +=1
            elif oY_test.tolist()[ii]==1 and Y_pred_thres[ii]==0 :      FN +=1
            elif oY_test.tolist()[ii]==0 and Y_pred_thres[ii]==1 :      FP +=1

        # If numbers are low, then the values of FP/FN/TP/TN are zero. Then,
        # selectivity/specificity cannot be calculated (err: division by zero)
        if (FP+TN) == 0:     EvalResults.iloc[2, ooL] = 0
        else:                EvalResults.iloc[2, ooL] = FP / (FP+TN)     # Specificity
        if (TP+FN) == 0:     EvalResults.iloc[3, ooL] = 0
        else:                EvalResults.iloc[3, ooL] = TP / (TP+FN)     # Sensitivity

        comparPred = pd.DataFrame( [oY_test.tolist(), oY_pred, Y_pred_thres, temp_accu] ).T
        comparPred.columns = ["oY_test", "oY_pred", "Y_pred_thres", "T-F"]

        if display_plot:
            print("Outer-loop: ", str(ooL+1))
            plot_metrics( outerMSE.iloc[ooL, :].tolist() , 'MSE', 'min')

    return EvalResults, comparPred, outerMSE, innerMSE




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
        if isinstance(vals, list):        yValues = vals
        else:                             yValues = vals[:,ii]

        plt.plot(xticks, np.array(yValues), '-v', color='blue', mfc='blue')

        # Determine minimum and visualize it
        from scipy.signal import argrelextrema
        if objective=='min':
            all_extr = argrelextrema( np.array(yValues) , np.less)[0]
        else:
            all_extr = argrelextrema( np.array(yValues) , np.greater)[0]

        # Handle case in which no local extrem point is found. Also, indexes
        # start at 0, so optimal_nLV must be adjusted (by -1) to plot correctly
        if all_extr.size == 0 :       idx = len(yValues)-1
        else:                         idx = all_extr[0]

        plt.plot(xticks[idx], np.array(yValues)[idx], 'P', ms=10, mfc='red')

    # Adjust plot parameters
    ax.set_xlabel('LVs in regression')
    ax.set_ylabel( ylabel )
    ax.set_xticks( xticks )
    plt.grid(color = "grey", linestyle='--')
    plt.title('Error in PLS models')

    plt.show()



def PLS_fit_model(M_X, M_Y, nLV, RespVar):
    # Perform PLS regression and return the transformed training sample scores
    #
    # INPUT:
    #   - M_X    - M_Y    - nLV
    #   - RespVar   : column name used from M_Y table
    # OUTPUT:
    #   - DF_scores : (DataFrame) with the latent var. scores
    #   - DF_loads  : (DataFrame) with the latent var. loadings

    # Define PLS object with nLV components. No scaling is necessary
    # (data was already scaled)
    from sklearn.cross_decomposition import PLSRegression
    plsr = PLSRegression(n_components= nLV , scale=False)
    plsr.fit( M_X, M_Y)

    # Store the LV' scores and loadings in DataFrames
    DF_X_scores = pd.DataFrame( data = plsr.x_scores_ ,
                                columns = ["Score_X_LV_"+str(xx+1) for xx in np.arange(nLV)] )

    DF_Y_scores = pd.DataFrame( data = plsr.y_scores_ ,
                                columns = ["Score_Y_LV_"+str(xx+1) for xx in np.arange(nLV)] )

    # For X loadings change the row indexes to the measured variables names
    DF_X_loads = pd.DataFrame( data = plsr.x_loadings_ ,
                               columns = ["Load_X_LV_"+str(xx+1)  for xx in np.arange(nLV)] )
    DF_X_loads = DF_X_loads.set_axis( M_X.columns.values.tolist() , axis='index')

    DF_Y_loads = pd.DataFrame( data = plsr.y_loadings_ ,
                               columns = ["Load_Y_LV_"+str(xx+1)  for xx in np.arange(nLV)] )

    return DF_X_scores, DF_Y_scores, DF_X_loads, DF_Y_loads



def CrossSelect_TwoTables( M_X, M_Y, RespVar, categories, transf_01):
    # The input are two matricex with the same ordered list of observations:
    #  - M_X is the variable matrix
    #  - M_Y is the predictors matrix
    # The function use M_Y to find the index of the rows where the value(s)
    # "categories" in column "RespVar" is True
    #
    # INPUT:
    #   - RespVar    : column name to use for selection
    #   - categories : is a list of 1+ elements
    #   - transf_01  : if there are only two categories, we can transform them
    #                  into simple binary 0-1 choise (0=min , 1=max)
    #
    # OUTPUT:
    #   - redX , redY:  equivalent DataFrames with reduced number of rows

    idx = []
    for cc in categories:
        idx = idx + M_Y.index[M_Y[RespVar] == cc].tolist()
    idx.sort()
    redX = M_X.loc[idx,:]
    redY = M_Y.loc[idx,:]

    #  if there are only two categories, convert into binary 0-1 data
    if transf_01 == True:
        if len(categories) == 2:
            temp_aa = np.zeros(len(redY[RespVar]), dtype=int)
            temp_aa[redY[RespVar]== categories[-1]] = 1
            redY[RespVar] = temp_aa
        else :
            print(" --- WARNING --- \n Cannot transform response to binary when there are more than 2 categories")

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
