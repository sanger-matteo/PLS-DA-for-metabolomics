#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ****************************************************************************

# Module contains a collection of functions to:
# - manipulate metabolomics datasets and tables
# - perform, cross validate and optimize PLS
# - plot PLS results

# ****************************************************************************

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

# ****************************************************************************

# COMMON variables: local function variables (NOT global) that are very
# similar in use and content, thus we use similar names
#
# - M_X        : (DataFrame) data to fit; cols are predictors
# - M_Y        : (DataFrame) target variable; cols are responses
# - nLV        : (int) number of latent variables to test
# - RespVar    : (str) M_Y column to use as response variable PLS-DA
# - RespVal_0  : lowest and highest categories in Y response variable
# - RespVal_1
# - P50_thrVal : threshold value (50%) to use to assign predictions to either
#                RespVal_0 or RespVal_1


def PLS_ModelPredict( X_train, X_test, Y_train, Y_test, nLV, P50_thrVal, RespVal_0, RespVal_1):
    # The function creates a PLS model with a train data set. Then use test to
    # create prediction (Y_pred). Finally performance metrics, MSE and accuracy
    from sklearn.cross_decomposition import PLSRegression
    from sklearn import metrics
    from sklearn import metrics

    plsr = PLSRegression(n_components = nLV)
    plsr.fit( X_train, Y_train )
    Y_pred = plsr.predict( X_test )
    Y_pred = Y_pred.astype(float).flatten()
    # Calculate performance metrics
    _, temp_EvalRes = binary_classification( Y_test.tolist(), Y_pred, P50_thrVal
                                             , RespVal_0, RespVal_1 )
    accu = temp_EvalRes.iloc[0,0]
    mse  = metrics.mean_squared_error( Y_test, Y_pred)

    perfo_metrics = [ accu , mse ]
    return Y_pred, perfo_metrics



def optimise_PLS_CrossVal( M_X, M_Y, Test_nLV, uniqueID_Col,
                           RespVar, RespVal_0, RespVal_1, P50_thrVal ,
                           outLoop, inLoop, outerPropT2T, innerPropT2T,
                           display_plot):
    # Execute a double cross validation for PLR regression.
    #
    # FUNC used:   > RandomSelect_P_TrainTest    > PLS_ModelPredict
    #
    # INPUT:
    #  - M_X          : (pd.df) predictors, num variables only; already scaled
    #  - M_Y          : (pd.df) responses,
    #  - Test_nLV     : maximum number of latent var. to test for each model
    #  - uniqueID_Col : (str) M_Y column to create training and test data sets
    #  - outLoop      : how many outer loops to run
    #  - inLoop       : how many inner loops to run
    #  - outerPropT2T : proportion of data to use as train set in outer loop
    #  - innerPropT2T : proportion of data to use as train set in inner loop
    #
    # OUTPUT:
    #  - PerfoMetric  : (pd.df) Performance metrics:
    #                 (0) accuracy (1) specificity (2) sensitivity (3) Best_nLV
    #  - comparPred   : test, train and prediction response variable used for
    #                   last (!) outer loop
    #                   (we need to implement MultiIndex to keep them all)

    # Initialize empty DataFrames to store results
    totalCAL = pd.DataFrame( [] )
    innerCAL = pd.DataFrame( 0, index   = [ "i"+str(xx+1) for xx in range(inLoop)] ,
                                columns = [ "lv"+str(xx+1) for xx in range(Test_nLV)])
    outerCAL = pd.DataFrame( 0, index   = [ "o"+str(xx+1) for xx in range(outLoop)] ,
                                columns = [ "lv"+str(xx+1) for xx in range(Test_nLV)])
    PerfoMetric = pd.DataFrame(0, index = ["Accuracy", "Specificity", "Sensitivity", "Best_nLV"] ,
                                columns = [ "o"+str(xx+1) for xx in range(outLoop)] )
    range_nLV = np.arange(1, Test_nLV+1)     # range of LV to iterate through

    # At each iteration of either inner or outer loop create train and test sets
    for ooL in range(outLoop):
        # Display inline text for the progression of cross-validation
        print("Iteration:   : "+str(ooL+1)+" of "+str(outLoop), sep=' ', end='\r', flush=True)

        # Split into "outer" training and test sets, based on uniqueID_Col
        oX_train, oX_test ,oY_train, oY_test, _,_ = RandomSelect_P_TrainTest( M_X, M_Y, uniqueID_Col, outerPropT2T )
        '''
        for iiL in range(inLoop):
            # Split outer Train into "inner" training and test sets, based on column uniqueID_Col
            iX_train, iX_test ,iY_train, iY_test, _,_ = RandomSelect_P_TrainTest( oX_train, oY_train, uniqueID_Col, innerPropT2T )
            # Take only the response column "RespVar"
            iY_train = iY_train.loc[:, RespVar ]
            iY_test  = iY_test.loc[ :, RespVar ]

            for nnLV in range_nLV:         # Create PLS models all range of number of LV
                iY_pred, tempMetric = PLS_ModelPredict( iX_train, iX_test, iY_train, iY_test,
                                                        nnLV, P50_thrVal , RespVal_0, RespVal_1)
                ACCU = tempMetric[0]
                innerCAL.iloc[ iiL, nnLV-1 ] = ACCU
        '''
        for nnLV in range_nLV:         # Create PLS models all range of number of LV
            for iiL in range(inLoop):
                # Split outer Train into "inner" training and test sets, based on column uniqueID_Col
                iX_train, iX_test ,iY_train, iY_test, _,_ = RandomSelect_P_TrainTest( oX_train, oY_train, uniqueID_Col, innerPropT2T )
                # Take only the response column "RespVar"
                iY_train = iY_train.loc[:, RespVar ]
                iY_test  = iY_test.loc[ :, RespVar ]

                iY_pred, tempMetric = PLS_ModelPredict( iX_train, iX_test, iY_train, iY_test,
                                                        nnLV, P50_thrVal , RespVal_0, RespVal_1)
                ACCU = tempMetric[0]
                innerCAL.iloc[ iiL, nnLV-1 ] = ACCU

        # totalCAL stores all and every (inner loop) model created
        # innerCAL.index = [ "i_"+str(ooL+1)+"_"+str(xx+1) for xx in range(innerCAL.shape[0])]
        totalCAL = pd.concat( [totalCAL, innerCAL], axis=0, join="outer", ignore_index=True)
        # OuterCal stores mean ACCU modeled with each N number latent variables.
        outerCAL.iloc[ooL, :] = innerCAL.mean(axis=0).tolist()
        # Find "leftmost" max ACCU, which is the optimal number of LV (opt_nLV)
        opt_nLV = np.argmax(outerCAL.iloc[ooL, :]) +1

        # Take only the response column "RespVar"
        oY_train = oY_train.loc[:, RespVar ]
        oY_test  = oY_test.loc[ :, RespVar ]
        # Obtain the predicted scores for the model using the test sets
        oY_pred, _ = PLS_ModelPredict( oX_train, oX_test, oY_train, oY_test,
                                       opt_nLV, P50_thrVal , RespVal_0, RespVal_1)
        comparPred, temp_EvalRes = binary_classification( oY_test.tolist(), oY_pred,
                                            P50_thrVal , RespVal_0, RespVal_1 )

        PerfoMetric.iloc[:, ooL] = temp_EvalRes.values
        PerfoMetric.iloc[3, ooL] = opt_nLV

    # Take the most common opt_nLV as overall optimal number of LV
    optimal_nLV = PerfoMetric.iloc[-1,:].mode()[0]

    if display_plot:
        bins = range(Test_nLV)
        data = PerfoMetric.iloc[-1,:]
        def bins_labels(bins, **kwargs):
            bin_w = 1
            plt.xticks(np.arange(min(bins)+bin_w/2, max(bins), bin_w), bins, **kwargs)
            plt.xlim( 0.5, bins[-1]+0.5)
            plt.ylim( 0, 1)
        fig = plt.figure(figsize = (4,3))
        ax  = fig.add_subplot(1,1,1)
        plt.hist( data , bins = bins, weights= np.zeros_like(data) + 1./data.size,
                  color = 'lightblue', edgecolor = 'gray')
        bins_labels(bins, fontsize=9)

    return PerfoMetric, comparPred, outerCAL, totalCAL, optimal_nLV



def binary_classification( Y_measured, Y_predicted, P50_thrVal, RespVal_0, RespVal_1 ):
    # Confront a Series of measured clinical events/conditions (Y_measured),
    # against a Series of predicted events/conditions (Y_predicted)
    # OUTPUT:
    # - comparPred    = DF with test/train/prediction response variable used
    # - perfo_metrics = Calculated performance metrics of the model
    #    0 - accuracy ------ tot_N True prediction / tot_N of predictions
    #    1 - specificity --- TrueNeg / (FalsPos+TrueNeg)    (i.e. test healthy)
    #    2 - sensitivity --- TruePos / (TruePos+FalsNeg)    (i.e. test sick)
    #    3 - Best_nLV ------ Best number of latent var. found in oo-th cycle

    perfo_metrics = pd.DataFrame( 0 , index = ["Accuracy", "Specificity", "Sensitivity", "Best_nLV"] ,
                                    columns= [ "temp" for xx in range(1)] )
    # Transform the Y_predicted in categorical values using P50_thrVal cutoff
    threshold_Y = []
    for yn in range(len(Y_predicted)):
        if Y_predicted[yn] > P50_thrVal:
            threshold_Y.append( RespVal_1 )
        elif Y_predicted[yn] <= P50_thrVal:
            threshold_Y.append( RespVal_0 )

    # Find T/F Positive and T/F Negative to calculate Specificity + Sensitivity
    # True Negat.      True Posit.      False Negat.      False Posit.
    TN = 0;            TP = 0;          FN = 0;           FP = 0;
    for ii in range(len(Y_measured)):
        if   Y_measured[ii]==0 and threshold_Y[ii]==0 :      TN +=1
        elif Y_measured[ii]==1 and threshold_Y[ii]==1 :      TP +=1
        elif Y_measured[ii]==1 and threshold_Y[ii]==0 :      FN +=1
        elif Y_measured[ii]==0 and threshold_Y[ii]==1 :      FP +=1

    # Now calculate performance metrics: ACCU, SPEC, SELE
    TF_predict = [ qq == ee for qq,ee in zip( Y_measured, threshold_Y) ]
    comparPred = pd.DataFrame( [Y_measured, Y_predicted, threshold_Y, TF_predict] ).T
    comparPred.columns = ["Y_measure", "Y_predict", "Y_thres", "T-F"]

    # If numbers are low, then the values of FP/FN/TP/TN are zero. Then,
    # specificity/Sensitivity cannot be calculated (err: division by zero)
    perfo_metrics.iloc[0] = (TP+TN) / (TP+TN+FP+FN)
    if (FP+TN) == 0:     perfo_metrics.iloc[1] = 0
    else:                perfo_metrics.iloc[1] = TN / (FP+TN)     # Specificity
    if (TP+FN) == 0:     perfo_metrics.iloc[2] = 0
    else:                perfo_metrics.iloc[2] = TP / (FN+TP)     # Sensitivity

    return comparPred, perfo_metrics



def plot_metrics(vals, ylabel, objective, do_mean):
    # Function to plot an np.array "vals" (either ACCU. MSE  or R2 from
    # validation of PLS) and display the min or max value with a cross
    # vals      = must be an np.array, each column is an individual line
    #             to plot on sequential x positions
    # ylabel    = (str) the name of the plotted y-values (e.g. MSE or R2)
    # objective = (str) highlight minimum or maximum data point ("min" or "max")
    # do_mean   = (T/F) make a mean of the table_data and plot with error-bar

    import matplotlib.pyplot as plt

    x_Npoints = vals.shape[0]
    y_Nlines  = vals.shape[1]

    fig, ax = plt.subplots(figsize=(8, 3))
    xticks  = np.arange( 1 , x_Npoints+1)

    # Plot all the line in the dataset vals
    for ii in range(y_Nlines):
        # Set the y_value to plot ii-th line
        if do_mean:
            yValues     = vals.mean(axis=1)
            std_yValues = vals.std(axis=1)
            plt.title('Mean Error in PLS models')
            plt.plot( xticks, np.array(yValues), '--', color='blue', mfc='blue')

            plt.fill_between(xticks, np.array(yValues) - std_yValues,
                                     np.array(yValues) + std_yValues,
                                     color='lightgray', alpha=0.05)
        else:
            yValues = vals[:,ii]
            plt.title('Error in PLS models')
            plt.plot( xticks, np.array(yValues), '--', color='blue', mfc='blue')

        # Determine minimum and visualize it
        if objective == 'min':
            idx = np.argmin( np.array(yValues))
        else:
            idx = np.argmax( np.array(yValues))

        plt.plot(xticks[idx], np.array(yValues)[idx], 'P', ms=10, mfc='red')

    # Adjust plot parameters
    ax.set_xlabel('LVs in regression')
    ax.set_ylabel( ylabel )
    ax.set_xticks( xticks )
    plt.grid(color = "grey", linestyle='--')
    plt.show()



def PLS_fit_model(M_X, M_Y, nLV, RespVar):
    # Perform PLS regression and return the transformed training sample scores
    #
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
    # (M_X and M_Y) are same ordered list of observations.

    import random
    # Generate unique list of identifiers to use to create the two sets
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
