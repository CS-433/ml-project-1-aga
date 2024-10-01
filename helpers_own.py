"""Some own helper functions for project 1."""
import csv
import numpy as np
import os


def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
    return np.exp(t) / (1 + np.exp(t))

def calculate_loss(y, tx, w, lambda1=0, lambda2=0):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred)) + lambda2 * np.squeeze(w.T.dot(w)) + lambda1 * np.sum(np.abs(w))
    return np.squeeze(-loss).item() * (1 / y.shape[0])

def calculate_gradient(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)
    """
    pred = sigmoid(tx.dot(w))
    ret = (1/y.shape[0])*tx.T.dot(pred-y)
    return ret

def learning_by_gradient_descent_ridge_lasso(y, tx, w, gamma, lambda1, lambda2):
    """
    Do one step of gradient descent using logistic regression. Return the loss and the updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: float

    Returns:
        loss: scalar number
        w: shape=(D, 1)
    """
    loss = calculate_loss(y, tx, w) + lambda2 * np.squeeze(w.T.dot(w)) + lambda1 * np.sum(np.abs(w))
    gradient = calculate_gradient(y, tx, w) + 2 * lambda2 * w + lambda1 * np.sign(w)
    w_new = w - gamma * gradient
    return loss, w_new

def standardize(x, mean_x=None, std_x=None):
    """Standardize the original data set."""
    if mean_x is None:
        mean_x = np.nanmean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.nanstd(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x

def build_k_indices(y, k_fold):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    indices = np.random.permutation(num_row)
    #retourne la k_fold number of arrays containing each interval number of indexes randomly permuted
    #this function creates an interval for each k from 0 to k-fold - 1
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def read_first_line(filename):
    """Read the first line of a CSV file

    Args:
        filename: name of the CSV file

    Returns:
        first_line: first line of the CSV file
    """
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        first_line = next(reader)
        return first_line
        
def extract_feature(name, filename, x):
    """Extract a feature from the data
    
    Args:
        name: name of the feature
        filename: name of the CSV file
        x: data

    Returns:
        x_new: extracted feature values
    """
    first_line = np.array(read_first_line(filename))
    index = np.where(first_line == name)
    ind = index[0].item()
    return x.copy()[:, ind-1]

def replace_mean(x, mean_value=None):
    """Replace NaN values with the mean of the feature
    
    Args:
        x: feature data
        mean_value: predetermined mean for replacement. If None, the mean is calculated from the non-NaN values

    Returns:
        x_new: feature data with NaN values replaced
    """
    if mean_value is None:
        mean_value = np.nanmean(x)
    x_new = x.copy()
    x_new[np.isnan(x_new)] = mean_value
    return x_new

def replace_mode(x, mode_value=None):
    """Replace NaN values with the mode of the feature
    
    Args:
        x: feature data
        mode_value: predetermined mode for replacement. If None, the mode is calculated from the non-NaN values

    Returns:
        x_new: feature data with NaN values replaced
    """
    if mode_value is None:
        unique, counts = np.unique(x[~np.isnan(x)], return_counts=True)
        mode_value = unique[np.argmax(counts)]
    x_new = x.copy()
    x_new[np.isnan(x_new)] = mode_value
    return x_new

def split_train_val(x, y, k_fold, k):
    """Split the data into training and validation sets
    
    Args:
        x: feature data
        y: labels
        k_fold: number of folds for cross-validation
        k: index of the fold to use for validation

    Returns:
        x_tr: training feature data
        x_te: validation feature data
        y_tr: training labels
        y_te: validation labels
    """
    k_indices = build_k_indices(y, k_fold) 
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]
    return x_tr, x_te, y_tr, y_te

def drop_nan(X, y):
    """Drop rows with NaN values
    
    Args:
        X: feature data
        y: labels

    Returns:
        X: feature data with NaN rows removed
        y: labels with NaN rows removed
    """
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    y = y[mask]
    return X, y

def make_data(filename_train, filename_test, x_train, x_test, y_train, replace=False):
    """Extract features from the training and test data, and process them.
    Selects the relevant features from the data, replaces NaN values, and standardizes the data.
    
    Args:
        filename_train: name of the training CSV file
        filename_test: name of the test CSV file
        x_train: training data
        x_test: test data
        y_train: training labels
        replace: whether to replace NaN values with the mean/mode of the feature

    Returns:
        X_train: processed training data
        Y_train: processed training labels
        X_val: processed validation data
        Y_val: processed validation labels
        X_test: processed test data
    """
    # First extract all of the relevant features from the training data

    ################Body mass idex - continuous feature
    _BMI5 = extract_feature('_BMI5', filename_train, x_train)

    ################High blood pressure - categorical feature (1 = no, 2 = yes, 9 = missing)
    _RFHYPE5 = extract_feature('_RFHYPE5', filename_train, x_train)
    _RFHYPE5[_RFHYPE5 == 9] = np.nan
    _RFHYPE5[_RFHYPE5 == 1] = 0
    _RFHYPE5[_RFHYPE5 == 2] = 1

    ################High cholesterol - categorical feature (1 = no, 2 = yes, 9 = missing)
    _RFCHOL = extract_feature('_RFCHOL', filename_train, x_train)
    _RFCHOL[_RFCHOL == 9] = np.nan
    _RFCHOL[_RFCHOL == 1] = 0
    _RFCHOL[_RFCHOL == 2] = 1        

    ################Smoking status - categorical feature (1 = every day, 2 = some days, 3 = formerly, 4 = never, 9 = missing)
    _SMOKER3 = extract_feature('_SMOKER3', filename_train, x_train)
    _SMOKER3[_SMOKER3 == 9] = np.nan        

    ################Has ever had a stroke  - categorical feature (1 = yes, 2 = no, 7 = don't know, 9 = missing)
    CVDSTRK3 = extract_feature('CVDSTRK3', filename_train, x_train)
    CVDSTRK3[CVDSTRK3 == 9] = np.nan
    CVDSTRK3[CVDSTRK3 == 7] = np.nan
    CVDSTRK3[CVDSTRK3 == 2] = 0        

    ################Cholesterol checked  - categorical feature (1 = within the last 5 years, 2 = more than 5 years ago, 3 = never, 9 = missing)
    _CHOLCHK = extract_feature('_CHOLCHK', filename_train, x_train)
    _CHOLCHK[_CHOLCHK == 9] = np.nan        

    ################Has ever had diabetes  - categorical feature (1 = yes, 2 = yes*, 3 = no, 4 = no - pre-diabetes, 7 = don't know, 9 = missing)
    DIABETE3 = extract_feature('DIABETE3', filename_train, x_train)
    DIABETE3[DIABETE3 == 9] = np.nan
    DIABETE3[DIABETE3 == 7] = np.nan
    DIABETE3[DIABETE3 == 3] = 0
    DIABETE3[DIABETE3 == 4] = 0
    DIABETE3[DIABETE3 == 2] = 1        

    ################Physical activity index  - categorical feature (1 = highly active, 2 = active, 3 = insufficiently active, 4 = inactive, 9 = missing)
    _PACAT1 = extract_feature('_PACAT1', filename_train, x_train)
    _PACAT1[_PACAT1 == 9] = np.nan        

    ################Total fruits consumed per day  - continuous feature (implied 2 dp)
    #_FRUTSUM = extract_feature('_FRUTSUM', filename_train, x_train)

    ################Total vegetables consumed per day  - continuous feature (implied 2 dp)
    #_VEGESUM = extract_feature('_VEGESUM', filename_train, x_train)

    ################Computed number of drinks of alcohol beverages per week  - continuous feature (99900 = missing)
    _DRNKWEK = extract_feature('_DRNKWEK', filename_train, x_train)
    _DRNKWEK[_DRNKWEK == 99900] = np.nan        

    ################Have any healthcare coverage  - categorical feature (1 = yes, 2 = no, 7 = don't know, 9 = missing)
    HLTHPLN1 = extract_feature('HLTHPLN1', filename_train, x_train)
    HLTHPLN1[HLTHPLN1 == 9] = np.nan
    HLTHPLN1[HLTHPLN1 == 7] = np.nan
    HLTHPLN1[HLTHPLN1 == 2] = 0        

    ################Could not see doctor because of cost  - categorical feature (1 = yes, 2 = no, 7 = don't know, 9 = missing)
    MEDCOST = extract_feature('MEDCOST', filename_train, x_train)
    MEDCOST[MEDCOST == 9] = np.nan
    MEDCOST[MEDCOST == 7] = np.nan
    MEDCOST[MEDCOST == 2] = 0        

    ################General health status  - categorical feature (1 = excellent, 2 = very good, 3 = good, 4 = fair, 5 = poor, 7 = don't know, 9 = missing)
    GENHLTH = extract_feature('GENHLTH', filename_train, x_train)
    GENHLTH[GENHLTH == 9] = np.nan
    GENHLTH[GENHLTH == 7] = np.nan        

    ################Number of days mental health not good  - continuous feature (88 = none, 77 = don't know, 99 = refused)
    MENTHLTH = extract_feature('MENTHLTH', filename_train, x_train)
    MENTHLTH[MENTHLTH == 88] = 0
    MENTHLTH[MENTHLTH == 77] = np.nan
    MENTHLTH[MENTHLTH == 99] = np.nan        

    ################Number of days physical health not good  - continuous feature (88 = none, 77 = don't know, 99 = refused)
    PHYSHLTH = extract_feature('PHYSHLTH', filename_train, x_train)
    PHYSHLTH[PHYSHLTH == 88] = 0
    PHYSHLTH[PHYSHLTH == 77] = np.nan
    PHYSHLTH[PHYSHLTH == 99] = np.nan        

    ################Difficulty walking or climbing stairs - categorical feature (1 = yes, 2 = no, 7 = don't know, 9 = missing)
    DIFFWALK = extract_feature('DIFFWALK', filename_train, x_train)
    DIFFWALK[DIFFWALK == 9] = np.nan
    DIFFWALK[DIFFWALK == 7] = np.nan
    DIFFWALK[DIFFWALK == 2] = 0        

    ################Sex - categorical feature (1 = male, 2 = female)
    SEX = extract_feature('SEX', filename_train, x_train)
    SEX[SEX == 2] = 0        

    ################Age  - categorical feature (1 = 18-24, ... 13 = 80+, 14 = missing)
    _AGEG5YR = extract_feature('_AGEG5YR', filename_train, x_train)
    _AGEG5YR[_AGEG5YR == 14] = np.nan        

    ################Education  - categorical feature (1 = none, ... 6 = college grad, 9 = missing)
    EDUCA = extract_feature('EDUCA', filename_train, x_train)
    EDUCA[EDUCA == 9] = np.nan        

    ################Income level  - categorical feature (1 = low, ... 5 = high, 9 = missing)
    _INCOMG = extract_feature('_INCOMG', filename_train, x_train)
    _INCOMG[_INCOMG == 9] = np.nan        

    # Here we add some interactions and polynomial features for feature expansion

    ################BMI x age
    BMIxAGE = _BMI5 * _AGEG5YR

    ################Age2
    AGE2 = _AGEG5YR ** 2

    ################Age3
    AGE3 = _AGEG5YR ** 3

    ################Drink2
    DRNK2 = _DRNKWEK ** 2

    ################Physhealth2
    PHYSHLTH2 = PHYSHLTH ** 2

    ################Menthealth2
    MENTHLTH2 = MENTHLTH ** 2

    ################CHOLxHYPE
    CHOLxHYPE = _RFCHOL * _RFHYPE5

    # TRIED TO ADD INTERACTIONS and POLYNOMIALS, DIDN'T SEEM TO IMPROVE THE MODEL

    #Here we stack the features together to have the our new X
    X = np.hstack((_BMI5.reshape(-1, 1), _RFHYPE5.reshape(-1, 1), _RFCHOL.reshape(-1, 1), _SMOKER3.reshape(-1, 1), CVDSTRK3.reshape(-1, 1), 
                _CHOLCHK.reshape(-1, 1), DIABETE3.reshape(-1, 1), _PACAT1.reshape(-1, 1), # _FRUTSUM.reshape(-1, 1), _VEGESUM.reshape(-1, 1), 
                _DRNKWEK.reshape(-1, 1), HLTHPLN1.reshape(-1, 1), MEDCOST.reshape(-1, 1), GENHLTH.reshape(-1, 1), MENTHLTH.reshape(-1, 1), PHYSHLTH.reshape(-1, 1), 
                DIFFWALK.reshape(-1, 1), SEX.reshape(-1, 1), _AGEG5YR.reshape(-1, 1), EDUCA.reshape(-1, 1), _INCOMG.reshape(-1, 1)))#,
                #BMIxAGE.reshape(-1, 1), AGE2.reshape(-1, 1), AGE3.reshape(-1, 1), DRNK2.reshape(-1, 1), PHYSHLTH2.reshape(-1, 1), MENTHLTH2.reshape(-1, 1), CHOLxHYPE.reshape(-1, 1)))
    
    # Change all the elements with -1 by 0
    y_train_working = y_train.copy()
    y_train_working[y_train_working == -1] = 0
    # Make y have the correct shape
    y_train_working = y_train_working.reshape(-1, 1)

    # Shuffle the data
    np.random.seed(6)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_train_working_shuffled = y_train_working[indices]

    # Split the data into training and validation sets (90% training, 10% validation)
    X_train, X_val, Y_train, Y_val = split_train_val(X_shuffled, y_train_working_shuffled, 10, 9)
    
    means_ids = [0, 8, 12, 13] # IDs of features that should be replaced with the mean instead of the mode
    if replace:
        # Replace the missing values with the mean or mode of the feature, depending on the feature
        for i in range(X_train.shape[1]):
            if i in means_ids:
                X_train[:, i] = replace_mean(X_train[:, i])
                mean_value = np.nanmean(X_train[:, i])
                X_val[:, i] = replace_mean(X_val[:, i])
            else:
                X_train[:, i] = replace_mode(X_train[:, i])
                unique, counts = np.unique(X_train[:, i][~np.isnan(X_train[:, i])], return_counts=True)
                mode_value = unique[np.argmax(counts)]
                X_val[:, i] = replace_mode(X_val[:, i], mode_value=mode_value)
    else:
        # Drop rows with NaN values in the training set
        X_train, Y_train = drop_nan(X_train, Y_train)

        # Replace NaN values with the mean or mode of the validation set
        for i in range(X_val.shape[1]):
            if i in means_ids:
                mean_value = np.nanmean(X_train[:, i])
                X_val[:, i] = replace_mean(X_val[:, i], mean_value=mean_value)
            else:
                unique, counts = np.unique(X_train[:, i][~np.isnan(X_train[:, i])], return_counts=True)
                mode_value = unique[np.argmax(counts)]
                X_val[:, i] = replace_mode(X_val[:, i], mode_value=mode_value)
        
        # X_val[:, 0] = replace_mean(X_val[:, 0]) # BMI
        # X_val[:, 1] = replace_mode(X_val[:, 1]) # RFHYPE5
        # X_val[:, 2] = replace_mode(X_val[:, 2]) # RFCHOL
        # X_val[:, 3] = replace_mode(X_val[:, 3]) # SMOKER3
        # X_val[:, 4] = replace_mode(X_val[:, 4]) # CVDSTRK3
        # X_val[:, 5] = replace_mode(X_val[:, 5]) # CHOLCHK
        # X_val[:, 6] = replace_mode(X_val[:, 6]) # DIABETE3
        # X_val[:, 7] = replace_mode(X_val[:, 7]) # PACAT1
        # X_val[:, 8] = replace_mean(X_val[:, 8]) # DRNKWEK
        # X_val[:, 9] = replace_mode(X_val[:, 9]) # HLTHPLN1
        # X_val[:, 10] = replace_mode(X_val[:, 10]) # MEDCOST
        # X_val[:, 11] = replace_mode(X_val[:, 11]) # GENHLTH
        # X_val[:, 12] = replace_mean(X_val[:, 12]) # MENTHLTH
        # X_val[:, 13] = replace_mean(X_val[:, 13]) # PHYSHLTH
        # X_val[:, 14] = replace_mode(X_val[:, 14]) # DIFFWALK
        # X_val[:, 15] = replace_mode(X_val[:, 15]) # SEX
        # X_val[:, 16] = replace_mode(X_val[:, 16]) # AGEG5YR
        # X_val[:, 17] = replace_mode(X_val[:, 17]) # EDUCA
        # X_val[:, 18] = replace_mode(X_val[:, 18]) # INCOMG

    # Standardize the data
    X_train, X_train_mean, X_train_std = standardize(X_train)
    X_val, _, _ = standardize(X_val, X_train_mean, X_train_std)

    # NOT SURE IF WE NEED TO AVOID STANDARDIZING CATEGORICAL FEATURES
    
    # _BMI5, _BMI5_mean, _BMI5_std = standardize(_BMI5)
    # _SMOKER3, _SMOKER3_mean, _SMOKER3_std = standardize(_SMOKER3)
    # _CHOLCHK,_ , _ = standardize(_CHOLCHK)
    # _PACAT1, _PACAT1_mean, _PACAT1_std = standardize(_PACAT1)
    # _DRNKWEK, _DRNKWEK_mean, _DRNKWEK_std = standardize(_DRNKWEK)
    # GENHLTH, GENHLTH_mean, GENHLTH_std = standardize(GENHLTH)
    # MENTHLTH, MENTHLTH_mean, MENTHLTH_std = standardize(MENTHLTH)
    # PHYSHLTH, PHYSHLTH_mean, PHYSHLTH_std = standardize(PHYSHLTH)
    # _AGEG5YR, _AGEG5YR_mean, _AGEG5YR_std = standardize(_AGEG5YR)
    # EDUCA, EDUCA_mean, EDUCA_std = standardize(EDUCA)
    # _INCOMG, _INCOMG_mean, _INCOMG_std = standardize(_INCOMG)


    # Now we extract the features from the test data
    ################Body mass idex - continuous feature
    _BMI5_test = extract_feature('_BMI5', filename_test, x_test)


    ################High blood pressure - categorical feature (1 = no, 2 = yes, 9 = missing)
    _RFHYPE5_test = extract_feature('_RFHYPE5', filename_test, x_test)
    _RFHYPE5_test[_RFHYPE5_test == 9] = np.nan
    _RFHYPE5_test[_RFHYPE5_test == 1] = 0
    _RFHYPE5_test[_RFHYPE5_test == 2] = 1

    ################High cholesterol - categorical feature (1 = no, 2 = yes, 9 = missing)
    _RFCHOL_test = extract_feature('_RFCHOL', filename_test, x_test)
    _RFCHOL_test[_RFCHOL_test == 9] = np.nan
    _RFCHOL_test[_RFCHOL_test == 1] = 0
    _RFCHOL_test[_RFCHOL_test == 2] = 1

    ################Smoking status - categorical feature (1 = every day, 2 = some days, 3 = formerly, 4 = never, 9 = missing)
    _SMOKER3_test = extract_feature('_SMOKER3', filename_test, x_test)
    _SMOKER3_test[_SMOKER3_test == 9] = np.nan

    ################Has ever had a stroke  - categorical feature (1 = yes, 2 = no, 7 = don't know, 9 = missing)
    CVDSTRK3_test = extract_feature('CVDSTRK3', filename_test, x_test)
    CVDSTRK3_test[CVDSTRK3_test == 9] = np.nan
    CVDSTRK3_test[CVDSTRK3_test == 7] = np.nan
    CVDSTRK3_test[CVDSTRK3_test == 2] = 0

    ################Cholesterol checked  - categorical feature (1 = within the last 5 years, 2 = more than 5 years ago, 3 = never, 9 = missing)
    _CHOLCHK_test = extract_feature('_CHOLCHK', filename_test, x_test)
    _CHOLCHK_test[_CHOLCHK_test == 9] = np.nan

    ################Has ever had diabetes  - categorical feature (1 = yes, 2 = yes*, 3 = no, 4 = no - pre-diabetes, 7 = don't know, 9 = missing)
    DIABETE3_test = extract_feature('DIABETE3', filename_test, x_test)
    DIABETE3_test[DIABETE3_test == 9] = np.nan
    DIABETE3_test[DIABETE3_test == 7] = np.nan
    DIABETE3_test[DIABETE3_test == 3] = 0
    DIABETE3_test[DIABETE3_test == 4] = 0
    DIABETE3_test[DIABETE3_test == 2] = 1

    ################Physical activity index  - categorical feature (1 = highly active, 2 = active, 3 = insufficiently active, 4 = inactive, 9 = missing)
    _PACAT1_test = extract_feature('_PACAT1', filename_test, x_test)
    _PACAT1_test[_PACAT1_test == 9] = np.nan

    ################Total fruits consumed per day  - continuous feature (implied 2 dp)
    #_FRUTSUM_test = extract_feature('_FRUTSUM', filename_test, x_test)

    ################Total vegetables consumed per day  - continuous feature (implied 2 dp)
    #_VEGESUM_test = extract_feature('_VEGESUM', filename_test, x_test)

    ################Computed number of drinks of alcohol beverages per week  - continuous feature (99900 = missing)
    _DRNKWEK_test = extract_feature('_DRNKWEK', filename_test, x_test)
    _DRNKWEK_test[_DRNKWEK_test == 99900] = np.nan

    ################Have any healthcare coverage  - categorical feature (1 = yes, 2 = no, 7 = don't know, 9 = missing)
    HLTHPLN1_test = extract_feature('HLTHPLN1', filename_test, x_test)
    HLTHPLN1_test[HLTHPLN1_test == 9] = np.nan
    HLTHPLN1_test[HLTHPLN1_test == 7] = np.nan
    HLTHPLN1_test[HLTHPLN1_test == 2] = 0

    ################Could not see doctor because of cost  - categorical feature (1 = yes, 2 = no, 7 = don't know, 9 = missing)
    MEDCOST_test = extract_feature('MEDCOST', filename_test, x_test)
    MEDCOST_test[MEDCOST_test == 9] = np.nan
    MEDCOST_test[MEDCOST_test == 7] = np.nan
    MEDCOST_test[MEDCOST_test == 2] = 0

    ################General health status  - categorical feature (1 = excellent, 2 = very good, 3 = good, 4 = fair, 5 = poor, 7 = don't know, 9 = missing)
    GENHLTH_test = extract_feature('GENHLTH', filename_test, x_test)
    GENHLTH_test[GENHLTH_test == 9] = np.nan
    GENHLTH_test[GENHLTH_test == 7] = np.nan

    ################Number of days mental health not good  - continuous feature (88 = none, 77 = don't know, 99 = refused)
    MENTHLTH_test = extract_feature('MENTHLTH', filename_test, x_test)
    MENTHLTH_test[MENTHLTH_test == 88] = 0
    MENTHLTH_test[MENTHLTH_test == 77] = np.nan
    MENTHLTH_test[MENTHLTH_test == 99] = np.nan

    ################Number of days physical health not good  - continuous feature (88 = none, 77 = don't know, 99 = refused)
    PHYSHLTH_test = extract_feature('PHYSHLTH', filename_test, x_test)
    PHYSHLTH_test[PHYSHLTH_test == 88] = 0
    PHYSHLTH_test[PHYSHLTH_test == 77] = np.nan
    PHYSHLTH_test[PHYSHLTH_test == 99] = np.nan

    ################Difficulty walking or climbing stairs - categorical feature (1 = yes, 2 = no, 7 = don't know, 9 = missing)
    DIFFWALK_test = extract_feature('DIFFWALK', filename_test, x_test)
    DIFFWALK_test[DIFFWALK_test == 9] = np.nan
    DIFFWALK_test[DIFFWALK_test == 7] = np.nan
    DIFFWALK_test[DIFFWALK_test == 2] = 0

    ################Sex - categorical feature (1 = male, 2 = female)
    SEX_test = extract_feature('SEX', filename_test, x_test)
    SEX_test[SEX_test == 2] = 0

    ################Age  - categorical feature (1 = 18-24, ... 13 = 80+, 14 = missing)
    _AGEG5YR_test = extract_feature('_AGEG5YR', filename_test, x_test)
    _AGEG5YR_test[_AGEG5YR_test == 14] = np.nan

    ################Education  - categorical feature (1 = none, ... 6 = college grad, 9 = missing)
    EDUCA_test = extract_feature('EDUCA', filename_test, x_test)
    EDUCA_test[EDUCA_test == 9] = np.nan

    ################Income level  - categorical feature (1 = low, ... 5 = high, 9 = missing)
    _INCOMG_test = extract_feature('_INCOMG', filename_test, x_test)
    _INCOMG_test[_INCOMG_test == 9] = np.nan

    #Here we stack the features together to have the our new X for the test data
    X_test = np.hstack((_BMI5_test.reshape(-1, 1), _RFHYPE5_test.reshape(-1, 1), _RFCHOL_test.reshape(-1, 1), _SMOKER3_test.reshape(-1, 1), CVDSTRK3_test.reshape(-1, 1), 
                _CHOLCHK_test.reshape(-1, 1), DIABETE3_test.reshape(-1, 1), _PACAT1_test.reshape(-1, 1), # _FRUTSUM_test.reshape(-1, 1), _VEGESUM_test.reshape(-1, 1), 
                _DRNKWEK_test.reshape(-1, 1), HLTHPLN1_test.reshape(-1, 1), MEDCOST_test.reshape(-1, 1), GENHLTH_test.reshape(-1, 1), MENTHLTH_test.reshape(-1, 1), PHYSHLTH_test.reshape(-1, 1), 
                DIFFWALK_test.reshape(-1, 1), SEX_test.reshape(-1, 1), _AGEG5YR_test.reshape(-1, 1), EDUCA_test.reshape(-1, 1), _INCOMG_test.reshape(-1, 1)))

    # We replace the missing values in the test set with the mean or mode of the feature, depending on the feature
    for i in range(X_test.shape[1]):
        if i in means_ids:
            mean_value = np.nanmean(X_train[:, i])
            X_test[:, i] = replace_mean(X_test[:, i], mean_value=mean_value)
        else:
            unique, counts = np.unique(X_train[:, i][~np.isnan(X_train[:, i])], return_counts=True)
            mode_value = unique[np.argmax(counts)]
            X_test[:, i] = replace_mode(X_test[:, i], mode_value=mode_value)
    
    # Standardize the test data
    X_test, _, _ = standardize(X_test, X_train_mean, X_train_std)


    return X_train, Y_train, X_val, Y_val, X_test


def undersampling(X, y):
    """Balance the data by taking only a certain number of values in y=0 s.t. the number of y=1 equals the number of y=0
    
    Args:
        X: feature data
        y: labels

    Returns:
        selected_X: balanced feature data
        selected_y: balanced labels
    """
    
    y = y.reshape(-1)
    indices_y_equals_1 = np.where(y == 1)[0]
    indices_y_equals_0 = np.where(y == 0)[0]
    num_positives = np.sum(y == 1)
    num_negatives = np.sum(y == 0)
    selected_indices_neg = np.random.choice(indices_y_equals_0, size=num_positives, replace=False)
    
    selected_indices = np.concatenate((selected_indices_neg, indices_y_equals_1))
    selected_X = X[selected_indices]
    selected_y = y[selected_indices].reshape(-1, 1)
    
    return selected_X, selected_y

def oversampling(X, y): 
    """Balance the data by duplicating the values in y=1 s.t. the number of y=1 equals the number of y=0
    
    Args:
        X: feature data
        y: labels

    Returns:
        X_balanced: balanced feature data
        y_balanced: balanced labels
    """

    y = y.reshape(-1)
    class_1 = X[y == 1]
    class_0 = X[y == 0]
    count_class_1 = class_1.shape[0]
    count_class_0 = class_0.shape[0]
    num_to_duplicate = count_class_0 - count_class_1
    duplicated_samples = np.tile(class_1, (num_to_duplicate // count_class_1 + 1, 1))[:num_to_duplicate]
    X_balanced = np.vstack((X, duplicated_samples))
    y_balanced = np.hstack((y, np.ones(num_to_duplicate))).reshape(-1, 1)
    return X_balanced, y_balanced

def undersampling_oversampling(X, y, ratio_majority=1, ratio_majority_to_minority=2):
    """Balance the data by undersampling the majority class and oversampling the minority class
    
    Args:
        X: feature data
        y: labels
        ratio_majority: desired factor with which to undersample the majority class
        ratio_majority_to_minority: desired ratio of majority class to minority class data to oversample the minority class

    Returns:
        X_balanced: balanced feature data
        y_balanced: balanced labels
    """

    y = y.reshape(-1)

    # Undersample the majority class
    indices_y_equals_1 = np.where(y == 1)[0]
    indices_y_equals_0 = np.where(y == 0)[0]
    num_positives = np.sum(y == 1)
    num_negatives = np.sum(y == 0)
    selected_indices_neg = np.random.choice(indices_y_equals_0, size=int(np.floor(ratio_majority*num_negatives)), replace=False)
    selected_indices = np.concatenate((selected_indices_neg, indices_y_equals_1))
    selected_X = X[selected_indices]
    selected_y = y[selected_indices]

    # Oversample the minority class
    num_to_duplicate = int(np.floor(ratio_majority*num_negatives) / ratio_majority_to_minority) - num_positives
    duplicated_samples = np.tile(X[y == 1], (num_to_duplicate // num_positives + 1, 1))[:num_to_duplicate]
    X_balanced = np.vstack((selected_X, duplicated_samples))
    y_balanced = np.hstack((selected_y, np.ones(num_to_duplicate)))

    # Shuffle the data
    np.random.seed(42)
    indices = np.arange(X_balanced.shape[0])
    np.random.shuffle(indices)
    X_balanced = X_balanced[indices]
    y_balanced = y_balanced[indices].reshape(-1, 1)
    
    return X_balanced, y_balanced