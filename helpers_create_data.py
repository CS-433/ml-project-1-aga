"""Some own helper functions for project 1."""
import csv
import numpy as np
import os

def sigmoid(t):
    return np.exp(t) / (1 + np.exp(t))

def standardize(x, mean_x=None, std_x=None):
    """
    Standardizes the given dataset by subtracting the mean and dividing by the standard deviation for each feature.
    If mean and standard deviation are not provided, they are computed from the data.
    
    Parameters:
    -----------
    x : numpy.ndarray
        The input data to be standardized, where rows are samples and columns are features.
        
    mean_x : numpy.ndarray, optional
        The precomputed mean of each feature. If None, the mean is calculated from the input data.
        
    std_x : numpy.ndarray, optional
        The precomputed standard deviation of each feature. If None, the standard deviation is calculated from the input data.
        
    Returns:
    --------
    x : numpy.ndarray
        The standardized data, where each feature has zero mean and unit variance.
        
    mean_x : numpy.ndarray
        The mean of each feature used for standardization.
        
    std_x : numpy.ndarray
        The standard deviation of each feature used for standardization.
    """

    if mean_x is None:
        mean_x = np.nanmean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.nanstd(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x


def read_first_line(filename):
    """
    Reads the first line of a CSV file.
    
    Parameters:
    -----------
    filename : str
        The name or path of the CSV file to be read.
        
    Returns:
    --------
    first_line : str
        The first line of the CSV file as a string.
    """
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        first_line = next(reader)
        return first_line
        
def extract_feature(name, filename, x):
    """
    Extracts a specified feature from the dataset.
    
    Parameters:
    -----------
    name : str
        The name of the feature to be extracted.
        
    filename : str
        The path of the CSV file from which the data is extracted.
        
    x : numpy.ndarray
        The dataset from which the feature will be extracted.
        
    Returns:
    --------
    x_new : numpy.ndarray
        A 1D array of the extracted feature, processed if necessary.
    """
    first_line = np.array(read_first_line(filename))
    index = np.where(first_line == name)
    ind = index[0].item()
    return x.copy()[:, ind-1]

def replace_mean(x, mean_value=None):
    """
    Replaces NaN values in the feature data with the mean of the feature.
    
    Parameters:
    -----------
    x : numpy.ndarray
        The feature data which may contain NaN values.
        
    mean_value : float, optional
        A predetermined mean value to replace NaNs. If None, the mean is calculated from the non-NaN values in `x`.
        
    Returns:
    --------
    x_new : numpy.ndarray
        The feature data with NaN values replaced by the mean.
    """
    if mean_value is None:
        mean_value = np.nanmean(x)
    x_new = x.copy()
    x_new[np.isnan(x_new)] = mean_value
    return x_new

def replace_mode(x, mode_value=None):
    """
    Replace NaN values with the mode of the feature.
    
    Parameters:
    -----
    x : numpy.ndarray
        Feature data which may contain NaN values.
        
    mode_value : float, optional
        A predetermined mode value for replacement. If None, the mode is calculated from the non-NaN values.
        
    Returns:
    --------
    x_new : numpy.ndarray
        Feature data with NaN values replaced by the mode.
    """
    if mode_value is None:
        unique, counts = np.unique(x[~np.isnan(x)], return_counts=True)
        mode_value = unique[np.argmax(counts)]
    x_new = x.copy()
    x_new[np.isnan(x_new)] = mode_value
    return x_new

def build_k_indices(y, k_fold):
    """
    Build k indices for k-fold. Taken directly from the ML course

    Parameters:
    -----
    y : shape=(N,)
    k_fold : K in K-fold, i.e. the fold num

    Returns:
    --------
    A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    indices = np.random.permutation(num_row)
    #this function creates an interval for each k from 0 to k-fold - 1
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def split_train_val(x, y, k_fold, k):
    """
    Split the dataset into training and validation sets.
    
    Parameters:
    -----
    x : numpy.ndarray
        Feature matrix (samples x features).
        
    y : numpy.ndarray
        Target values corresponding to the samples in `x`.
        
    k_fold : int
        The total number of folds
        
    k : int
        The index of the fold beeing taken for the validation set
    
    Returns:
    --------
    x_tr : numpy.ndarray
        Training feature data.
        
    x_te : numpy.ndarray
        Validation feature data.
        
    y_tr : numpy.ndarray
        Training target data.
        
    y_te : numpy.ndarray
        Validation target data.
    """
    
    k_indices = build_k_indices(y, k_fold) 
    te_indice = k_indices[k]
    #Take all the folds exept the kth one
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]
    return x_tr, x_te, y_tr, y_te

def drop_nan(X, y):
    """
    Drop rows with NaN values from both feature data and labels.
    
    Parameters:
    -----
    X : numpy.ndarray
        Feature data where rows with NaN values might exist.
        
    y : numpy.ndarray
        Labels corresponding to the feature data.
    
    Returns:
    --------
    X : numpy.ndarray
        Feature data with rows containing NaN values removed.
        
    y : numpy.ndarray
        Labels with rows containing NaN values removed (corresponding to the removed feature rows).
    """
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    y = y[mask]
    return X, y

def one_hot_encode(tx_train, tx_test, tx_val = None):
    """
    Perform one-hot encoding on categorical feature arrays for training, validation, and test datasets.
    
    Parameters:
    -----
    tx_train : numpy.ndarray
        A 1D array of categorical feature data for the training set.
    
    tx_val : numpy.ndarray
        A 1D array of categorical feature data for the validation set.
    
    tx_test : numpy.ndarray
        A 1D array of categorical feature data for the test set.
    
    Returns:
    --------
    tx_train_one_hot : numpy.ndarray
        A 2D array representing the one-hot encoded version of the training set.
    
    tx_val_one_hot : numpy.ndarray
        A 2D array representing the one-hot encoded version of the validation set.
    
    tx_test_one_hot : numpy.ndarray
        A 2D array representing the one-hot encoded version of the test set.
    """
    # Combine all datasets to find unique values
    if tx_val is None:
        combined = np.concatenate((tx_train, tx_test))
    else:
        combined = np.concatenate((tx_train, tx_val, tx_test))
    unique_values = np.unique(combined)
    n_unique = unique_values.size
    
    # Initialize one-hot encoded arrays
    tx_train_one_hot = np.zeros((tx_train.shape[0], n_unique))
    if tx_val is not None:
        tx_val_one_hot = np.zeros((tx_val.shape[0], n_unique))
    tx_test_one_hot = np.zeros((tx_test.shape[0], n_unique))
    
    # Encode training set
    for i in range(tx_train.shape[0]):
        tx_train_one_hot[i, np.where(unique_values == tx_train[i])[0]] = 1
    
    # Encode validation set
    if tx_val is not None:
        for i in range(tx_val.shape[0]):
            tx_val_one_hot[i, np.where(unique_values == tx_val[i])[0]] = 1
    else:
        tx_val_one_hot = None
    
    # Encode test set
    for i in range(tx_test.shape[0]):
        tx_test_one_hot[i, np.where(unique_values == tx_test[i])[0]] = 1

    return tx_train_one_hot, tx_test_one_hot, tx_val_one_hot

def extract_features(filename_train, filename_test, x_train, x_test, onehotencode):
    """
    Extracts and preprocesses a set of features from the provided data.
    
    Parameters:
    -----------
    filename_train : str
        Name of the CSV file from which the training data is extracted.
    filename_test : str
        Name of the CSV file from which the test data is extracted.
    x_train : array-like
        The training data matrix or feature set from which individual features are extracted.
    x_test : array-like
        The test data matrix or feature set from which individual features are extracted.
    onehotencode : bool
        If True, certain categorical features will be one-hot encoded.

    Returns:
    --------
    X_train : numpy.ndarray
        A 2D array of extracted and processed features for the training set.
    X_test : numpy.ndarray
        A 2D array of extracted and processed features for the test set.
    """

    ################Body mass idex - continuous feature
    _BMI5_train = extract_feature('_BMI5', filename_train, x_train)

    ################High blood pressure - categorical feature (1 = no, 2 = yes, 9 = missing)
    _RFHYPE5_train = extract_feature('_RFHYPE5', filename_train, x_train)
    _RFHYPE5_train[_RFHYPE5_train == 9] = np.nan
    _RFHYPE5_train[_RFHYPE5_train == 1] = 0
    _RFHYPE5_train[_RFHYPE5_train == 2] = 1

    ################High cholesterol - categorical feature (1 = no, 2 = yes, 9 = missing)
    _RFCHOL_train = extract_feature('_RFCHOL', filename_train, x_train)
    _RFCHOL_train[_RFCHOL_train == 9] = np.nan
    _RFCHOL_train[_RFCHOL_train == 1] = 0
    _RFCHOL_train[_RFCHOL_train == 2] = 1        

    ################Smoking status - categorical feature (1 = every day, 2 = some days, 3 = formerly, 4 = never, 9 = missing)
    _SMOKER3_train = extract_feature('_SMOKER3', filename_train, x_train)
    _SMOKER3_train[_SMOKER3_train == 9] = np.nan 
    #Here we have 3 or more categories -> hot encoding makes sense

    ################Has ever had a stroke  - categorical feature (1 = yes, 2 = no, 7 = don't know, 9 = missing)
    CVDSTRK3_train = extract_feature('CVDSTRK3', filename_train, x_train)
    CVDSTRK3_train[CVDSTRK3_train == 9] = np.nan
    CVDSTRK3_train[CVDSTRK3_train == 7] = np.nan
    CVDSTRK3_train[CVDSTRK3_train == 2] = 0        

    ################Cholesterol checked  - categorical feature (1 = within the last 5 years, 2 = more than 5 years ago, 3 = never, 9 = missing)
    _CHOLCHK_train = extract_feature('_CHOLCHK', filename_train, x_train)
    _CHOLCHK_train[_CHOLCHK_train == 9] = np.nan        

    ################Has ever had diabetes  - categorical feature (1 = yes, 2 = yes*, 3 = no, 4 = no - pre-diabetes, 7 = don't know, 9 = missing)
    DIABETE3_train = extract_feature('DIABETE3', filename_train, x_train)
    DIABETE3_train[DIABETE3_train == 9] = np.nan
    DIABETE3_train[DIABETE3_train == 7] = np.nan
    DIABETE3_train[DIABETE3_train == 3] = 0
    DIABETE3_train[DIABETE3_train == 4] = 0
    DIABETE3_train[DIABETE3_train == 2] = 1        

    ################Physical activity index  - categorical feature (1 = highly active, 2 = active, 3 = insufficiently active, 4 = inactive, 9 = missing)
    _PACAT1_train = extract_feature('_PACAT1', filename_train, x_train)
    _PACAT1_train[_PACAT1_train == 9] = np.nan
    #Here we have 3 or more categories -> hot encoding makes sense

    ################Computed number of drinks of alcohol beverages per week  - continuous feature (99900 = missing)
    _DRNKWEK_train = extract_feature('_DRNKWEK', filename_train, x_train)
    _DRNKWEK_train[_DRNKWEK_train == 99900] = np.nan        

    ################Have any healthcare coverage  - categorical feature (1 = yes, 2 = no, 7 = don't know, 9 = missing)
    HLTHPLN1_train = extract_feature('HLTHPLN1', filename_train, x_train)
    HLTHPLN1_train[HLTHPLN1_train == 9] = np.nan
    HLTHPLN1_train[HLTHPLN1_train == 7] = np.nan
    HLTHPLN1_train[HLTHPLN1_train == 2] = 0        

    ################Could not see doctor because of cost  - categorical feature (1 = yes, 2 = no, 7 = don't know, 9 = missing)
    MEDCOST_train = extract_feature('MEDCOST', filename_train, x_train)
    MEDCOST_train[MEDCOST_train == 9] = np.nan
    MEDCOST_train[MEDCOST_train == 7] = np.nan
    MEDCOST_train[MEDCOST_train == 2] = 0        

    ################General health status  - categorical feature (1 = excellent, 2 = very good, 3 = good, 4 = fair, 5 = poor, 7 = don't know, 9 = missing)
    GENHLTH_train = extract_feature('GENHLTH', filename_train, x_train)
    GENHLTH_train[GENHLTH_train == 9] = np.nan
    GENHLTH_train[GENHLTH_train == 7] = np.nan 
    #Here we have 3 or more categories -> hot encoding makes sense

    ################Number of days mental health not good  - continuous feature (88 = none, 77 = don't know, 99 = refused)
    MENTHLTH_train = extract_feature('MENTHLTH', filename_train, x_train)
    MENTHLTH_train[MENTHLTH_train == 88] = 0
    MENTHLTH_train[MENTHLTH_train == 77] = np.nan
    MENTHLTH_train[MENTHLTH_train == 99] = np.nan        

    ################Number of days physical health not good  - continuous feature (88 = none, 77 = don't know, 99 = refused)
    PHYSHLTH_train = extract_feature('PHYSHLTH', filename_train, x_train)
    PHYSHLTH_train[PHYSHLTH_train == 88] = 0
    PHYSHLTH_train[PHYSHLTH_train == 77] = np.nan
    PHYSHLTH_train[PHYSHLTH_train == 99] = np.nan        

    ################Difficulty walking or climbing stairs - categorical feature (1 = yes, 2 = no, 7 = don't know, 9 = missing)
    DIFFWALK_train = extract_feature('DIFFWALK', filename_train, x_train)
    DIFFWALK_train[DIFFWALK_train == 9] = np.nan
    DIFFWALK_train[DIFFWALK_train == 7] = np.nan
    DIFFWALK_train[DIFFWALK_train == 2] = 0        

    ################Sex - categorical feature (1 = male, 2 = female)
    SEX_train = extract_feature('SEX', filename_train, x_train)
    SEX_train[SEX_train == 2] = 0        

    ################Age  - categorical feature (1 = 18-24, ... 13 = 80+, 14 = missing)
    _AGEG5YR_train = extract_feature('_AGEG5YR', filename_train, x_train)
    _AGEG5YR_train[_AGEG5YR_train == 14] = np.nan 
    #Here we have 3 or more categories -> hot encoding makes sense

    ################Education  - categorical feature (1 = none, ... 6 = college grad, 9 = missing)
    EDUCA_train = extract_feature('EDUCA', filename_train, x_train)
    EDUCA_train[EDUCA_train == 9] = np.nan  
    #Here we have 3 or more categories -> hot encoding makes sense

    ################Income level  - categorical feature (1 = low, ... 5 = high, 9 = missing)
    _INCOMG_train = extract_feature('_INCOMG', filename_train, x_train)
    _INCOMG_train[_INCOMG_train == 9] = np.nan 
    #Here we have 3 or more categories -> hot encoding makes sense


    # Now we do the same for the test set


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
    #Here we have 3 or more categories -> hot encoding makes sense

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
    #Here we have 3 or more categories -> hot encoding makes sense

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
    #Here we have 3 or more categories -> hot encoding makes sense

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
    #Here we have 3 or more categories -> hot encoding makes sense

    ################Education  - categorical feature (1 = none, ... 6 = college grad, 9 = missing)
    EDUCA_test = extract_feature('EDUCA', filename_test, x_test)
    EDUCA_test[EDUCA_test == 9] = np.nan  
    #Here we have 3 or more categories -> hot encoding makes sense

    ################Income level  - categorical feature (1 = low, ... 5 = high, 9 = missing)
    _INCOMG_test = extract_feature('_INCOMG', filename_test, x_test)
    _INCOMG_test[_INCOMG_test == 9] = np.nan 
    #Here we have 3 or more categories -> hot encoding makes sense

    if onehotencode:
        _SMOKER3_train, _SMOKER3_test, _ = one_hot_encode(_SMOKER3_train, _SMOKER3_test)
        _PACAT1_train, _PACAT1_test, _ = one_hot_encode(_PACAT1_train, _PACAT1_test)
        GENHLTH_train, GENHLTH_test, _ = one_hot_encode(GENHLTH_train, GENHLTH_test)
        _AGEG5YR_train, _AGEG5YR_test, _ = one_hot_encode(_AGEG5YR_train, _AGEG5YR_test)
        EDUCA_train, EDUCA_test, _ = one_hot_encode(EDUCA_train, EDUCA_test)
        _INCOMG_train, _INCOMG_test, _ = one_hot_encode(_INCOMG_train, _INCOMG_test)

    X_train = []
    X_test = []
    #Here we stack the features together (depending if we want to hot one encode or not) to have the our new X
    if onehotencode:
        X_train = np.hstack((_BMI5_train.reshape(-1, 1), _RFHYPE5_train.reshape(-1, 1), _RFCHOL_train.reshape(-1, 1), _SMOKER3_train, CVDSTRK3_train.reshape(-1, 1), _CHOLCHK_train.reshape(-1, 1), DIABETE3_train.reshape(-1, 1), _PACAT1_train, _DRNKWEK_train.reshape(-1, 1), HLTHPLN1_train.reshape(-1, 1), MEDCOST_train.reshape(-1, 1), GENHLTH_train, MENTHLTH_train.reshape(-1, 1), PHYSHLTH_train.reshape(-1, 1), DIFFWALK_train.reshape(-1, 1), SEX_train.reshape(-1, 1), _AGEG5YR_train, EDUCA_train, _INCOMG_train))
        X_test = np.hstack((_BMI5_test.reshape(-1, 1), _RFHYPE5_test.reshape(-1, 1), _RFCHOL_test.reshape(-1, 1), _SMOKER3_test, CVDSTRK3_test.reshape(-1, 1), _CHOLCHK_test.reshape(-1, 1), DIABETE3_test.reshape(-1, 1), _PACAT1_test, _DRNKWEK_test.reshape(-1, 1), HLTHPLN1_test.reshape(-1, 1), MEDCOST_test.reshape(-1, 1), GENHLTH_test, MENTHLTH_test.reshape(-1, 1), PHYSHLTH_test.reshape(-1, 1), DIFFWALK_test.reshape(-1, 1), SEX_test.reshape(-1, 1), _AGEG5YR_test, EDUCA_test, _INCOMG_test))
    else:
        X_train = np.hstack((_BMI5_train.reshape(-1, 1), _RFHYPE5_train.reshape(-1, 1), _RFCHOL_train.reshape(-1, 1), _SMOKER3_train.reshape(-1, 1), CVDSTRK3_train.reshape(-1, 1), _CHOLCHK_train.reshape(-1, 1), DIABETE3_train.reshape(-1, 1), _PACAT1_train.reshape(-1, 1), _DRNKWEK_train.reshape(-1, 1), HLTHPLN1_train.reshape(-1, 1), MEDCOST_train.reshape(-1, 1), GENHLTH_train.reshape(-1, 1), MENTHLTH_train.reshape(-1, 1), PHYSHLTH_train.reshape(-1, 1), DIFFWALK_train.reshape(-1, 1), SEX_train.reshape(-1, 1), _AGEG5YR_train.reshape(-1, 1), EDUCA_train.reshape(-1, 1), _INCOMG_train.reshape(-1, 1)))
        X_test = np.hstack((_BMI5_test.reshape(-1, 1), _RFHYPE5_test.reshape(-1, 1), _RFCHOL_test.reshape(-1, 1), _SMOKER3_test.reshape(-1, 1), CVDSTRK3_test.reshape(-1, 1), _CHOLCHK_test.reshape(-1, 1), DIABETE3_test.reshape(-1, 1), _PACAT1_test.reshape(-1, 1), _DRNKWEK_test.reshape(-1, 1), HLTHPLN1_test.reshape(-1, 1), MEDCOST_test.reshape(-1, 1), GENHLTH_test.reshape(-1, 1), MENTHLTH_test.reshape(-1, 1), PHYSHLTH_test.reshape(-1, 1), DIFFWALK_test.reshape(-1, 1), SEX_test.reshape(-1, 1), _AGEG5YR_test.reshape(-1, 1), EDUCA_test.reshape(-1, 1), _INCOMG_test.reshape(-1, 1)))
    return X_train, X_test
    

def make_data(filename_train, filename_test, x_train, x_test, y_train, replace=False, onehotecode = False):
    """
    Extract features from the training and test data, and process them.
    Selects the relevant features from the data, processes NaN values, and standardizes the data.
    
    Parameters:
    -----------
        filename_train: str
            Name of the training CSV file.
        filename_test: str
            Name of the test CSV file.
        x_train: numpy.ndarray
            Training data.
        x_test: numpy.ndarray
            Test data.
        y_train: numpy.ndarray
            Training labels.
        replace: bool
            Whether to replace NaN values with the mean/mode of the feature.
        onhotecode: bool
            Whether to one-hot-encode some of the categorical features
    
    Returns:
    --------
        X_train: numpy.ndarray
            Processed training data.
        Y_train: numpy.ndarray
            Processed training labels.
        X_val: numpy.ndarray
            Processed validation data.
        Y_val: numpy.ndarray
            Processed validation labels.
        X_test: numpy.ndarray
            Processed test data.
    """
    # First extract all of the relevant features from the training data
    X_train, X_test = extract_features(filename_train, filename_test, x_train, x_test, onehotecode)
    
    # Change all the elements with -1 by 0
    y_train_working = y_train.copy()
    y_train_working[y_train_working == -1] = 0
    # Make y have the correct shape
    y_train_working = y_train_working.reshape(-1, 1)

    # Shuffle the data
    np.random.seed(6)
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_shuffled = X_train[indices]
    y_train_working_shuffled = y_train_working[indices]

    # Split the data into training and validation sets (90% training, 10% validation)
    X_train, X_val, Y_train, Y_val = split_train_val(X_shuffled, y_train_working_shuffled, 10, 9)
    
    means_ids = [0, 8, 12, 13] # IDs of features that should be replaced with the mean instead of the mode
    if replace:
        # Replace the missing values with the mean or mode of the feature, depending on the feature
        # Mean for continuous features and mode for others
        # Do it for train and validation sets. For validation set, we replace by the mean/mode of the train set
        for i in range(X_train.shape[1]):
            if i in means_ids:
                X_train[:, i] = replace_mean(X_train[:, i])
                mean_value = np.nanmean(X_train[:, i])
                std_value = np.nanstd(X_train[:, i])
                # Standardize the data
                X_train[:, i] = (X_train[:, i] - mean_value) / std_value

                # Replace the missing values in the validation set with the mean of the feature in the train set and standardize
                X_val[:, i] = replace_mean(X_val[:, i], mean_value=mean_value)
                X_val[:, i] = (X_val[:, i] - mean_value) / std_value
                
                # Replace the missing values in the test set with the mean of the feature in the train set and standardize
                X_test[:, i] = replace_mean(X_test[:, i], mean_value=mean_value)
                X_test[:, i] = (X_test[:, i] - mean_value) / std_value
            else:
                X_train[:, i] = replace_mode(X_train[:, i])
                unique, counts = np.unique(X_train[:, i][~np.isnan(X_train[:, i])], return_counts=True)
                mode_value = unique[np.argmax(counts)]
                X_val[:, i] = replace_mode(X_val[:, i], mode_value=mode_value)
                X_test[:, i] = replace_mode(X_test[:, i], mode_value=mode_value)

    else:
        # Drop rows with NaN values in the training set
        X_train, Y_train = drop_nan(X_train, Y_train)

        # In validation set replace NaN values with the mean or mode of the train set
        for i in range(X_val.shape[1]):
            if i in means_ids:
                mean_value = np.nanmean(X_train[:, i])
                std_value = np.nanstd(X_train[:, i])

                X_val[:, i] = replace_mean(X_val[:, i], mean_value=mean_value)
                X_val[:, i] = (X_val[:, i] - mean_value) / std_value

                X_test[:, i] = replace_mean(X_test[:, i], mean_value=mean_value)
                X_test[:, i] = (X_test[:, i] - mean_value) / std_value
            else:
                unique, counts = np.unique(X_train[:, i][~np.isnan(X_train[:, i])], return_counts=True)
                mode_value = unique[np.argmax(counts)]
                X_val[:, i] = replace_mode(X_val[:, i], mode_value=mode_value)
                X_test[:, i] = replace_mode(X_test[:, i], mode_value=mode_value)

    if not onehotecode:
        # Standardize the data
        X_train, X_train_mean, X_train_std = standardize(X_train)
        # We standardize the validation set with the mean/std of the train set 
        X_val, _, _ = standardize(X_val, X_train_mean, X_train_std)
        X_test, _, _ = standardize(X_test, X_train_mean, X_train_std)

    
    return X_train, Y_train, X_val, Y_val, X_test


def undersampling(X, y):
    """
    Balance the data by taking only a certain number of values in y=0 so that the number of y=1 equals the number of y=0.
    
    Parameters:
    -----
        X: numpy.ndarray
            Feature data.
        y: numpy.ndarray
            Labels.
    
    Returns:
    --------
        selected_X: numpy.ndarray
            Balanced feature data.
        selected_y: numpy.ndarray
            Balanced labels.
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
    """
    Balance the data by duplicating the values in y=1 so that the number of y=1 equals the number of y=0.
    
    Args:
    -----
        X: numpy.ndarray
            Feature data.
        y: numpy.ndarray
            Labels.
    
    Returns:
    --------
        X_balanced: numpy.ndarray
            Balanced feature data.
        y_balanced: numpy.ndarray
            Balanced labels.
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
    """
    Balance the data by undersampling the majority class and oversampling the minority class.
    
    Args:
    -----
        X: numpy.ndarray
            Feature data.
        y: numpy.ndarray
            Labels.
        ratio_majority: float
            Desired factor with which to undersample the majority class.
            (ratio_majority = 0.5 => half of the points will be dropped in the majority class).
        ratio_majority_to_minority: float
            Desired ratio of majority class to minority class data to oversample the minority class.
            (ratio_majority_to_minority = 2 => number of points in the majority class will be set as two times the number of points in the minority class).
    
    Returns:
    --------
        X_balanced: numpy.ndarray
            Balanced feature data.
        y_balanced: numpy.ndarray
            Balanced labels.
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
    indices = np.arange(X_balanced.shape[0])
    np.random.shuffle(indices)
    X_balanced = X_balanced[indices]
    y_balanced = y_balanced[indices].reshape(-1, 1)
    
    return X_balanced, y_balanced

def prediction(tx_test, w):
    """
    Predict binary labels for the test data using logistic regression.

    Parameters:
    -----
        tx_test: numpy.ndarray
            The feature matrix for the test data, with shape (n_samples, n_features).
        w: numpy.ndarray
            The weights of the logistic regression model, with shape (n_features,).

    Returns:
    --------
        y_test: numpy.ndarray
            The predicted binary labels for the test data, with shape (n_samples,).
    """
    compute = sigmoid(np.dot(tx_test, w))
    y_test = (compute >= 0.5).astype(int)
    return y_test
    
def compute_accuracy(true_labels, predicted_labels):
    """
    Calculate the percentage of correctly predicted labels.

    Parameters:
    -----
        true_labels: numpy.ndarray
            The true binary labels, with shape (n_samples,).
        predicted_labels: numpy.ndarray
            The predicted binary labels, with shape (n_samples,).

    Returns:
    --------
        percentage_right: float
            The percentage of correctly predicted labels.
    
    Raises:
    -------
        ValueError: If the lengths of true_labels and predicted_labels do not match.
    """
    # Check if both vectors have the same length
    if len(true_labels) != len(predicted_labels):
        raise ValueError("The two vectors must have the same length.")
    # Calculate the number of wrongly predicted points
    num_right = np.sum(true_labels == predicted_labels)
    # Calculate the percentage of wrongly predicted points
    percentage_right = (num_right / len(true_labels)) * 100
    return percentage_right


def f1(y_pred, y_true):
    """
    Calculate the F1 score, a measure of a test's accuracy that c
    onsiders both precision and recall.

    Args:
    -----
        y_pred: numpy.ndarray
            The predicted binary labels, with shape (n_samples,).
        y_true: numpy.ndarray
            The true binary labels, with shape (n_samples,).

    Returns:
    --------
        f1: float
            The F1 score, a value between 0 and 1.
    """
    tp = np.sum(y_pred[y_true == 1] == 1)
    fp = np.sum(y_pred[y_true == 0] == 1)
    fn = np.sum(y_pred[y_true == 1] == 0)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if (precision + recall) == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def drop_rows_with_nan(dataset,y, threshold=0.6):
    # Calculate the number of columns in the dataset
    num_cols = dataset.shape[1]
    
    # Calculate the threshold for the number of NaN values
    nan_threshold = threshold * num_cols
    
    # Create a mask where the number of NaNs in each row is less than or equal to the threshold
    non_nan_rows = np.sum(np.isnan(dataset), axis=1) <= nan_threshold
    
    # Return the filtered dataset
    return dataset[non_nan_rows], y[non_nan_rows]


def process_datasets(x_train, x_val, x_test, unique_values_thresh=10):
    """
    Processes the datasets by handling categorical and numerical columns and removing columns with a threshold of unique value.
    
    Parameters:
    -----------
    x_train : numpy.ndarray
        The training dataset to process, of shape (n_samples, n_features).
    x_val : numpy.ndarray
        The val dataset to process, of shape (n_samples, n_features).
    x_test : numpy.ndarray
        The test dataset to process, of shape (n_samples, n_features).
        
    Returns:
    --------
    numpy.ndarray
        The processed datasets (train/val/test) with categorical columns replaced by mode and numerical columns with NaNs replaced by the mean.
        Columns with a small number of values are removed for both the x_train, the x_val, the x_test
    """
    processed_x_train = []
    processed_x_val = []
    processed_x_test = []
    threshold = 1e-6

    # Iterate over each column
    for i in range(x_train.shape[1]):
        col_train = x_train[:, i]
        col_val = x_val[:,i]
        col_test = x_test[:,i]
        unique_values_train = np.unique(col_train[~np.isnan(col_train)])  # Get unique non-NaN values of train

        if len(unique_values_train) < 2:
            # If there's only one unique value in either x_train, skip this column (remove it)
            continue
            
        elif len(unique_values_train) <= unique_values_thresh:
            # If the column has a very low variance we just drop the column because this column doesn't give us any information and 
            # we might get a 0 variance whenever we standardize which is very annoying
            if np.nanstd(col_train) < threshold:
                continue
            else:
                # Categorical column: Replace NaNs with the mode
                # Use np.unique with return_counts to find the mode
                values, counts = np.unique(col_train[~np.isnan(col_train)], return_counts=True)
                mode = values[np.argmax(counts)]
                col_train[np.isnan(col_train)] = mode
                # we also replace the NaNs of the test set with the mode of the train set
                col_val[np.isnan(col_val)] = mode
                col_test[np.isnan(col_test)] = mode

                # Apply one-hot encoding
                col_train_encoded, col_test_encoded, col_val_encoded = one_hot_encode(col_train, col_test, tx_val=col_val)

                # Append the encoded columns to the result
                processed_x_train.append(col_train_encoded)
                processed_x_val.append(col_val_encoded)
                processed_x_test.append(col_test_encoded)

        else:
            # Numerical column: Replace NaNs with the mean
            mean = np.nanmean(col_train)
            col_train[np.isnan(col_train)] = mean
            col_val[np.isnan(col_val)] = mean
            col_test[np.isnan(col_test)] = mean

            # Standardize the column
            std = np.nanstd(col_train)
            col_train = (col_train - mean) / std
            col_val = (col_val - mean) / std
            col_test = (col_test - mean) / std

            # Append the processed column to the result
            processed_x_train.append(col_train.reshape(-1, 1))
            processed_x_val.append(col_val.reshape(-1, 1))
            processed_x_test.append(col_test.reshape(-1, 1))

    # Convert lists back to a NumPy array
    processed_x_train = np.hstack(processed_x_train)
    processed_x_val = np.hstack(processed_x_val)
    processed_x_test = np.hstack(processed_x_test)
    
    return processed_x_train, processed_x_val, processed_x_test