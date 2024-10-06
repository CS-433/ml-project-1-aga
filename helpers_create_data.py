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

def one_hot_encode(tx):
    """
    Perform one-hot encoding on a categorical feature array.
    
    Parameters:
    -----
    tx : numpy.ndarray
        A 1D array of categorical feature data to be one-hot encoded.
    
    Returns:
    --------
    tx_one_hot : numpy.ndarray
        A 2D array representing the one-hot encoded version of the input data.
        Each unique category in `tx` is represented by a separate column, 
        except for the 'nan' value, that is given up on.
    """
    unique_values = np.unique(tx)
    n_unique = unique_values.size
    tx_one_hot = np.zeros((tx.shape[0], n_unique-1))
    
    for i in range(tx.shape[0]):
        tx_one_hot[i, np.where(unique_values == tx[i])[0]] = 1

    return tx_one_hot

def extract_features(filename, x, onehotencode):
    """
    Extracts and preprocesses a set of features from the provided data.
    
    Parameters:
    -----------
    filename : str
        Name of the CSV file from which the data is extracted
        
    x : array-like
        The data matrix or feature set from which individual features are extracted.
        
    onehotencode : bool
        If True, certain categorical features will be one-hot encoded.

    Returns:
    --------
    X : numpy.ndarray
        A 2D array of extracted and processed features, where rows correspond to samples 
        and columns correspond to different features. The shape depends on the number of 
        features and whether one-hot encoding is applied.
    """

    #Here we manually extract all the features. 
    #We replace the values corresponding to missig data / people not willing to answer by nans 
    #that we will drop or replace by mean/mode later
    
    ################Body mass idex - continuous feature
    _BMI5 = extract_feature('_BMI5', filename, x)

    ################High blood pressure - categorical feature (1 = no, 2 = yes, 9 = missing)
    _RFHYPE5 = extract_feature('_RFHYPE5', filename, x)
    _RFHYPE5[_RFHYPE5 == 9] = np.nan
    _RFHYPE5[_RFHYPE5 == 1] = 0
    _RFHYPE5[_RFHYPE5 == 2] = 1

    ################High cholesterol - categorical feature (1 = no, 2 = yes, 9 = missing)
    _RFCHOL = extract_feature('_RFCHOL', filename, x)
    _RFCHOL[_RFCHOL == 9] = np.nan
    _RFCHOL[_RFCHOL == 1] = 0
    _RFCHOL[_RFCHOL == 2] = 1        

    ################Smoking status - categorical feature (1 = every day, 2 = some days, 3 = formerly, 4 = never, 9 = missing)
    _SMOKER3 = extract_feature('_SMOKER3', filename, x)
    _SMOKER3[_SMOKER3 == 9] = np.nan 
    #Here we have 3 or more categories -> hot encoding makes sense
    if onehotencode: 
        _SMOKER3 = one_hot_encode(_SMOKER3)

    ################Has ever had a stroke  - categorical feature (1 = yes, 2 = no, 7 = don't know, 9 = missing)
    CVDSTRK3 = extract_feature('CVDSTRK3', filename, x)
    CVDSTRK3[CVDSTRK3 == 9] = np.nan
    CVDSTRK3[CVDSTRK3 == 7] = np.nan
    CVDSTRK3[CVDSTRK3 == 2] = 0        

    ################Cholesterol checked  - categorical feature (1 = within the last 5 years, 2 = more than 5 years ago, 3 = never, 9 = missing)
    _CHOLCHK = extract_feature('_CHOLCHK', filename, x)
    _CHOLCHK[_CHOLCHK == 9] = np.nan        

    ################Has ever had diabetes  - categorical feature (1 = yes, 2 = yes*, 3 = no, 4 = no - pre-diabetes, 7 = don't know, 9 = missing)
    DIABETE3 = extract_feature('DIABETE3', filename, x)
    DIABETE3[DIABETE3 == 9] = np.nan
    DIABETE3[DIABETE3 == 7] = np.nan
    DIABETE3[DIABETE3 == 3] = 0
    DIABETE3[DIABETE3 == 4] = 0
    DIABETE3[DIABETE3 == 2] = 1        

    ################Physical activity index  - categorical feature (1 = highly active, 2 = active, 3 = insufficiently active, 4 = inactive, 9 = missing)
    _PACAT1 = extract_feature('_PACAT1', filename, x)
    _PACAT1[_PACAT1 == 9] = np.nan
    #Here we have 3 or more categories -> hot encoding makes sense
    if onehotencode: 
        _PACAT1 = one_hot_encode(_PACAT1)

    ################Computed number of drinks of alcohol beverages per week  - continuous feature (99900 = missing)
    _DRNKWEK = extract_feature('_DRNKWEK', filename, x)
    _DRNKWEK[_DRNKWEK == 99900] = np.nan        

    ################Have any healthcare coverage  - categorical feature (1 = yes, 2 = no, 7 = don't know, 9 = missing)
    HLTHPLN1 = extract_feature('HLTHPLN1', filename, x)
    HLTHPLN1[HLTHPLN1 == 9] = np.nan
    HLTHPLN1[HLTHPLN1 == 7] = np.nan
    HLTHPLN1[HLTHPLN1 == 2] = 0        

    ################Could not see doctor because of cost  - categorical feature (1 = yes, 2 = no, 7 = don't know, 9 = missing)
    MEDCOST = extract_feature('MEDCOST', filename, x)
    MEDCOST[MEDCOST == 9] = np.nan
    MEDCOST[MEDCOST == 7] = np.nan
    MEDCOST[MEDCOST == 2] = 0        

    ################General health status  - categorical feature (1 = excellent, 2 = very good, 3 = good, 4 = fair, 5 = poor, 7 = don't know, 9 = missing)
    GENHLTH = extract_feature('GENHLTH', filename, x)
    GENHLTH[GENHLTH == 9] = np.nan
    GENHLTH[GENHLTH == 7] = np.nan 
    #Here we have 3 or more categories -> hot encoding makes sense
    if onehotencode: 
        GENHLTH = one_hot_encode(GENHLTH)

    ################Number of days mental health not good  - continuous feature (88 = none, 77 = don't know, 99 = refused)
    MENTHLTH = extract_feature('MENTHLTH', filename, x)
    MENTHLTH[MENTHLTH == 88] = 0
    MENTHLTH[MENTHLTH == 77] = np.nan
    MENTHLTH[MENTHLTH == 99] = np.nan        

    ################Number of days physical health not good  - continuous feature (88 = none, 77 = don't know, 99 = refused)
    PHYSHLTH = extract_feature('PHYSHLTH', filename, x)
    PHYSHLTH[PHYSHLTH == 88] = 0
    PHYSHLTH[PHYSHLTH == 77] = np.nan
    PHYSHLTH[PHYSHLTH == 99] = np.nan        

    ################Difficulty walking or climbing stairs - categorical feature (1 = yes, 2 = no, 7 = don't know, 9 = missing)
    DIFFWALK = extract_feature('DIFFWALK', filename, x)
    DIFFWALK[DIFFWALK == 9] = np.nan
    DIFFWALK[DIFFWALK == 7] = np.nan
    DIFFWALK[DIFFWALK == 2] = 0        

    ################Sex - categorical feature (1 = male, 2 = female)
    SEX = extract_feature('SEX', filename, x)
    SEX[SEX == 2] = 0        

    ################Age  - categorical feature (1 = 18-24, ... 13 = 80+, 14 = missing)
    _AGEG5YR = extract_feature('_AGEG5YR', filename, x)
    _AGEG5YR[_AGEG5YR == 14] = np.nan 
    #Here we have 3 or more categories -> hot encoding makes sense
    if onehotencode: 
        _AGEG5YR = one_hot_encode(_AGEG5YR)

    ################Education  - categorical feature (1 = none, ... 6 = college grad, 9 = missing)
    EDUCA = extract_feature('EDUCA', filename, x)
    EDUCA[EDUCA == 9] = np.nan  
    #Here we have 3 or more categories -> hot encoding makes sense
    if onehotencode: 
        EDUCA = one_hot_encode(EDUCA)

    ################Income level  - categorical feature (1 = low, ... 5 = high, 9 = missing)
    _INCOMG = extract_feature('_INCOMG', filename, x)
    _INCOMG[_INCOMG == 9] = np.nan 
    #Here we have 3 or more categories -> hot encoding makes sense
    if onehotencode: 
        _INCOMG = one_hot_encode(_INCOMG)

    X = []
    #Here we stack the features together (depending if we want to hot one encode or not) to have the our new X
    if onehotencode:
        X = np.hstack((_BMI5.reshape(-1, 1), _RFHYPE5.reshape(-1, 1), _RFCHOL.reshape(-1, 1), _SMOKER3, CVDSTRK3.reshape(-1, 1), _CHOLCHK.reshape(-1, 1), DIABETE3.reshape(-1, 1), _PACAT1, _DRNKWEK.reshape(-1, 1), HLTHPLN1.reshape(-1, 1), MEDCOST.reshape(-1, 1), GENHLTH, MENTHLTH.reshape(-1, 1), PHYSHLTH.reshape(-1, 1), DIFFWALK.reshape(-1, 1), SEX.reshape(-1, 1), _AGEG5YR, EDUCA, _INCOMG))

    else:
        X = np.hstack((_BMI5.reshape(-1, 1), _RFHYPE5.reshape(-1, 1), _RFCHOL.reshape(-1, 1), _SMOKER3.reshape(-1, 1), CVDSTRK3.reshape(-1, 1), _CHOLCHK.reshape(-1, 1), DIABETE3.reshape(-1, 1), _PACAT1.reshape(-1, 1), _DRNKWEK.reshape(-1, 1), HLTHPLN1.reshape(-1, 1), MEDCOST.reshape(-1, 1), GENHLTH.reshape(-1, 1), MENTHLTH.reshape(-1, 1), PHYSHLTH.reshape(-1, 1), DIFFWALK.reshape(-1, 1), SEX.reshape(-1, 1), _AGEG5YR.reshape(-1, 1), EDUCA.reshape(-1, 1), _INCOMG.reshape(-1, 1)))

    return X

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
    X = extract_features(filename_train, x_train, onehotecode)
    
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
        # Mean for continuous features and mode for others
        # Do it for train and validation sets. For validation set, we replace by the mean/mode of the train set
        for i in range(X_train.shape[1]):
            if i in means_ids:
                X_train[:, i] = replace_mean(X_train[:, i])
                mean_value = np.nanmean(X_train[:, i])
                X_val[:, i] = replace_mean(X_val[:, i], mean_value=mean_value)
            else:
                X_train[:, i] = replace_mode(X_train[:, i])
                unique, counts = np.unique(X_train[:, i][~np.isnan(X_train[:, i])], return_counts=True)
                mode_value = unique[np.argmax(counts)]
                X_val[:, i] = replace_mode(X_val[:, i], mode_value=mode_value)
    else:
        # Drop rows with NaN values in the training set
        X_train, Y_train = drop_nan(X_train, Y_train)

        # In validation set replace NaN values with the mean or mode of the train set
        for i in range(X_val.shape[1]):
            if i in means_ids:
                mean_value = np.nanmean(X_train[:, i])
                X_val[:, i] = replace_mean(X_val[:, i], mean_value=mean_value)
            else:
                unique, counts = np.unique(X_train[:, i][~np.isnan(X_train[:, i])], return_counts=True)
                mode_value = unique[np.argmax(counts)]
                X_val[:, i] = replace_mode(X_val[:, i], mode_value=mode_value)

    # Standardize the data
    X_train, X_train_mean, X_train_std = standardize(X_train)
    # We strandardize the validation set with the mean/std of the train set 
    X_val, _, _ = standardize(X_val, X_train_mean, X_train_std)

    
    X_test = extract_features(filename_test, x_test, onehotecode)

    # We replace the missing values in the test set with the mean/mode (of the train set) of the feature, depending on the feature
    for i in range(X_test.shape[1]):
        if i in means_ids:
            mean_value = np.nanmean(X_train[:, i])
            X_test[:, i] = replace_mean(X_test[:, i], mean_value=mean_value)
        else:
            unique, counts = np.unique(X_train[:, i][~np.isnan(X_train[:, i])], return_counts=True)
            mode_value = unique[np.argmax(counts)]
            X_test[:, i] = replace_mode(X_test[:, i], mode_value=mode_value)
    
    # Standardize the test data (with mean/std of the train set)
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
    
def percentage_well_predicted(true_labels, predicted_labels):
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
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1