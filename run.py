import random
from datetime import datetime

import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import time

from helpers_create_data import *
from implementations import *

from helpers import load_csv_data
from helpers import create_csv_submission

#We import the data here
x_train, x_test, y_train, train_ids, test_ids = load_csv_data("./dataset", sub_sample=False)

# Change all the elements with -1 by 0
y_train_working = y_train.copy()
y_train_working[y_train_working == -1] = 0
# Make y have the correct shape
y_train_working = y_train_working.reshape(-1, 1)

# Shuffle the data
np.random.seed(6)
indices = np.arange(x_train.shape[0])
np.random.shuffle(indices)
X_shuffled = x_train[indices]
y_train_working_shuffled = y_train_working[indices]

# Split the data into training and validation sets (90% training, 10% validation)
X_train_all, X_val_all, Y_train_all, Y_val_all = split_train_val(X_shuffled, y_train_working_shuffled, 10, 9)

# we drop the rows that has a NaN percentage of {threshold} because we assume that they don't offer much information
X_tr_all, Y_tr_all = drop_rows_with_nan(X_train_all, Y_train_all, threshold=0.4)

# we process the dataset by replacing the remaining NaNs by column with the mode of the feature column that have less than 10 unique values and by its 
# mean if the feature column has more than 10 unique values. Also we remove the columns that have extremely low variance as this column 
# doesn't offer any information and we might encouter numerical issues when standardizing.
X_tr_all, X_val_all, X_test_all = process_datasets(X_tr_all, X_val_all, x_test, unique_values_thresh=10)

# We now balance the data to a slightly more balanced ratio of 0s and 1s
X_tr_all, Y_tr_all = undersampling_oversampling(X_tr_all, Y_tr_all, ratio_majority=0.5, ratio_majority_to_minority=2)

# We add a column of ones (bias term) to the dataset
X_tr_all = np.c_[np.ones((X_tr_all.shape[0], 1)), X_tr_all]
X_val_all = np.c_[np.ones((X_val_all.shape[0], 1)), X_val_all]
X_test_all = np.c_[np.ones((X_test_all.shape[0], 1)), X_test_all]

#Gamma = 0.05 was giving us good results while converging in acceptable time
gamma = 0.05
max_iter = 10000

#Reshape y_train form (#points,1) to (#points,) in order to use the implemented logistic regression function
Y_tr_all = Y_tr_all.reshape(-1)
#Create a new w in order to match the number of selected feature and has shape (1 + #features, )
w_reg = np.zeros((X_tr_all.shape[1], 1)).reshape(-1)
#Train model (-> our train set) using logistic regression
w, loss = logistic_regression(Y_tr_all, X_tr_all, w_reg, max_iter, gamma)

#y_pred_test are the predicted labels for the validation set
y_pred_test = prediction(X_val_all, w)
Y_val = Y_val_all.reshape(-1)
print('Accuracy:', compute_accuracy(Y_val, y_pred_test))
print('F1: ', f1(y_pred_test, Y_val))

#Here we actually make the prediction for the test set
y_pred = prediction(X_test_all, w)
y_pred[y_pred == 0] = -1

create_csv_submission(test_ids, y_pred, "Submission_27.10.2024_21_56")