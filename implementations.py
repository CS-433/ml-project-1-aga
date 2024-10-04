import numpy as np
import os

#--------------------------------------------------Linear gradient--------------

def compute_loss_linear(y, tx, w):
    y_predicted = tx.dot(w)
    return 1/2 * np.mean((y - y_predicted) ** 2)

def compute_gradient(y, tx, w):
    N = y.shape[0]
    return -1/N * tx.T.dot(y - tx.dot(w))

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Performs gradient descent to minimize the mean squared error (MSE) loss.

    Parameters:
    -----------
    y : numpy.ndarray
        The target values, of shape (n_samples,).
    
    tx : numpy.ndarray
        The feature matrix, of shape (n_samples, n_features).
    
    initial_w : numpy.ndarray
        The initial weights, of shape (n_features,).
    
    max_iters : int
        The number of iterations to run gradient descent.
    
    gamma : float
        The learning rate or step size.

    Returns:
    --------
    w : numpy.ndarray
        The final weights after gradient descent, of shape (n_features,).

    loss : numpy.ndarray
        The final loss value (MSE) after gradient descent, with shape (0,).
    """
    w = initial_w
    loss = compute_loss_linear(y, tx, w)
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient
        loss = compute_loss_linear(y, tx, w)
    return w, np.array(loss)

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Performs stochastic gradient descent (SGD) to minimize the mean squared error (MSE) loss.
    In each iteration, a random one-dimentional sample from the dataset is used to compute the gradient and update the weights.

    Parameters:
    -----------
    y : numpy.ndarray
        The target values, of shape (n_samples,).
    
    tx : numpy.ndarray
        The feature matrix, of shape (n_samples, n_features).
    
    initial_w : numpy.ndarray
        The initial weights, of shape (n_features,).
    
    max_iters : int
        The number of iterations to run stochastic gradient descent.
    
    gamma : float
        The learning rate or step size.

    Returns:
    --------
    w : numpy.ndarray
        The final weights after stochastic gradient descent, of shape (n_features,).

    loss : numpy.ndarray
        The final loss value (MSE) after stochastic gradient descent, with shape (0,).
    """
    w = initial_w
    loss = compute_loss_linear(y, tx, w)
    for n_iter in range(max_iters):
        random_idx = np.random.randint(0, len(y))
        y_single = np.array([y[random_idx]])
        tx_single = tx[random_idx, :].reshape(1, -1)
        gradient = compute_gradient(y_single, tx_single, w)
        w = w - gamma * gradient
        loss = compute_loss_linear(y, tx, w)
    return w, np.array(loss)

#----------------------------------------------------Least square------------------------------------

def least_squares(y, tx):
    """
    Performs least squares regression to find the optimal weights.

    Parameters:
    -----------
    y : numpy.ndarray
        The target values, of shape (n_samples,).
    
    tx : numpy.ndarray
        The feature matrix, of shape (n_samples, n_features).

    Returns:
    --------
    w : numpy.ndarray
        The optimal weights that minimize the least squares loss, of shape (n_features,).

    loss : numpy.ndarray
        The final loss value (MSE) associated with the optimal weights, with shape (0,).
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss_linear(y, tx, w)
    return w, np.array(loss)

def ridge_regression(y, tx, lambda_):
    """
    Performs least square with added ridge term to find the optimal weights with L2 regularization.

    Parameters:
    -----------
    y : numpy.ndarray
        The target values, of shape (n_samples,).
    
    tx : numpy.ndarray
        The feature matrix, of shape (n_samples, n_features).
    
    lambda_ : float
        The regularization parameter (lambda) to control the strength of regularization.

    Returns:
    --------
    w : numpy.ndarray
        The optimal weights that minimize the ridge regression loss, of shape (n_features,).

    loss : numpy.ndarray
        The final loss value (MSE) associated with the optimal weights, with shape (0,).
    """
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss_linear(y, tx, w)
    return w, np.array(loss)
    
#----------------------------------------------------logistic---------------------------------------------
def sigmoid(t):
    return np.exp(t) / (1 + np.exp(t))

def calculate_loss_logistic(y, tx, w):
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(-loss).item() * (1 / y.shape[0])

def calculate_gradient(y, tx, w):
    pred = sigmoid(tx.dot(w))
    ret = (1/y.shape[0])*tx.T.dot(pred-y)
    return ret

def learning_by_gradient_descent_ridge(y, tx, w, gamma, lambda_):
    gradient = calculate_gradient(y, tx, w) + 2 * lambda_ * w
    w_new = w - gamma * gradient
    loss = calculate_loss_logistic(y, tx, w_new)
    return loss, w_new

def learning_by_gradient_descent(y, tx, w, gamma):
    gradient = calculate_gradient(y, tx, w)
    w_new = w - gamma * gradient
    loss = calculate_loss_logistic(y, tx, w_new)
    return loss, w_new

def logistic_regression(y, x, initial_w, max_iters, gamma):
    """
    Performs logistic regression using gradient descent.

    Parameters:
    -----------
    y : numpy.ndarray
        The target binary values (0 or 1), of shape (n_samples,).
    
    x : numpy.ndarray
        The feature matrix, of shape (n_samples, n_features).
    
    initial_w : numpy.ndarray
        The initial weights for the logistic regression model, of shape (n_features,).
    
    max_iters : int
        The maximum number of iterations for the gradient descent algorithm.
    
    gamma : float
        The learning rate or step size for weight updates.

    Returns:
    --------
    w : numpy.ndarray
        The final weights after training, of shape (n_features,).
    
    loss : numpy.ndarray
        The final loss value associated with the optimal weights, with shape (0,).
    """
    # init parameters
    threshold = 1e-8
    losses = []

    y_shaped = y[:, np.newaxis]
    w_shaped = initial_w[:, np.newaxis]
    loss = calculate_loss_logistic(y_shaped, x, w_shaped)

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w_shaped = learning_by_gradient_descent(y_shaped, x, w_shaped, gamma)
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w_shaped.ravel(), np.array(loss) 


def reg_logistic_regression(y, x, lambda_, initial_w, max_iters, gamma):
    """
    Performs logistic regression with L2 regularization using gradient descent.

    Parameters:
    -----------
    y : numpy.ndarray
        The target binary values (0 or 1), of shape (n_samples,).
    
    x : numpy.ndarray
        The feature matrix, of shape (n_samples, n_features).
    
    lambda_ : float
        The regularization parameter to control the amount of L2 regularization.
    
    initial_w : numpy.ndarray
        The initial weights for the logistic regression model, of shape (n_features,).
    
    max_iters : int
        The maximum number of iterations for the gradient descent algorithm.
    
    gamma : float
        The learning rate or step size for weight updates.

    Returns:
    --------
    w : numpy.ndarray
        The final weights after training with L2 regularization, of shape (n_features,).
    
    loss : numpy.ndarray
        The final loss value associated with the optimal weights, with shape (0,).
    """
     # init parameters
    threshold = 1e-8
    losses = []

    y_shaped = y[:, np.newaxis]
    w_shaped = initial_w[:, np.newaxis]
    loss = calculate_loss_logistic(y_shaped, x, w_shaped)

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w_shaped = learning_by_gradient_descent_ridge(y_shaped, x, w_shaped, gamma, lambda_)
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w_shaped.ravel(), np.array(loss)
















