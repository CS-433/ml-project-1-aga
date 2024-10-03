import numpy as np
import os

def sigmoid(t):
    return np.exp(t) / (1 + np.exp(t))


def compute_loss_linear(y, tx, w):
    y_predicted = tx.dot(w)
    return 1/2 * np.mean((y - y_predicted) ** 2)

def compute_gradient(y, tx, w):
    N = y.shape[0]
    return -1/N * tx.T.dot(y - tx.dot(w))

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    loss = 0
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss_linear(y, tx, w)
        w = w - gamma * gradient
    return loss, w

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    return 2

def calculate_loss_logistic(y, tx, w):
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
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

def learning_by_gradient_descent_ridge(y, tx, w, gamma, lambda_):
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
    loss = calculate_loss_logistic(y, tx, w)
    gradient = calculate_gradient(y, tx, w) + 2 * lambda_ * w
    w_new = w - gamma * gradient
    return loss, w_new

def reg_logistic_regression(y, x, lambda_, initial_w, max_iters, gamma):
     # init parameters
    threshold = 1e-7
    losses = []

    # build tx
    w = np.zeros((x.shape[1], 1))
    y_shaped = y[:, np.newaxis]

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_gradient_descent_ridge(y_shaped, x, w, gamma, lambda_)
        # converge criterion
        print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
        print(w)
    return w, loss
















