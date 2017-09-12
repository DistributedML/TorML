from __future__ import division
from numpy.linalg import norm
import numpy as np
import utils
import pdb

lammy = 0.1
verbose = 1
X = 0
y = 0
iteration = 1
alpha = 1e-2
d = 0
hist_grad = 0

class logRegL2:
    
    # Logistic Regression
    def __init__(self, X, y, lammy, verbose=1, maxEvals=100):
        self.lammy = lammy
        self.verbose = verbose
        self.maxEvals = maxEvals

        self.X = X
        self.y = y

        self.iter = 1
        self.init_alpha = 1e-2
        self.alpha = self.init_alpha

        n, self.d = self.X.shape
        self.w = np.zeros(self.d)
        self.hist_grad = np.zeros(self.d)

    def funObj(self, w, X, y):

        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    # Reports the direct change to w, based on the given one.
    # Batch size could be 1 for SGD, or 0 for full gradient.
    def privateFun(self, theta, ww, batch_size=0):

        # Define constants and params
        nn, dd = self.X.shape
        threshold = int(self.d * theta)
        self.iter = self.iter + 1

        if batch_size > 0 and batch_size < nn:
            idx = np.random.choice(nn, batch_size, replace=False)
        else:
            # Just take the full range
            idx = range(nn)

        f, g = self.funObj(ww, self.X[idx, :], self.y[idx])

        # AdaGrad
        self.hist_grad += g**2
        ada_grad = g / (1e-6 + np.sqrt(self.hist_grad))

        # Determine the actual step magnitude
        delta = -self.init_alpha * ada_grad

        # Weird way to get NON top k values
        if theta < 1:
            param_filter = np.argpartition(
                abs(delta), -threshold)[:self.d - threshold]
            delta[param_filter] = 0

        w_new = ww + delta
        f_new, g_new = self.funObj(w_new, self.X[idx, :], self.y[idx])

        return (delta, f_new, g_new)

def init(dataset):

    data = utils.load_dataset(dataset)

    global X
    X = data['X']

    global y
    y = data['y']

    global d
    d = X.shape[1]

    global hist_grad
    hist_grad = np.zeros(d)

    return d


def funObj(ww, X, y):
    yXw = y * X.dot(ww)

    # Calculate the function value
    f = np.sum(np.logaddexp(0, -yXw)) + 0.5 * lammy * ww.T.dot(ww)

    # Calculate the gradient value
    res = - y / np.exp(np.logaddexp(0, yXw))
    g = X.T.dot(res) + lammy * ww

    return f, g


# Reports the direct change to w, based on the given one.
# Batch size could be 1 for SGD, or 0 for full gradient.
def privateFun(theta, ww, batch_size=0):

    global iteration
    ww = np.array(ww)

    # Define constants and params
    nn, dd = X.shape
    threshold = int(d * theta)

    if batch_size > 0 and batch_size < nn:
        idx = np.random.choice(nn, batch_size, replace=False)
    else:
        # Just take the full range
        idx = range(nn)

    f, g = funObj(ww, X[idx, :], y[idx])

    # AdaGrad
    global hist_grad
    hist_grad += g**2

    ada_grad = g / (1e-6 + np.sqrt(hist_grad))

    # Determine the actual step magnitude
    delta = -alpha * ada_grad

    # Weird way to get NON top k values
    if theta < 1:
        param_filter = np.argpartition(
            abs(delta), -threshold)[:d - threshold]
        delta[param_filter] = 0

    w_new = ww + delta
    f_new, g_new = funObj(w_new, X[idx, :], y[idx])
    iteration = iteration + 1

    return delta
