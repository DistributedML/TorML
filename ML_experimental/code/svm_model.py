from __future__ import division
import numpy as np
import math
import minimizers
import utils
import pdb
from numpy.linalg import norm

class hingeSVM:
    # Logistic Regression
    def __init__(self, X, y, verbose=1, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.X = X
        self.y = y
        self.init_alpha = 1e-2
        self.alpha = self.init_alpha
        self.iter = 1

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
        delta = -self.alpha * ada_grad

        # Weird way to get NON top k values
        if theta < 1:
            param_filter = np.argpartition(
                abs(delta), -threshold)[:self.d - threshold]
            delta[param_filter] = 0

        w_new = ww + delta
        f_new, g_new = self.funObj(w_new, self.X[idx, :], self.y[idx])

        return (delta, f_new, g_new)

    def sgd_fit(self, theta, batch_size=0, *args):

        print "Training model."

        # Parameters of the Optimization
        optTol = 1e-4
        i = 0
        n, d = self.X.shape

        # Initial guess
        self.w = np.zeros(d)
        funEvals = 1

        while True:

            (delta, f_new, g) = self.privateFun(
                theta, self.w, batch_size, *args)
            funEvals += 1

            # Print progress
            if self.verbose > 0:
                print("%d - loss: %.3f" % (funEvals, f_new))
                print("%d - g_norm: %.3f" % (funEvals, norm(g)))

            # Update parameters
            self.w = self.w + delta

            # Test termination conditions
            optCond = norm(g, float('inf'))

            if optCond < optTol:
                if self.verbose:
                    print("Problem solved up to optimality tolerance %.3f" % optTol)
                break

            if funEvals >= self.maxEvals:
                if self.verbose:
                    print("Reached maximum number of function evaluations %d" %
                          self.maxEvals)
                break

        print "Done fitting."

    def fit(self):

        (self.w, self.alpha, f, _) = minimizers.findMin(self.funObj, self.w, self.alpha,
                                                        self.maxEvals,
                                                        self.verbose,
                                                        self.X,
                                                        self.y)

        print("Training error: %.3f" %
              utils.classification_error(self.predict(self.X), self.y))

    def getParameters(self):
        return self.w

    def predict(self, X):
        w = self.w
        yhat = np.dot(X, w)
        return np.sign(yhat)

    '''
    Original model
    def privatePredict(self, X, scale):
        _, d = X.shape
        w = self.w + utils.exp_noise(scale=scale, size=d)
        yhat = np.dot(X, w)
        return np.sign(yhat)
    '''

    def privatePredict(self, X, epsilon):
        nn, dd = X.shape
        w = self.w
        yhat = np.dot(X, w)

        # TODO: Estimate the L1 Sensitivity
        sens = 0.25 * dd * dd + 3 * dd

        return np.sign(yhat + utils.lap_noise(loc=0, scale=sens / epsilon, size=nn))


class logRegL2(logReg):

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

        #utils.check_gradient(self, self.X, self.y)

    def funObj(self, ww, X, y):
        yXw = y * X.dot(ww)

        # Calculate the function value
        f = np.sum(np.logaddexp(0, -yXw)) + 0.5 * self.lammy * ww.T.dot(ww)

        # Calculate the gradient value
        res = - y / np.exp(np.logaddexp(0, yXw))
        g = X.T.dot(res) + self.lammy * ww

        return f, g

    def privatePredict(self, X, epsilon):
        nn, dd = X.shape
        w = self.w
        yhat = np.dot(X, w)

        # TODO: Estimate the L1 Sensitivity
        # sens = 0.25 * dd * dd + 3 * dd
        sens = 2 / (nn * self.lammy)

        return np.sign(yhat + utils.lap_noise(loc=0, scale=sens / epsilon, size=nn))