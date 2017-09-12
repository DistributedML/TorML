from __future__ import division
import numpy as np
import minimizers
import utils
import pdb


class linReg:

    # Q3 - One-vs-all Least Squares for multi-class classification
    def __init__(self, X, y, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.X = X
        self.y = y
        self.alpha = 1e-2
        self.iter = 0

        n, self.d = self.X.shape
        self.w = np.zeros(self.d)
        self.hist_grad = np.zeros(self.d)

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

    def funObj(self, w, X, y):

        xwy = (X.dot(w) - y)
        f = 0.5 * xwy.T.dot(xwy)
        g = X.T.dot(xwy)

        return f, g

    def fit(self):

        (self.w, self.alpha, f, _) = minimizers.findMin(self.funObj, self.w,
                                                        self.alpha,
                                                        self.maxEvals,
                                                        self.verbose,
                                                        self.X,
                                                        self.y)

    def predict(self, X):
        w = self.w
        yhat = np.dot(X, w)
        return yhat

    def privatePredict(self, X, epsilon):
        nn, dd = X.shape
        w = self.w
        yhat = np.dot(X, w)

        # TODO: Estimate the L1 Sensitivity in a better way
        sens = (dd * dd + 2 * dd + 1) * 2

        y_private = yhat + \
            utils.lap_noise(loc=0, scale=sens / epsilon, size=nn)
        return y_private


class linRegL2(linReg):

    def __init__(self, X, y, lammy, verbose=0, maxEvals=100):
        self.lammy = lammy
        self.verbose = verbose
        self.maxEvals = maxEvals

        self.X = X
        self.y = y
        self.alpha = 1e-2
        self.iter = 0

        n, self.d = self.X.shape
        self.w = np.zeros(self.d)
        self.hist_grad = np.zeros(self.d)

    def funObj(self, ww, X, y):

        xwy = (X.dot(ww) - y)
        f = 0.5 * xwy.T.dot(xwy) + 0.5 * self.lammy * ww.T.dot(ww)
        g = X.T.dot(xwy) + self.lammy * ww

        return f, g
