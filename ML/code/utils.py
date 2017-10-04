from __future__ import division
import pickle
import os
import sys
import numpy as np
from numpy.linalg import norm
import pandas as pd
import pdb

def sliceup(numSplits, dataset):

    data = load_dataset(dataset)

    X, y = data['X'], data['y']
    Xvalid, yvalid = data['Xvalid'], data['yvalid']

    randseed = np.random.permutation(X.shape[0])
    X = X[randseed, :]
    y = y[randseed]

    numRows = int(X.shape[0] / numSplits)

    for i in range(numSplits):
        dataslice = np.hstack((X[(i * numRows):((i + 1) * numRows), :],
                        y[(i * numRows):((i + 1) * numRows)].reshape((numRows, 1))))
        np.savetxt("../data/" + dataset + str(i + 1) + ".csv", dataslice, delimiter=",")

    numTestRows = Xvalid.shape[0]
    datatest = np.hstack((Xvalid, yvalid.reshape((numTestRows, 1))))
    np.savetxt("../data/" + dataset + "test.csv", datatest, delimiter=",")


def load_dataset(dataset_name):

    # Load and standardize the data and add the bias term
    if dataset_name == "logisticData":
        # data = load_pkl(os.path.join('..', "data", 'logisticData.pkl'))
        # Made change here
        data = load_pkl(os.path.join('../ML', "data", 'logisticData.pkl'))

        X, y = data['X'], data['y']
        Xvalid, yvalid = data['Xvalidate'], data['yvalidate']

        n, _ = X.shape

        randseed = np.random.permutation(n)
        X = X[randseed, :]
        y = y[randseed]

        X, mu, sigma = standardize_cols(X)
        Xvalid, _, _ = standardize_cols(Xvalid, mu, sigma)

        X = np.hstack([np.ones((X.shape[0], 1)), X])
        Xvalid = np.hstack([np.ones((Xvalid.shape[0], 1)), Xvalid])

        return {"X": X, "y": y,
                "Xvalid": Xvalid,
                "yvalid": yvalid}

    elif dataset_name == "slices":

        slices = pd.read_csv(os.path.join(
            '../ML', "data", 'slice_localization_data.csv'))
        n, d = slices.shape

        npslices = slices.ix[np.random.permutation(n), :].as_matrix()
        split = int(n * 0.70)

        X = npslices[0:split, 0:d - 1]
        y = npslices[0:split, -1]

        Xvalid = npslices[(split + 1):n, 0:d - 1]
        yvalid = npslices[(split + 1):n, -1]

        X, mu, sigma = standardize_cols(X)
        Xvalid, _, _ = standardize_cols(Xvalid, mu, sigma)

        X = np.hstack([np.ones((X.shape[0], 1)), X])
        Xvalid = np.hstack([np.ones((Xvalid.shape[0], 1)), Xvalid])

        X = normalize_rows(X)
        Xvalid = normalize_rows(Xvalid)

        return {"X": X, "y": y,
                "Xvalid": Xvalid,
                "yvalid": yvalid}

    elif dataset_name == "magic":

        magic = pd.read_csv(os.path.join('..', "data", 'magic04.data'))
        nn, dd = magic.shape

        y = magic.ix[:, dd - 1].as_matrix()
        y[np.where(y == 'g')] = 1
        y[np.where(y == 'h')] = -1

        npmagic = magic.ix[np.random.permutation(
            nn), :].as_matrix().astype(int)
        split = int(nn * 0.70)

        X = npmagic[0:split - 1, 0:dd - 2]
        y = npmagic[0:split - 1, dd - 1]
        Xvalid = npmagic[split:nn - 1, 0:dd - 2]
        yvalid = npmagic[split:nn - 1, dd - 1]

        X, mu, sigma = standardize_cols(X)
        Xvalid, _, _ = standardize_cols(Xvalid, mu, sigma)

        X = np.hstack([np.ones((X.shape[0], 1)), X])
        Xvalid = np.hstack([np.ones((Xvalid.shape[0], 1)), Xvalid])

        X = normalize_rows(X)
        Xvalid = normalize_rows(Xvalid)

        return {"X": X, "y": y,
                "Xvalid": Xvalid,
                "yvalid": yvalid}

    elif dataset_name == "sns":

        sns = pd.read_csv(os.path.join('..', 'data', 'sns.txt'), sep="\t")
        nn, dd = sns.shape

        npsns = sns.ix[np.random.permutation(nn), :].as_matrix().astype(int)
        split = int(nn * 0.70)

        X = npsns[0:split - 1, 0:dd - 2]
        y = ((npsns[0:split - 1, dd - 1] - 1.5) * 2).astype(int)
        Xvalid = npsns[split:nn - 1, 0:dd - 2]
        yvalid = ((npsns[split:nn - 1, dd - 1] - 1.5) * 2).astype(int)

        X, mu, sigma = standardize_cols(X)
        Xvalid, _, _ = standardize_cols(Xvalid, mu, sigma)

        X = np.hstack([np.ones((X.shape[0], 1)), X])
        Xvalid = np.hstack([np.ones((Xvalid.shape[0], 1)), Xvalid])

        X = normalize_rows(X)
        Xvalid = normalize_rows(Xvalid)

        return {"X": X, "y": y,
                "Xvalid": Xvalid,
                "yvalid": yvalid}

    else:

        # This is the main section typically called by the tor client.
        # Thus we have hardcoded an absolute path from that location
        data = pd.read_csv(os.path.join('../ML', "data", dataset_name + '.csv'))
        d = data.shape[1]

        data = data.as_matrix()

        X = data[:, 0:d - 1]
        y = data[:, -1]

        return {"X": X, "y": y}

def normalize_rows(X):

    # Sets all rows to have L2 norm of 1. Needed for diff priv
    nn, dd = X.shape

    for i in xrange(nn):
        X[i, ] = X[i, ] / norm(X[i, ], 2)

    return X


def standardize_cols(X, mu=None, sigma=None):
    # Standardize each column with mean 0 and variance 1
    n_rows, n_cols = X.shape

    if mu is None:
        mu = np.mean(X, axis=0)

    if sigma is None:
        sigma = np.std(X, axis=0)
        sigma[sigma < 1e-8] = 1.

    return (X - mu) / sigma, mu, sigma


def standardize_outputs(y, mu=None, sigma=None):

    if mu is None:
        mu = np.mean(y)

    if sigma is None:
        sigma = np.std(y)
        if sigma < 1e-8:
            sigma = 1.

    return (y - mu) / sigma, mu, sigma


def check_gradient(model, X, y):
    # This checks that the gradient implementation is correct
    w = np.random.rand(model.w.size)
    f, g = model.funObj(w, X, y)

    # Check the gradient
    estimated_gradient = approx_fprime(w,
                                       lambda w: model.funObj(w, X, y)[0],
                                       epsilon=1e-6)

    implemented_gradient = model.funObj(w, X, y)[1]

    if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
        raise Exception('User and numerical derivatives differ:\n%s\n%s' %
                        (estimated_gradient[:5], implemented_gradient[:5]))
    else:
        print('User and numerical derivatives agree.')


def lap_noise(loc=0, scale=1, size=1):
    return np.random.laplace(loc=loc, scale=scale, size=size)


def exp_noise(scale=1, size=1):
    return np.random.exponential(scale=scale, size=size)


def approx_fprime(x, f_func, epsilon=1e-7):
    # Approximate the gradient using the complex step method
    n_params = x.size
    e = np.zeros(n_params)
    gA = np.zeros(n_params)
    for n in range(n_params):
        e[n] = 1.
        val = f_func(x + e * np.complex(0, epsilon))
        gA[n] = np.imag(val) / epsilon
        e[n] = 0

    return gA


def regression_error(y, yhat):
    return 0.5 * np.sum(np.square((y - yhat)) / float(yhat.size))


def classification_error(y, yhat):
    return np.sum(y != yhat) / float(yhat.size)


def load_pkl(fname):
    """Reads a pkl file.

    Parameters
    ----------
    fname : the name of the .pkl file

    Returns
    -------
    data :
        Returns the .pkl file as a 'dict'
    """
    if not os.path.isfile(fname):
        raise ValueError('File {} does not exist.'.format(fname))

    if sys.version_info[0] < 3:
        # Python 2
        with open(fname, 'rb') as f:
            data = pickle.load(f)
    else:
        # Python 3
        with open(fname, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

    return data
