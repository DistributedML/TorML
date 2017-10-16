from __future__ import division
import numpy as np
import utils
import pdb

data = utils.load_dataset("credittest")
Xtest, ytest = data['X'], data['y']

data = utils.load_dataset("credittrain")
XBin, yBin = data['X'], data['y']


def train_error(ww):
    ww = np.array(ww)
    yhat = np.sign(np.dot(XBin, ww))
    error = np.sum(yhat != yBin) / float(yBin.size)
    return error


def test_error(ww):
    ww = np.array(ww)
    yhat = np.sign(np.dot(Xtest, ww))
    error = np.sum(yhat != ytest) / float(yhat.size)
    return error
