from __future__ import division
import numpy as np




def train_error(ww, XBin, yBin):
    ww = np.array(ww)
    yhat = np.sign(np.dot(XBin, ww))
    error = np.sum(yhat != yBin) / float(yBin.size)
    return error


def test_error(ww, dataset, Xtest, ytest):
    ww = np.array(ww)
    yhat = np.sign(np.dot(Xtest, ww))
    error = np.sum(yhat != ytest) / float(yhat.size)
    return error
