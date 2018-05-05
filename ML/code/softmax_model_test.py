from __future__ import division
import numpy as np
import utils
import matplotlib.pyplot as plt
import pdb

datatest = utils.load_dataset("kddcup/kddcup_test", npy=True)
Xtest, ytest = datatest['X'], datatest['y']

datatrain = utils.load_dataset("kddcup/kddcup_train", npy=True)
XBin, yBin = datatrain['X'], datatrain['y']


def train_error(ww):

    # hardcoded for MNIST
    W = np.reshape(ww, (23, 41))
    yhat = np.argmax(np.dot(XBin, W.T), axis=1)
    error = np.mean(yhat != yBin)
    return error


def test_error(ww):
    W = np.reshape(ww, (23, 41))
    yhat = np.argmax(np.dot(Xtest, W.T), axis=1)
    error = np.mean(yhat != ytest)
    return error


def kappa(ww, delta):
    
    ww = np.array(ww)
    yhat = np.argmax(np.dot(Xtest, ww), axis=1)
    
    ww2 = np.array(ww + delta)
    yhat2 = np.argmax(np.dot(Xtest, ww2), axis=1)

    P_A = np.mean(yhat == yhat2)
    P_E = 0.5

    return (P_A - P_E) / (1 - P_E)


def roni(ww, delta):
    
    ww = np.array(ww)
    yhat = np.argmax(np.dot(Xtest, ww), axis=1)
    
    ww2 = np.array(ww + delta)
    yhat2 = np.argmax(np.dot(Xtest, ww2), axis=1)

    g_err = np.mean(yhat != ytest)
    new_err = np.mean(yhat2 != ytest)

    # How much does delta improve the validation error?
    return g_err - new_err


def plot(data):

    data = np.loadtxt("lossflush.csv", delimiter=',')
    fig = plt.figure()
    plt.plot(data)
    fig.savefig("loss.jpeg")

    return 1