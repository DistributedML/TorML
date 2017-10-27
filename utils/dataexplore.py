import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.spatial.distance as ds
import pdb


def readData(datafile):
    data = pd.read_csv(datafile, sep=',', header=None)

    nn, dd = data.shape

    # remove the last label
    labels = data.ix[:, dd - 1]
    data = data.ix[:, 0:dd - 2].as_matrix()
    data = data.T

    return data, labels


def distanceMatrix(data, distance=ds.euclidean):

    # find the number of models
    nPoints, nModels = data.shape

    distances = np.empty([nModels, nModels], dtype=float)

    for i in range(nModels):
        for j in range(nModels):
            distances[i, j] = distance(data[:, i], data[:, j])

    return distances


def makeGraph(data):

    # find the number of models
    nPoints, nModels = data.shape

    plt.plot(data[:, 0], color='black')
    plt.plot(data[:, 1:nModels], color='green')
    plt.plot(data[:, 3], color='red')
    plt.show()


if __name__ == "__main__":
    pdb.set_trace()
