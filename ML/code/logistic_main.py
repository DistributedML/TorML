from __future__ import division
from numpy.linalg import norm
import logistic_model
import logistic_model_test
import numpy as np
import utils
import pdb


if __name__ == "__main__":

    dataset = "logisticData"
    data = utils.load_dataset(dataset)

    X = data['X']
    y = data['y']
    oweights = np.zeros(X.shape[1])

    batch_size = 5

    # Object
    model1 = logistic_model.logRegL2(X, y, lammy=0.1)

    for i in xrange(20000):
        (deltas, f, g) = model1.privateFun(1, oweights, batch_size)
        oweights = oweights + deltas

    print("Test error: %d", logistic_model_test.test(oweights))

    # Global
    numFeatures = logistic_model.init(dataset)
    weights = np.zeros(numFeatures)

    for i in xrange(20000):
        deltas = logistic_model.privateFun(1, weights, batch_size)
        weights = weights + deltas

    print("Test error: %d", logistic_model_test.test(weights))
