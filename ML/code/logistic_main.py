from __future__ import division
from numpy.linalg import norm
import logistic_model
import logistic_model_test
import numpy as np
import utils
import pdb


if __name__ == "__main__":

    dataset = "susy1"
    data = utils.load_dataset(dataset)

    X = data['X']
    y = data['y']
    oweights = np.zeros(X.shape[1])

    batch_size = 5

    # Global
    numFeatures = logistic_model.init(dataset)
    weights = np.zeros(numFeatures)

    for i in xrange(20000):
        deltas = logistic_model.privateFun(1, weights, batch_size)
        weights = weights + deltas

    print("Train error: %d", logistic_model_test.train_error(weights))
    print("Test error: %d", logistic_model_test.test_error(weights))
