from __future__ import division
from numpy.linalg import norm
import matplotlib.pyplot as plt 
import logistic_model
import logistic_model_test
import numpy as np
import utils
import pdb


if __name__ == "__main__":

    dataset = "creditbad"

    batch_size = 50
    iterations = 200000

    # Global
    numFeatures = logistic_model.init(dataset)
    weights = np.random.rand(numFeatures)
    
    rolling_average = np.zeros([iterations, numFeatures])
    all_delta = np.zeros([iterations, numFeatures])
    progress = np.zeros(iterations)
    train_progress = np.zeros(iterations)
    test_progress = np.zeros(iterations)

    for i in xrange(iterations):
        deltas = logistic_model.privateFun(1, weights, batch_size)
        weights = weights + deltas
        train_progress[i] = logistic_model_test.train_error(weights)
        test_progress[i] = logistic_model_test.test_error(weights)
        if i % 10000 == 0:
            print(i)

    plt.plot(train_progress, "green")
    plt.plot(test_progress, "red")

    plt.show()

    print("Train error: %d", logistic_model_test.train_error(weights))
    print("Test error: %d", logistic_model_test.test_error(weights))
