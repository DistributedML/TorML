from __future__ import division
from numpy.linalg import norm
import matplotlib.pyplot as plt

import logistic_model
import logistic_model_test
import logistic_aggregator
import softmax_model
import softmax_model_test
import softmax_model_obj
import poisoning_compare

import numpy as np
import utils

import pdb

# Just a simple sandbox for testing out python code, without using Go.

def basic_conv():

    dataset = "mnist_train"

    batch_size = 10
    iterations = 4000
    epsilon = 5

    # Global
    numFeatures = softmax_model.init(dataset, epsilon=epsilon)

    print("Start training")

    weights = np.random.rand(numFeatures) / 1000.0

    train_progress = np.zeros(iterations)
    test_progress = np.zeros(iterations)

    for i in xrange(iterations):
        deltas = softmax_model.privateFun(1, weights, batch_size)
        weights = weights + deltas

        if i % 100 == 0:
            print("Train error: %d", softmax_model_test.train_error(weights))
            print("Test error: %d", softmax_model_test.test_error(weights))

    print("Done iterations!")
    print("Train error: %d", softmax_model_test.train_error(weights))
    print("Test error: %d", softmax_model_test.test_error(weights))


def synchronous_non_iid(model_names):

    batch_size = 10
    iterations = 4000
    epsilon = 5
    numFeatures = 7840

    list_of_models = []

    for dataset in model_names:
        list_of_models.append(softmax_model_obj.SoftMaxModel(dataset, epsilon=epsilon))

    numClients = len(list_of_models)

    print("Start training")

    weights = np.random.rand(numFeatures) / 1000.0

    for i in xrange(iterations):

        total_delta = np.zeros((numClients, numFeatures))

        for k in range(len(list_of_models)):
            total_delta[k, :] = list_of_models[k].privateFun(1, weights, batch_size)

        delta, nnbs = logistic_aggregator.euclidean_binning(total_delta, numFeatures, numClients)
        weights = weights + delta

        print(nnbs)
        
        if i % 400 == 0:
            print("Train error: %d", softmax_model_test.train_error(weights))
            print("Test error: %d", softmax_model_test.test_error(weights))

    print("Done iterations!")
    print("Train error: %d", softmax_model_test.train_error(weights))
    print("Test error: %d", softmax_model_test.test_error(weights))
    return weights


if __name__ == "__main__":

    full_model = softmax_model_obj.SoftMaxModel("mnist_train", epsilon=1)
    Xtest, ytest = full_model.get_data()

    models = ["mnist0", "mnist1", "mnist2", "mnist3", "mnist4", "mnist5","mnist6","mnist7","mnist8","mnist9",
        "mnist_bad_49", "mnist_bad_49", "mnist_bad_49", "mnist_bad_49", "mnist_bad_49", "mnist_bad_49"]
    weights = synchronous_non_iid(models)

    poisoning_compare.eval(Xtest, ytest, weights)

    pdb.set_trace()
