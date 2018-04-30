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


def attack_simulation(model_names, distance):

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
    heur_distances = np.zeros(iterations)
    train_progress = []

    for i in xrange(iterations):

        total_delta = np.zeros((numClients, numFeatures))

        for k in range(len(list_of_models)):
            total_delta[k, :] = list_of_models[k].privateFun(1, weights, batch_size)

        delta, nnbs, hd = logistic_aggregator.lsh_sieve(total_delta, numFeatures,
                                                        numClients, distance)

        heur_distances[i] = hd
        weights = weights + delta

        if i % 50 == 0:
            error = softmax_model_test.train_error(weights)
            print("Train error: %d", error)
            train_progress.append(error)

    # fig = plt.figure()
    # plt.plot(heur_distances)
    # fig.savefig("heur_distances.jpeg")

    print("Done iterations!")
    print("Train error: %d", softmax_model_test.train_error(weights))
    print("Test error: %d", softmax_model_test.test_error(weights))
    return weights


if __name__ == "__main__":

    full_model = softmax_model_obj.SoftMaxModel("mnist_train", epsilon=1)
    Xtest, ytest = full_model.get_data()

    models = ["mnist05", "mnist16", "mnist27", "mnist38", "mnist49",
              "mnist_bad", "mnist_bad", "mnist_bad", "mnist_bad",
              "mnist_bad", "mnist_bad"]

    distances = [1.0 / (80 * 7840), 1.0 / (100 * 7840)]
    results = np.zeros((10, len(distances)))

    for j in range(len(distances)):
        for test in xrange(10):
            weights = attack_simulation(models, distances[j])
            score = poisoning_compare.eval(Xtest, ytest, weights)
            results[test, j] = score

    pdb.set_trace()
