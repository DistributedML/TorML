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

    print_interval = 50
    progress = iterations/print_interval

    for dataset in model_names:
        list_of_models.append(softmax_model_obj.SoftMaxModel(dataset, epsilon=epsilon))

    numClients = len(list_of_models)
    logistic_aggregator.init(numClients, numFeatures)

    print("Start training")

    weights = np.random.rand(numFeatures) / 1000.0
    heur_distances = np.zeros(iterations)
    train_progress = []

    for i in xrange(iterations):

        total_delta = np.zeros((numClients, numFeatures))

        for k in range(len(list_of_models)):
            total_delta[k, :] = list_of_models[k].privateFun(1, weights, batch_size)

        distance = logistic_aggregator.search_distance(total_delta, 1.0)
        delta, hm = logistic_aggregator.euclidean_binning_hm(total_delta, distance)
        #print(distance)
        #print(nnbs)

        heur_distances[i] = distance
        weights = weights + delta

        if i % 50 == 0:
            error = softmax_model_test.train_error(weights)
            progress -= 1
            print("Train error: %.10f \t %d iterations left" % (error, progress))
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

    models = ["mnist0", "mnist1", "mnist2", "mnist3", "mnist4",
              "mnist5", "mnist6", "mnist7", "mnist8", "mnist9",
              "mnist_bad_17", "mnist_bad_17", "mnist_bad_17", "mnist_bad_17",
              "mnist_bad_17", "mnist_bad_17", "mnist_bad_17", "mnist_bad_17",
              "mnist_bad_17", "mnist_bad_17","mnist_bad_17", "mnist_bad_17"]

    distance = 1.0 / (150 * 7840)

    weights = attack_simulation(models, distance)
    score = poisoning_compare.eval(Xtest, ytest, weights)

    pdb.set_trace()
