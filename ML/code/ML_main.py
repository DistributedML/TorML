from __future__ import division
from numpy.linalg import norm
import matplotlib.pyplot as plt



import logistic_aggregator
import softmax_model
import softmax_model_test
import softmax_model_obj
import poisoning_compare

import numpy as np
import utils

import pdb
import sys
import json

# Just a simple sandbox for testing out python code, without using Go.

def debug_signal_handler(signal, frame):
    import pdb
    pdb.set_trace()
import signal
signal.signal(signal.SIGINT, debug_signal_handler)


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


def non_iid(model_names):

    batch_size = 10
    iterations = 10000
    epsilon = 5
    #numFeatures = 943
    numFeatures = 7840
    list_of_models = []

    for dataset in model_names:
        list_of_models.append(softmax_model_obj.SoftMaxModel(dataset, epsilon))

    numClients = len(list_of_models)
    logistic_aggregator.init(numClients, numFeatures)

    print("Start training across " + str(numClients) + " clients.")

    weights = np.random.rand(numFeatures) / 100.0
    train_progress = []
    poisoned_per_it = []
    hm_per_it = []
    poisoned = np.zeros(numClients)
    for i in xrange(iterations):

        total_delta = np.zeros((numClients, numFeatures))

        for k in range(len(list_of_models)):
            total_delta[k, :] = list_of_models[k].privateFun(1, weights, batch_size)
        # distance, p = logistic_aggregator.search_distance_euc2(total_delta, 1.0, False, [], np.zeros(numClients), 0)
        # delta, dist, nnbs = logistic_aggregator.euclidean_binning_hm(total_delta, distance)
        #distance, p = logistic_aggregator.search_distance_euc2(total_delta, 1.0, False, [], np.zeros(numClients), 0)
        #print(distance)
        distance = .11
        delta, dist, nnbs = logistic_aggregator.euclidean_binning_hm(total_delta, distance)

        #poisoned += p
        weights = weights + delta

        if i % 100 == 0:
            error = softmax_model_test.train_error(weights)
            print("Train error: %.10f" % error)
            train_progress.append(error)
            hm_per_it.append(np.matrix(logistic_aggregator.hit_matrix))
            poisoned_per_it.append(list(poisoned))

    # fig = plt.figure()
    # plt.plot(heur_distances)
    # fig.savefig("heur_distances.jpeg")
    # pdb.set_trace()
    print("Done iterations!")
    print("Train error: %d", softmax_model_test.train_error(weights))
    print("Test error: %d", softmax_model_test.test_error(weights))
    return weights


#amazon: 50 classes, 10000 features
#mnist: 10 classes, 784 features
#kdd: 23 classes, 41 features
if __name__ == "__main__":

    full_model = softmax_model_obj.SoftMaxModel("mnist/mnist_train", epsilon=1)
    Xtest, ytest = full_model.get_data()

    models = []

    for i in range(10):
        models.append("mnist/mnist" + str(i))

    models.append("mnist/mnist_bad_17")
    models.append("mnist/mnist_bad_17")
    models.append("mnist/mnist_bad_17")

    weights = non_iid(models)
    pdb.set_trace()
    score = poisoning_compare.mnist_eval(Xtest, ytest, weights, 1, 7)
    # score = poisoning_compare.eval(Xtest, ytest, weights, 1, 7)
    pdb.set_trace()
