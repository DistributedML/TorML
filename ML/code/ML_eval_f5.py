from __future__ import division
from numpy.linalg import norm
import matplotlib.pyplot as plt


import logistic_aggregator
import softmax_model
import softmax_model_test
import softmax_model_obj
import softmax_validator
import poisoning_compare

import numpy as np
import utils

import pdb
import sys
np.set_printoptions(suppress=True)

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

    for i in xrange(iterations):
        deltas = softmax_model.privateFun(1, weights, batch_size)
        weights = weights + deltas

        if i % 100 == 0:
            print("Train error: %d", softmax_model_test.train_error(weights))
            print("Test error: %d", softmax_model_test.test_error(weights))

    print("Done iterations!")
    print("Train error: %d", softmax_model_test.train_error(weights))
    print("Test error: %d", softmax_model_test.test_error(weights))


def non_iid(model_names, numClasses, numParams, softmax_test, iter=3000):

    batch_size = 50
    iterations = iter
    epsilon = 5

    list_of_models = []

    for dataset in model_names:
        list_of_models.append(softmax_model_obj.SoftMaxModel(dataset, epsilon, numClasses))

    numClients = len(list_of_models)
    logistic_aggregator.init(numClients, numParams)

    print("Start training across " + str(numClients) + " clients.")

    weights = np.random.rand(numParams) / 100.0
    train_progress = []

    cs = np.zeros((numClients, numClients))

    all_roni = np.zeros((iter, numClients))

    for i in xrange(iterations):

        total_delta = np.zeros((numClients, numParams))

        for k in range(len(list_of_models)):
            sub_delta = list_of_models[k].privateFun(1, weights, batch_size)
            all_roni[i, k] = str(softmax_validator.roni(weights, sub_delta))
            total_delta[k, :] = sub_delta

        # scs = logistic_aggregator.get_cos_similarity(total_delta)
        # cs = cs + scs
        # delta = logistic_aggregator.cos_aggregate(total_delta, cs, i)

        delta = logistic_aggregator.average(total_delta)
        weights = weights + delta

        if i % 100 == 0:
            error = softmax_test.train_error(weights)
            print("Train error: %.10f" % error)
            train_progress.append(error)

    print("Done iterations!")
    print("Train error: %d", softmax_test.train_error(weights))
    print("Test error: %d", softmax_test.test_error(weights))

    return weights, all_roni


if __name__ == "__main__":
    argv = sys.argv[1:]
    dataset = argv[0]

    if (dataset == "mnist"):
        numClasses = 10
        numFeatures = 784
    elif (dataset == "kddcup"):
        numClasses = 23
        numFeatures = 41
    elif (dataset == "amazon"):
        numClasses = 50
        numFeatures = 10000
    else:
        print("Dataset " + dataset + " not found. Available datasets: mnist kddcup amazon")

    numParams = numClasses * numFeatures
    dataPath = dataset + "/" + dataset

    full_model = softmax_model_obj.SoftMaxModel(dataPath + "_train", 1, numClasses)
    Xtest, ytest = full_model.get_data()

    # Over poisoners from 0 to 9
    results = np.zeros((10, 4))

    for sanity in range(5):

        print("Start attack with 5")
        attack = "1_7"
        models = []

        for i in range(numClasses):
            models.append(dataPath + str(i))

        for i in range(5):
            models.append(dataPath + "_bad_" + attack)

        softmax_test = softmax_model_test.SoftMaxModelTest(dataset, numClasses, numFeatures)
        weights, roni = non_iid(models, numClasses, numParams, softmax_test, 3000)

        overall, correct, misslabel_correct, attacked = poisoning_compare.eval(
            Xtest, ytest, weights, 1, 7, numClasses, numFeatures)

        np.save("all_roni_scores_" + str(sanity), roni)

    pdb.set_trace()
