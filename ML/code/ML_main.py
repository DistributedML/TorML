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


def non_iid(model_names, numClasses, numParams, softmax_test):

    batch_size = 10
    iterations = 3000
    epsilon = 5

    list_of_models = []

    for dataset in model_names:
        list_of_models.append(softmax_model_obj.SoftMaxModel(dataset, epsilon, numClasses))

    numClients = len(list_of_models)
    logistic_aggregator.init(numClients, numParams)

    print("Start training across " + str(numClients) + " clients.")

    weights = np.random.rand(numParams) / 100.0
    train_progress = []
    poisoned_per_it = []
    hm_per_it = []
    poisoned = np.zeros(numClients)
    for i in xrange(iterations):

        total_delta = np.zeros((numClients, numParams))

        for k in range(len(list_of_models)):
            total_delta[k, :] = list_of_models[k].privateFun(1, weights, batch_size)
        # distance, p = logistic_aggregator.search_distance_euc2(total_delta, 1.0, False, [], np.zeros(numClients), 0)
        # delta, dist, nnbs = logistic_aggregator.euclidean_binning_hm(total_delta, distance)
        #distance, p = logistic_aggregator.search_distance_euc2(total_delta, 1.0, False, [], np.zeros(numClients), 0)
        #print(distance)
        #distance = .11
        distance, poisoned = logistic_aggregator.search_distance_euc(total_delta, np.random.rand()*10, False, [], np.zeros(numClients), 0)
        #pdb.set_trace()
        delta, dist, nnbs = logistic_aggregator.euclidean_binning_hm(total_delta, distance, logistic_aggregator.get_nnbs_euc_cos)

        #poisoned += p
        weights = weights + delta

        if i % 100 == 0:
            error = softmax_test.train_error(weights)
            print("Train error: %.10f" % error)
            train_progress.append(error)
            hm_per_it.append(np.matrix(logistic_aggregator.hit_matrix))
            poisoned_per_it.append(list(poisoned))

    print("Done iterations!")
    print("Train error: %d", softmax_test.train_error(weights))
    print("Test error: %d", softmax_test.test_error(weights))
    return weights


#amazon: 50 classes, 10000 features
#mnist: 10 classes, 784 features
#kdd: 23 classes, 41 features
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

    models = []

    for i in range(numClasses):
        models.append(dataPath + str(i))

    for attack in argv[1:]:
        for i in range(int(attack[0])):
            models.append(dataPath + "_bad_" + attack[1:])


    softmax_test = softmax_model_test.SoftMaxModelTest(dataset, numClasses, numFeatures)
    weights = non_iid(models, numClasses, numParams, softmax_test)


    for attack in argv[1:]:
        from_idx = int(attack[1])
        to_idx = int(attack[2])
        score = poisoning_compare.eval(Xtest, ytest, weights, from_idx, to_idx, numClasses, numFeatures)
    #pdb.set_trace()
