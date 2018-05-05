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
def kdd_test():

    dataset = "kddcup/kddcup_train"

    batch_size = 100
    iterations = 4000
    epsilon = 5    

    # Global
    numFeatures = softmax_model.init(dataset, epsilon=epsilon)

    print("Got " + str(numFeatures) + " features")

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


def faces_test():

    dataset = "faces"
    
    batch_size = 10
    iterations = 4000
    epsilon = 5    

    # Global
    numFeatures = softmax_model.init(dataset, epsilon=epsilon)

    print("Got " + str(numFeatures) + " features")

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
    iterations = 5000
    epsilon = 5
    numFeatures = 943

    list_of_models = []

    for dataset in model_names:
        list_of_models.append(softmax_model_obj.SoftMaxModel(dataset, epsilon))

    numClients = len(list_of_models)
    logistic_aggregator.init(numClients, numFeatures)

    print("Start training across " + str(numClients) + " clients.")

    weights = np.random.rand(numFeatures) / 100.0
    train_progress = []

    for i in xrange(iterations):

        total_delta = np.zeros((numClients, numFeatures))

        for k in range(len(list_of_models)):
            total_delta[k, :] = list_of_models[k].privateFun(1, weights, batch_size)

        delta, hm = logistic_aggregator.average(total_delta)
        weights = weights + delta

        if i % 100 == 0:
            error = softmax_model_test.train_error(weights)
            print("Train error: %.10f" % error)
            train_progress.append(error)

    # fig = plt.figure()
    # plt.plot(heur_distances)
    # fig.savefig("heur_distances.jpeg")

    print("Done iterations!")
    print("Train error: %d", softmax_model_test.train_error(weights))
    print("Test error: %d", softmax_model_test.test_error(weights))
    return weights


def attack_simulation(model_names, distance):

    batch_size = 10
    iterations = 4000
    epsilon = 5
    numFeatures = 7840

    list_of_models = []

    print_interval = 50
    progress = iterations/print_interval
    num_printed = 0

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

        distance = logistic_aggregator.search_distance_euc2(total_delta, 1.0, False, [], np.zeros(numClients), 0)

        # pdb.set_trace()
        # distance = logistic_aggregator.search_distance_euc(total_delta, 1.0)
        # delta, distance, nnbs = logistic_aggregator.euclidean_binning_hm(total_delta, distance)
        # distance = logistic_aggregator.search_distance_lsh(total_delta, 1.0, -float('Inf'), False)
        # delta, hm, nnbs = logistic_aggregator.lsh_sieve(total_delta, distance)
        # print(distance)
        # print(nnbs)

        heur_distances[i] = distance
        weights = weights + delta

        if i % 50 == 0:
            error = softmax_model_test.train_error(weights)
            progress -= 1
            num_printed += 1
            print("Train error: %.10f \t %d iterations left" % (error, progress))
            train_progress.append(error)

            if (num_printed >= 10 and train_progress[num_printed-10] - train_progress[num_printed-1] < 0.001):
                print("Not improving much...Quiting...")
                break

    # fig = plt.figure()
    # plt.plot(heur_distances)
    # fig.savefig("heur_distances.jpeg")

    print("Done iterations!")
    print("Train error: %d", softmax_model_test.train_error(weights))
    print("Test error: %d", softmax_model_test.test_error(weights))
    return weights


if __name__ == "__main__":

    full_model = softmax_model_obj.SoftMaxModel("kddcup/kddcup_train", epsilon=1)
    Xtest, ytest = full_model.get_data()

    models = []

    for i in range(23):
        models.append("kddcup/kddcup" + str(i))

    models.append("kddcup/kddcup_bad")
    models.append("kddcup/kddcup_bad")
    models.append("kddcup/kddcup_bad")

    weights = non_iid(models)

    score = poisoning_compare.kdd_eval(Xtest, ytest, weights, 0, 11)
    # score = poisoning_compare.eval(Xtest, ytest, weights, 1, 7)
