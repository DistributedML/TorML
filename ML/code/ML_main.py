from __future__ import division
from numpy.linalg import norm
import matplotlib.pyplot as plt 

import logistic_model
import logistic_model_test
import softmax_model
import softmax_model_test
import softmax_model_obj

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


def synchronous_non_iid():

    batch_size = 10
    iterations = 4000
    epsilon = 5
    numFeatures = 7840

    # Global
    model0 = softmax_model_obj.SoftMaxModel("mnist05", epsilon=epsilon)
    model1 = softmax_model_obj.SoftMaxModel("mnist16", epsilon=epsilon)
    model2 = softmax_model_obj.SoftMaxModel("mnist27", epsilon=epsilon)
    model3 = softmax_model_obj.SoftMaxModel("mnist38", epsilon=epsilon)
    model4 = softmax_model_obj.SoftMaxModel("mnist49", epsilon=epsilon)
    
    print("Start training")

    weights = np.random.rand(numFeatures) / 1000.0

    for i in xrange(iterations):
        
        total_delta = model0.privateFun(1, weights, batch_size)
        total_delta = total_delta + model1.privateFun(1, weights, batch_size)
        total_delta = total_delta + model2.privateFun(1, weights, batch_size)
        total_delta = total_delta + model3.privateFun(1, weights, batch_size)
        total_delta = total_delta + model4.privateFun(1, weights, batch_size)
        total_delta = total_delta / 5

        weights = weights + total_delta

        if i % 100 == 0:
            print("Train error: %d", softmax_model_test.train_error(weights))
            print("Test error: %d", softmax_model_test.test_error(weights))

    print("Done iterations!")
    print("Train error: %d", softmax_model_test.train_error(weights))
    print("Test error: %d", softmax_model_test.test_error(weights))


if __name__ == "__main__":

    synchronous_non_iid()
