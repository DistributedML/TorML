import numpy as np
import pandas as pd
import pdb


def eval(Xtest, ytest, weights):

    # hardcoded for MNIST
    W = np.reshape(weights, (10, 784))
    yhat = np.argmax(np.dot(Xtest, W.T), axis=1)

    targetIdx = np.where(ytest == 1)
    otherIdx = np.where(ytest != 1)
    overall = np.mean(yhat[otherIdx] == ytest[otherIdx])
    correct1 = np.mean(yhat[targetIdx] == 1)
    attacked1 = np.mean(yhat[targetIdx] == 7)

    print("Overall Error: " + str(overall))
    print("Target Training Accuracy on 1s: " + str(correct1))
    print("Target Attack Rate (1 to 7): " + str(attacked1) + "\n")

    return attacked1


def main():

    dataTrain = np.load("../data/mnist_train.npy")
    dataTest = np.load("../data/mnist_train.npy")

    df = pd.read_csv("../../DistSys/modelflush_pure.csv", header=None)
    pure_model = df.ix[0, :7839].as_matrix().astype(float)

    Xtrain = dataTest[:, :784]
    ytrain = dataTest[:, 784]

    eval(Xtrain, ytrain, pure_model)

    df = pd.read_csv("../../DistSys/modelflush_1p.csv", header=None)
    poison_model = df.ix[0, :7839].as_matrix().astype(float)

    eval(Xtrain, ytrain, poison_model)

    df = pd.read_csv("../../DistSys/modelflush_2p.csv", header=None)
    poison_model = df.ix[0, :7839].as_matrix().astype(float)

    eval(Xtrain, ytrain, poison_model)

    pdb.set_trace()


if __name__ == "__main__":

    main()
