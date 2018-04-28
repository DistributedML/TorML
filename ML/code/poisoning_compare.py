import numpy as np
import pandas as pd
import pdb


def main():

    dataTrain = np.load("../data/mnist_train.npy")
    dataTest = np.load("../data/mnist_train.npy")

    df1 = pd.read_csv("../../DistSys/modelflush_1p.csv", header=None)
    poison_model = df1.ix[0, :7839].as_matrix().astype(float)

    df2 = pd.read_csv("../../DistSys/modelflush_pure.csv", header=None)
    pure_model = df2.ix[0, :7839].as_matrix().astype(float)

    Xtrain = dataTrain[:, :784]
    ytrain = dataTrain[:, 784]

    # hardcoded for MNIST
    W = np.reshape(pure_model, (10, 784))
    yhat = np.argmax(np.dot(Xtrain, W.T), axis=1)

    targetIdx = np.where(ytrain == 1)
    correct1 = np.mean(yhat[targetIdx] == 1)
    attacked1 = np.mean(yhat[targetIdx] == 7)

    print("Target Training Error on 1s: " + str(correct1))
    print("Target Attack Rate (1 to 7): " + str(attacked1))

    # hardcoded for MNIST
    W = np.reshape(poison_model, (10, 784))
    yhat = np.argmax(np.dot(Xtrain, W.T), axis=1)

    targetIdx = np.where(ytrain == 1)
    correct1 = np.mean(yhat[targetIdx] == 1)
    attacked1 = np.mean(yhat[targetIdx] == 7)

    print("Target Training Error on 1s: " + str(correct1))
    print("Target Attack Rate (1 to 7): " + str(attacked1))


    pdb.set_trace()


if __name__ == "__main__":

    main()
