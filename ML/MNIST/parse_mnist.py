from mnist import MNIST
from sklearn import svm, linear_model, neural_network
import pdb
import numpy as np
import matplotlib.pyplot as plt


def main():

    mndata = MNIST('.')

    images, labels = mndata.load_training()
    images_test, labels_test = mndata.load_testing()

    n = len(images)
    d = len(images[0])
    t = len(images_test)

    Xtrain = np.zeros((n, d))
    Xtest = np.zeros((t, d))

    ytrain = np.asarray(labels)
    ytest = np.asarray(labels_test)

    for i in range(n):
        Xtrain[i, :] = np.asarray(images[i])

    for q in range(t):
        Xtest[q, :] = np.asarray(images_test[q])

    print("Training classifier.")

    clf = linear_model.SGDClassifier(loss='log', max_iter=1000, tol=0.01)
    clf.fit(Xtrain, ytrain)

    # Training error
    y_hat = clf.predict(Xtrain)
    train_error = np.mean(y_hat != ytrain)
    print("Training Err: " + str(train_error))

    y_hat_test = clf.predict(Xtest)
    test_error = np.mean(y_hat_test != ytest)
    print("Test Err: " + str(test_error))

    nn = neural_network.MLPClassifier()
    nn.fit(Xtrain, ytrain)

    # Training error
    y_hat = nn.predict(Xtrain)
    train_error = np.mean(y_hat != ytrain)
    print("Training Err: " + str(train_error))

    y_hat_test = nn.predict(Xtest)
    test_error = np.mean(y_hat_test != ytest)
    print("Test Err: " + str(test_error))

    pdb.set_trace()


def slice_for_tm(by_digit=True):

    mndata = MNIST('.')

    images, labels = mndata.load_training()
    images_test, labels_test = mndata.load_testing()

    n = len(images)
    d = len(images[0])
    t = len(images_test)

    Xtrain = np.zeros((n, d))
    Xtest = np.zeros((t, d))

    ytrain = np.asarray(labels)
    ytest = np.asarray(labels_test)

    for i in range(n):
        Xtrain[i, :] = np.asarray(images[i])

    for q in range(t):
        Xtest[q, :] = np.asarray(images_test[q])

    for m in range(10):
        idx = np.where(ytrain == m)[0]

        class_slice = Xtrain[idx]
        data_slice = np.hstack((class_slice, ytrain[idx][:, None]))
        print("Saving mnist" + str(m))
        np.save("mnist" + str(m), data_slice)


    for k in range(5):

        idx = np.where((ytrain == k) + (ytrain == k + 5))[0]

        class_slice = Xtrain[idx]
        data_slice = np.hstack((class_slice, ytrain[idx][:, None]))

        print("slice " + str(k) + str(k + 5) + " is shape " + str(data_slice.shape))
        np.save("mnist" + str(k) + str(k + 5), data_slice)

    train_slice = np.hstack((Xtrain, np.reshape(ytrain, (len(ytrain), 1))))
    np.save("mnist_train", train_slice)

    test_slice = np.hstack((Xtest, np.reshape(ytest, (len(ytest), 1))))
    np.save("mnist_test", test_slice)


def show_digit(image):

    plt.imshow(image, cmap='gray')
    plt.show()


if __name__ == "__main__":

    slice_for_tm()
