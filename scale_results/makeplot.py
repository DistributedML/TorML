import matplotlib.pyplot as plt
import numpy as np
import pylab
import pdb

if __name__ == "__main__":

    # data1 = np.loadtxt("loss_10_hot_timed.csv", delimiter=',')
    # data2 = np.loadtxt("loss_50_hot_timed.csv", delimiter=',')
    # data3 = np.loadtxt("loss_100_hot_timed.csv", delimiter=',')
    # data4 = np.loadtxt("loss_250_hot_timed.csv", delimiter=',')

    # pylab.plot(data1[0,:], data1[1,:], color="black", label="10 clients")
    # pylab.plot(data2[0,:], data2[1,:], color="red", label="50 clients")
    # pylab.plot(data3[0,:], data3[1,:], color="orange", label="100 clients")
    # pylab.plot(data4[0,:], data4[1,:], color="green", label="250 clients")

    data1nt = np.loadtxt("loss_10_nt_timed.csv", delimiter=',')
    data2nt = np.loadtxt("loss_50_nt_timed.csv", delimiter=',')
    data3nt = np.loadtxt("loss_100_nt_timed.csv", delimiter=',')
    data4nt = np.loadtxt("loss_200_nt_timed.csv", delimiter=',')

    pylab.plot(data1nt[0,:], data1nt[1,:], color="black", label="10 clients")
    pylab.plot(data2nt[0,:], data2nt[1,:], color="red", label="50 clients")
    pylab.plot(data3nt[0,:], data3nt[1,:], color="orange", label="100 clients")
    pylab.plot(data4nt[0,:], data4nt[1,:], color="green", label="250 clients")

    pylab.legend(loc='upper right')

    #pylab.xticks(np.arange(0, 400, 10.0))

    pylab.title("Convergence over time without Tor, varying number of clients")
    pylab.xlabel("Time (s)")
    pylab.ylabel("Training Error")

    pylab.show()

