import matplotlib.pyplot as plt
import numpy as np
import pylab

if __name__ == "__main__":

    data1 = np.loadtxt("lossflush_10_nt.csv", delimiter=',')
    data2 = np.loadtxt("lossflush_50_nt.csv", delimiter=',')
    data3 = np.loadtxt("lossflush_100_nt.csv", delimiter=',')
    data4 = np.loadtxt("lossflush_200_nt.csv", delimiter=',')

    pylab.plot(data1, color="black", label="10 clients")
    pylab.plot(data2, color="red", label="50 clients")
    pylab.plot(data3, color="orange", label="100 clients")
    pylab.plot(data4, color="green", label="200 clients")

    pylab.legend(loc='upper right')

    #pylab.xticks(np.arange(0, 400, 10.0))

    pylab.title("Convergence over time, varying number of clients")
    pylab.xlabel("Time (s)")
    pylab.ylabel("Training Error")

    pylab.show()

