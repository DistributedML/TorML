import matplotlib.pyplot as plt
import numpy as np
import pdb

if __name__ == "__main__":

    fig, ax = plt.subplots()

    for clients in (10, 50, 100, 200):

        median_data = np.zeros(5)

        for k in (1, 2, 3, 4, 5):

            data = np.loadtxt("loss_" + str(clients) + "_" + str(k) + ".csv", delimiter=',')
            median_data[k-1] = data.shape[0]

        print str(clients) + " median is " + str(np.median(median_data))
        print str(clients) + " stddev is " + str(np.std(median_data))

    data1 = np.loadtxt("loss_10_2.csv", delimiter=',')
    data2 = np.loadtxt("loss_50_2.csv", delimiter=',')
    data3 = np.loadtxt("loss_100_2.csv", delimiter=',')
    data4 = np.loadtxt("loss_200_2.csv", delimiter=',')
    
    plt.plot(data1, color="black", label="10 clients", lw=5)
    plt.plot(data2, color="red", label="50 clients", lw=5)
    plt.plot(data3, color="orange", label="100 clients", lw=5)
    plt.plot(data4, color="green", label="200 clients", lw=5)

    plt.legend(loc='best', ncol=1, fontsize=18)

    plt.xlabel("Time (s)", fontsize=22)
    plt.ylabel("Training Error", fontsize=22)

    axes = plt.gca()
    axes.set_ylim([0, 0.5])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.setp(ax.get_xticklabels(), fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=18)

    plt.show()