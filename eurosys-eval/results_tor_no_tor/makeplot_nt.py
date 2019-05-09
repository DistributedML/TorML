import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    fig, ax = plt.subplots(figsize=(10,5))

    for clients in (10, 50, 100, 200):

        median_data = np.zeros(5)

        for k in (1, 2, 3, 4, 5):

            data = np.loadtxt("notor/loss_" + str(clients) + "_nt_" + str(k) + ".csv", delimiter=',')
            median_data[k-1] = data.shape[0]

        print str(clients) + " median is " + str(np.median(median_data))
        print str(clients) + " stddev is " + str(np.std(median_data))

    data1nt = np.loadtxt("notor/loss_10_nt_1.csv", delimiter=',')
    data2nt = np.loadtxt("notor/loss_50_nt_1.csv", delimiter=',')
    data3nt = np.loadtxt("notor/loss_100_nt_1.csv", delimiter=',')
    data4nt = np.loadtxt("notor/loss_200_nt_1.csv", delimiter=',')

    plt.plot(data1nt, color="black", label="10 clients", lw=5)
    plt.plot(data2nt, color="red", label="50 clients", lw=5)
    plt.plot(data3nt, color="orange", label="100 clients", lw=5)
    plt.plot(data4nt, color="green", label="200 clients", lw=5)

    plt.legend(loc='best', ncol=1, fontsize=18)

    plt.xlabel("Time (s)", fontsize=18)
    plt.ylabel("Training Error", fontsize=18)

    axes = plt.gca()
    axes.set_ylim([0, 0.5])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.setp(ax.get_xticklabels(), fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=18)

    plt.tight_layout()
    plt.show()