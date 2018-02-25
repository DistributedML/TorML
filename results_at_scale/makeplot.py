import matplotlib.pyplot as plt
import numpy as np
import pdb

if __name__ == "__main__":

    fig, ax = plt.subplots()

    data1 = np.loadtxt("loss_10_hot_timed.csv", delimiter=',')
    data2 = np.loadtxt("loss_50_hot_timed.csv", delimiter=',')
    data3 = np.loadtxt("loss_100_hot_timed.csv", delimiter=',')
    data4 = np.loadtxt("loss_200_1_timed.csv", delimiter=',')

    plt.plot(data1[0, :], data1[1, :], color="black", label="10 clients", lw=5)
    plt.plot(data2[0, :], data2[1, :], color="red", label="50 clients", lw=5)
    plt.plot(data3[0, :], data3[1, :], color="orange", label="100 clients", lw=5)
    plt.plot(data4[0, :], data4[1, :], color="green", label="200 clients", lw=5)

    # data1nt = np.loadtxt("loss_10_nt_timed.csv", delimiter=',')
    # data2nt = np.loadtxt("loss_50_nt_timed.csv", delimiter=',')
    # data3nt = np.loadtxt("loss_100_nt_timed.csv", delimiter=',')
    # data4nt = np.loadtxt("loss_200_nt_timed.csv", delimiter=',')

    # plt.plot(data1nt[0, :], data1nt[1, :], color="black", label="10 clients", lw=5)
    # plt.plot(data2nt[0, :], data2nt[1, :], color="red", label="50 clients", lw=5)
    # plt.plot(data3nt[0, :], data3nt[1, :], color="orange", label="100 clients", lw=5)
    # plt.plot(data4nt[0, :], data4nt[1, :], color="green", label="200 clients", lw=5)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='best', ncol=1, fontsize=18)
    #plt.legend(loc='upper right', ncol=2, fontsize=18)

    plt.xlabel("Time (s)", fontsize=22)
    plt.ylabel("Training Error", fontsize=22)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.setp(ax.get_xticklabels(), fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=18)


    plt.show()