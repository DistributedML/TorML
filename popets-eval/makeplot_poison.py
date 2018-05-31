import matplotlib.pyplot as plt
import numpy as np
import pdb

if __name__ == "__main__":

    fig, ax = plt.subplots()

    # data1 = np.loadtxt("poison/loss_24_50p_05p_t_1.csv", delimiter=',')
    # data2 = np.loadtxt("poison/loss_24_50p_1p_t_1.csv", delimiter=',')
    # data3 = np.loadtxt("poison/loss_24_50p_2p_t_1.csv", delimiter=',')
    # data4 = np.loadtxt("poison/loss_24_50p_5p_t_1.csv", delimiter=',')

    # plt.plot(data4, color="green", label="5% RONI", lw=5)
    # plt.plot(data3, color="orange", label="2% RONI", lw=5)
    # plt.plot(data2, color="red", label="1% RONI", lw=5)
    # plt.plot(data1, color="black", label="0.5% RONI", lw=5)

    data1 = np.loadtxt("lossflush_p25_2.csv", delimiter=',')
    data2 = np.loadtxt("lossflush_p50_2.csv", delimiter=',')
    data3 = np.loadtxt("lossflush_p75_2.csv", delimiter=',')

    plt.plot(np.arange(data3.shape[0]), data3,
             color="red", label="75% poisoners", lw=5)
    plt.plot(np.arange(data2.shape[0]), data2,
             color="orange", label="50% poisoners", lw=5)
    plt.plot(np.arange(data1.shape[0]), data1,
             color="green", label="25% poisoners", lw=5)
    # plt.plot(np.arange(data0.shape[0]), data0,
    #          color="black", label="0% poisoners", lw=5)

    plt.legend(loc='best', ncol=1, fontsize=18)

    ax.set_xlim(0, 1000)

    plt.xlabel("Time (s)", fontsize=22)
    plt.ylabel("Training Error", fontsize=22)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.setp(ax.get_xticklabels(), fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=18)

    plt.tight_layout()
    plt.show()
