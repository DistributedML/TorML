import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    # data1 = np.loadtxt("lossflush_10_nt.csv", delimiter=',')
    # data2 = np.loadtxt("lossflush_50_nt.csv", delimiter=',')
    # data3 = np.loadtxt("lossflush_100_nt.csv", delimiter=',')
    # data4 = np.loadtxt("lossflush_200_nt.csv", delimiter=',')
    data5 = np.loadtxt("loss_e1_fp.csv", delimiter=',')
    data6 = np.loadtxt("loss_e5_fp.csv", delimiter=',')

    fig, ax = plt.subplots()

    plt.plot(data5, color="black", label=r'$\varepsilon$ = 1', lw=5)
    plt.plot(data6, color="red", label=r'$\varepsilon$ = 5', lw=5)
    # plt.plot(data3, color="orange", label="100 clients", lw=5)
    # plt.plot(data4, color="green", label="200 clients", lw=5)

    plt.legend(loc='upper right')

    plt.xlabel("Time (s)")
    plt.ylabel("Training Error")

    plt.legend(loc='upper right', ncol=4, fontsize=18)

    plt.xlabel("Time (s)", fontsize=18)
    plt.ylabel("Training Error", fontsize=18)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.show()
