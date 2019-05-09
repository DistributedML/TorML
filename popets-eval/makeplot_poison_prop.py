import matplotlib.pyplot as plt
import numpy as np
import pdb

if __name__ == "__main__":

    length = 250
    fig, ax = plt.subplots(figsize=(10, 5))

    # Avg over 3 runs
    data1 = np.zeros((3, length))
    data1[0] = np.loadtxt("lossflush_p0_1.csv", delimiter=',')[0:length]
    data1[1] = np.loadtxt("lossflush_p0_2.csv", delimiter=',')[0:length]
    data1[2] = np.loadtxt("lossflush_p0_3.csv", delimiter=',')[0:length]

    data2 = np.zeros((3, length))
    data2[0] = np.loadtxt("lossflush_p25_1.csv", delimiter=',')[0:length]
    data2[1] = np.loadtxt("lossflush_p25_2.csv", delimiter=',')[0:length]
    data2[2] = np.loadtxt("lossflush_p25_3.csv", delimiter=',')[0:length]

    data3 = np.zeros((3, length))
    data3[0, 0:121] = np.loadtxt("lossflush_p50_1.csv", delimiter=',')[0:121]
    data3[1] = np.loadtxt("lossflush_p50_2.csv", delimiter=',')[0:length]
    data3[2] = np.loadtxt("lossflush_p50_3.csv", delimiter=',')[0:length]

    data4 = np.zeros((3, length))
    data4[0] = np.loadtxt("lossflush_p75_1.csv", delimiter=',')[0:length]
    data4[1] = np.loadtxt("lossflush_p75_2.csv", delimiter=',')[0:length]
    data4[2] = np.loadtxt("lossflush_p75_3.csv", delimiter=',')[0:length]

    plt.plot(np.mean(data4, axis=0),
             color="red", label="75% poisoners", lw=3)
    plt.plot(np.mean(data3, axis=0),
             color="orange", label="50% poisoners", lw=3)
    plt.plot(np.mean(data2, axis=0),
             color="green", label="25% poisoners", lw=3)
    plt.plot(np.mean(data1, axis=0),
             color="black", label="0% poisoners", lw=3)

    plt.legend(loc='best', ncol=1, fontsize=18)

    ax.set_xlim(0, length)

    plt.xlabel("Time (s)", fontsize=22)
    plt.ylabel("Training Error", fontsize=22)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.setp(ax.get_xticklabels(), fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=18)

    plt.tight_layout()
    plt.savefig('poison.pdf')
    plt.show()
