import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import re
import pdb


def plot():

    uniform_data = np.random.rand(10, 10)
    ax = sns.heatmap(uniform_data, linewidth=0.5, annot=True, fmt=".2f")

    plt.xlabel("Source Label", fontsize=18)
    plt.ylabel("Target Label", fontsize=18)

    plt.show()


def collect():

    x = 10
    y = 10
    dataset = 'mnist'
    iterations = 3000
    grid_data = []

    for i in range(x):
        row = []
        for j in range(y):
            if i == j:
                row.append(1)
            else:
                filename = 'autologs/f2_new/' + dataset + ' ' + str(iterations) + ' 5_' + str(i) + '_' + str(j) + '.log'
                with open(filename, 'r') as logfile:
                    data = logfile.read()
                    #print(data)
                    print(str(i) + str(j))
                    attack_rate_match = re.search('Target Attack Rate.*:\s+([0-9]*.[0-9]*)', data)
                    attack_rate = float(attack_rate_match.group(1))
                    row.append(attack_rate)
        grid_data.append(row)

    data = np.array(grid_data)
    np.savetxt("f2n.csv", data, delimiter=',')
    return data

if __name__ == "__main__":

    data = collect()
    pdb.set_trace()
