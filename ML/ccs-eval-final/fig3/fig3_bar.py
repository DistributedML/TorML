import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

fig, ax = plt.subplots(figsize=(10, 5))

prop = [20, 40, 60, 80, 100]
datasets = ['mnist', 'kddcup', 'amazon']
plotobj = np.zeros((5, 3))

for d in range(3):

    dataset = datasets[d]

    for i in range(5):

        df = pd.read_csv("fig3results_" + dataset + "_" + str(i) + ".csv", header=None)
        data = df.values

        plotobj[:, d] += data[:, 3] / 5

plt.plot(prop, plotobj[:, 0], color="black", label="MNIST", lw=5)
plt.plot(prop, plotobj[:, 1], color="red", label="KDDCup", lw=5)
plt.plot(prop, plotobj[:, 2], color="orange", label="Amazon", lw=5)

plt.legend(loc='center right', ncol=1, fontsize=18)

plt.xlabel("Proportion of Mixed Data", fontsize=22)
plt.ylabel("Attack Rate", fontsize=22)

axes = plt.gca()
axes.set_ylim([0, 0.01])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.setp(ax.get_xticklabels(), fontsize=18)
plt.setp(ax.get_yticklabels(), fontsize=18)

fig.savefig("fig3_mixing.pdf", bbox_inches='tight')
