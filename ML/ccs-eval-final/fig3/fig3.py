import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

fig, ax = plt.subplots(figsize=(10, 5))

width = 0.25
prop = [20, 40, 60, 80, 100]
datasets = ['mnist', 'kddcup', 'amazon']
plotobj = np.zeros((5, 3))

for d in range(3):

    dataset = datasets[d]

    for i in range(5):

        df = pd.read_csv("fig3results_" + dataset + "_" + str(i) + ".csv", header=None)
        data = df.values

        plotobj[:, d] += data[:, 3] / 5

# plt.plot(prop, plotobj[:, 0], color="black", label="MNIST", lw=5)
# plt.plot(prop, plotobj[:, 1], color="red", label="KDDCup", lw=5)
# plt.plot(prop, plotobj[:, 2], color="orange", label="Amazon", lw=5)

plotobj[plotobj < 0.0002] = 0.00005

p1 = ax.bar(np.arange(5), plotobj[:, 0], width, hatch='/')
p2 = ax.bar(np.arange(5) + width, plotobj[:, 1], width, hatch='\\')
p3 = ax.bar(np.arange(5) + 2 * width, plotobj[:, 2], width, hatch='.')

ax.set_xticks(np.arange(6) + width)
ax.set_xticklabels(prop, fontsize=16)

ax.set_yticklabels(np.arange(0, 0.5, 0.1))
plt.setp(ax.get_yticklabels(), fontsize=16)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.xlabel("Proportion of Mixed Data", fontsize=22)
plt.ylabel('Error/Rate (%)', fontsize=22)

plt.ylim(0, 0.005)

totals = []

# find the values and append to list
for i in ax.patches:
    totals.append(i.get_height())

# set individual bar lables using above list
total = sum(totals)
str(round((i.get_height() / total), 2))
# set individual bar lables using above list
for i in ax.patches:
    # get_x pulls left or right; get_height pushes up or down
    height = str(i.get_height() * 100)[0:5]
    if i.get_height() < 0.0001:
        height = "  0"

    print(i.get_height())
    ax.text(i.get_x(), i.get_height() + .0001, height, fontsize=14, color='black')


ax.legend((p1[0], p2[0], p3[0]),
          ('MNIST', 'KDDCup', 'Amazon'),
          loc='best', ncol=3, fontsize=18)

fig.tight_layout(pad=0.1)
fig.savefig("fig3_mixing.pdf")

plt.show()
