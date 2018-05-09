import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

fig, ax = plt.subplots(figsize=(10, 5))

df = pd.read_csv("fig1results_fed.csv", header=None)
data1 = df.values[:, 3]

df = pd.read_csv("fig1results_krum_niid.csv", header=None)
data2 = df.values[:, 3]

df = pd.read_csv("fig1results_foolsgold.csv", header=None)
data3 = df.values[:, 3]

plt.plot(data1, color="black", label="Baseline", lw=5)
plt.plot(data2, color="red", label="Krum", lw=5)
plt.plot(data3, color="orange", label="FoolsGold", lw=5)

plt.legend(loc='center right', ncol=1, fontsize=18)

plt.xlabel("# of Poisoners", fontsize=22)
plt.ylabel("Attack Rate", fontsize=22)

axes = plt.gca()
axes.set_ylim([0, 1])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.setp(ax.get_xticklabels(), fontsize=18)
plt.setp(ax.get_yticklabels(), fontsize=18)

fig.tight_layout(pad=0.1)
fig.savefig("fig1_baselines.pdf")

plt.show()