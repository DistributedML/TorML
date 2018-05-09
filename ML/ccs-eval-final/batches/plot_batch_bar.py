import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb


df1 = pd.read_csv("mnist_batch.csv", header=None)
data1 = df1.values

df2 = pd.read_csv("kddbatch.csv", header=None)
data2 = df2.values

df3 = pd.read_csv("amazonbatch.csv", header=None)
data3 = df3.values

# plt.plot(data1[0], np.mean(data1[1:3], axis=0), color="black", label="MNIST", lw=3)
# plt.plot(data2[0], np.mean(data2[1:3], axis=0), color="red", label="KDDCup", lw=3)
# plt.plot(data3[0], np.mean(data3[1:3], axis=0), color="orange", label="Amazon", lw=3)

N = 6
width = 0.25
fig, ax = plt.subplots(figsize=(10, 5))

ticklabels = ['1', '5', '10', '20', '50', '100']

p1 = ax.bar(np.arange(6), np.mean(data1[1:3], axis=0), width)
p2 = ax.bar(np.arange(6), np.mean(data2[1:3], axis=0), width)
p3 = ax.bar(np.arange(6), np.mean(data3[1:3], axis=0), width)
ax.set_xticks(np.arange(6) + width)
ax.set_xticklabels(ticklabels, fontsize=14)

plt.setp(ax.get_yticklabels(), fontsize=14)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# totals = []

# # find the values and append to list
# for i in ax.patches:
#     totals.append(i.get_height())

# # set individual bar lables using above list
# total = sum(totals)
# str(round((i.get_height() / total), 2))
# # set individual bar lables using above list
# for i in ax.patches:
#     # get_x pulls left or right; get_height pushes up or down
#     height = str(i.get_height())[0:4]
#     if i.get_height() == 0:
#         height = "0.00"

#     ax.text(i.get_x(), i.get_height() + .03, height, fontsize=14, color='black')

# ##############################


plt.ylabel('Error/Rate', fontsize=16)

ax.legend((p1[0], p2[0], p3[0]),
          ('MNIST', 'KDDCup', 'Amazon'),
          loc='best', ncol=3)

fig.tight_layout(pad=0.1)
fig.savefig("fig_batch_bar.pdf")

plt.show()
