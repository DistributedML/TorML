import pandas as pd
import pdb
import matplotlib.pyplot as plt
import numpy as np

dat = pd.read_csv("results.csv")

ind = np.arange(5)
width = 0.20

# AVERAGING

e1 = (36.008,63.093,36.848,61.745,34.229)
e01 = (36.008,34.540,34.723,34.973,34.229)
e001 = (36.008,34.246,34.708,34.662,34.229)
'''

e1 = (0.209,0.217,0.206,0.219,0.208)
e01 = (0.209,0.209,0.206,0.208,0.208)
e001 = (0.209,0.208,0.208,0.208,0.208)
'''

fig, ax = plt.subplots()
rects1 = ax.bar(ind, e1, width, color='r')
rects2 = ax.bar(ind + width, e01, width, color='y')
rects3 = ax.bar(ind + 2 * width, e001, width, color='g')

ax.set_ylabel("Error")
ax.set_title("Squared Error for Various Aggregation Methods")

ax.set_xticks(ind + width)
ax.set_xticklabels(('Local', 'Average', 'Feature', 'K-Transfer', 'Full'))
#ax.set_ylim(ymin=0.18)
ax.legend((rects1[0], rects2[0], rects3[0]), (r'$\beta$=0.1', r'$\beta$=0.01', r'$\beta$=0.001'))

plt.grid(True, which='both', axis="y", linestyle=":", color="gray")
plt.show()


# ITERATIVE TRAINING
'''
x = (0.1, 0.25, 0.5, 1)
local = (0.208, 0.208, 0.208, 0.208)
err = (0.209, 0.209, 0.210, 0.211)
full = (0.2081, 0.2081, 0.2081, 0.2081)


local = (36.008, 36.008, 36.008, 36.008)
err = (34.993, 34.638, 34.354, 34.282)
full = (34.229, 34.229, 34.229, 34.229)


plt.title("Squared Error for Global Iterative Training, for Various Theta")
plt.plot(x, local)
plt.plot(x, err)
plt.plot(x, full)

plt.xlabel("Theta (Proportion of parameters used per iteration)")
plt.ylabel("Error")
plt.legend(['Local Model', 'Globally Trained', 'Full'])
plt.show()

'''

'''
k = (1,2,3,4,5)
LocalL0Err = (0.235, 0.235, 0.235, 0.235, 0.235)
LocalL1Err = (0.185, 0.185, 0.185, 0.185, 0.185)
L0err = (0.288, 0.21, 0.204, 0.134, 0.094)
L1err = (0.16, 0.122, 0.086, 0.066, 0.07)
FullL1 = (0.052, 0.052, 0.052, 0.052, 0.052)
FullL0 = (0.018, 0.018, 0.018, 0.018, 0.018)

LocalL1select = (41.8, 41.8, 41.8, 41.8, 41.8)
LocalL0select = (17.2, 17.2, 17.2, 17.2, 17.2)
L0select = (3,6,8,17,55)
L1select = (9,14,33,61,88)
FL1select = (71,71,71,71,71)
FL0select = (26,26,26,26,26)


plt.title("Classification Error for Private Feature Selection, for Various Thresholds")
plt.plot(k, LocalL0Err)
plt.plot(k, LocalL1Err)
plt.plot(k, L0err)
plt.plot(k, L1err)
plt.plot(k, FullL0)
plt.plot(k, FullL1)
plt.ylabel("Error")

plt.title("Number of Features Selected for Private Feature Selection, for Various Thresholds")
plt.plot(k, LocalL0select)
plt.plot(k, LocalL1select)
plt.plot(k, L0select)
plt.plot(k, L1select)
plt.plot(k, FL0select)
plt.plot(k, FL1select)
plt.ylabel("Number of Selected Features")

plt.xlabel("k-threshold")
plt.xticks(k)
plt.legend(['Local L0', 'Local L1', 'Global L0', 'Global L1', 'Full L0', 'Full L1'],
	loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
'''