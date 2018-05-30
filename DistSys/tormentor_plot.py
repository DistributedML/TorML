import numpy as np 
import matplotlib.pyplot as plt

data = np.loadtxt("lossflush10_3.csv", delimiter=',')
fig = plt.figure()
plt.plot(data)
fig.savefig("loss10_3.jpeg")