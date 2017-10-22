import numpy as np 
import matplotlib.pyplot as plt

data = np.loadtxt("lossflush.csv", delimiter=',')
fig = plt.figure()
plt.plot(data)
fig.savefig("loss.jpeg")