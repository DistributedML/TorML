import numpy as np 
import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":

	filename = sys.argv[1]

	print filename

	data = np.loadtxt(filename, delimiter=',')
	fig = plt.figure()
	plt.plot(data)
	fig.savefig("loss.jpeg")