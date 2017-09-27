import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb


def makeGraph(datafile):

    data = pd.read_csv(datafile, sep=',', header=None).as_matrix()

    # find the number of models
    nModels, nPoints = data.shape
    data = data.T

    #pdb.set_trace()

    plt.plot(data[:,0], color='black')
    plt.plot(data[:,1:nModels])
    plt.show()


if __name__ == "__main__":

	makeGraph("models_allbad.csv")
	pdb.set_trace()