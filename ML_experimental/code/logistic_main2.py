from __future__ import division
import utils
import logistic_model
import global_model2
import pdb
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt

# Load Binary and Multi -class data
#data = utils.load_dataset("logisticData")


dataset = "skin"
data = utils.load_dataset(dataset)
XBin = data['X']
yBin = data['y']

(n,d) = XBin.shape
yBin = yBin.reshape(n,1)
all_data = np.hstack((XBin, yBin))
np.random.shuffle(all_data)

XBin = all_data[:,:d]
yBin = all_data[:,d:]
yBin = yBin.reshape(n)


print(n)
print(d)



if __name__ == "__main__":

    model1 = logistic_model.logRegL2(XBin, yBin,
                                     lammy=0.1, verbose=0, maxEvals=4000)


    minValues = []

          
    # GLOBAL MODEL with private SGD
    global_model_sgd = global_model2.globalModel2(logistic=True, verbose=0, maxEvals=100000)
    global_model_sgd.add_model(model1)

    training_batch_size = 100
    #print("STOCHASTIC BS is %.0f" % training_batch_size)
    
    collection = [0.3, 0.5, 1, 1.5]
    collection2 = [10, 50, 100, 200]
    #collection = [1]
    k = 0
    for j in collection2:
        fig = plt.figure()
        for alpha in collection:
          
            global_model_sgd.sgd_fit_private(alpha, eta=0.01, batch_size=j, dataset=dataset)
            #minValues.append(x)
        k = k+1
        s = dataset + str(k) + ".jpeg"
        fig.savefig(s, bbox_inches='tight')
#    collection2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4,
#                   1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
#    for alpha in collection2:
#        a=global_model_sgd.sgd_fit_private(alpha, eta=0.01, batch_size=training_batch_size)
#        minValues.append(a)
#    #plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
#    plt.plot(collection2,minValues)
#    plt.ylabel('Final value of objective')
#    plt.xlabel('alpha')
#    #plt.title('alpha = [0.3, 0.5, 1, 1.5]')
#    #plt.legend()
