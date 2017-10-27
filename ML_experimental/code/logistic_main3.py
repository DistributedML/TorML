from __future__ import division
import utils
import logistic_model3
import global_model3
import numpy as np
import matplotlib.pyplot as plt

# Load Binary and Multi -class data
data = utils.load_dataset("magic")
XBin, yBin = data['X'], data['y']
XBinValid, yBinValid = data['Xvalid'], data['yvalid']


#Shuffle data

(n,d) = XBin.shape
(nn,dd) = XBinValid.shape

yBin = yBin.reshape(n,1)
yBinValid = yBinValid.reshape(nn,1)

all_data = np.hstack((XBin, yBin))
all_data2 = np.hstack((XBinValid, yBinValid))
np.random.shuffle(all_data)
np.random.shuffle(all_data2)

XBin = all_data[:,:d]
yBin = all_data[:,d:]
yBin = yBin.reshape(n)

XBinValid = all_data2[:,:dd]
yBinValid = all_data2[:,dd:]
yBinValid = yBinValid.reshape(nn)


cut1 = int(XBin.shape[0] * 0.5)
cut2 = XBin.shape[0]

cutVal = int(XBinValid.shape[0] * 0.5)
cutVal2 = XBinValid.shape[0]

if __name__ == "__main__":

    model1 = logistic_model3.logRegL2(XBin[0:cut1, :], yBin[0:cut1],
                                     lammy=0.1, verbose=0, maxEvals=400)


    model2 = logistic_model3.logRegL2(XBin[cut1 + 1:cut2, :], yBin[cut1 + 1:cut2],
                                     lammy=0.1, verbose=0, maxEvals=400)


          
    # GLOBAL MODEL with private SGD
    global_model_sgd = global_model3.globalModel3(logistic=True, verbose=0, maxEvals=100000)
    global_model_sgd.add_model(model1)
    global_model_sgd.add_model(model2)

    training_batch_size = 100
    
    #collection = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,
                  #1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0]
    
    #collection = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
    collection = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,10.5,11.5,12.5,13.5,14.5,15.5]
    
    eucdiffs = []
    
    for alpha in collection:
    
        x = global_model_sgd.sgd_fit_private(0.1, alpha, 0.01, training_batch_size)
        eucdiffs.append(x)
    
    
    plt.plot(collection,eucdiffs)
    plt.ylabel('Euclidean distance')
    plt.xlabel('alpha')
    

