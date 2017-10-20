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


collection4 = ["creditcard", "diabetic", "eye", "logisticData", 
                   "magic","occupancy", "pulsars", "skin", "tom", "transfusion", "twitter"]   

for ds in collection4:

    #dataset = "bank"
    data = utils.load_dataset(ds)
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
              
        # GLOBAL MODEL with private SGD
        global_model_sgd = global_model2.globalModel2(logistic=True, verbose=0, maxEvals=100000)
        global_model_sgd.add_model(model1)
    
        training_batch_size = 100
        #print("STOCHASTIC BS is %.0f" % training_batch_size)
        
    # -------------------------------------------------------------------------------------
        
      # Plotting objective value against iterations  
        
    #    collection = [0.3, 0.5, 1, 1.5]
        collection2 = [10, 50, 100, 200]
    #    #collection = [1]
    #    k = 0
    #    for j in collection2:
    #        fig = plt.figure()
    #        for alpha in collection:
    #          
    #            global_model_sgd.sgd_fit_private(alpha, eta=0.01, batch_size=j, dataset=dataset)
    #            
    #        k = k+1
    #        s = dataset + str(k) + ".jpeg"
    #        fig.savefig(s, bbox_inches='tight')
    
    # -------------------------------------------------------------------------------------
    
      # Plotting final objective value against alpha
         
        collection3 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 
                       1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 
                       2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 
                       3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 
                       4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 
                       5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0]
        
        k = 0
        for j in collection2:
            minValues = []
            fig = plt.figure()
            for alpha in collection3:
                
                a=global_model_sgd.sgd_fit_private(alpha, eta=0.01, batch_size=j)
                minValues.append(a)
                
            s2 = 'dataset = ' + ds + ' & batch size = ' + str(j)
            plt.plot(collection3,minValues)
            plt.ylabel('Final value of objective')
            plt.xlabel('alpha')
            plt.title(s2)
            #plt.legend()
            
            k = k+1
            s = ds + str(k) + ".jpeg"
            fig.savefig(s, bbox_inches='tight')
            

