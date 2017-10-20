from __future__ import division
import utils
import logistic_model
import global_model4
import numpy as np
import matplotlib.pyplot as plt



collection4 = ["bank", "creditcard", "diabetic", "eye", "logisticData", 
                   "magic","occupancy", "pulsars", "skin", "tom", "transfusion", "twitter"]   

for ds in collection4:

    data = utils.load_dataset(ds)
    XBin, yBin = data['X'], data['y']
    XBinValid, yBinValid = data['Xvalid'], data['yvalid']
    
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

    print("n = ",n)
    print("d = ",d)
    
    
    
    if __name__ == "__main__":
    
        model1 = logistic_model.logRegL2(XBin, yBin,
                                         lammy=0.1, verbose=0, maxEvals=4000)
              
        global_model_sgd = global_model4.globalModel4(logistic=True, verbose=0, maxEvals=100000)
        global_model_sgd.add_model(model1)
        
        
        
    
    # -------------------------------------------------------------------------------------
        
      # Plotting training and validation error against iterations  
        
        collection = [0.3, 0.5, 1, 1.5]
        #collection2 = [10, 50, 100, 200]
        collection2 = [100]
        #collection = [1]
        k = 0
        for j in collection2:
            for alpha in collection:
              
                fig = plt.figure()

                global_model_sgd.sgd_fit_private(alpha, XBin, yBin, XBinValid, yBinValid, eta=0.01, batch_size=j, dataset=ds)
                
                k = k+1
                s = ds + str(k) + ".jpeg"
                fig.savefig(s, bbox_inches='tight')
    

            

