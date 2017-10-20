from __future__ import division
import utils
import logistic_model
import global_model
import pdb
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
import numpy as np

# Load Binary and Multi -class data
data = utils.load_dataset("logisticData")
XBin, yBin = data['X'], data['y']
XBinValid, yBinValid = data['Xvalid'], data['yvalid']


#Shuffle data

(n,d) = XBin.shape
#all_data = np.hstack((XBin, yBin))
#np.random.shuffle(all_data)
#all_data_valid = np.hstack((XBinValid, yBinValid))

#XBin,yBin = np.hsplit(all_data,d)
#XBinValid, yBinValid = np.hsplit(all_data_valid,d)

cut1 = int(XBin.shape[0] * 0.2)
cut2 = int(XBin.shape[0] * 0.4)
cut3 = int(XBin.shape[0] * 0.6)
cut4 = int(XBin.shape[0] * 0.8)
cut5 = XBin.shape[0]

cutVal = int(XBinValid.shape[0] * 0.5)
cutVal2 = XBinValid.shape[0]

if __name__ == "__main__":

    model1 = logistic_model.logRegL2(XBin[0:cut1, :], yBin[0:cut1],
                                     lammy=0.1, verbose=0, maxEvals=400)
    model1.fit()

    model2 = logistic_model.logRegL2(XBin[cut1 + 1:cut2, :], yBin[cut1 + 1:cut2],
                                     lammy=0.1, verbose=0, maxEvals=400)
    model2.fit()

    model3 = logistic_model.logRegL2(XBin[cut2 + 1:cut3, :], yBin[cut2 + 1:cut3],
                                     lammy=0.1, verbose=0, maxEvals=400)
    model3.fit()

    model4 = logistic_model.logRegL2(XBin[cut3 + 1:cut4, :], yBin[cut3 + 1:cut4],
                                     lammy=0.1, verbose=0, maxEvals=400)
    model4.fit()

    model5 = logistic_model.logRegL2(XBin[cut4 + 1:cut5, :], yBin[cut4 + 1:cut5],
                                     lammy=0.1, verbose=0, maxEvals=400)
    model5.fit()

    print("model1 Validation error %.3f" %
          utils.classification_error(model1.predict(XBinValid), yBinValid))
    print("model2 Validation error %.3f" %
          utils.classification_error(model2.predict(XBinValid), yBinValid))
    print("model3 Validation error %.3f" %
          utils.classification_error(model3.predict(XBinValid), yBinValid))
    print("model4 Validation error %.3f" %
          utils.classification_error(model4.predict(XBinValid), yBinValid))
    print("model5 Validation error %.3f" %
          utils.classification_error(model5.predict(XBinValid), yBinValid))

    clf = SGDClassifier(loss="hinge", penalty="l2")
    clf.fit(XBin, yBin)
    print("sklearn sgd validation error %.3f" %
          utils.classification_error(clf.predict(XBinValid), yBinValid))

    svmclf = LinearSVC()
    svmclf.fit(XBin, yBin)
    print("sklearn SVM validation error %.3f" %
          utils.classification_error(svmclf.predict(XBinValid), yBinValid))

    # GLOBAL MODEL
    global_model_gd = global_model.globalModel(
        logistic=True, verbose=0, maxEvals=500)
    global_model_gd.add_model(model1)
    global_model_gd.add_model(model2)
    global_model_gd.add_model(model3)
    global_model_gd.add_model(model4)
    global_model_gd.add_model(model5)

    global_model_gd.fit(theta=1)
    print("global 1 GD Training error %.3f" %
          utils.classification_error(global_model_gd.predict(XBin), yBin))
    print("global 1 GD Validation error %.3f" %
          utils.classification_error(global_model_gd.predict(XBinValid), yBinValid))

    # GLOBAL MODEL with SGD
    global_model_sgd = global_model.globalModel(
        logistic=True, verbose=0, maxEvals=100000)
    global_model_sgd.add_model(model1)
    global_model_sgd.add_model(model2)
    global_model_sgd.add_model(model3)
    global_model_sgd.add_model(model4)
    global_model_sgd.add_model(model5)
    training_batch_size = 5
    print("STOCHASTIC BS is %.0f" % training_batch_size)

    global_model_sgd.fit(theta=1, batch_size=training_batch_size)
    print("global 1 SGD Training error %.3f" %
          utils.classification_error(global_model_sgd.predict(XBin), yBin))
    print("global 1 SGD Validation error %.3f" %
          utils.classification_error(global_model_sgd.predict(XBinValid), yBinValid))
          
    # GLOBAL MODEL with private SGD
    global_model_sgd = global_model.globalModel(logistic=True, verbose=0, maxEvals=100000)
    global_model_sgd.add_model(model1)
    global_model_sgd.add_model(model2)
    global_model_sgd.add_model(model3)
    global_model_sgd.add_model(model4)
    global_model_sgd.add_model(model5)
    training_batch_size = 5
    print("STOCHASTIC BS is %.0f" % training_batch_size)
          
    global_model_sgd.sgd_fit_private(alpha=1, eta=0.01, batch_size=training_batch_size)
    print("global 1 SGD Training error %.3f" %
          utils.classification_error(global_model_sgd.predict(XBin), yBin))
    print("global 1 SGD Validation error %.3f" %
          utils.classification_error(global_model_sgd.predict(XBinValid), yBinValid))

    

    # GLOBAL MODEL with PEGASOS
    global_model_pegasos = global_model.globalModelSVM(
        logistic=True, verbose=0, maxEvals=100000)
    global_model_pegasos.add_model(model1)
    global_model_pegasos.add_model(model2)
    global_model_pegasos.add_model(model3)
    global_model_pegasos.add_model(model4)
    global_model_pegasos.add_model(model5)
    global_model_pegasos.fit(batch_size=training_batch_size)

    print("global SVM Training error %.3f" %
          utils.classification_error(global_model_pegasos.predict(XBin), yBin))
    print("global SVM Validation error %.3f" %
          utils.classification_error(global_model_pegasos.predict(XBinValid), yBinValid))

    # FULL
    sk_full = logistic_model.logRegL2(XBin, yBin,
                                      lammy=0.1, verbose=0, maxEvals=100000)
    sk_full.sgd_fit(theta=1, batch_size=training_batch_size)
    print("full Training error %.3f" %
          utils.classification_error(sk_full.predict(XBin), yBin))
    print("full Validation error %.3f" %
          utils.classification_error(sk_full.predict(XBinValid), yBinValid))

    # RAW AVERAGE
    print("----------------------------------------------")
    print("global-averaging e=0.1 Validation error %.3f" %
          utils.classification_error(global_model_gd.predictAverage(
              XBinValid, epsilon=0.1), yBinValid))

    print("global-averaging e=0.01 Validation error %.3f" %
          utils.classification_error(global_model_gd.predictAverage(
              XBinValid, epsilon=0.01), yBinValid))

    print("global-averaging e=0.001 Validation error %.3f" %
          utils.classification_error(global_model_gd.predictAverage(
              XBinValid, epsilon=0.001), yBinValid))

    # WEIGHTED AVERAGE on public labelled
    global_model_gd.fitWeightedAverage(
        XBinValid[0:cutVal, :], yBinValid[0:cutVal], epsilon=0.1)
    print("global-weighted e=0.1 Validation error %.3f" %
          utils.classification_error(global_model_gd.predictWeightedAverage(
              XBinValid[cutVal + 1:cutVal2, :]), yBinValid[cutVal + 1:cutVal2]))

    # WEIGHTED AVERAGE on public labelled
    global_model_gd.fitWeightedAverage(
        XBinValid[0:cutVal, :], yBinValid[0:cutVal], epsilon=0.01)
    print("global-weighted e=0.01 Validation error %.3f" %
          utils.classification_error(global_model_gd.predictWeightedAverage(
              XBinValid[cutVal + 1:cutVal2, :]), yBinValid[cutVal + 1:cutVal2]))

    # WEIGHTED AVERAGE on public labelled
    global_model_gd.fitWeightedAverage(
        XBinValid[0:cutVal, :], yBinValid[0:cutVal], epsilon=0.001)
    print("global-weighted e=0.001 Validation error %.3f" %
          utils.classification_error(global_model_gd.predictWeightedAverage(
              XBinValid[cutVal + 1:cutVal2, :]), yBinValid[cutVal + 1:cutVal2]))

    '''
    ### KNOWLEDGE TRANSFER on public unlabelled
    ypub = global_model_gd.predictAverage(XBinValid[0:cutVal,:], epsilon=0.1)
    global_kt = logistic_model.logRegL2(XBinValid[0:cutVal,:], ypub, lammy=0.1, verbose=0, maxEvals=400)
    global_kt.fit()
    print("global-knowledge-transfer e=0.1 Validation error %.3f" % 
        utils.classification_error(global_kt.predict(
            XBinValid[cutVal+1:cutVal2,:]), yBinValid[cutVal+1:cutVal2]))

        ### KNOWLEDGE TRANSFER on public unlabelled
    ypub = global_model_gd.predictAverage(XBinValid[0:cutVal,:], epsilon=0.01)
    global_kt = logistic_model.logRegL2(XBinValid[0:cutVal,:], ypub, lammy=0.1, verbose=0, maxEvals=400)
    global_kt.fit()
    print("global-knowledge-transfer e=0.01 Validation error %.3f" % 
        utils.classification_error(global_kt.predict(
            XBinValid[cutVal+1:cutVal2,:]), yBinValid[cutVal+1:cutVal2]))

        ### KNOWLEDGE TRANSFER on public unlabelled
    ypub = global_model_gd.predictAverage(XBinValid[0:cutVal,:], epsilon=0.001)
    global_kt = logistic_model.logRegL2(XBinValid[0:cutVal,:], ypub, lammy=0.1, verbose=0, maxEvals=400)
    global_kt.fit()
    print("global-knowledge-transfer e=0.001 Validation error %.3f" % 
        utils.classification_error(global_kt.predict(
            XBinValid[cutVal+1:cutVal2,:]), yBinValid[cutVal+1:cutVal2]))
    '''
