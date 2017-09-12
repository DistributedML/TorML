from __future__ import division
import utils
import logistic_model
import global_model
import pdb

# Load Binary and Multi -class data
data = utils.load_dataset("logisticData")
XBin, yBin = data['X'], data['y']
XBinValid, yBinValid = data['Xvalid'], data['yvalid']

cut1 = int(XBin.shape[0] * 0.20)
cut2 = int(XBin.shape[0] * 0.40)
cut3 = int(XBin.shape[0] * 0.60)
cut4 = int(XBin.shape[0] * 0.80)
cut5 = XBin.shape[0]

cutVal = int(XBinValid.shape[0] * 0.5)
cutVal2 = XBinValid.shape[0]

if __name__ == "__main__":

    ## LOCAL MODELS
    model1 = logistic_model.logRegL1(XBin[0:cut1,:], yBin[0:cut1], verbose=0, lammy=1, maxEvals=400)
    model2 = logistic_model.logRegL1(XBin[cut1+1:cut2,:], yBin[cut1+1:cut2], verbose=0, lammy=1, maxEvals=400)
    model3 = logistic_model.logRegL1(XBin[cut2+1:cut3,:], yBin[cut2+1:cut3], verbose=0, lammy=1, maxEvals=400)
    model4 = logistic_model.logRegL1(XBin[cut3+1:cut4,:], yBin[cut3+1:cut4], verbose=0, lammy=1, maxEvals=400)
    model5 = logistic_model.logRegL1(XBin[cut4+1:cut5,:], yBin[cut4+1:cut5], verbose=0, lammy=1, maxEvals=400)

    model1.fit()
    model2.fit()
    model3.fit()
    model4.fit()
    model5.fit()

    print("model1 Training error %.3f" % 
        utils.classification_error(model1.predict(XBin[0:cut1,:]), yBin[0:cut1]))
    print("model2 Training error %.3f" % 
        utils.classification_error(model2.predict(XBin[cut1+1:cut2,:]), yBin[cut1+1:cut2]))
    print("model3 Training error %.3f" % 
        utils.classification_error(model3.predict(XBin[cut2+1:cut3,:]), yBin[cut2+1:cut3]))
    print("model4 Training error %.3f" % 
        utils.classification_error(model4.predict(XBin[cut3+1:cut4,:]), yBin[cut3+1:cut4]))
    print("model5 Training error %.3f" % 
        utils.classification_error(model5.predict(XBin[cut4+1:cut5,:]), yBin[cut4+1:cut5]))

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

    print("model1 Non zeros %.1f" % sum(model1.w != 0))
    print("model2 Non zeros %.1f" % sum(model2.w != 0))
    print("model3 Non zeros %.1f" % sum(model3.w != 0))
    print("model4 Non zeros %.1f" % sum(model4.w != 0))
    print("model5 Non zeros %.1f" % sum(model5.w != 0))

    ## GLOBAL MODEL
    global_model = global_model.globalModel(verbose=0, maxEvals=400)
    global_model.add_model(model1)
    global_model.add_model(model2)
    global_model.add_model(model3)
    global_model.add_model(model4)
    global_model.add_model(model5)

    global_mock_model = logistic_model.logRegL0(XBin, yBin, verbose=0, lammy=1, maxEvals=400)

    for k in range(1, 6):
        selected = global_model.selectFeatures(minVotes=k)    
        print("Global method selected %.0f" % len(selected))

        if len(selected) == 0:
            continue

        # Train a model with the given features, as if it had all the data
        global_mock_model.fitSelected(selected)
        print("Global L0, removing all with", k, "Validation error %.3f" % 
            utils.classification_error(global_mock_model.predict(XBinValid), yBinValid))
        print("Global L0, removing all with", k, "method selected %.0f" % sum(global_mock_model.w != 0))

    ## FULL MODEL
    full = logistic_model.logRegL1(XBin, yBin, verbose=0, lammy=1, maxEvals=400)
    full.fit()

    print("full L1 Training error %.3f" % 
        utils.classification_error(full.predict(XBin), yBin))
    print("full L1 Validation error %.3f" % 
        utils.classification_error(full.predict(XBinValid), yBinValid))
    print("Full L1 method selected %.0f" % sum(full.w != 0))

    L0model = logistic_model.logRegL0(XBin, yBin, verbose=0, lammy=1, maxEvals=400)
    L0model.fit()

    print("Full L0 Training error %.3f" % 
        utils.classification_error(L0model.predict(XBin), yBin))
    print("Full L0 Validation error %.3f" % 
        utils.classification_error(L0model.predict(XBinValid), yBinValid))
    print("Full L0 method selected %.0f" % sum(L0model.w != 0))
