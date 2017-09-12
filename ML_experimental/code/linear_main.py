from __future__ import division
import utils
import linear_model
import global_model
import pdb

# Load Binary and Multi -class data
data = utils.load_dataset("slices")
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
    model1 = linear_model.linRegL2(XBin[0:cut1,:], yBin[0:cut1], lammy=0.1, maxEvals=400)
    model2 = linear_model.linRegL2(XBin[cut1+1:cut2,:], yBin[cut1+1:cut2], lammy=0.1, maxEvals=400)
    model3 = linear_model.linRegL2(XBin[cut2+1:cut3,:], yBin[cut2+1:cut3], lammy=0.1, maxEvals=400)
    model4 = linear_model.linRegL2(XBin[cut3+1:cut4,:], yBin[cut3+1:cut4], lammy=0.1, maxEvals=400)
    model5 = linear_model.linRegL2(XBin[cut4+1:cut5,:], yBin[cut4+1:cut5], lammy=0.1, maxEvals=400)

    model1.fit()
    model2.fit()
    model3.fit()
    model4.fit()
    model5.fit()  

    print("model1 Training error %.3f" % 
        utils.regression_error(model1.predict(XBin[0:cut1,:]), yBin[0:cut1]))
    print("model2 Training error %.3f" % 
        utils.regression_error(model2.predict(XBin[cut1+1:cut2,:]), yBin[cut1+1:cut2]))
    print("model3 Training error %.3f" % 
        utils.regression_error(model3.predict(XBin[cut2+1:cut3,:]), yBin[cut2+1:cut3]))
    print("model4 Training error %.3f" % 
        utils.regression_error(model4.predict(XBin[cut3+1:cut4,:]), yBin[cut3+1:cut4]))
    print("model5 Training error %.3f" % 
        utils.regression_error(model5.predict(XBin[cut4+1:cut5,:]), yBin[cut4+1:cut5]))

    print("model1 Validation error %.3f" % 
        utils.regression_error(model1.predict(XBinValid), yBinValid))
    print("model2 Validation error %.3f" % 
        utils.regression_error(model2.predict(XBinValid), yBinValid))
    print("model3 Validation error %.3f" % 
        utils.regression_error(model3.predict(XBinValid), yBinValid))
    print("model4 Validation error %.3f" % 
        utils.regression_error(model4.predict(XBinValid), yBinValid))
    print("model5 Validation error %.3f" % 
        utils.regression_error(model5.predict(XBinValid), yBinValid))

    ## GLOBAL MODEL
    global_model = global_model.globalModel(verbose=0, maxEvals=400)
    global_model.add_model(model1)
    global_model.add_model(model2)
    global_model.add_model(model3)
    global_model.add_model(model4)
    global_model.add_model(model5)

    # global_model.fit(theta=0.1)
    # print("global 0.1 Training error %.3f" % 
    #     utils.regression_error(global_model.predict(XBin), yBin))
    # print("global 0.1 Validation error %.3f" % 
    #     utils.regression_error(global_model.predict(XBinValid), yBinValid))

    # global_model.fit(theta=0.25)
    # print("global 0.25 Training error %.3f" % 
    #     utils.regression_error(global_model.predict(XBin), yBin))
    # print("global 0.25 Validation error %.3f" % 
    #     utils.regression_error(global_model.predict(XBinValid), yBinValid))

    global_model.fit(theta=0.5)
    print("global 0.5 Training error %.3f" % 
        utils.regression_error(global_model.predict(XBin), yBin))
    print("global 0.5 Validation error %.3f" % 
        utils.regression_error(global_model.predict(XBinValid), yBinValid))

    global_model.fit(theta=1)
    print("global 1 Training error %.3f" % 
        utils.regression_error(global_model.predict(XBin), yBin))
    print("global 1 Validation error %.3f" % 
        utils.regression_error(global_model.predict(XBinValid), yBinValid))

    print("global average e=0.1 validation error %.3f" % 
        utils.regression_error(global_model.predictAverage(
            XBinValid, epsilon=0.1), yBinValid))

    print("global average e=0.01 validation error %.3f" % 
    utils.regression_error(global_model.predictAverage(
        XBinValid, epsilon=0.01), yBinValid))

    print("global average e=0.001 validation error %.3f" % 
    utils.regression_error(global_model.predictAverage(
        XBinValid, epsilon=0.001), yBinValid))

    global_model.fitWeightedAverage(XBinValid[0:cutVal,:], yBinValid[0:cutVal], epsilon=0.1)
    print("global-weighted e=0.1 Validation error %.3f" % 
        utils.regression_error(global_model.predictWeightedAverage(
            XBinValid[cutVal+1:cutVal2,:]), yBinValid[cutVal+1:cutVal2]))

    global_model.fitWeightedAverage(XBinValid[0:cutVal,:], yBinValid[0:cutVal], epsilon=0.01)
    print("global-weighted e=0.01 Validation error %.3f" % 
        utils.regression_error(global_model.predictWeightedAverage(
            XBinValid[cutVal+1:cutVal2,:]), yBinValid[cutVal+1:cutVal2]))

    global_model.fitWeightedAverage(XBinValid[0:cutVal,:], yBinValid[0:cutVal], epsilon=0.001)
    print("global-weighted e=0.001 Validation error %.3f" % 
        utils.regression_error(global_model.predictWeightedAverage(
            XBinValid[cutVal+1:cutVal2,:]), yBinValid[cutVal+1:cutVal2]))

    # ### KNOWLEDGE TRANSFER on public unlabelled
    # ypub = global_model.predictAverage(XBinValid[0:cutVal,:], epsilon=0.1)
    # global_kt = linear_model.linRegL2(XBinValid[0:cutVal,:], ypub, lammy=0.1, verbose=0, maxEvals=400)
    # global_kt.fit()
    # print("global-knowledge-transfer e=0.1 Validation error %.3f" % 
    #     utils.regression_error(global_kt.predict(
    #         XBinValid[cutVal+1:cutVal2,:]), yBinValid[cutVal+1:cutVal2]))

    # ### KNOWLEDGE TRANSFER on public unlabelled
    # ypub = global_model.predictAverage(XBinValid[0:cutVal,:], epsilon=0.01)
    # global_kt = linear_model.linRegL2(XBinValid[0:cutVal,:], ypub, lammy=0.1, verbose=0, maxEvals=400)
    # global_kt.fit()
    # print("global-knowledge-transfer e=0.01 Validation error %.3f" % 
    #     utils.regression_error(global_kt.predict(
    #         XBinValid[cutVal+1:cutVal2,:]), yBinValid[cutVal+1:cutVal2]))

    # ### KNOWLEDGE TRANSFER on public unlabelled
    # ypub = global_model.predictAverage(XBinValid[0:cutVal,:], epsilon=0.001)
    # global_kt = linear_model.linRegL2(XBinValid[0:cutVal,:], ypub, lammy=0.1, verbose=0, maxEvals=400)
    # global_kt.fit()
    # print("global-knowledge-transfer e=0.001 Validation error %.3f" % 
    #     utils.regression_error(global_kt.predict(
    #         XBinValid[cutVal+1:cutVal2,:]), yBinValid[cutVal+1:cutVal2]))

    ## FULL MODEL
    full = linear_model.linRegL2(XBin, yBin, lammy=0.1, maxEvals=400)
    full.fit()

    print("full Training error %.3f" % 
        utils.regression_error(full.predict(XBin), yBin))
    print("full Validation error %.3f" % 
        utils.regression_error(full.predict(XBinValid), yBinValid))


