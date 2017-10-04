from __future__ import division
import numpy as np
import minimizers
import utils
from numpy.linalg import norm
import emcee
#import matplotlib.pyplot as pl
import random
import math
import minimizers
import utils
import pdb



class logReg:
    # Logistic Regression
    def __init__(self, X, y, verbose=1, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.X = X
        self.y = y
        self.init_alpha = 1e-2
        self.alpha = self.init_alpha

        n, self.d = self.X.shape
        self.w = np.zeros(self.d)
        self.hist_grad = np.zeros(self.d)

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    # Reports the direct change to w, based on the given one.
    # Batch size could be 1 for SGD, or 0 for full gradient.
    def privateFun(self, theta, ww, batch_size=0):

        # Define constants and params
        nn, dd = self.X.shape
        threshold = int(self.d * theta)

        if batch_size > 0 and batch_size < nn:
            idx = np.random.choice(nn, batch_size, replace=False)
        else:
            # Just take the full range
            idx = range(nn)

        f, g = self.funObj(ww, self.X[idx, :], self.y[idx])

        # AdaGrad
        self.hist_grad += g**2
        ada_grad = g / (1e-6 + np.sqrt(self.hist_grad))

        # Determine the actual step magnitude
        delta = -self.init_alpha * ada_grad

        # Weird way to get NON top k values
        if theta < 1:
            param_filter = np.argpartition(abs(delta),
                                           -threshold)[:self.d - threshold]
            delta[param_filter] = 0

        w_new = ww + delta
        f_new, g_new = self.funObj(w_new, self.X[idx, :], self.y[idx])

        return (delta, f_new, g_new)
    
    
    def privateFun2(self, eta, alpha, ww,  Z, Xbatch, ybatch, batch_size=0): 
    
        nn, dd = self.X.shape

        if batch_size > 0 and batch_size < nn:
            f, g = self.funObj(ww, Xbatch, ybatch, True, batch_size, nn)
        else:
            # Just take the full range
            f, g = self.funObj(ww, self.X, self.y,True)
    
    
        # step magnitude as implemented in Song, Chaudhuri and Sarwate (2013)
        delta = (-1)*eta*(g+(1/batch_size)*Z)
        w_new = ww + delta
        
        if batch_size > 0 and batch_size < nn:
            f_new, g_new = self.funObj(w_new, Xbatch, ybatch, True, batch_size, nn)
        else: 
            f_new, g_new = self.funObj(w_new, self.X, self.y,True)
            
        return (delta, f_new, g_new)
    

    def svmUpdate(self, ww, batch_size=1):

        nn, dd = self.X.shape

        # Sample k examples from the data
        idx = np.random.choice(nn, batch_size, replace=False)

        # indicator. if expression is less than zero, we keep the delta.
        indic = (self.X[idx, :].dot(ww) * self.y[idx] < 1).astype(int)

        # delta. remove all examples with indic first.
        grad = (indic * self.y[idx]).dot(self.X[idx, :])
        delta = grad / batch_size

        return delta
    
    
    def sgd_private_fit(self, alpha, eta, batch_size=0, *args):

        n, d = self.X.shape
        batchesX = []
        batchesY = []
        within = True
        i=0
        
        while within:
            if (i+batch_size <= n):
                batchesX.append(self.X[i:i+batch_size,:])
                batchesY.append(self.y[i:i+batch_size])
                i+=batch_size
            else:
                within = False
      
    
        print ("Training model via private SGD.")
        # Parameters of the Optimization
        optTol = 1e-2

        # Initial guess
        self.w = np.zeros(d)
        self.hist_grad = 0
        funEvals = 1
    #---------------------------------------------------------------------------
        #Generate random samples from isotropic multivariate laplace distribution using emcee
    
        def lnprob(x,alpha):
            return -(alpha/2)*np.linalg.norm(x)
    
        ndim = d
        nwalkers = 3*d
        p0 = np.random.rand(ndim*nwalkers).reshape((nwalkers,ndim))
        #p0 = np.random.randn(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,args=[alpha])
        
        pos, prob, state = sampler.run_mcmc(p0, 100)
        sampler.reset()
        
        #sampler.run_mcmc(pos, 1000)
        sampler.run_mcmc(pos, 1000, rstate0=state)
        
        print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))
        #print("Autocorrelation time:", sampler.get_autocorr_time())
        
        sample = sampler.flatchain
    #---------------------------------------------------------------------------
    
        while True:
            d1,d2 = sample.shape
            z = np.random.randint(0,d1)
            Z = sample[z]
        
            l = np.random.randint(0,len(batchesX))
            Xbatch = batchesX[l]
            ybatch = batchesY[l]
             
    
            (delta, f_new, g) = self.privateFun2(eta,alpha,self.w, Z, Xbatch, ybatch,batch_size, *args)
            funEvals += 1

            # Print progress
            if self.verbose > 0:
                print("%d - loss: %.3f" % (funEvals, f_new))
                print("%d - g_norm: %.3f" % (funEvals, norm(g)))

            # Update parameters
            self.w = self.w + delta

            # Test termination conditions
            optCond = norm(g, float('inf'))

            if optCond < optTol:
                if self.verbose:
                    print("Problem solved up to optimality tolerance %.3f" % optTol)
                    break

            if funEvals >= self.maxEvals:
                if self.verbose:
                    print("Reached maximum number of function evaluations %d" % self.maxEvals)
                    break

        print ("Done fitting.")

    

    def sgd_fit(self, theta, batch_size=0, *args):

        print ("Training model via SGD.")

        # Parameters of the Optimization
        optTol = 1e-2
        n, d = self.X.shape

        # Initial guess
        self.w = np.zeros(d)
        self.hist_grad = 0
        funEvals = 1

        while True:

            (delta, f_new, g) = self.privateFun(
                theta, self.w, batch_size, *args)
            funEvals += 1

            # Print progress
            if self.verbose > 0:
                print("%d - loss: %.3f" % (funEvals, f_new))
                print("%d - g_norm: %.3f" % (funEvals, norm(g)))

            # Update parameters
            self.w = self.w + delta

            # Test termination conditions
            optCond = norm(g, float('inf'))

            if optCond < optTol:
                if self.verbose:
                    print("Problem solved up to optimality tolerance %.3f" % optTol)
                break

            if funEvals >= self.maxEvals:
                if self.verbose:
                    print("Reached maximum number of function evaluations %d" %
                          self.maxEvals)
                break

        print ("Done fitting.")

    def fit(self):

        print ("Normal Fit.")

        (self.w, self.alpha, f, _) = minimizers.findMin(self.funObj, self.w, self.alpha,
                                                        self.maxEvals,
                                                        self.verbose,
                                                        self.X,
                                                        self.y)

        print("Training error: %.3f" %
              utils.classification_error(self.predict(self.X), self.y))

    def getParameters(self):
        return self.w

    def predict(self, X):
        w = self.w
        yhat = np.dot(X, w)
        return np.sign(yhat)

    '''
    Original model
    def privatePredict(self, X, scale):
        _, d = X.shape
        w = self.w + utils.exp_noise(scale=scale, size=d)
        yhat = np.dot(X, w)
        return np.sign(yhat)
    '''

    def privatePredict(self, X, epsilon):
        nn, dd = X.shape
        w = self.w
        yhat = np.dot(X, w)

        # TODO: Estimate the L1 Sensitivity
        sens = 0.25 * dd * dd + 3 * dd

        return np.sign(yhat + utils.lap_noise(loc=0, scale=sens / epsilon, size=nn))


class logRegL2(logReg):

    def __init__(self, X, y, lammy, verbose=1, maxEvals=100):
        self.lammy = lammy
        self.verbose = verbose
        self.maxEvals = maxEvals

        self.X = X
        self.y = y

        self.iter = 1
        self.init_alpha = 1e-2
        self.alpha = self.init_alpha

        n, self.d = self.X.shape
        self.w = np.zeros(self.d)
        self.hist_grad = np.zeros(self.d)

        #utils.check_gradient(self, self.X, self.y)

    def funObj(self, ww, X, y, scale=False, batch_size=1, n=1):
        yXw = y * X.dot(ww)

        # Calculate the function value
        f = (1/n)*np.sum(np.logaddexp(0, -yXw)) + 0.5 * self.lammy * ww.T.dot(ww)

        # Calculate the gradient value
        res = - y / np.exp(np.logaddexp(0, yXw))
        if scale:
            s = np.linalg.norm(X.T.dot(res))
            if (s > 1):
                g = (1/batch_size)*X.T.dot(res)/s + self.lammy * ww
            else:
                g = (1/batch_size)*X.T.dot(res) + self.lammy * ww
        else:
            g = (1/batch_size)*X.T.dot(res) + self.lammy * ww
            
                

        return f, g

    def privatePredict(self, X, epsilon):
        nn, dd = X.shape
        w = self.w
        yhat = np.dot(X, w)

        # TODO: Estimate the L1 Sensitivity
        # sens = 0.25 * dd * dd + 3 * dd
        sens = 2 / (nn * self.lammy)

        return np.sign(yhat + utils.lap_noise(loc=0, scale=sens / epsilon, size=nn))


class logRegL1(logReg):

    def __init__(self, X, y, lammy, verbose=1, maxEvals=100):
        self.lammy = lammy
        self.verbose = verbose
        self.maxEvals = maxEvals

        self.X = X
        self.y = y
        self.alpha = 1

        n, self.d = self.X.shape
        self.w = np.zeros(self.d)

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.logaddexp(0, -yXw))

        # Calculate the gradient value
        res = - y / np.exp(np.logaddexp(0, yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self):

        nn, dd = self.X.shape

        # Initial guess
        self.w = np.zeros(dd)
        (self.w, f) = minimizers.findMinL1(self.funObj,
                                           self.w,
                                           self.lammy,
                                           self.maxEvals,
                                           self.verbose,
                                           self.X, self.y)

# L0 Regularized Logistic Regression


class logRegL0(logReg):
    # this is class inheritance:
    # we "inherit" the funObj and predict methods from logReg
    # and we overwrite the __init__ and fit methods below.
    # Doing it this way avoids copy/pasting code.
    # You can get rid of it and copy/paste
    # the code from logReg if that makes you feel more at ease.
    def __init__(self, X, y, lammy=1.0, verbose=1, maxEvals=400):
        self.verbose = verbose
        self.lammy = lammy
        self.maxEvals = maxEvals

        self.X = X
        self.y = y
        self.alpha = 1

        n, self.d = self.X.shape
        self.w = np.zeros(self.d)

    def fitSelected(self, selected):
        n, d = self.X.shape
        w0 = np.zeros(self.d)

        def minimize(ind): return minimizers.findMin(self.funObj,
                                                     w0[ind],
                                                     self.alpha,
                                                     self.maxEvals,
                                                     self.verbose,
                                                     self.X[:, ind], self.y)

        # re-train the model one last time using the selected features
        self.w = w0
        self.w[selected], _, _, _ = minimize(selected)

    def fit(self):
        n, d = self.X.shape
        w0 = np.zeros(self.d)

        def minimize(ind): return minimizers.findMin(self.funObj,
                                                     w0[ind],
                                                     self.alpha,
                                                     self.maxEvals,
                                                     self.verbose,
                                                     self.X[:, ind], self.y)
        selected = set()
        selected.add(0)  # always include the bias variable
        minLoss = np.inf
        oldLoss = 0
        bestFeature = -1

        while minLoss != oldLoss:
            oldLoss = minLoss
            if self.verbose > 1:
                print("Epoch %d " % len(selected))
                print("Selected feature: %d" % (bestFeature))
                print("Min Loss: %.3f\n" % minLoss)

            for i in range(d):
                if i in selected:
                    continue

                selected_new = selected | {i}  # add "i" to the set
                # TODO: Fit the model with 'i' added to the features,
                # then compute the score and update the minScore/minInd

                # TODO: Fit the model with 'i' added to the features,
                # then compute the score and update the minScore/minInd
                sl = list(selected_new)
                temp_w, _, _, _ = minimize(sl)

                # pdb.set_trace()

                loss, _ = self.funObj(temp_w, self.X[:, sl], self.y)

                if loss < minLoss:
                    minLoss = loss
                    bestFeature = i

            selected.add(bestFeature)

        # re-train the model one last time using the selected features
        self.w = w0
        self.w[list(selected)], _, _, _ = minimize(list(selected))
