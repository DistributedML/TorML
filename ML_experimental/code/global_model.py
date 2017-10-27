from __future__ import division
import numpy as np
import minimizers
import utils
import pdb
from numpy.linalg import norm
import emcee
#import matplotlib.pyplot as pl
import random
import math
import minimizers
import utils



class globalModel:

    # Logistic Regression
    def __init__(self, logistic=False, verbose=1, maxEvals=400):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.models = []
        self.weights = np.empty(0)
        self.logistic = logistic

    def add_model(self, model):
        self.models.append(model)

    def fit(self, theta, batch_size=0, *args):

        print ("Training global model.")

        # Parameters of the Optimization
        optTol = 1e-2
        i = 0
        n, d = self.models[0].X.shape

        # Initial guess
        self.w = np.zeros(d)
        funEvals = 1

        while True:

            (delta, f_new, g) = self.models[i % len(self.models)].privateFun(
                theta, self.w, batch_size, *args)
            funEvals += 1
            i += 1

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

        print ("Done fitting global model.")
    
    
    def sgd_fit_private(self, alpha, eta, batch_size=0, *args):
        
        print ("Training model via private SGD.")
        
        # Parameters of the Optimization
        optTol = 1e-2
        i = 0
        n, d = self.models[0].X.shape
        collectionX = []
        collectionY = []
        
        for k in range(0,len(self.models)):
            batchesX = []
            batchesY = []
            within = True
            j=0
            while within:
                if (j+batch_size <= n):
                    batchesX.append(self.models[k].X[j:j+batch_size,:])
                    batchesY.append(self.models[k].y[j:j+batch_size])
                    j+=batch_size
                else:
                    within = False
            
            collectionX.append(batchesX)
            collectionY.append(batchesY)

        #---------------------------------------------------------------------------
        #Generate random samples from isotropic multivariate laplace distribution using emcee
        
        def lnprob(x,alpha):
            return -(alpha/2)*np.linalg.norm(x)
        
        ndim = d
        #ndim=10
        nwalkers = max(4*d,250)
        #nwalkers=50
        p0 = [np.random.rand(ndim) for i in range(nwalkers)]
        #p0 = np.random.randn(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,args=[alpha])
        
        pos, prob, state = sampler.run_mcmc(p0,100)
        sampler.reset()
        print(d)
        
        #sampler.run_mcmc(pos, 1000)
        sampler.run_mcmc(pos, 1000,rstate0=state)
        
        print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))
        #print("Autocorrelation time:", sampler.get_autocorr_time())
        
        sample = sampler.flatchain
        #---------------------------------------------------------------------------
        
        
        # Initial guess
        self.w = np.zeros(d)
        funEvals = 1
        
        while True:
            
            d1,d2 = sample.shape
            z = np.random.randint(0,d1)
            Z = sample[z]
            
            l = np.random.randint(0,len(collectionX[i% len(self.models)]))
            Xbatch = collectionX[i% len(self.models)][l]
            ybatch = collectionY[i% len(self.models)][l]

            (delta, f_new, g) = self.models[i % len(self.models)].privateFun2(eta,alpha,self.w, Z, Xbatch, ybatch, batch_size, *args)

            funEvals += 1
            i += 1
        
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

        print ("Done fitting global model.")
    
    

    

    def selectFeatures(self, minVotes):

        n, d = self.models[0].X.shape
        votes = np.zeros(d)

        for i in range(len(self.models)):
            votes += (self.models[i].w == 0).astype(int)

        return np.where(votes < minVotes)[0]

    def predict(self, X):
        w = self.w

        if self.logistic:
            yhat = np.sign(np.dot(X, w))
        else:
            yhat = np.dot(X, w)

        return yhat

    def predictAverage(self, X, epsilon=1):
        n, d = X.shape
        yhats = {}
        yhat_total = np.zeros(n)

        # Aggregation function
        for i in range(len(self.models)):
            yhats[i] = self.models[i].privatePredict(X, epsilon)
            yhat_total = yhat_total + yhats[i]

        if self.logistic:
            yhat = np.sign(yhat_total)
        else:
            yhat = yhat_total / float(len(self.models))

        return yhat

    def fitWeightedAverage(self, X, y, epsilon=1):

        n, d = X.shape
        k = len(self.models)

        modelX = np.zeros(shape=(n, k))

        for i in range(k):
            modelX[:, i] = self.models[i].privatePredict(X, epsilon)

        A = np.dot(modelX.T, modelX)
        B = np.dot(modelX.T, y)

        self.weights = np.linalg.solve(A, B)

    def predictWeightedAverage(self, X):

        n, d = X.shape
        k = len(self.models)

        modelX = np.zeros(shape=(n, k))

        for i in range(k):
            modelX[:, i] = self.models[i].predict(X)

        if self.logistic:
            yhat = np.sign(np.dot(modelX, self.weights.T))
        else:
            yhat = np.dot(modelX, self.weights.T)

        return yhat


class globalModelSVM(globalModel):

     # Logistic Regression
    def __init__(self, logistic=False, verbose=1, maxEvals=400):

        self.verbose = verbose
        self.maxEvals = maxEvals
        self.models = []

        self.weights = np.empty(0)
        self.logistic = logistic

    def add_model(self, model):
        self.models.append(model)

    # Uses the pegasos algorithm for the global SVM fitting
    def fit(self, batch_size=0, *args):

        print ("Training global model with Pegasos algorithm")

        # Parameters of the Optimization
        optTol = 1e-2
        i = 1
        n, d = self.models[0].X.shape

        # Initial guess
        self.w = np.zeros(d)

        while True:

            learn_rate = 1 / (0.1 * i)
            delta = self.models[
                i % len(self.models)].svmUpdate(self.w, batch_size, *args)
            i += 1

            # Update parameters
            self.w = (1 - 0.1 * learn_rate) * self.w + learn_rate * delta

            if i >= self.maxEvals:
                if self.verbose:
                    print("Reached maximum number of function evaluations %d" %
                          self.maxEvals)
                break

        print ("Done fitting global model.")
