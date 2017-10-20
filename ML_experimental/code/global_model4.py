from __future__ import division
import numpy as np
import logistic_model_test
from numpy.linalg import norm
import emcee
#import matplotlib.pyplot as pl
import matplotlib.pyplot as plt




class globalModel4:

    # Logistic Regression
    def __init__(self, logistic=False, verbose=1, maxEvals=400):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.models = []
        self.weights = np.empty(0)
        self.logistic = logistic

    def add_model(self, model):
        self.models.append(model)

    
    
    
    def sgd_fit_private(self, alpha, XBin, yBin, XBinValid, yBinValid, eta, batch_size=0,dataset='', *args):
        
        #print ("Training model via private SGD.")
        
        # Parameters of the Optimization
        optTol = 1e-2
        n, d = self.models[0].X.shape
        collectionX = []
        collectionY = []
        #print(alpha)
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
        #print(nwalkers)
        p0 = [np.random.rand(ndim) for i in range(nwalkers)]
        #p0 = np.random.randn(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,args=[alpha])
        
        pos, prob, state = sampler.run_mcmc(p0,100)
        sampler.reset()
        #print(d)
        
        #sampler.run_mcmc(pos, 1000)
        sampler.run_mcmc(pos, 1000,rstate0=state)
        
        #print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))
        #print("Autocorrelation time:", sampler.get_autocorr_time())
        
        sample = sampler.flatchain
        #---------------------------------------------------------------------------
        
        fValues = []
        iterations = []
        
        # Initial guess
        self.w = np.random.rand(d)
        #self.w = np.zeros(d)
        funEvals = 1
        i=1
        
        train_progress = []
        test_progress = []
        
        while True:
            
            d1,d2 = sample.shape
            z = np.random.randint(0,d1)
            Z = sample[z]
            
            l = np.random.randint(0,len(collectionX[0]))
            Xbatch = collectionX[0][l]
            ybatch = collectionY[0][l]

            (delta, f_new, g) = self.models[0].privateFun2(eta,alpha,self.w, Z, Xbatch, ybatch, batch_size, *args)
            
            
            #if i%1000 == 0:
            iterations.append(i)
            fValues.append(f_new)
        
    
            i+=1
            funEvals += 1
    
        
            # Print progress
            if self.verbose > 0:
                print("%d - loss: %.3f" % (funEvals, f_new))
                print("%d - g_norm: %.3f" % (funEvals, norm(g)))
        
            # Update parameters
            self.w = self.w + delta
            
            train_progress.append(logistic_model_test.train_error(self.w, XBin, yBin))
            test_progress.append(logistic_model_test.test_error(self.w, dataset, XBinValid, yBinValid))
            
            # Test termination conditions
            optCond = norm(g, float('inf'))
            
            #print("alpha = ", alpha)
            
            if optCond < optTol:
                #print("1",f_new)
                if self.verbose:
                    print("Problem solved up to optimality tolerance %.3f" % optTol)
                break

            if funEvals >= self.maxEvals:
                #print("2",f_new)
                if self.verbose:
                    print("Reached maximum number of function evaluations %d" %
                          self.maxEvals)
                break 
   
# -------------------------------------------------------------------------------------
         
        #For plotting training and validation error against iterations     
            
            
        #s = 'alpha = ' + str(alpha)
        s2 = 'dataset = ' + dataset + ' & alpha size = ' + str(alpha)
        #fig = plt.figure()
        plt.plot(train_progress,label="Training")
        plt.plot(test_progress,label="Validation")
        plt.ylabel('Training & Validation error')
        plt.xlabel('Number of iterations')
        plt.title(s2)
        plt.legend()



    
    


