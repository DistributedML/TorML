from __future__ import division
import numpy as np
import minimizers
import utils
import pdb
from numpy.linalg import norm
import emcee
#import matplotlib.pyplot as pl
import matplotlib.pyplot as plt
import random
import math
import minimizers
import utils
from matplotlib.backends.backend_pdf import PdfPages



class globalModel3:

    # Logistic Regression
    def __init__(self, logistic=False, verbose=1, maxEvals=400):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.models = []
        self.weights = np.empty(0)
        self.logistic = logistic

    def add_model(self, model):
        self.models.append(model)

    
    
    def sgd_fit_private(self, lammy, alpha, eta, batch_size=0, *args):
        
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
        
        #fValues = []
        #iterations = []
        
        # Initial guess for victim's isolated SGD
        wvic = np.random.rand(d)
        # Initial guess for attacker and victim together
        self.w = np.random.rand(d)
        #self.w = np.zeros(d)
        funEvals1 = 1
        funEvals2 = 1
        i1=1
        i2=1
        
        # collection of victim's deduced (differentially private) gradients
        spying = []
        # collection of victim's own gradients
        victim = []
        
        #1 is attacker and 2 is victim
        
        # First victim's own SGD
        
        while True:
       
            l2 = np.random.randint(0,len(collectionX[1]))

            Xbatch2 = collectionX[1][l2]
            ybatch2 = collectionY[1][l2]
            
            g3 = self.models[1].normal(eta, wvic, Xbatch2, ybatch2, batch_size, *args)
            victim.append(g3)
            
            #if i%1000 == 0:
            #iterations.append(i)
            #fValues.append(f_new)
        
    
            i1+=1
            funEvals1 += 1
    
            # Update parameters

            deltavic = (-1)*eta*(g3 + lammy * wvic)
            wvic = wvic + deltavic
            
            # Test termination conditions
            optCond = norm(g3 + lammy * wvic, float('inf'))
            
            
            if i1 % 1000 == 0:
                print(i1)
            
            if optCond < optTol:
                if self.verbose:
                    print("Problem solved up to optimality tolerance %.3f" % optTol)
                break

            if funEvals1 >= self.maxEvals:
                if self.verbose:
                    print("Reached maximum number of function evaluations %d" %
                          self.maxEvals)
                break 
           
        # Now attacker & victim together
        
        while True:
            
            d1,d2 = sample.shape
            z1 = np.random.randint(0,d1)
            z2 = np.random.randint(0,d1)

            Z1 = sample[z1]
            Z2 = sample[z2]

            l1 = np.random.randint(0,len(collectionX[0]))
            l2 = np.random.randint(0,len(collectionX[1]))

            Xbatch1 = collectionX[0][l1]
            ybatch1 = collectionY[0][l1]
            
            Xbatch2 = collectionX[1][l2]
            ybatch2 = collectionY[1][l2]
            

            g1 = self.models[0].privateFun2(eta,alpha,self.w, Z1, Xbatch1, ybatch1, batch_size, *args)
            
            # victim's gradient
            g2 = self.models[1].privateFun2(eta,alpha,self.w, Z2, Xbatch2, ybatch2, batch_size, *args)
            
            # attacker can deduce victim's gradient
            spying.append(g2)
            
            #if i%1000 == 0:
            #iterations.append(i)
            #fValues.append(f_new)
        
    
            i2+=1
            funEvals2 += 1
    
            # Update parameters
            delta = (-1)*eta*(g1 + g2 + lammy * self.w)
            self.w = self.w + delta
            
            # Test termination conditions
            optCond = norm(g1 + g2 + lammy * self.w, float('inf'))
            
            
            if i2 % 10000 == 0:
                print(i2)
            
            if optCond < optTol:
                if self.verbose:
                    print("Problem solved up to optimality tolerance %.3f" % optTol)
                break

            if funEvals2 >= self.maxEvals:
                if self.verbose:
                    print("Reached maximum number of function evaluations %d" %
                          self.maxEvals)
                break 
            
        
        wspy = np.random.rand(d)
        for g in spying:
            
            delta = (-1)*eta*(g + lammy * wspy)
            wspy = wspy + delta
            
        dist = np.linalg.norm(wspy-wvic)
        return dist
            
            
         
#        i=0 
#        step=10
#        faverage=[]
#        while i < len(fValues):
#            end=min(i+step,len(fValues))
#            a=fValues[i:end]
#            b=np.mean(a)*np.ones((len(a),1))
#            faverage.append(b)
#            i=i+step
#        
#        a = faverage[0]
#        for j in range(1,len(faverage)):
#            a=np.concatenate((a,faverage[j]))
#            
#        fValuesAverage = a 
#        
#        
#            
#            
#        s = 'alpha = ' + str(alpha)
#        #'alpha = [0.3, 0.5, 1, 1.5]'
#        s2 = 'dataset = ' + dataset + ' & batch size = ' + str(batch_size)
#        #fig = plt.figure()
#        plt.plot(iterations,fValuesAverage,label=s)
#        plt.ylabel('Value of objective')
#        plt.xlabel('Number of iterations')
#        plt.title(s2)
#        plt.legend()
#        #fig.savefig("foo.jpeg", bbox_inches='tight')
#        #print("alpha = ", alpha)
#        #pylab.title('alpha= %s'%(alpha))
#
#        #print ("Done fitting global model.")
#        #plot1 = plotGraph(tempDLstats, tempDLlabels)
#        #plot2 = plotGraph(tempDLstats_1, tempDLlabels_1)
#        #plot3 = plotGraph(tempDLstats_2, tempDLlabels_2)
#        #pp = PdfPages('foo.pdf')
#        #pp.savefig(plot1)
#        #pp.savefig(plot2)
#        #pp.savefig(plot3)
#        #pp.close()
#        return f_new  

    
    


