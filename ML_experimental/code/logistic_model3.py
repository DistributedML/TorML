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
        
        
        
    def normal(self, eta, ww, Xbatch, ybatch, batch_size=0): 
    
        nn, dd = self.X.shape

        if batch_size > 0 and batch_size < nn:
            g = self.funObj(ww, Xbatch, ybatch, True, batch_size, nn)
        else:
            # Just take the full range
            g = self.funObj(ww, self.X, self.y,True)
    
    
        # step magnitude as implemented in Song, Chaudhuri and Sarwate (2013)
        #delta = (-1)*eta*(g+(1/batch_size)*Z)
        
        #if batch_size > 0 and batch_size < nn:
            #g_new = self.funObj(w_new, Xbatch, ybatch, True, batch_size, nn)
        #else:
            #g_new = self.funObj(w_new, self.X, self.y,True)
            
        return g

    


    
    def privateFun2(self, eta, alpha, ww,  Z, Xbatch, ybatch, batch_size=0): 
    
        nn, dd = self.X.shape

        if batch_size > 0 and batch_size < nn:
            g = self.funObj(ww, Xbatch, ybatch, True, batch_size, nn)
        else:
            # Just take the full range
            g = self.funObj(ww, self.X, self.y,True)
    
    
        # step magnitude as implemented in Song, Chaudhuri and Sarwate (2013)
        #delta = (-1)*eta*(g+(1/batch_size)*Z)
        gpriv = g+(1/batch_size)*Z
        
        #if batch_size > 0 and batch_size < nn:
            #g_new = self.funObj(w_new, Xbatch, ybatch, True, batch_size, nn)
        #else:
            #g_new = self.funObj(w_new, self.X, self.y,True)
            
        return gpriv


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
        #f = (1/n)*np.sum(np.logaddexp(0, -yXw)) + 0.5 * self.lammy * ww.T.dot(ww)

        # Calculate the gradient value
        res = - y / np.exp(np.logaddexp(0, yXw))
        if scale:
            s = np.linalg.norm(X.T.dot(res))
            #g = (1/batch_size)*X.T.dot(res)/max(s,1) + self.lammy * ww
            g = (1/batch_size)*X.T.dot(res)/max(s,1)
    
        else:
            #g = (1/batch_size)*X.T.dot(res) + self.lammy * ww
            g = (1/batch_size)*X.T.dot(res)
            
            
        return g




