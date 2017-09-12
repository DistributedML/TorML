from __future__ import division
import numpy as np
from numpy.linalg import norm


def findMin(funObj, w, init_alpha, maxEvals, verbose, *args):
    """
    Uses gradient descent to optimize the objective function
    
    This uses quadratic interpolation in its line search to
    determine the step size alpha
    """
    # Parameters of the Optimization
    optTol = 1e-2
    gamma = 1e-4

    # Evaluate the initial function value and gradient
    f, g = funObj(w,*args)
    funEvals = 1
    optTolSwitch = 0

    alpha = init_alpha
    while True:

        # Line-search using quadratic interpolation to find an acceptable value of alpha
        gg = g.T.dot(g)

        while True:
            w_new = w - alpha * g
            f_new, g_new = funObj(w_new, *args)

            funEvals += 1

            if f_new <= f - gamma * alpha*gg:
                break

            if verbose > 1:
                print("f_new: %.3f - f: %.3f - Backtracking..." % (f_new, f))
         
            # Update step size alpha
            alpha = (alpha**2) * gg/(2.*(f_new - f + alpha*gg))

        # Print progress
        if verbose > 0:
            print("%d - g_norm: %.3f" % (funEvals, norm(g_new)))
            print("%d - loss: %.3f" % (funEvals, f_new))

        # Update step-size for next iteration
        y = g_new - g
        alpha = -alpha * np.dot(y.T, g) / np.dot(y.T, y)
       
        # Safety guards
        if np.isnan(alpha) or alpha < 1e-10 or alpha > 1e10:
            alpha = 1.

        if verbose > 1:
            print("alpha: %.6f" % (alpha))

        # Update parameters/function/gradient
        w = w_new
        f = f_new
        g = g_new

        # Test termination conditions
        optCond = norm(g, float('inf'))

        if optCond < optTol:
            if verbose:
                print("Problem solved up to optimality tolerance %.3f" % optTol)
                optTolSwitch = 1
            break

        if funEvals >= maxEvals:
            if verbose:
                print("Reached maximum number of function evaluations %d" % maxEvals)
            break

    return w, alpha, f, optTolSwitch

def findMinL1(funObj, ww, L1, maxEvals, verbose, *args):
    """
    Uses the L1 proximal gradient descent to optimize the objective function
    
    The line search algorithm divides the step size by 2 until
    it find the step size that results in a decrease of the L1 regularized
    objective function
    """
    # Parameters of the Optimization
    optTol = 1e-2
    gamma = 1e-4

    # Evaluate the initial function value and gradient
    f, g = funObj(ww,*args)
    funEvals = 1

    alpha = 1.
    proxL1 = lambda ww, alpha: np.sign(ww) * np.maximum(abs(ww)- L1*alpha,0)
    L1Term = lambda ww: L1 * np.sum(np.abs(ww))
    
    while True:
        gtd = None
        # Start line search to determine alpha      
        while True:

            w_new = ww - alpha * g
            w_new = proxL1(w_new, alpha)

            if gtd is None:
                gtd = g.T.dot(w_new - ww)

            f_new, g_new = funObj(w_new, *args)
            funEvals += 1

            if f_new + L1Term(w_new) <= f + L1Term(ww) + gamma*alpha*gtd:
                # Wolfe condition satisfied, end the line search
                break

            if verbose > 1:
                print("Backtracking... f_new: %.3f, f: %.3f" % (f_new, f))
            
            # Update alpha
            alpha /= 2.

        # Print progress
        if verbose > 0:
            print("%d - alpha: %.3f - loss: %.3f" % (funEvals, alpha, f_new))

        # Update step-size for next iteration
        y = g_new - g
        alpha = -alpha*np.dot(y.T,g) / np.dot(y.T,y)

        # Safety guards
        if np.isnan(alpha) or alpha < 1e-10 or alpha > 1e10:
            alpha = 1.

        # Update parameters/function/gradient
        ww = w_new
        f = f_new
        g = g_new

        # Test termination conditions
        optCond = norm(proxL1(ww - g, 1.0), float('inf'))

        if optCond < optTol:
            if verbose:
                print("Problem solved up to optimality tolerance %.3f" % optTol)
            break

        if funEvals >= maxEvals:
            if verbose:
                print("Reached maximum number of function evaluations %d" % maxEvals)
            break

    return ww, f