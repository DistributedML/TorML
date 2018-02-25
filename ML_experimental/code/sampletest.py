import emcee
import numpy as np
import utils
import pdb


def lnprob(x, alpha):
    return -(alpha / 2) * np.linalg.norm(x)


data1 = utils.load_dataset("creditpositive")
X1, y1 = data['X'], data['y']

data2 = utils.load_dataset("creditnegative1")
X2, y2 = data['X'], data['y']

if __name__ == "__main__":

    # SAMPLING CODE
    alpha = 5.0
    dd = 25
    ndim = 25
    nwalkers = max(4 * dd, 250)
    # print(nwalkers)
    p0 = [np.random.rand(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[alpha])

    pos, prob, state = sampler.run_mcmc(p0, 100)
    sampler.reset()
    sampler.run_mcmc(pos, 1000, rstate0=state)

    print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))

    sample = sampler.flatchain

    d1,d2 = sample.shape
    z = np.random.randint(0,d1)
    Z = sample[z]
