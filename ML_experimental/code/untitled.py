import emcee
import numpy as np
import pdb


def lnprob(x, alpha):
    return -(alpha / 2) * np.linalg.norm(x)


if __name__ == "__main__":

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

    pdb.set_trace()
