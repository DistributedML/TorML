from __future__ import division
import numpy as np
import pdb

GROUP_SIZE = 5


# Performs a krum aggregation
def krum(ww, deltas):
    return ww + deltas[np.argmin(get_krum_scores(deltas, GROUP_SIZE))]


def get_krum_scores(X, groupsize):

    krum_scores = np.zeros(len(X))

    # Calculate distances
    distances = np.sum(X**2, axis=1)[:, None] + np.sum(
        X**2, axis=1)[None] - 2 * np.dot(X, X.T)

    for i in range(len(X)):
        krum_scores[i] = np.sum(np.sort(distances[i])[1:(groupsize + 1)])

    return krum_scores


if __name__ == "__main__":
    sample = np.random.rand(50, 5)
    pdb.set_trace()
