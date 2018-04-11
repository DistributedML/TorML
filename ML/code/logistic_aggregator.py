from __future__ import division
import numpy as np
import pdb
import falconn


def setup_falconn(full_deltas):

    n, d = full_deltas.shape

    params = falconn.get_default_parameters(n, d)
    fln = falconn.LSHIndex(params)
    fln.setup(full_deltas)
    qob = fln.construct_query_object()

    # Get the k nearest
    # qob.find_k_nearest_neighbors(full_deltas[0], 5)

    # Within a distance threshold
    for i in range(n):
        num_neighbors = qob.find_near_neighbors(full_deltas[i], 1.0 / d)
        print str(i) + " has " + str(num_neighbors)


# Returns the index of the row that should be used in Krum
def krum(full_deltas, dd, groupsize):

    # assume deltas is an array of size group * d
    deltas = np.reshape(full_deltas, (groupsize, dd))
    return np.argmin(get_krum_scores(deltas, groupsize))


def get_krum_scores(X, groupsize):

    krum_scores = np.zeros(len(X))

    # Calculate distances
    distances = np.sum(X**2, axis=1)[:, None] + np.sum(
        X**2, axis=1)[None] - 2 * np.dot(X, X.T)

    for i in range(len(X)):
        krum_scores[i] = np.sum(np.sort(distances[i])[1:(groupsize - 1)])

    return krum_scores


if __name__ == "__main__":

    sample = np.random.rand(50, 5)
    setup_falconn(sample)
