from __future__ import division
import numpy as np
import pdb
import falconn


def lsh_sieve(full_deltas, d, n):

    deltas = np.reshape(full_deltas, (n, d))
    centred_deltas = (deltas - np.mean(deltas, axis=0))

    params = falconn.get_default_parameters(n, d)
    fln = falconn.LSHIndex(params)
    fln.setup(centred_deltas)
    qob = fln.construct_query_object()

    # Greedy merge within a distance
    # all_sets = list()

    full_grad = np.zeros(d)

    # The number of neighbors
    nnbs = []

    heur_distance = np.min(np.std(centred_deltas, axis=1)) / n
    test_distance = 1.0 / (50 * d)

    for i in range(n):
        neighbors = qob.find_near_neighbors(centred_deltas[i], test_distance)
        nnbs.append(len(neighbors))
        full_grad = full_grad + (deltas[i] / len(neighbors))

    # pdb.set_trace()

    return full_grad, nnbs


def average(full_deltas, d, n):

    deltas = np.reshape(full_deltas, (n, d))
    return np.mean(deltas, axis=0), 0


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

    good = (np.random.rand(50, 5) - 0.5) * 2
    attackers = np.random.rand(10, 5) + 0.5

    sample = np.vstack((good, attackers))

    lsh_sieve(sample.flatten(), 5, 60)

    pdb.set_trace()
