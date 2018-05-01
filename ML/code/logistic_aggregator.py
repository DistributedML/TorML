from __future__ import division
import numpy as np
import pdb
import falconn

n = 0
d = 0
hit_matrix = np.zeros(1)
it = 0


def init(num_clients, num_features):

    global d
    d = num_features

    global n
    n = num_clients

    global hit_matrix
    hit_matrix = np.zeros((n, n))


def lsh_sieve(full_deltas, test_distance):

    deltas = np.reshape(full_deltas, (n, d))
    centred_deltas = (deltas - np.mean(deltas, axis=0))

    params = falconn.get_default_parameters(n, d)
    fln = falconn.LSHIndex(params)
    fln.setup(centred_deltas)
    qob = fln.construct_query_object()

    # Greedy merge within a distance
    # all_sets = list()

    full_grad = np.zeros(d)

    heur_distance = 1.5 * np.mean(np.std(centred_deltas, axis=1)) / n

    for i in range(n):
        neighbors = qob.find_near_neighbors(centred_deltas[i], test_distance)
        hit_matrix[i][neighbors] += 1
        # full_grad += (deltas[i] / len(neighbors))

    global hit_matrix
    hit_matrix = hit_matrix - np.eye(n)

    # Take the inverse L2 norm
    wv = 1.0 / (np.linalg.norm(hit_matrix, axis=1) + 1)

    # Normalize to have sum equal to number of clients
    wv = wv * n / np.linalg.norm(wv)

    # Apply the weights
    full_grad += np.dot(deltas.T, wv)

    global it
    it += 1

    return full_grad, heur_distance

def euclidean_binning(full_deltas, d, n):
    deltas = np.reshape(full_deltas, (n, d))
    centered_deltas = (deltas - np.mean(deltas, axis=0))

    # distance range of the euclidean norm to be considered a near neighbor
    threshhold = 0.001

    full_grad = np.zeros(d)
    nnbs = []

    for i in range(n):
        nnb = 1
        # Count nearby gradients within threshhold
        for j in range(n):
            # print(np.linalg.norm(centered_deltas[i] - centered_deltas[j]))
            if i != j and np.linalg.norm(centered_deltas[i] - centered_deltas[j]) < threshhold:
                nnb += 1
        nnbs.append(nnb)
        full_grad = full_grad + deltas[i] / nnbs[i]
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
