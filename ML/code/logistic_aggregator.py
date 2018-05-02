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
    nnbs = []

    full_grad = np.zeros(d)

    heur_distance = 1.5 * np.mean(np.std(centred_deltas, axis=1)) / n

    for i in range(n):
        neighbors = qob.find_near_neighbors(centred_deltas[i], test_distance)
        hit_matrix[i][neighbors] += 1
        nnbs.append(len(neighbors))


    global hit_matrix
    hit_matrix = hit_matrix - np.eye(n)

    new_hm = np.zeros((n, n))
    collusions = np.zeros(n);
    # Find number of collusions per node
    for i in range(n):
        collusions[i] = np.sum(hit_matrix[i])
    # Reweight based on differences in sum
    for i in range(n):
        for j in range(n):
            # add 1 for divby0 and so that poisoners don't benefit from this
            new_hm[i][j] = hit_matrix[i][j] / (np.abs(collusions[i] - collusions[j]) + 1)

    # Take the inverse L2 norm
    wv = 1.0 / (np.linalg.norm(new_hm, axis=1) + 1)

    # Apply the weights
    full_grad += np.dot(deltas.T, wv)

    global it
    it += 1

    return full_grad, heur_distance, nnbs

def search_distance_euc(full_deltas, distance):
    std = np.std(get_nnbs_euc(full_deltas, distance))

    if std == 0:
        # no poisoners
        if distance < 1e-10:
            return 0
        else:
            return search_distance_euc(full_deltas, distance/2)
    std_left = np.std(get_nnbs_euc(full_deltas, distance/2))
    std_right = np.std(get_nnbs_euc(full_deltas, distance + distance/2))
    # print("Std_left: " + str(std_left) + "distance left: " + str(distance/2) + " Std_right: " + str(std_right) + "distance: " + str(distance) + " std: " + str(std))
    if std_left <= std and std_right <= std:
        return distance
    if std_left > std_right:
        return search_distance_euc(full_deltas, distance/2)
    else:
        return search_distance_euc(full_deltas, distance + distance/2)

def search_distance_lsh(full_deltas, distance, prev_std, typical_set):
    std = np.std(get_nnbs_lsh(full_deltas, distance))
    # First run to initialize prev_std
    if prev_std == -float('Inf'):
        return search_distance_lsh(full_deltas, distance/2, std, typical_set)
    # Until you reach typical set, lsh surprisingly returns same nnbs
    if prev_std == std:
        # no poisoners
        if distance < 1e-10:
            return 0
        else:
            return search_distance_lsh(full_deltas, distance/2, std, False)

    std_left = np.std(get_nnbs_lsh(full_deltas, distance/2))
    std_right = np.std(get_nnbs_lsh(full_deltas, distance + distance/2))
    # print("Std_left: " + str(std_left) + "distance left: " + str(distance/2) + " Std_right: " + str(std_right) + "distance: " + str(distance) + " std: " + str(std))
    if std_left <= std and std_right <= std:
        return distance
    if std_left > std_right:
        return search_distance_lsh(full_deltas, distance/2, std, True)
    else:
        return search_distance_lsh(full_deltas, distance + distance/2, std, True)


def get_nnbs_euc(full_deltas, distance):
    deltas = np.reshape(full_deltas, (n, d))
    centered_deltas = (deltas - np.mean(deltas, axis=0))

    nnbs = []

    for i in range(n):
        nnb = 1
        # Count nearby gradients within threshhold
        for j in range(n):
            # print(np.linalg.norm(centered_deltas[i] - centered_deltas[j]))
            if i != j and np.linalg.norm(centered_deltas[i] - centered_deltas[j]) < distance:
                nnb += 1
        nnbs.append(nnb)
    return nnbs

def get_nnbs_lsh(full_deltas, distance):
    deltas = np.reshape(full_deltas, (n, d))
    centred_deltas = (deltas - np.mean(deltas, axis=0))

    params = falconn.get_default_parameters(n, d)
    fln = falconn.LSHIndex(params)
    fln.setup(centred_deltas)
    qob = fln.construct_query_object()

    nnbs = []

    for i in range(n):
        neighbors = qob.find_near_neighbors(centred_deltas[i], distance)
        nnbs.append(len(neighbors))
    return nnbs

def euclidean_binning_hm(full_deltas, distance):
    global hit_matrix
    deltas = np.reshape(full_deltas, (n, d))
    centered_deltas = (deltas - np.mean(deltas, axis=0))

    full_grad = np.zeros(d)
    nnbs = []
    for i in range(n):
        nnb = 1
        # Count nearby gradients within distance
        for j in range(n):
            # print(np.linalg.norm(centered_deltas[i] - centered_deltas[j]))
            if i != j and np.linalg.norm(centered_deltas[i] - centered_deltas[j]) < distance:
                hit_matrix[i][j] += 1
                nnb += 1
        nnbs.append(nnb)
    #print(nnbs)
    # TODO:: Make thsi faster/inplace\
    # Reweight graph based on connectivity heuristic
    new_hm = np.zeros((n, n))
    collusions = np.zeros(n);
    # Find number of collusions per node
    for i in range(n):
        collusions[i] = np.sum(hit_matrix[i])
    # Reweight based on differences in sum
    for i in range(n):
        for j in range(n):
            # add 1 for divby0 and so that poisoners don't benefit from this
            new_hm[i][j] = hit_matrix[i][j] / (np.abs(collusions[i] - collusions[j]) + 1)

    # Take the inverse L2 norm
    wv = 1.0 / (np.linalg.norm(new_hm, axis=1) + 1)

    # Normalize to have sum equal to number of clients
    #wv = wv * n / np.linalg.norm(wv)

    # Apply the weights
    full_grad += np.dot(deltas.T, wv)

    global it
    it += 1

    return full_grad, distance, nnbs

def euclidean_binning(full_deltas, distance):
    deltas = np.reshape(full_deltas, (n, d))
    centered_deltas = (deltas - np.mean(deltas, axis=0))

    full_grad = np.zeros(d)
    nnbs = []

    for i in range(n):
        nnb = 1
        # Count nearby gradients within threshhold
        for j in range(n):
            # print(np.linalg.norm(centered_deltas[i] - centered_deltas[j]))
            if i != j and np.linalg.norm(centered_deltas[i] - centered_deltas[j]) < distance:
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
