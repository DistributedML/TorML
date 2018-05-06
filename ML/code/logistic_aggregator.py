from __future__ import division
import numpy as np
import pdb
import falconn
import sklearn.metrics.pairwise as smp

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
            new_hm[i][j] = hit_matrix[i][j] / (np.abs(np.linalg.norm(collusions[i]) - np.linalg.norm(collusions[j])) + 1)


    # Take the inverse L2 norm
    # for i in range(n):
    #     for j in range(n):
    #         new_hm[i][j] = max(new_hm[i][j] - (it/5), 0)
    wv = 1.0 / (np.linalg.norm(new_hm, axis=1) + 1)

    wv = (np.log(wv / (1 - wv)) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0

    # Apply the weights
    full_grad += np.dot(deltas.T, wv)
    return full_grad, heur_distance, nnbs

def get_nnbs_lsh2(full_deltas, distance):
    deltas = np.reshape(full_deltas, (n, d))
    centred_deltas = (deltas - np.mean(deltas, axis=0))

    params = falconn.get_default_parameters(n, d)
    fln = falconn.LSHIndex(params)
    fln.setup(centred_deltas)
    qob = fln.construct_query_object()

    nnbs = []
    graph = np.zeros((n,n))

    for i in range(n):
        neighbors = qob.find_near_neighbors(centred_deltas[i], distance)
        graph[i][neighbors] += 1
        nnbs.append(len(neighbors))
    graph = graph - np.eye(n)
    return nnbs, graph

def search_distance_lsh2(full_deltas, distance, typical_set, prev, poisoned, last_distance):
    nnbs, graph = get_nnbs_lsh2(full_deltas, distance)
    #first run
    if len(prev) == 0:
        return search_distance_lsh2(full_deltas, distance/2, typical_set, nnbs, poisoned, distance)

    if distance <= np.finfo(float).eps:
        return last_distance, poisoned
    #Keep halving till you reach the minimum value of search space: [1 1 ... 1]
    if not(typical_set):
        if np.sum(nnbs) != len(nnbs):
            return search_distance_lsh2(full_deltas, distance/2, typical_set, nnbs, poisoned, distance)
        else:
            return search_distance_lsh2(full_deltas, distance, True, nnbs, poisoned, distance)

    #### Found largest distance s.t nnbs = [1... 1] ####

    #if distances make all nodes overlap, return the last_distance
    if not(1 in nnbs):
       return last_distance, poisoned

    new_poisoned = np.array(poisoned)
    #while there are solo nodes left...
    if 0 in poisoned:
        for i in range(n):
            if prev[i] == 1 and poisoned[i] == 0:
                if nnbs[i] != 1:
                    for j in range(n):
                        if graph[i][j] == 1:
                            if poisoned[j] == 2:
                                new_poisoned[i] = 1
                                break
                            elif poisoned[j] == 0:
                                new_poisoned[i] = 2
                                last_distance = distance
                                break
                            else:
                                new_poisoned[i] = 1
                                break
        return search_distance_lsh2(full_deltas, distance*2, typical_set, nnbs, new_poisoned, last_distance)
    else:
        return last_distance, poisoned




#poisoned[i]: 0 for undefined, 1 for checked off good, 2 for poison
def search_distance_euc2(full_deltas, distance, typical_set, prev, poisoned, last_distance):

    nnbs, graph = get_nnbs_euc_cos(full_deltas, distance)
    #first run
    if len(prev) == 0:
        return search_distance_euc2(full_deltas, distance/2, typical_set, nnbs, poisoned, distance)
    if distance <= np.finfo(float).eps:
        return last_distance, poisoned
    #Keep halving till you reach the minimum value of search space: [1 1 ... 1]
    if not(typical_set):
        if np.sum(nnbs) != len(nnbs):
            return search_distance_euc2(full_deltas, distance/2, typical_set, nnbs, poisoned, distance)
        else:
            return search_distance_euc2(full_deltas, distance, True, nnbs, poisoned, distance)

    #### Found largest distance s.t nnbs = [1... 1] ####

    #if distances make all nodes overlap, return the last_distance
    if not(1 in nnbs):
       return last_distance, poisoned

    new_poisoned = np.array(poisoned)
    #while there are solo nodes left...
    if 0 in poisoned:
        for i in range(n):
            if prev[i] == 1 and poisoned[i] == 0:
                if nnbs[i] != 1:
                    for j in range(n):
                        if graph[i][j] == 1:
                            if poisoned[j] == 2:
                                new_poisoned[i] = 1
                                break
                            else:
                                new_poisoned[i] = 2
                                last_distance = distance
                                break
        return search_distance_euc2(full_deltas, distance*2, typical_set, nnbs, new_poisoned, last_distance)
    else:
        return last_distance, poisoned


def euclidean_binning_hm(full_deltas, distance):
    global hit_matrix
    deltas = np.reshape(full_deltas, (n, d))
    centered_deltas = (deltas - np.mean(deltas, axis=0))
    full_grad = np.zeros(d)

    nnbs = []
    graph = (smp.cosine_similarity(centered_deltas) >= (1-distance)).astype(int)
    nnbs = np.sum(graph, axis=1)
    graph -= np.eye(n)
    hit_matrix += graph

    new_hm = np.zeros((n, n))
    collusions = np.zeros(n);
    # Find number of collusions per node
    for i in range(n):
        collusions[i] = np.sum(hit_matrix[i])
    # Reweight based on differences in sum
    for i in range(n):
        for j in range(n):
            # add 1 for divby0 and so that poisoners don't benefit from this
            if (collusions[j] > collusions[i]):
                new_hm[i][j] = hit_matrix[i][j] / (np.abs(collusions[i] - collusions[j]) + 1)

    global it
    it += 1
    # Take the inverse L2 norm
    wv = 1 / (np.linalg.norm(new_hm, axis=1) + 1)

    wv = (np.log(wv / (1 - wv)) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0


    # Apply the weights
    full_grad += np.dot(deltas.T, wv)

    return full_grad, distance, nnbs

def get_nnbs_euc_cos(full_deltas, distance):
    deltas = np.reshape(full_deltas, (n, d))
    centered_deltas = (deltas - np.mean(deltas, axis=0))

    nnbs = []
    graph = (smp.cosine_similarity(centered_deltas) >= (1-distance)).astype(int)
    nnbs = np.sum(graph, axis=1)
    return nnbs, graph

def get_nnbs_euc2(full_deltas, distance):
    deltas = np.reshape(full_deltas, (n, d))
    centered_deltas = (deltas - np.mean(deltas, axis=0))

    nnbs = []
    graph = np.zeros((n, n))
    for i in range(n):
        nnb = 1
        # Count nearby gradients within threshhold
        for j in range(n):
            # print(np.linalg.norm(centered_deltas[i] - centered_deltas[j]))
            if i != j and np.linalg.norm(centered_deltas[i] - centered_deltas[j]) < distance:
                nnb += 1
                graph[i][j] += 1
        nnbs.append(nnb)
    return nnbs, graph


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


def average(full_deltas, d, n):

    deltas = np.reshape(full_deltas, (n, d))
    return np.mean(deltas, axis=0), 0


# Returns the index of the row that should be used in Krum
def krum(full_deltas, dd, groupsize):

    # assume deltas is an array of size group * d
    deltas = np.reshape(full_deltas, (groupsize, dd))
    best_idx = np.argmin(get_krum_scores(deltas, groupsize))
    return deltas[best_idx], 0


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
