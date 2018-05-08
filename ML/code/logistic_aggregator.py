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




'''
Heuristic search to calculate the optimal distance
Decrease distance until all nodes are alone
Increase distance by factor of 2 and label nodes as:
    poisoner if the collusions are with non-poisoners
    non-poisoner if the collusions are with poisoners

poisoned[i]: 0 for undefined, 1 for checked off good, 2 for poison
'''
def search_distance_euc(full_deltas, distance, typical_set, prev, poisoned, last_distance, scs):

    nnbs, graph = get_nnbs_euc_cos(full_deltas, distance, scs)
    #first run

    if len(prev) == 0:
        return search_distance_euc(full_deltas, distance/2, typical_set, nnbs, poisoned, distance, scs)

    # Keep halving till you reach the minimum value of search space: [1 1 ... 1]
    if not(typical_set):

        if distance <= 0.00000001:
            idx = np.where(nnbs != 1)
            poisoned[idx] = 2
            return search_distance_euc(full_deltas, distance, True, nnbs, poisoned, distance, scs)

        if np.sum(nnbs) != len(nnbs):
            return search_distance_euc(full_deltas, distance/2, typical_set, nnbs, poisoned, distance, scs)
        else:
            return search_distance_euc(full_deltas, distance, True, nnbs, poisoned, distance, scs)



    #### Found largest distance s.t nnbs = [1... 1] ####

    # if distances make all nodes overlap, return the last_distance
    if distance >= 0.8:
        return last_distance, poisoned

    if not(1 in nnbs):
       return last_distance, poisoned

    # Jump size
    jump = np.var(smp.cosine_similarity(full_deltas)) * 0.1

    new_poisoned = np.array(poisoned)
    # while there are solo nodes left...
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

        return search_distance_euc(full_deltas, distance *2, typical_set, nnbs, new_poisoned, last_distance, scs)
    else:
        return last_distance, poisoned


'''
Given a distance, update the hit matrix using get_nnbs_
Calculate new edge weights based on the heuristic that benefits nodes that collude with poisoners often
Weight gradients based on logit
'''
def euclidean_binning_hm(full_deltas, distance, get_nnbs, scs):
    full_grad = np.zeros(d)
    deltas = np.reshape(full_deltas, (n, d))
    nnbs, graph = get_nnbs(full_deltas, distance, scs)
    graph = graph - np.eye(n)

    #Update global hit_matrix
    global hit_matrix
    hit_matrix += graph

    new_hm = np.zeros((n, n))
    collusions = np.zeros(n);
    # Find number of collusions per node
    for i in range(n):
        collusions[i] = np.sum(hit_matrix[i])
    # Reweight based on differences in sum
    # for i in range(n):
    #     for j in range(n):
    #         # add 1 for divby0 and so that poisoners don't benefit from this
    #         if (collusions[j] > collusions[i]):
    #             new_hm[i][j] = hit_matrix[i][j] / (np.abs(collusions[i] - collusions[j]) + 1)
    #         else:
    #             new_hm[i][j] = hit_matrix[i][j]

    global it
    it += 1

    # Take the inverse L2 norm
    wv = 1 / (np.linalg.norm(hit_matrix, axis=1) + 1)

    # Rescale so that max value is wv
    wv = wv / np.max(wv)

    # Logit function (map 0-1 space to inf)
    wv = (np.log(wv / (1 - wv)) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0

    # Sigmoid function for collusions
    # Rolling forward, only strike if total collusions / it > 2
    # tp = collusions / (it * n)
    # wv = 1.0 / (1 + np.exp(-10 * wv + 5))

    # Apply the weights
    full_grad += np.dot(deltas.T, wv)
    return full_grad, distance, nnbs

'''
Returns the hit matrix and nnbs from binning distance away in cosine similarity
'''
def get_nnbs_euc_cos(full_deltas, distance, scs):
    # deltas = np.reshape(full_deltas, (n, d))
    # centered_deltas = (deltas - np.mean(deltas, axis=0))

    nnbs = []
    graph = (scs >= (1-distance)).astype(int)
    nnbs = np.sum(graph, axis=1)
    return nnbs, graph

'''
Returns the hit matrix and nnbs from binning by euclidean distance
'''
def get_nnbs_euc(full_deltas, distance):
    deltas = np.reshape(full_deltas, (n, d))
    centered_deltas = (deltas - np.mean(deltas, axis=0))

    nnbs = []
    graph = np.zeros((n, n))
    for i in range(n):
        nnb = 1
        for j in range(n):
            if i != j and np.linalg.norm(centered_deltas[i] - centered_deltas[j]) < distance:
                nnb += 1
                graph[i][j] += 1
        nnbs.append(nnb)
    return nnbs, graph

'''
Return
'''
def get_cos_similarity(full_deltas):
    deltas = np.reshape(full_deltas, (n, d))
    centered_deltas = (deltas - np.mean(deltas, axis=0))
    if True in np.isnan(centered_deltas):
        pdb.set_trace()
    return smp.cosine_similarity(centered_deltas)


'''
Aggregates history of gradient directions
'''
def cos_aggregate_sum(full_deltas, sum_deltas, i):
    deltas = np.reshape(full_deltas, (n, d))
    full_grad = np.zeros(d)

    cs = smp.cosine_similarity(sum_deltas) - np.eye(n)
    maxcs = np.max(cs, axis=1)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i]/maxcs[j]
    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0


    full_grad += np.dot(deltas.T, wv)

    return full_grad

'''
Aggregates history of gradient directions
'''
def cos_aggregate_sum_norecalib(full_deltas, sum_deltas, i):
    deltas = np.reshape(full_deltas, (n, d))
    full_grad = np.zeros(d)

    cs = smp.cosine_similarity(sum_deltas) - np.eye(n)
    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0
    # Rescale so that max value is wv
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99
    wv = (np.log(wv / (1 - wv)) + 0.5)

    wv[(np.isinf(wv) + wv > 1)] = 1

    wv[(wv < 0)] = 0

    full_grad += np.dot(deltas.T, wv)

    return full_grad



'''
Aggregates history of gradient directions
'''
def cos_aggregate_sum(full_deltas, sum_deltas, i):
    deltas = np.reshape(full_deltas, (n, d))
    full_grad = np.zeros(d)

    cs = smp.cosine_similarity(sum_deltas) - np.eye(n)
    maxcs = np.max(cs, axis=1)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i]/maxcs[j]
    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0
    # Rescale so that max value is wv
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99
    wv = (np.log(wv / (1 - wv)) + 0.5)

    wv[(np.isinf(wv) + wv > 1)] = 1

    wv[(wv < 0)] = 0

    full_grad += np.dot(deltas.T, wv)

    return full_grad
'''
Aggregates history of gradient directions
'''
def cos_aggregate_sum_nomem(full_deltas):
    deltas = np.reshape(full_deltas, (n, d))
    full_grad = np.zeros(d)

    cs = smp.cosine_similarity(full_deltas) - np.eye(n)
    maxcs = np.max(cs, axis=1)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i]/maxcs[j]
    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0
    # Rescale so that max value is wv
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99
    wv = (np.log(wv / (1 - wv)) + 0.5)

    wv[(np.isinf(wv) + wv > 1)] = 1

    wv[(wv < 0)] = 0

    full_grad += np.dot(deltas.T, wv)

    return full_grad

'''
Aggregates history of cosine similarities
'''
def cos_aggregate(full_deltas, cs, i):
    if True in np.isnan(full_deltas):
        pdb.set_trace()
    deltas = np.reshape(full_deltas, (n, d))
    full_grad = np.zeros(d)

    ncs = cs - (i+1)*np.eye(n)
    wv = 1 - (np.max(ncs, axis=1) / (i+1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0
    # Rescale so that max value is wv
    wv = wv / np.max(wv)

    wv = (np.log(wv / (1 - wv)) + 0.5)

    wv[(np.isinf(wv) + wv > 1)] = 1

    wv[(wv < 0)] = 0

    full_grad += np.dot(deltas.T, wv)
    return full_grad

def cos_aggregate_nomem(full_deltas, scs):
    deltas = np.reshape(full_deltas, (n, d))
    full_grad = np.zeros(d)

    scs = scs - np.eye(n)
    wv = 1 - (np.max(scs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0
    # Rescale so that max value is wv
    wv = wv / np.max(wv)

    wv = (np.log(wv / (1 - wv)) + 0.5)

    wv[(np.isinf(wv) + wv > 1)] = 1

    wv[(wv < 0)] = 0

    full_grad += np.dot(deltas.T, wv)
    return full_grad

def average(full_deltas):

    deltas = np.reshape(full_deltas, (n, d))
    return np.mean(deltas, axis=0)


# Returns the index of the row that should be used in Krum
def krum(deltas, clip):

    # assume deltas is an array of size group * d
    scores = get_krum_scores(deltas, n - clip)

    good_idx = np.argpartition(scores, n - clip)[:(n - clip)]
    return np.mean(deltas[good_idx], axis=0)


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
