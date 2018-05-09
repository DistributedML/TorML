
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
Using locality sensitive hashing
'''
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

def get_nnbs_lsh(full_deltas, distance):
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

def search_distance_lsh(full_deltas, distance, typical_set, prev, poisoned, last_distance):
    nnbs, graph = get_nnbs_lsh(full_deltas, distance)
    #first run
    if len(prev) == 0:
        return search_distance_lsh(full_deltas, distance/2, typical_set, nnbs, poisoned, distance)

    if distance <= np.finfo(float).eps:
        return last_distance, poisoned
    #Keep halving till you reach the minimum value of search space: [1 1 ... 1]
    if not(typical_set):
        if np.sum(nnbs) != len(nnbs):
            return search_distance_lsh(full_deltas, distance/2, typical_set, nnbs, poisoned, distance)
        else:
            return search_distance_lsh(full_deltas, distance, True, nnbs, poisoned, distance)

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
        return search_distance_lsh(full_deltas, distance*2, typical_set, nnbs, new_poisoned, last_distance)
    else:
        return last_distance, poisoned




'''
Using std to guide search for optimal distance
Doesn't work well if the size of sybil sets is small
'''

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
