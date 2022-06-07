import numpy as np
from itertools import chain, combinations

def powerset(X):
    return set(chain.from_iterable(combinations(X, r) for r in range(len(X)+1)))

def Quine_McCluskey(f, only_generate_prime_implicants=False):
    #inspired by https://github.com/int-main/Quine-McCluskey/blob/master/Quine%20McCluskey.py
    f_domain = [i for i, e in enumerate(f) if e]
    C = {}
    n, d = int(np.log2(len(f))), sum(f)
    groups = [[] for _ in range(n+1)]
    for i, minterm in enumerate(f_domain):
        as_bin = bin(minterm)[2:].zfill(n)
        groups[sum(map(int, as_bin))].append((as_bin))
        C[as_bin] = [i]

    canreduce = True
    while canreduce:
        canreduce = False
        newgroups = [g.copy() for g in groups]
        for i in range(n):
            for t1 in groups[i]:
                for t2 in groups[i+1]:
                    #Check if t1 and t2 differ by one bit
                    differs, k = 0, 0
                    for j in range(n):
                        if t1[j] != t2[j]:
                            differs += 1
                            k = j

                    #If they differ by one bit, combine them in newgroups
                    if differs == 1:
                        combine = t1[:k] + "2" + t2[k+1:]
                        if not combine in newgroups[i]:
                            newgroups[i].append(combine)
                            C[combine] = C[t1] + C[t2]
                        if t1 in newgroups[i]:
                            newgroups[i].remove(t1)
                        if t2 in newgroups[i+1]:
                            newgroups[i+1].remove(t2)
                        canreduce = True
        groups = [g.copy() for g in newgroups]
    V = [t for g in groups for t in g]

    #For benchmarking QM vs PrimeImplicants
    if only_generate_prime_implicants:
        return V

    #By now we cannot reduce the Prime Implicants further, so construct Table
    T = np.zeros((len(V), sum(f)))
    for i, pi in enumerate(V):
        for j in C[pi]:
            T[i, j] = 1
    res = set()
    covered = []
    for j in range(d):
        s = T[:, j].sum()
        if s == 1:
            for i,v  in enumerate(V):
                if T[i, j] == 1:
                    res.add(v)
                    for k in range(d):
                        if T[i, k]:
                            covered.append(f_domain[k])
    remaining = [v for v in V if not v in res]
    remaining_to_cover = set(e for e in f_domain if not e in covered)
    if len(remaining) == 1 and len(remaining_to_cover) > 0:
        res.add(remaining[0])
    elif len(remaining) > 1 and len(remaining_to_cover) > 0:
        #In this code, we brute force the optimal solution - to get an exact result
        #that we can compare to the other algorithms
        S = powerset(remaining) 
        dnfs = []
        for t in S:
            cov = set(f_domain[j] for v in t for j in C[v])
            if remaining_to_cover.issubset(cov):
                dnfs.append(t)
        for i in min(dnfs, key=len):
            res.add(i)
    return res


def QMtoString(phi, n):
    res = []
    for t in phi:
        res.append(tuple("-"*(x == "0") + "x" + str(i) for i, x in enumerate(t[::-1]) if x != "2"))
    return res

