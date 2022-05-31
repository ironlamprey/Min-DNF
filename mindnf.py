import numpy as np

#Helper function to find argmax
def argmax(X, key=None):
    m = max(X)
    if not key==None:
        m = max(X, key=key)
    return [i for i in range(len(X)) if X[i] == m][-1]


def FindTerms(f, verbose=False):
    N = len(f)
    Nlog3 = int(N**np.log2(3))+1
    V = []
    t = [0 for _ in range(Nlog3)]
    #Initialise t and C for full terms of f
    for i in range(N):
        #Interpret bin(i) as ternary
        j = int(bin(i)[2:], 3)
        if f[i]:
            t[j] = 1
            V.append(j)
    #Now check all term of f:
    for i in range(Nlog3):
        #Shout out to numpy!
        T = list(np.base_repr(i,base=3))
        if "2" in T:
            #Find the most significant bit equal to 2
            T = list(T)
            for k in range(len(T)):
                if T[k] == "2":
                    break
            #Find x, where kth bit of T has been replaced by 0
            tmp = list(T)
            tmp[k] = "0"
            x = int("".join(tmp),3)
            #Find x, where kth bit of T has been replaced by 1
            tmp = list(T)
            tmp[k] = "1"
            y = int("".join(tmp), 3)
            if t[x] == 1 and t[y] == 1:
                t[i] = 1
                V.append(i)
    return V


def PrimeImplicants(f, verbose=False):
    V = FindTerms(f)
    if verbose:
        print("All terms:", V)
    N = len(f)
    Nlog3 = int(N**np.log2(3))+1
    PI = [0 for _ in range(Nlog3)]
    for v in V:
        PI[v] = 1
    for v in V[::-1]:
            T = list(np.base_repr(v,base=3).zfill(int(np.log(N))+1))
            for k, b in enumerate(T):
                    if b == "2":
                        #Find x, where kth bit of T has been replaced by 0
                        tmp = list(T)
                        tmp[k] = "0"
                        x = int("".join(tmp),3)
                        PI[x] = 0
                        #Find x, where kth bit of T has been replaced by 1
                        tmp = list(T)
                        tmp[k] = "1"
                        y = int("".join(tmp), 3)
                        PI[y] = 0
    PI = [v for v in V if PI[v]]
    return PI



def GreedyMinDNF(f, verbose=False):
    N = len(f)
    f_domain = sum(f)

    #Find Prime implicants:
    V = PrimeImplicants(f, verbose)
    C = findOnes(V, f)
    m = len(max(C, key=lambda x: len(x)))
    #Map ones of f to terms covering them
    index_covering_ones = [[] for _ in range(N)]
    for i in V:
        for j in C[i]:
            index_covering_ones[j].append(i)
    sizes = [len(s) for s in C]
    A = [set() for i in range(max(sizes)+1)]
    for j in V:
        A[sizes[j]].add(j)
    m = len(A)-1
    
    if verbose:
        print("N =", N)
        print("N^log3 =", int(N**np.log2(3))+1)
        print("Prime Implicants:", V)
        print("Ones of f:", C)
        print("index_covering_ones:", index_covering_ones)
        print("sizes:", sizes)
        print("A:", A)
        print()
    #For each size, find which idx has this size
    TermsCovered = set()
    phi = []
    while len(TermsCovered) < f_domain:
        #TODO Remove Try-catch when it works again
        T  = A[m].pop()
        phi.append(T)
        L = [t for t in C[T] if not t in TermsCovered]
        TermsCovered.update(C[T])
        #for ones_covered in [ones for ones in ones_covered_by_index[maxT]]:
        for ones_covered in L:
            for k in index_covering_ones[ones_covered]:
                if k != T:
                    A[sizes[k]].remove(k)
                    sizes[k] -= 1
                    A[sizes[k]].add(k)
        #Make sure we do not pop from an empty set
        while not A[m] and m > 0:
            m -= 1
        if verbose:
            print("Term picked:", T)
            print("phi:", phi)
            print("Terms covered:", TermsCovered)
            print("sizes:", sizes)
            print("A:", A)
            print("m:", m)
            print()
    return phi

def findOnes(V, f):
    N = len(f)
    Nlog3 = int(N**np.log2(3))+1
    C = [[] for _ in range(Nlog3)]
    for v in V:
        T = list(np.base_repr(v,base=3))
        k = T.count("2")
        A = [i for i, x in enumerate(T) if x == "2"]
        for i in range(2**k):
            b = bin(i)[2:].zfill(k)
            x = list(T)
            for j in range(k):
                x[A[j]] = b[j]
            y = int("".join(x), 2)
            C[v].append(y)
    return C

#For pretty printing
def toString(phi, n):
    res = []
    for t in phi:
        T = np.base_repr(t, 3).zfill(n)
        res.append(tuple("-"*(x == "0") + "x" + str(i) for i, x in enumerate(T[::-1]) if int(x) < 2))
    return res

# Phi(f) = sum(|T| for T in Prime Implicants)
def Phi(f):
    V = PrimeImplicants(f)
    C = findOnes(V, f)
    return sum([len(c) for c in C])


def RandomizedMinDNF(f):
    N = len(f)
    V = set(PrimeImplicants(f))
    Nlog3 = int(N**np.log2(3))+1
    n = int(np.log2(N))
    s = 1
    while s < N:
        #Take a random sample of 1s of f not covered yet
        notCovered = set(i for i in range(N) if f[i])
        phi = set()
        while notCovered:
            I = np.random.choice(list(notCovered), min(s, len(notCovered)), replace=False)
            #Choose T that maximizes [i for i in f if T covers i]
            maxT = None 
            m = [0 for i in range(Nlog3)]
            for i in I:
                for j in range(N):
                    T = list(bin(i)[2:].zfill(n))
                    for k, bit in enumerate(bin(j)[2:].zfill(n)):
                        if bit == "1":
                            T[k] = "2"
                    T = int("".join(T), 3)
                    m[T] += 1
            maxT = None
            max_cover = 0
            for T in V:
                if m[T] > max_cover:
                    maxT = T
                    max_cover = m[T]
            phi.add(maxT)
            for i in I:
                if term_covers_bit(maxT, i, n):
                    notCovered.remove(i)
        if len(phi) <= s:
            break
        s*= 2
    return phi


def term_covers_bit(T, b, n):
    bi = bin(b)[2:].zfill(n)
    t = np.base_repr(T, 3).zfill(n)
    for i in range(n):
        if t[i] == "1" and bi[i] == "0":
            return False
        if t[i] == "0" and bi[i] == "1":
            return False 
    return True
