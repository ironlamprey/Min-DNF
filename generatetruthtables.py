def DoubleParity(n):
    f = [0 for _ in range(2**n)]
    for i in range(2**n):
        d = list(map(int, bin(i)[2:]))
        for _ in range(n - len(d)):
            d.insert(0, 0)
        if (d[n-1] == 0 and sum(d[0:n//2]) % 2 == 1) or (d[n-1] == 1 and sum(d[n//2: n-1]) % 2 == 1):
            f[i] = 1
    return f

def Threshhold(n, t):
    f = [0 for _ in range(2**n)]
    for i in range(2**n):
        d = list(map(int, bin(i)[2:]))
        if sum(d) >= t:
            f[i] = 1
    return f

def Maj(n):
    return Threshhold(n, n//2)

def Sym(n):
    f = [1 for _ in range(2**n)]
    f[(2**n)//2-1] = 0
    f[(2**n)//2 ] = 0
    return f

def Ones(n):
    return [1 for i in range(2**n)]

def ThreePhase(n, a, b):
    f = [0 for i in range(2**n)]
    for i in range(2**n):
        d = bin(i)[2:].zfill(n)
        if d.count("1") >= a and d.count("0") >= b:
            f[i] = 1
    return f

def And(n):
    return [0 for i in range(2**n - 1)] + [1]

def Or(n):
    return [0] +  [1 for i in range(2**n - 1)]

DParity = [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0]
#print(DParity)
#print(DoubleParity(5))