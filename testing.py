from more_itertools import only
import numpy as np
import matplotlib.pyplot as plt
from mindnf import GreedyMinDNF, PrimeImplicants, RandomizedMinDNF, findOnes, toString, Phi
from quinemccluskey import QMtoString, Quine_McCluskey
from generatetruthtables import DoubleParity, Or, ThreePhase, Threshhold, Maj, Sym, Ones, And
import time
import matplotlib as mpl


#Uncomment to generate TikZ code
#mpl.use("pgf")

OR = [0, 1, 1, 1]
print(toString(GreedyMinDNF(OR), 2))
print(toString(PrimeImplicants(OR), 2))
print(QMtoString(Quine_McCluskey(OR),2))
print(toString(RandomizedMinDNF(OR), 2))
print()

XOR = [0, 1, 1, 0]
print(toString(GreedyMinDNF(XOR), 2))
print(toString(PrimeImplicants(XOR), 2))
print(QMtoString(Quine_McCluskey(XOR),2))
print(toString(RandomizedMinDNF(XOR), 2))
print()

f = [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1]
print(toString(GreedyMinDNF(f), 4))
print(toString(PrimeImplicants(f), 4))
print(QMtoString(Quine_McCluskey(f),4))
print(toString(RandomizedMinDNF(f), 4))
print()

AND = [0, 0, 0, 0, 0, 0, 0, 1]
print(toString(GreedyMinDNF(AND), 3))
print(toString(PrimeImplicants(AND), 3))
print(QMtoString(Quine_McCluskey(AND),3))
print(toString(RandomizedMinDNF(AND), 3))
print()

x1 = [0, 0, 1, 1, 0, 0, 1, 1]
print(toString(GreedyMinDNF(x1), 3))
print(toString(PrimeImplicants(x1), 3))
print(QMtoString(Quine_McCluskey(x1),3))
print(toString(RandomizedMinDNF(x1), 3))
print()

MAJ = [0, 0, 0, 1, 0, 1, 1, 1]
print(toString(GreedyMinDNF(MAJ), 3))
print(toString(PrimeImplicants(MAJ), 3))
print(QMtoString(Quine_McCluskey(MAJ),3))
print(toString(RandomizedMinDNF(MAJ), 3))
print()

#The truth table from Wikipedias article on Quine-McCluskey
f = [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1]
print(toString(GreedyMinDNF(f), 4))
print(toString(PrimeImplicants(f), 4))
print(QMtoString(Quine_McCluskey(f),4))
print(toString(RandomizedMinDNF(f), 4))
print()


f = [1, 1, 1 ,1, 0, 1, 1, 1]
print(toString(GreedyMinDNF(f), 3))
print(toString(PrimeImplicants(f), 3))
print(QMtoString(Quine_McCluskey(f),3))
print(toString(RandomizedMinDNF(f), 3))
print()

print("here")
f = [1, 1, 1, 0, 0, 1, 1, 1]
print(toString(GreedyMinDNF(f), 3))
print(toString(PrimeImplicants(f), 3))
print(QMtoString(Quine_McCluskey(f),3))
print(toString(RandomizedMinDNF(f), 3))
print()


NAND = [1, 1, 1, 1, 1, 1, 1, 0]
print(toString(GreedyMinDNF(NAND), 3))
print(toString(PrimeImplicants(NAND), 3))
print(QMtoString(Quine_McCluskey(NAND),3))
print(toString(RandomizedMinDNF(NAND), 3))
print()

allTerms = [1, 1, 1, 1, 1, 1, 1, 1]
print(toString(GreedyMinDNF(allTerms), 3))
print(toString(PrimeImplicants(allTerms), 3))
print(QMtoString(Quine_McCluskey(allTerms),3))
print(toString(RandomizedMinDNF(allTerms), 3))
print()

DParity5 = [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0]
print(toString(GreedyMinDNF(DParity5), 5))
print(toString(PrimeImplicants(DParity5), 5))
print(QMtoString(Quine_McCluskey(DParity5),5))
print(toString(RandomizedMinDNF(DParity5), 5))
print()

nDParity = DoubleParity(11)
print(len(GreedyMinDNF(nDParity)))
print(len(PrimeImplicants(nDParity)))
print()

Savicky = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1]
print(toString(GreedyMinDNF(Savicky), 4))
print(toString(PrimeImplicants(Savicky), 4))
print(QMtoString(Quine_McCluskey(Savicky),4))
print(toString(RandomizedMinDNF(Savicky), 4))
print()




#Let's make a plot!
def benchmark(n, filename, title, qm=False, randomized=True, function=lambda x: x, log=False):
    x = np.array(range(1, n+1))
    y1 = np.zeros(n)
    y2 = np.zeros(n)
    y3 = np.zeros(n)
    y4 = np.zeros(n)
    for i in range(1, n+1):
        if randomized:
            f = [np.random.randint(0, 2) for _ in range(2**i)]
        else:
            f = function(i)
        if qm:
            t1 = time.time()
            PrimeImplicants(f)
            y2[i-1] = time.time() - t1
            t1 = time.time()
            Quine_McCluskey(f, only_generate_prime_implicants=True)
            y4[i-1] = time.time() - t1
        else: 
            t1 = time.time()
            GreedyMinDNF(f)
            y1[i-1] = time.time() - t1
            t1 = time.time()
            RandomizedMinDNF(f)
            y3[i-1] = time.time() - t1
    if qm:
        plt.plot(x, y4, "b-o", label="Quine-McCluskey")
        plt.plot(x, y2, "y-o", label="Prime Implicants")
    else:
        plt.plot(x, y1, "b-o",label="Greedy-MinDNF", )
        plt.plot(x, y3, "y-o", label="Randomized-MinDNF")
    plt.legend()
    if log:
        plt.yscale('log')
    plt.xlabel("Number of variables")
    plt.ylabel("Time in seconds")
    plt.title(title)
    #plt.savefig(filename+".png")

    #Uncomment to generate TikZ code
    #plt.savefig(filename+".pgf")
    #plt.close()
    #plt.show()

benchmark(6, "", "")