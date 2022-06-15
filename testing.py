from more_itertools import only
import numpy as np
import matplotlib.pyplot as plt
from mindnf import GreedyMinDNF, PrimeImplicants, RandomizedMinDNF, findOnes, toString, Phi
from quinemccluskey import QMtoString, Quine_McCluskey
from generatetruthtables import DoubleParity, Or, ThreePhase, Threshhold, Maj, Sym, Ones, And, XORAdrr
import time
import matplotlib as mpl


#Uncomment to generate TikZ code
#mpl.use("pgf")

def test1(verbose=False):
    OR = [0, 1, 1, 1]
    print(toString(GreedyMinDNF(OR, verbose), 2))
    print(toString(PrimeImplicants(OR), 2))
    print(QMtoString(Quine_McCluskey(OR),2))
    print(toString(RandomizedMinDNF(OR), 2))
    print()

def test2(verbose=False):
    XOR = [0, 1, 1, 0]
    print(toString(GreedyMinDNF(XOR, verbose), 2))
    print(toString(PrimeImplicants(XOR), 2))
    print(QMtoString(Quine_McCluskey(XOR),2))
    print(toString(RandomizedMinDNF(XOR), 2))
    print()

def test3(verbose=False):
    f = [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1]
    print(toString(GreedyMinDNF(f, verbose), 4))
    print(toString(PrimeImplicants(f), 4))
    print(QMtoString(Quine_McCluskey(f),4))
    print(toString(RandomizedMinDNF(f), 4))
    print()

def test4(verbose=False):
    AND = [0, 0, 0, 0, 0, 0, 0, 1]
    print(toString(GreedyMinDNF(AND, verbose), 3))
    print(toString(PrimeImplicants(AND), 3))
    print(QMtoString(Quine_McCluskey(AND),3))
    print(toString(RandomizedMinDNF(AND), 3))
    print()

def test5(verbose=False):
    x1 = [0, 0, 1, 1, 0, 0, 1, 1]
    print(toString(GreedyMinDNF(x1, verbose), 3))
    print(toString(PrimeImplicants(x1), 3))
    print(QMtoString(Quine_McCluskey(x1),3))
    print(toString(RandomizedMinDNF(x1), 3))
    print()

def test6(verbose=False):
    MAJ = [0, 0, 0, 1, 0, 1, 1, 1]
    print(toString(GreedyMinDNF(MAJ, verbose), 3))
    print(toString(PrimeImplicants(MAJ), 3))
    print(QMtoString(Quine_McCluskey(MAJ),3))
    print(toString(RandomizedMinDNF(MAJ), 3))
    print()

#The truth table from Wikipedias article on Quine-McCluskey
def test7(verbose=False):
    f = [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1]
    print(toString(GreedyMinDNF(f, verbose), 4))
    print(toString(PrimeImplicants(f), 4))
    print(QMtoString(Quine_McCluskey(f),4))
    print(toString(RandomizedMinDNF(f), 4))
    print()

def test8(verbose=False):
    f = [1, 1, 1 ,1, 0, 1, 1, 1]
    print(toString(GreedyMinDNF(f, verbose), 3))
    print(toString(PrimeImplicants(f), 3))
    print(QMtoString(Quine_McCluskey(f),3))
    print(toString(RandomizedMinDNF(f), 3))
    print()

def test9(verbose=False):
    f = [1, 1, 1, 0, 0, 1, 1, 1]
    print(toString(GreedyMinDNF(f, verbose), 3))
    print(toString(PrimeImplicants(f), 3))
    print(QMtoString(Quine_McCluskey(f),3))
    print(toString(RandomizedMinDNF(f), 3))
    print()

def test10(verbose=False):
    NAND = [1, 1, 1, 1, 1, 1, 1, 0]
    print(toString(GreedyMinDNF(NAND, verbose), 3))
    print(toString(PrimeImplicants(NAND), 3))
    print(QMtoString(Quine_McCluskey(NAND),3))
    print(toString(RandomizedMinDNF(NAND), 3))
    print()

def test11(verbose=False):
    allTerms = [1, 1, 1, 1, 1, 1, 1, 1]
    print(toString(GreedyMinDNF(allTerms, verbose), 3))
    print(toString(PrimeImplicants(allTerms), 3))
    print(QMtoString(Quine_McCluskey(allTerms),3))
    print(toString(RandomizedMinDNF(allTerms), 3))
    print()

def test12(verbose=False):
    DParity5 = [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0]
    print(toString(GreedyMinDNF(DParity5, verbose), 5))
    print(toString(PrimeImplicants(DParity5), 5))
    print(QMtoString(Quine_McCluskey(DParity5),5))
    print(toString(RandomizedMinDNF(DParity5), 5))
    print()

def test13(verbose=False):
    nDParity = DoubleParity(11)
    print(len(GreedyMinDNF(nDParity, verbose)))
    print(len(PrimeImplicants(nDParity)))
    print()

def test14(verbose=False):
    Savicky = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1]
    print(toString(GreedyMinDNF(Savicky, verbose), 4))
    print(toString(PrimeImplicants(Savicky), 4))
    print(QMtoString(Quine_McCluskey(Savicky),4))
    print(toString(RandomizedMinDNF(Savicky), 4))
    print()

def testAll():
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()
    test7()
    test8()
    test9()
    test10()
    test11()
    test12()
    test13()
    test14()

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
    plt.savefig("plots/" +filename+".png")

    #Uncomment to generate TikZ code
    #plt.savefig(filename+".pgf")
    plt.show()

benchmark(3, "XORAddr", "XORAddr", qm=True, randomized=False, function=XORAdrr, log=True)

#testAll()