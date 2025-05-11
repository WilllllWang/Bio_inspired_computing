import time
import numpy as np
from matplotlib import pyplot as plt
from pso import AWPSO
from coa import COA


def main():
    # Algorithm
    # exper(AWPSO)
    exper(COA)



def exper(META):
    # Objective function definition
    FOBJ = rastrigin        # Function
    d = 6                   # Problem dimension
    lu = np.zeros((2, d))   # Boundaires
    lu[0, :] = -5.12        # Lower boundaires
    lu[1, :] = 5.12         # Upper boundaries

    # Experimanetal variables
    iterMax = 5000          # Number of iterations
    nExp = 30               # Number of experiments
    t = time.time()         # Time counter (and initial value)
    gBests = np.zeros(nExp) # Experiments costs (for stats.)
    opIts = np.full(nExp, iterMax)
    success = 0
    for i in range(nExp):
        globalBest, globalBestParams, globalBestPerIter, optimumIter = META(lu, iterMax, FOBJ)
        gBests[i] = globalBest
        if optimumIter > -1: opIts[i] = optimumIter
        if gBests[i] < 1e-6: success += 1
        print("Experiment ", i+1, ", Best: ", globalBest, ", Iteration when optimum found: ", opIts[i], ", time (s): ", time.time()-t)
        t = time.time()

        plt.plot(globalBestPerIter)
        plt.xlabel("Iteration")
        plt.ylabel("Average Global Best")
        plt.title("Convergence Curve")
        plt.grid(True)
        plt.show(block=False)
        plt.pause(2)
        plt.close()

    
    # Show the statistics
    print(f"\nSuccess rate = {success} / {nExp} = {success / nExp} %\n")
    print("Iteration when optimum found: \n(min., avg., median, max., std.)")
    print([np.min(opIts), np.mean(opIts), np.median(opIts), np.max(opIts), np.std(opIts)])
    print("\nGlobal optimum: \n(min., avg., median, max., std.)")
    print([np.min(gBests), np.mean(gBests), np.median(gBests), np.max(gBests), np.std(gBests)])

    return gBests, opIts



def sphere(X):
    y = np.sum(X**2)
    return y


def rastrigin(X):
    return 10 * len(X) + np.sum(X**2 - 10 * np.cos(2 * np.pi * X))


def schwefel_226(X):
    return 418.9829 * len(X) - np.sum(X * np.sin(np.sqrt(np.abs(X))))

if __name__ == "__main__":
    main()