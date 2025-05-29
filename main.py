import time
import numpy as np
from numba import njit
from matplotlib import pyplot as plt
from pso import AWPSO
from coa import COA
from ga import MVCGA
from wo import WO


def main():
    # Algorithm
    # exper(MVCGA)
    # exper(AWPSO)
    # exper(COA)
    exper(WO)


def exper(META):
    d = 6                           # Problem dimension

    FOBJ = rastrigin                # Function
    lu = np.zeros((2, d))           # Boundaries
    lu[0, :] = -5.12                # Lower boundaries
    lu[1, :] = 5.12                 # Upper boundaries
    target = 0.0                    # Global Optimum

    # FOBJ = schwefel_226             # Function
    # lu = np.zeros((2, d))           # Boundaries
    # lu[0, :] = -500                 # Lower boundaries
    # lu[1, :] = 500                  # Upper boundaries
    # target = 0.0                    # Global Optimum

    # Experimental variables
    iterMax = 5000                  # Number of iterations
    nExp = 30                       # Number of experiments
    t = time.time()                 # Time counter (and initial value)
    gBests = np.zeros(nExp)         # Experiments costs (for stats.)
    opIts = np.full(nExp, iterMax)
    allGlobalBestPerIter = []       # Store convergence curves for all experiments
    success = 0

    for i in range(nExp):
        res = META(lu, iterMax, FOBJ, target)
        globalBest = res[0]
        globalBestParams = res[1]
        globalBestPerIter = res[2]
        optimumIter = res[3] 
        gBests[i] = globalBest
        if optimumIter > -1:
            opIts[i] = optimumIter
        else:
            opIts[i] = iterMax
        allGlobalBestPerIter.append(globalBestPerIter)
        if gBests[i] < 1e-6:
            success += 1

        print(f"Experiment {i+1}, Best: {globalBest}, Iteration when optimum found: {opIts[i]}, time (s): {time.time()-t:.2f}")
        t = time.time()

        # Optional
        fig = plt.figure()
        manager = plt.get_current_fig_manager()
        manager.window.state('zoomed')
        
        plt.plot(globalBestPerIter)
        plt.xlabel("Iteration")
        plt.ylabel("Global Best")
        plt.title(f"Global best = {globalBest} at iteration {opIts[i]}")
        plt.grid(True)

        


    # # Select best 
    # min_gBest = np.min(gBests)
    # candidates = np.where(gBests == min_gBest)[0]
    # best_index = candidates[np.argmin(opIts[candidates])]

    # # Save best
    # plt.plot(allGlobalBestPerIter[best_index])
    # plt.xlabel("Iteration")
    # plt.ylabel("Global Best")
    # plt.grid(True)
    # plt.savefig(f"COA_schwefel226.png")  # Change it according to function and algorithm
    # plt.close()

    # Show the statistics
    print(f"\nSuccess rate = {success} / {nExp} = {success / nExp * 100:.2f} %\n")
    print("Iteration when optimum found: \n(min., avg., median, max., std.)")
    print([np.min(opIts), np.mean(opIts), np.median(opIts), np.max(opIts), np.std(opIts)])
    print("\nGlobal optimum: \n(min., avg., median, max., std.)")
    print([np.min(gBests), np.mean(gBests), np.median(gBests), np.max(gBests), np.std(gBests)])


# [-100, 100]
@njit
def sphere(X):
    y = np.sum(X**2)
    return y

# [-5.12, 5.12]
@njit
def rastrigin(X):
    return 10 * len(X) + np.sum(X**2 - 10 * np.cos(2 * np.pi * X))

# [-500, 500]
@njit
def schwefel_226(X):
    return 418.9829 * len(X) - np.sum(X * np.sin(np.sqrt(np.abs(X))))


if __name__ == "__main__":
    main()
