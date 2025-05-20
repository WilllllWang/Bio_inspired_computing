# Implementation of Micro Variation Chaotic Genetic Algorithm with Paper-Based Adaptive Crossover
import numpy as np
from numba import njit

def MVCGA(lu, iterMax, FOBJ, target):
    # Parameters
    nPop = 70
    pcMin = 0.5                 # Crossover probability lower bound
    pcMax = 0.8                 # Crossover probability upper bound
    pmMin = 0.001               # Mutation probability lower bound
    pmMax = 0.05                # Mutation probability upper bound
    microPm = 0.001             # Micro variation probability
    degradation = 0.2           # Micro variation blending factor
    alpha = np.random.rand()    # Initial value for chaotic map

    # Search range
    d = lu.shape[1]
    varMin = lu[0]
    varMax = lu[1]

    # Initialize 
    pop = varMin + np.random.rand(nPop, d) * (varMax - varMin)
    costs = np.array([FOBJ(g) for g in pop])

    # Global best
    bestIdx = np.argmin(costs)
    globalBestParams = pop[bestIdx].copy()
    globalBest = costs[bestIdx]
    globalBestPerIter = np.zeros(iterMax)
    optimumIter = -1

    for iter in range(iterMax):
        # Update chaotic map 
        alpha = 4 * alpha * (1 - alpha)

        offspringPop = []
        for _ in range(nPop // 2):
            idx1 = rouletteWheelSelection(costs)
            idx2 = rouletteWheelSelection(costs)

            parent1 = chaoticMicroVariation(pop[idx1].copy(), varMin, varMax, microPm, degradation, alpha)
            parent2 = chaoticMicroVariation(pop[idx2].copy(), varMin, varMax, microPm, degradation, alpha)

            f1 = FOBJ(parent1)
            f2 = FOBJ(parent2)
            fPair = max(f1, f2)
            fMax = np.max(costs)
            fAvgPop = np.mean(costs)  

            # Crossover
            if fPair < fAvgPop:
                pcPair = pcMax
            else:
                sigmoid = 1 / (1 + safe_exp(-iter) + 1e-12) 
                numerator = fPair - fAvgPop
                denominator = fMax - fAvgPop + 1e-12 
                pcPair = max(0, pcMin - ((pcMax - pcMin) / (sigmoid + 1e-12)) * (numerator / denominator))

            if np.random.rand() < pcPair:
                crossAlpha = np.random.rand()
                child1 = crossAlpha * parent1 + (1 - crossAlpha) * parent2
                child2 = crossAlpha * parent2 + (1 - crossAlpha) * parent1
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            offspringPop.extend([child1, child2])

        # Replace population and costs
        pop = np.array(offspringPop)
        costs = np.array([FOBJ(g) for g in pop])

        # Adaptive Mutation
        for i in range(nPop):
            fMax = np.max(costs)
            fAvgPop = np.mean(costs)
            fi = costs[i]

            if fi < fAvgPop:
                pmInd = pmMax
            else:
                numerator = fi - fAvgPop
                denominator = fMax - fAvgPop + 1e-12
                sigmoid = 1 / (1 + safe_exp(iter) + 1e-12)
                pmInd = max(0, pmMax - ((pmMax - pmMin) / (sigmoid + 1e-12)) * (numerator / denominator))

            if np.random.rand() < pmInd:
                mutation = (varMax - varMin) * (np.random.rand(d) - 0.5)
                pop[i] += mutation
                pop[i] = np.clip(pop[i], varMin, varMax)

        # Re-evaluate after mutation
        costs = np.array([FOBJ(g) for g in pop])

        # Chaos perturbation (apply to worse half)
        sortedIndices = np.argsort(costs)
        worseIndices = sortedIndices[nPop // 2:]

        for i in worseIndices:
            eta = np.random.rand()
            pop[i] = pop[i] * eta + alpha * (varMax - pop[i]) * (1 - eta)
            pop[i] = np.clip(pop[i], varMin, varMax)

        # Re-evaluate costs after chaos
        costs = np.array([FOBJ(g) for g in pop])

        # Update best
        bestIdx = np.argmin(costs)
        if costs[bestIdx] < globalBest:
            globalBest = costs[bestIdx]
            globalBestParams = pop[bestIdx].copy()

        globalBestPerIter[iter] = globalBest
        if globalBest == target and optimumIter == -1:
            optimumIter = iter
            break

    return globalBest, globalBestParams, globalBestPerIter, optimumIter


def rouletteWheelSelection(fitness):
    fitnessInv = 1 / (fitness + 1e-12)
    probs = fitnessInv / np.sum(fitnessInv)
    return np.random.choice(len(fitness), p=probs)


def chaoticMicroVariation(individual, varMin, varMax, pm=0.001, degradation=0.2, alpha=None):
    if np.random.rand() < pm:
        if alpha is None:
            alpha = np.random.rand()
        mutaFactor = varMin + (varMax - varMin) * alpha
        individual = (1 - degradation) * individual + degradation * mutaFactor
        individual = np.clip(individual, varMin, varMax)
    return individual


@njit
def safe_exp(x, clip_val=20.0):
    if x < -clip_val:
        x_clip = -clip_val
    elif x > clip_val:
        x_clip = clip_val
    else:
        x_clip = x
    return np.exp(x_clip)


def FOBJ(x):
    return np.sum(x**2)


if __name__ == "__main__":
    d = 3
    lu = np.zeros((2, d))
    lu[0, :] = -1
    lu[1, :] = 1
    iterMax = 5000
    target = 0.0
    result = MVCGA(lu, iterMax, FOBJ, target)
    print("Best:", result[0], "at iteration:", result[3])
