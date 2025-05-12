# Implementation of Micro Variation Chaotic Genetic Algorithm
import numpy as np


def MVCGA(lu, iterMax, FOBJ):
    # Parameters
    nPop = 50
    Pc = 0.8                    # Crossover probability
    gamma = 0.3                 # Blend crossover range
    Pm = 0.15                   # Standard mutation probability
    microPm = 0.001             # Micro variation probability
    mu = 0.2                    # Mutation rate
    degradation = 0.2           # Micro variation blending factor
    alpha = np.random.rand()    # Initial value for chaotic map

    # Search range
    D = lu.shape[1]
    VarMin = lu[0]
    VarMax = lu[1]

    # Initialize population
    pop = VarMin + np.random.rand(nPop, D) * (VarMax - VarMin)
    costs = np.array([FOBJ(g) for g in pop])

    # Best solution
    bestIdx = np.argmin(costs)
    globalBestParams = pop[bestIdx].copy()
    globalBest = costs[bestIdx]
    globalBestPerIter = np.zeros(iterMax)
    optimumIter = -1

    for iter in range(iterMax):
        offspringPop = []

        for _ in range(nPop // 2):
            # Select and micro-mutate parents
            idx1 = roulette_wheel_selection(costs)
            idx2 = roulette_wheel_selection(costs)

            parent1 = chaotic_micro_variation(pop[idx1].copy(), VarMin, VarMax, microPm, degradation, alpha)
            parent2 = chaotic_micro_variation(pop[idx2].copy(), VarMin, VarMax, microPm, degradation, alpha)

            # Crossover
            if np.random.rand() < Pc:
                crossAlpha = np.random.uniform(-gamma, 1 + gamma, D)
                child1 = crossAlpha * parent1 + (1 - crossAlpha) * parent2
                child2 = crossAlpha * parent2 + (1 - crossAlpha) * parent1
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            offspringPop.extend([child1, child2])

        # Update chaotic map value (logistic map)
        alpha = 4 * alpha * (1 - alpha)

        # Mutation
        for i in range(nPop):
            if np.random.rand() < Pm:
                mutation = mu * (VarMax - VarMin) * np.random.randn(D)
                offspringPop[i] += mutation
                offspringPop[i] = np.clip(offspringPop[i], VarMin, VarMax)

        # Evaluate
        pop = np.array(offspringPop)
        costs = np.array([FOBJ(g) for g in pop])

        # Update best
        bestIdx = np.argmin(costs)
        if costs[bestIdx] < globalBest:
            globalBest = costs[bestIdx]
            globalBestParams = pop[bestIdx].copy()

        globalBestPerIter[iter] = globalBest
        if globalBest == 0.0 and optimumIter == -1:
            optimumIter = iter

    return globalBest, globalBestParams, globalBestPerIter, optimumIter


def roulette_wheel_selection(fitness):
    fitnessINV = 1 / (fitness + 1e-12)
    probs = fitnessINV / np.sum(fitnessINV)
    return np.random.choice(len(fitness), p=probs)


def chaotic_micro_variation(individual, VarMin, VarMax, pm=0.001, degradation=0.2, alpha=None):
    if np.random.rand() < pm:
        if alpha is None:
            alpha = np.random.rand()
        mutaFactor = VarMin + (VarMax - VarMin) * alpha
        individual = (1 - degradation) * individual + degradation * mutaFactor
        individual = np.clip(individual, VarMin, VarMax)
    return individual


def FOBJ(X):
    return np.sum(X**2)


if __name__ == "__main__":
    d = 3
    lu = np.zeros((2, d))
    lu[0, :] = -1
    lu[1, :] = 1
    iterMax = 200
    result = MVCGA(lu, iterMax, FOBJ)
    print("Best:", result[0], "at iteration:", result[3])
