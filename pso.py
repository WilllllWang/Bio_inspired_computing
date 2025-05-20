# Implementation of Adaptive Weight Particle Swarm Optimization
import numpy as np


def AWPSO(lu, iterMax, FOBJ, target):
    # Parameters
    nPop = 50
    wMax = 0.9
    wMin = 0.4
    D = lu.shape[1]
    VarMin = lu[0]
    VarMax = lu[1]
    Vmax = 0.2 * (VarMax - VarMin)
    Vmin = -Vmax
    searchSpace = VarMax[0] - VarMin[0]
    a = 0.000035 * searchSpace  # steepness parameter for sigmoid function

    # Initialize particle positions and velocities
    positions = VarMin + np.random.rand(nPop, D) * (VarMax - VarMin)
    velocities = Vmin + np.random.rand(nPop, D) * (Vmax - Vmin)

    # Initialize local bests
    localBestParams = positions.copy()
    localBest = np.array([FOBJ(p) for p in positions])

    # Initialize global best
    bestIdx = np.argmin(localBest)
    globalBestParams = localBestParams[bestIdx].copy()
    globalBest = localBest[bestIdx]
    globalBestPerIter = np.zeros(iterMax)
    optimumIter = -1

    # Main loop
    for iter in range(iterMax):
        r1 = np.random.rand(nPop, D)
        r2 = np.random.rand(nPop, D)

        # Differences to pbest and gbest
        diff_pbest = localBestParams - positions
        diff_gbest = globalBestParams - positions

        # Euclidean distances as paper stated
        gpi = np.linalg.norm(diff_pbest, axis=1, keepdims=True)
        ggi = np.linalg.norm(diff_gbest, axis=1, keepdims=True)

        # Acceleration coefficients (sigmoid-based)
        cgpi = F(gpi, a)  
        cggi = F(ggi, a)

        # Velocity update
        w = wMax - ((wMax - wMin) * iter / iterMax)
        velocities = w * velocities + cgpi * r1 * diff_pbest + cggi * r2 * diff_gbest
        velocities = np.clip(velocities, Vmin, Vmax)

        # Position update
        positions += velocities
        positions = np.clip(positions, VarMin, VarMax)

        # Evaluate fitness
        newFitness = np.array([FOBJ(p) for p in positions])

        # Update personal bests
        betterBool = newFitness < localBest
        localBestParams[betterBool] = positions[betterBool]
        localBest[betterBool] = newFitness[betterBool]

        # Update global best
        bestIdx = np.argmin(localBest)
        if localBest[bestIdx] < globalBest:
            globalBest = localBest[bestIdx]
            globalBestParams = localBestParams[bestIdx].copy()
            

        globalBestPerIter[iter] = globalBest
        if globalBest == target and optimumIter == -1: 
            optimumIter = iter
            break

    return globalBest, globalBestParams, globalBestPerIter, optimumIter


def F(distance, a):
    b = 0.5
    c = 0
    d = 1.5
    return b / (1 + np.exp(-a * (distance - c))) + d


def FOBJ(X):
    return np.sum(X**2)


if __name__ == "__main__":
    d = 6               
    lu = np.zeros((2, d))   
    lu[0, :] = -1       
    lu[1, :] = 1
    iterMax = 10
    target = 0.0
    AWPSO(lu, iterMax, FOBJ, target)



