# Implementation of Coyote Optimization Algorithm
import numpy as np

def COA(lu, iterMax, FOBJ, target):
    # Optimization problem variables
    D = lu.shape[1]
    VarMin = lu[0]
    VarMax = lu[1]
    nPacks = 10     # Must be at least 2
    nCoy = 5        # Must be at least 3
    nPop = nPacks * nCoy

    # Probability of leaving a pack
    p_leave = 0.005 * (nCoy**2)
    Ps = 1 / D

    # Packs initialization (Eq. 2)
    coyotes = VarMin + np.random.rand(nPop, D) * (VarMax - VarMin)
    ages = np.zeros(nPop)
    packs = np.random.permutation(nPop).reshape(nPacks, nCoy)

    # Evaluate coyotes adaptation (Eq. 3)
    costs = np.array([FOBJ(c) for c in coyotes])

    # Output variables
    bestIdx = np.argmin(costs)
    globalBestParams = coyotes[bestIdx].copy()
    globalBest = costs[bestIdx]
    globalBestPerIter = np.zeros(iterMax)
    optimumIter = -1

    # Main loop
    for iter in range(iterMax):  
        # Execute the operations inside each pack
        for p in range(nPacks):
            # Get the coyotes that belong to each pack
            packP = packs[p]
            coyotesPackP = coyotes[packP]
            costsPackP = costs[packP]
            agesPackP = ages[packP]

            # Detect alphas according to the costs (Eq. 5)
            alphaIdx = np.argmin(costsPackP)
            alphaC = coyotesPackP[alphaIdx]

            # Compute the social tendency of the pack (Eq. 6)
            tendency = np.median(coyotesPackP, axis=0)
            for c in range(nCoy):
                rc1 = 0
                while rc1 == c:
                    rc1 = np.random.randint(0, nCoy)
                rc2 = 0
                while rc2 == c or rc2 == rc1:
                    rc2 = np.random.randint(0, nCoy)

                # Try to update the social condition according
                # to the alpha and the pack tendency(Eq. 12)
                newCoyote = coyotesPackP[c] + np.random.rand() * (alphaC - coyotesPackP[rc1]) + np.random.rand() * (tendency - coyotesPackP[rc2])

                # Keep the coyotes in the search space (optimization problem constraint)
                newCoyote = np.clip(newCoyote, VarMin, VarMax)

                # Evaluate the new social condition (Eq. 13)
                newCostC = FOBJ(newCoyote)

                # Adaptation (Eq. 14)
                if newCostC < costsPackP[c]:
                    costsPackP[c] = newCostC
                    coyotesPackP[c] = newCoyote

            # Birth of a new coyote from random parents (Eq. 7 and Alg. 1)
            parents = np.random.permutation(nCoy)[:2]
            pdr = np.random.permutation(D)
            p1 = np.zeros(D)
            p2 = np.zeros(D)
            p1[pdr[0]] = 1  # Guarantee 1 charac. per individual
            p2[pdr[1]] = 1  # Guarantee 1 charac. per individual
            r = np.random.rand(1, D-2)
            p1[pdr[2:]] = r < (1 - Ps) / 2
            p2[pdr[2:]] = r > (1 + Ps) / 2

            # Eventual noise
            n = np.logical_not(np.logical_or(p1, p2))

            # Generate the pup considering intrinsic and extrinsic influence
            pup = p1 * coyotesPackP[parents[0]] + p2 * coyotesPackP[parents[1]] + n * (VarMin + np.random.rand(D) * (VarMax - VarMin))

            # Verify if the pup will survive
            pupCost = FOBJ(pup)
            worst = np.flatnonzero(costsPackP > pupCost)
            if len(worst) > 0:
                older = np.argsort(agesPackP[worst])
                which = worst[older[::-1]]
                coyotesPackP[which[0]] = pup
                costsPackP[which[0]] = pupCost
                agesPackP[which[0]] = 0

            # Update the pack information
            coyotes[packs[p]] = coyotesPackP
            costs[packs[p]] = costsPackP
            ages[packs[p]] = agesPackP

        # A coyote can leave a pack and enter in another pack (Eq. 4)
        # Swap their respective pack index
        if np.random.rand() < p_leave:
            rp = np.random.permutation(nPacks)[:2]
            rc = [np.random.randint(0, nCoy), np.random.randint(0, nCoy)]
            tmp = packs[rp[0], rc[0]]
            packs[rp[0], rc[0]] = packs[rp[1], rc[1]]
            packs[rp[1], rc[1]] = tmp

        # Update coyotes ages
        ages += 1

        # Output variables (best alpha coyote among all alphas)
        bestIdx = np.argmin(costs)
        globalBest = costs[bestIdx]
        globalBestParams = coyotes[bestIdx]
        globalBestPerIter[iter] = globalBest
        if globalBest == target and optimumIter == -1: 
            optimumIter = iter
            break

    return globalBest, globalBestParams, globalBestPerIter, optimumIter


def FOBJ(X):
    return np.sum(X**2)


if __name__ == "__main__":
    d = 6               
    lu = np.zeros((2, d))   
    lu[0, :] = -1       
    lu[1, :] = 1
    iterMax = 1000
    target = 0.0
    arg = COA(lu, iterMax, FOBJ, target)
    print(arg[3])
