import numpy as np




def GA():
    print("Hello")




def FOBJ(X):
    return np.sum(X**2)


if __name__ == "__main__":
    d = 6               
    lu = np.zeros((2, d))   
    lu[0, :] = -1       
    lu[1, :] = 1
    iterMax = 1000
    arg = GA(lu, iterMax, FOBJ)
    print(arg[3])