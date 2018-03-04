import numpy as np
from labfuns import *
from lab3 import *

def assignment1():
    X, lables = genBlobs()
    mu, sigma = mlParams(X, lables)
    plotGaussian(X, lables, mu, sigma)

def assignment2():
    X, lables = genBlobs()
    mu, sigma = mlParams(X, lables)
    priors = computePrior(lables)
    classifyBayes(X, priors, mu, sigma)

def assignment3():
    # (1)
    print("Färdig")

def assignment4():
    X, lables = genBlobs()
    W = np.ones((X.shape[0], 1))*(1/X.shape[0])
    muOrig, sigmaOrig = mlParams(X, lables)
    muW, sigmaW = mlParams(X, lables, W)
    print(muOrig)
    print(sigmaOrig)
    print(muW)
    print(sigmaW)

def assignment5():
    X, lables = genBlobs()
    W = np.ones((X.shape[0], 1)) * np.random.uniform(0, 5, ((X.shape[0], 1)))
    priors = computePrior(lables, W)
    print(np.sum(priors, 0))





#assignment1()
#assignment2()
#assignment4()
assignment5()