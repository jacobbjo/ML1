import numpy as np
from lab3py.labfuns import *
from lab3py.lab3 import *

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



#assignment1()
assignment2()