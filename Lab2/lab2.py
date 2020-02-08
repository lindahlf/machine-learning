import numpy as np
import random 
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def generateData():
    "Generates random data"
    "Returns array of samples (inputs) and corresponding class labels (targets)"
    np.random.seed(100)

    classA = np.concatenate(
        (np.random.randn(10,2) * 0.2 + [1.5, 0.5],
        np.random.randn(10,2) * 0.2 + [-1.5, 0.5])) 
    classB = np.random.randn(20,2) * 0.2 + [0.0, -0.5]

    inputs = np.concatenate((classA, classB))
    targets = np.concatenate(
        (np.ones(classA.shape[0]),
        -np.ones(classB.shape[0]))
    ) 

    N = inputs.shape[0] # Number of rows (samples)

    # Randomly reorders samples
    permute = list(range(N))
    np.random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]
    return inputs, targets, N

def createP(x,t):
    "Creates the P matrix"   
    P = np.outer(t,t)
    for i in range(N):
        for j in range(N):
            P[i,j] *= kernel(t[i],t[j])
    return P

def kernel(x1, x2):
    "Performs calculation defined by given kernel of columns x1 and x2. Returns a scalar"
    return np.dot(x1,x2)

def objective(alpha):
    firstTerm = 0
    for i in range(N):
        for j in range(N):
            firstTerm += alpha[i]*alpha[j]*P[i,j]
    return 0.5*firstTerm - np.sum(alpha)

inputs, targets, N = generateData()
P = createP(inputs,targets)
ret = minimize(objective)


x = np.array([1, 2, 3])
y = np.array([1, 2, 3])
