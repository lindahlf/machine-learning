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
            P[i,j] *= kernel(x[i],x[j])
    return P

def kernel(x1, x2):
    "Performs calculation defined by given kernel of columns x1 and x2. Returns a scalar"
    return np.dot(x1,x2)

def objective(alpha):
    "Objective function to be minimized"
    return 0.5*np.dot(np.dot(alpha,P),alpha) - np.sum(alpha)

def zerofun(alpha):
    "Optimization constraint"
    return np.dot(alpha,targets) 

inputs, targets, N = generateData()
P = createP(inputs,targets)
C = None
ret = minimize(objective, np.zeros(N), 
            bounds = [(0,C) for b in range(N)], constraints={'type':'eq', 'fun':zerofun})

a = ret['x']


y = np.array([1, 2, 3])

print(a)