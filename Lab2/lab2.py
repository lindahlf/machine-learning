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
        (np.random.randn(10, 2) * 0.2 + [1.5, 0.5],
         np.random.randn(10, 2) * 0.2 + [-1.5, 0.5]))
    classB = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]

    inputs = np.concatenate((classA, classB))
    targets = np.concatenate(
        (np.ones(classA.shape[0]),
         -np.ones(classB.shape[0]))
    )

    N = inputs.shape[0]  # Number of rows (samples)

    # Randomly reorders samples
    permute = list(range(N))
    np.random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]
    return inputs, targets, N, classA, classB


def createP(x, t):
    """Creates the P matrix"""
    P = np.outer(t, t)
    for i in range(N):
        for j in range(N):
            P[i, j] *= kernel(x[i], x[j])
    return P


def kernel(x1, x2):
    """Performs calculation defined by given kernel of columns x1 and x2. Returns a scalar"""
    return np.dot(x1, x2)


def objective(alpha):
    """Objective function to be minimized"""
    return 0.5 * np.dot(np.dot(alpha, P), alpha) - np.sum(alpha)


def zerofun(alpha):
    """Optimization constraint"""
    return np.dot(alpha, targets)


def indicator(x, y):
    """Indicator function"""
    s = np.array([x,y])
    return


inputs, targets, N, classA, classB = generateData()
P = createP(inputs, targets)
C = None
ret = minimize(objective, np.zeros(N),
               bounds=[(0, C) for b in range(N)], constraints={'type': 'eq', 'fun': zerofun})

a = ret['x']


#### Create contours for decision boundary
xgrid = np.linspace(-5, 5)
ygrid = np.linspace(-4, 4)



grid = np.array([[indicator(x, y)
                  for x in xgrid]
                 for y in ygrid])

plt.contour(xgrid, ygrid, grid,
            (-1, 0, 1),
            colors=('red', 'black', 'blue'),
            linewidths=(1, 3, 1))

#### Plotting the data points
plt.plot([p[0] for p in classA],
         [p[1] for p in classA],
         'b.')
plt.plot([p[0] for p in classB],
         [p[1] for p in classB],
         'r.')
plt.axis('equal')
plt.savefig('svmplot.pdf')
plt.show()
