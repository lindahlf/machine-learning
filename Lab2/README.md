# Assignment 2: answers

#### Relate the parameters of the non-linear kernels to the size of the decision boundary and bias-variance trade-off. 

##### The radial basis function kernel: 
With a larger value of sigma, the decision boundary becomes larger and vice versa, exemplified by the two figures below. 

Sigma = 0.01             |  Sigma = 2
:-------------------------:|:-------------------------:
![](https://github.com/lindahlf/machine-learning/blob/master/Lab2/svmplot_sigma001.png)  |  ![](https://github.com/lindahlf/machine-learning/blob/master/Lab2/svmplot_sigma2.png)

The larger the decision boundary is, the scope of data points that we can fit into the red class expands, giving us a large **variance** as shown when sigma = 2. By contrast, when sigma = 0.01 
