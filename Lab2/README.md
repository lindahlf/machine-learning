# Assignment 2: answers

#### Relate the parameters of the non-linear kernels to the size of the decision boundary and bias-variance trade-off. 

##### The radial basis function kernel: 
With a larger value of sigma, the decision boundary becomes larger and vice versa if we decrease the value, exemplified by the two figures below. 

Sigma = 0.01             |  Sigma = 2
:-------------------------:|:-------------------------:
![](https://github.com/lindahlf/machine-learning/blob/master/Lab2/svmplot_sigma001.png)  |  ![](https://github.com/lindahlf/machine-learning/blob/master/Lab2/svmplot_sigma2.png)

In the left image, we can see that the decision boundary is extremely tailored to fit the data, implying that it has overfitted the data. Thus, we find that the smaller the value of sigma, the higher the variance since the decision boundary then is more likely to overfit the data. By contrast, when sigma is large, the decision boundary is large and the data is relatively underfitted, implying a higher bias instead.  

##### The polynomial kernel: 

The higher the degree of the polynomial the higher the complexity of the decision boundary with respect to the data. Thus, higher the degree, the higher the variance. 
